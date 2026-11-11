/* bn_pentode_6550_own.c — C port of the project's OWN 6550 pentode (the US edition).
 *
 * EXACTLY the model of bn_pentode_own.c (EL34) with another set of constants:
 * Koren's eq. 5 form + a sigma split + exact current bookkeeping, without the
 * EL34's cutoff tail. The constants come from `pentode_6550_own.py --fit`
 * against the 6550 data sheet; they are never edited by hand.
 *
 * Fit residuals against the sheet: plate current 2.4 %, screen current 13.8 %.
 * MU2 is FIXED at the published value (8, the "triode amplification factor"):
 * left free, the fit's total cost fell but the plate current's error rose from
 * 2.4 % to 5.6 %.
 */
#include "bn_pentode_6550_own.h"

#include <math.h>

/* --- parameters FITTED to the 6550 data sheet (pentode_6550_own.py --fit) --- */
#define G_     0.0021930554
#define KP     59.775867
#define GAMMA  1.4523756
#define MU2    8.0
#define MUA    234.98748
#define SINF   0.97003224
#define KVB    14.583727
#define GMIN   1e-10
#define TAU    1.0

#define TWO_OVER_PI 0.63661977236758134308

/* --- CONTROL-GRID CURRENT
 * The same argument and law as the EL34's (bn_pentode_own.c): without it the
 * grid runs to tens of volts positive and the screen to ~76 W per valve against
 * a 6 W limit. ig = GG * softplus(vg1k, CG)^XI + IG0 (Dempwolf & Zölzer,
 * Table 1 of the paper, RSD-1); CG, XI and IG0 as they are (the SHAPE of the
 * conduction knee), GG scaled by THIS valve's cathode perveance (Child-Langmuir:
 * it goes with the emitting area).
 * ⚠️ Scaling GG is OUR INFERENCE: the 6550 data sheet publishes no grid
 * current either. */
#define GG_TRIODE  6.177e-4          /* Table 1 (RSD-1), 12AX7 */
#define G_TRIODE   2.11116986e-03    /* the project triode's cathode perveance */
#define CG         9.901
#define XI         1.314
#define IG0        8.025e-8
#define GG         (GG_TRIODE * (G_ / G_TRIODE))

static inline double softplus_c(double x, double c)
{
    const double z = c * x;
    if (z > 40.0) { return x; }
    return log1p(exp(z)) / c;
}

static inline double sigmoid_c(double x, double c)
{
    const double z = c * x;
    if (z > 40.0)  { return 1.0; }
    if (z < -40.0) { return 0.0; }
    return 1.0 / (1.0 + exp(-z));
}

/* stable log(1+exp(z)), and its derivative (the sigmoid) */
static inline double softplus_raw(double z)
{
    if (z > 40.0) { return z; }
    if (z < -40.0) { return exp(z); }
    return log1p(exp(z));
}

static inline double sigmoid_raw(double z)
{
    if (z > 40.0) { return 1.0; }
    if (z < -40.0) { return exp(z); }
    return 1.0 / (1.0 + exp(-z));
}

void bn_pentode_6550_own_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double e1 = (g2 / KP) * softplus_raw(z);
    const double ik = (e1 > 0.0) ? G_ * pow(e1, GAMMA) : 0.0;
    const double a = TAU * softplus_raw(va / TAU);
    const double s = SINF * TWO_OVER_PI * atan(a / KVB);

    /* the grid takes its share of the cathode current: Ik = Ia + Ig2 + Ig1 */
    const double hg = softplus_c(vg1, CG);
    const double ig1 = GG * pow(hg, XI) + IG0;
    const double ikr = ik - ig1;
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}

/* ANALYTIC Jacobian, branch by branch. J[r*3+c] = d i[r] / d v[c].
 * Checked against centred differences in tools/bn_test_pentode_own.c (a wrong
 * derivative at a knee is a silent error). */
void bn_pentode_6550_own(const double* v, double* i, double* J, void* user)
{
    (void)user;
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);          /* otherwise g2 is clamped */
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double L = softplus_raw(z);
    const double S = sigmoid_raw(z);
    const double e1 = (g2 / KP) * L;
    const double ik = (e1 > 0.0) ? G_ * pow(e1, GAMMA) : 0.0;
    /* dIk/dE1 = G*GAMMA*E1^(GAMMA-1) */
    const double dik_de1 = (e1 > 0.0) ? G_ * GAMMA * pow(e1, GAMMA - 1.0) : 0.0;

    /* dE1 with respect to each voltage */
    const double de1_dvg1 = S;
    const double de1_dva_ctrl = S / MUA;
    const double de1_dvg2 = g2_live ? (L / KP - S * w / g2) : 0.0;

    const double dik_dvg1 = dik_de1 * de1_dvg1;
    const double dik_dvg2 = dik_de1 * de1_dvg2;
    const double dik_dva = dik_de1 * de1_dva_ctrl;

    const double a = TAU * softplus_raw(va / TAU);
    const double da_dva = sigmoid_raw(va / TAU);
    const double t = a / KVB;
    const double s = SINF * TWO_OVER_PI * atan(t);
    const double ds_dva = SINF * TWO_OVER_PI * da_dva / (KVB * (1.0 + t * t));

    const double hg = softplus_c(vg1, CG);
    const double sg = sigmoid_c(vg1, CG);
    const double ig1 = GG * pow(hg, XI) + IG0;
    const double dig1 = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    const double ikr = ik - ig1;

    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;

    /* d(ikr)/dv = d(ik)/dv minus the grid's share (which depends on vg1 only) */
    const double dikr_dvg1 = dik_dvg1 - dig1;
    J[0] = dig1; J[1] = 0.0; J[2] = 0.0;
    J[3] = dikr_dvg1 * (1.0 - s);
    J[4] = dik_dvg2 * (1.0 - s);
    J[5] = dik_dva * (1.0 - s) - ikr * ds_dva;
    J[6] = dikr_dvg1 * s;
    J[7] = dik_dvg2 * s;
    J[8] = dik_dva * s + ikr * ds_dva + GMIN;
}

#ifdef BN_6550_TAB
/* The 6550 TABULATED: the EL34's table variant (bn_pentode_own.c) with the 6550's constants (no cutoff tail), so the
 * US editions avoid the closed form's pow x3, exp, log1p and atan per evaluation. Parity against the closed form:
 * tools/bn_test_pentode_own.c, its BN_6550_TAB block. */
#define BN_PT_N 16384
/* q^GAMMA has infinite curvature at q = 0 (deep cutoff): 64k points, as for the EL34 */
#define BN_PT_NQ 65536
/* the split ramp's knee at va ~ 0 over a 1000 V range: 64k points (0.016 V), as for the EL34 */
#define BN_PT_NV 65536
static double pt_L[BN_PT_N], pt_S[BN_PT_N];
static double pt_P[BN_PT_NQ], pt_dP[BN_PT_NQ];
static double pt_s[BN_PT_NV], pt_ds[BN_PT_NV];
static double pt_ig[BN_PT_N], pt_dig[BN_PT_N];
static const double ZLO = -40.0, ZHI = 40.0;
static const double QLO = 0.0,   QHI = 64.0;
static const double VA0 = -50.0, VA1 = 1000.0;
static const double G0  = -4.0,  G1  = 12.0;
static int pt_ready = 0;


void bn_pentode_6550_own_tables_init(void)
{
    if (pt_ready) { return; }
    for (int n = 0; n < BN_PT_N; ++n) {
        const double z = ZLO + (ZHI - ZLO) * n / (BN_PT_N - 1);
        pt_L[n] = softplus_raw(z);
        pt_S[n] = sigmoid_raw(z);
        const double g = G0 + (G1 - G0) * n / (BN_PT_N - 1);
        const double hg = softplus_c(g, CG);
        const double sg = sigmoid_c(g, CG);
        pt_ig[n]  = GG * pow(hg, XI) + IG0;
        pt_dig[n] = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    }
    for (int n = 0; n < BN_PT_NV; ++n) {
        const double va = VA0 + (VA1 - VA0) * n / (BN_PT_NV - 1);
        const double a = TAU * softplus_raw(va / TAU);
        const double t = a / KVB;
        pt_s[n]  = SINF * TWO_OVER_PI * atan(t);
        pt_ds[n] = SINF * TWO_OVER_PI * sigmoid_raw(va / TAU) / (KVB * (1.0 + t * t));
    }
    for (int n = 0; n < BN_PT_NQ; ++n) {
        const double q = QLO + (QHI - QLO) * n / (BN_PT_NQ - 1);
        pt_P[n]  = (q > 0.0) ? pow(q, GAMMA) : 0.0;
        pt_dP[n] = (q > 0.0) ? GAMMA * pow(q, GAMMA - 1.0) : 0.0;
    }
    pt_ready = 1;
}

static inline double pt_lutv(const double* t, double u)
{
    const double f = (u - VA0) * ((BN_PT_NV - 1) / (VA1 - VA0));
    const int    n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}

static inline double pt_lutq(const double* t, double u)
{
    const double f = (u - QLO) * ((BN_PT_NQ - 1) / (QHI - QLO));
    const int    n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}

static inline double pt_lut(const double* t, double u0, double u1, double u)
{
    const double f = (u - u0) * ((BN_PT_N - 1) / (u1 - u0));
    const int    n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}

void bn_pentode_6550_own_tab(const double* v, double* i, double* J, void* user)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    /* below the vg1 table the grid current is IG0 (a clamp, as in the triode);
     * everything else out of range goes to the closed form */
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        bn_pentode_6550_own(v, i, J, user);
        return;
    }
    const double L = pt_lut(pt_L, ZLO, ZHI, z);
    const double S = pt_lut(pt_S, ZLO, ZHI, z);
    const double e1 = (g2 / KP) * L;
    if (!(e1 < QHI - 1e-9)) { bn_pentode_6550_own(v, i, J, user); return; }
    const double ik = G_ * pt_lutq(pt_P, e1);
    const double dik_de1 = G_ * pt_lutq(pt_dP, e1);
    const double de1_dvg2 = g2_live ? (L / KP - S * w / g2) : 0.0;
    const double dik_dvg1 = dik_de1 * S;
    const double dik_dvg2 = dik_de1 * de1_dvg2;
    const double dik_dva  = dik_de1 * S / MUA;
    const double s = pt_lutv(pt_s, va);
    const double ds_dva = pt_lutv(pt_ds, va);
    const double ig1 = pt_lut(pt_ig, G0, G1, gc);
    const double dig1 = pt_lut(pt_dig, G0, G1, gc);
    const double dh = 1.0;
    const double ikr = ik - ig1;
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
    const double dikr_dvg1 = dh * (dik_dvg1 - dig1);
    J[0] = dig1; J[1] = 0.0; J[2] = 0.0;
    J[3] = dikr_dvg1 * (1.0 - s);
    J[4] = dh * dik_dvg2 * (1.0 - s);
    J[5] = dh * dik_dva * (1.0 - s) - ikr * ds_dva;
    J[6] = dikr_dvg1 * s;
    J[7] = dh * dik_dvg2 * s;
    J[8] = dh * dik_dva * s + ikr * ds_dva + GMIN;
}

void bn_pentode_6550_own_tab_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        bn_pentode_6550_own_currents(v, i);
        return;
    }
    const double e1 = (g2 / KP) * pt_lut(pt_L, ZLO, ZHI, z);
    if (!(e1 < QHI - 1e-9)) { bn_pentode_6550_own_currents(v, i); return; }
    const double ik = G_ * pt_lutq(pt_P, e1);
    const double s = pt_lutv(pt_s, va);
    const double ig1 = pt_lut(pt_ig, G0, G1, gc);
    const double ikr = ik - ig1;
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}

#ifdef BN_GRID_TAB
/* The control-grid law alone, for the send system (src/prepig/bn_prepig_device.c), which only needs ig1 and
 * dig1/dvg1. The same table lookup the table variant makes for ig1; outside it the closed form, evaluated at a rest
 * screen and plate. */
void bn_pentode_6550_own_grid_tab(double vg1, double* ig, double* dig)
{
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(gc < G1 - 1e-9)) {
        const double v3[3] = { vg1, 400.0, 400.0 };
        double i3[3], J3[9];
        bn_pentode_6550_own(v3, i3, J3, 0);
        *ig = i3[0];
        if (dig) { *dig = J3[0]; }
        return;
    }
    *ig = pt_lut(pt_ig, G0, G1, gc);
    if (dig) { *dig = pt_lut(pt_dig, G0, G1, gc); }
}
#endif
#endif /* BN_6550_TAB */
