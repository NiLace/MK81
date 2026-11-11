/* bn_pentode_own.c — C port of the project's own EL34 pentode (dsp/prototype/pentode_own.py).
 *
 * Model:
 *     g2 = max(vg2k, 1)                    (guard: g2 is a divisor)
 *     w  = vg1k + vak/MUA
 *     z  = KP*(1/MU2 + w/g2)
 *     L  = log1p(exp(z))                   (= z for large z)
 *     E1 = (g2/KP)*L                       (Koren eq. 5 + an anode term)
 *     Ik = G_*E1^GAMMA
 *     s  = SINF*(2/pi)*atan(TAU*softplus(vak/TAU)/KVB)   (smoothed split ramp)
 *     ig1 = GG*softplus(vg1k,CG)^XI + IG0        (see the grid block)
 *     Ikr = h(Ik - ig1)                          (Ik = ia+ig2+ig1; h the cutoff tail)
 *     ia = Ikr*s + GMIN*vak ; ig2 = Ikr*(1-s)
 *
 * The constants are never edited by hand: they come from `pentode_own.py --fit`
 * (as in bn_tube_own.c).
 */
#include "bn_pentode_own.h"

#include <math.h>

/* --- parameters FITTED to the EL34 data sheet (pentode_own.py --fit) --- */
#define G_     0.0098251541
#define KP     52.814461
#define GAMMA  1.2137747
#define MU2    13.98095
#define MUA    251.32573
#define SINF   0.89395901
#define KVB    9.2169857
#define GMIN   1e-10
/* The cutoff tail, h(x) = x*|x|^n/(|x|^n + I0^n) on the cathode current left after the grid, fitted to the same
 * data sheet's cutoff curve (dsp/prototype/pentode_own.py, I0_TAIL / N_TAIL). */
#define I0_TAIL  7.153e-3
#define N_TAIL   2.814
#define I0N_TAIL 9.173390161821437e-07   /* I0_TAIL^N_TAIL */
/* The knee's foot is smoothed: TAU*softplus(va/TAU) instead of max(va,0), which
 * puts a kink exactly at va=0. A numerical choice (like the triode's GMIN), not
 * physics, and a necessary one: the plate swings down to almost zero, so the
 * kink would sit INSIDE the travel and break Newton's Jacobian there. With
 * TAU = 1 V the effect is negligible by va = 5 V (softplus(5) = 5.007). */
#define TAU    1.0

#define TWO_OVER_PI 0.63661977236758134308

/* --- CONTROL-GRID CURRENT ---------------------------------------------------
 * WHY: with ig1 = 0 nothing loads the grid, which then rises to tens of volts
 * positive, impossible in a real amp (the grid conducts as soon as it passes
 * zero and clamps at a few volts, loading the phase inverter). The chain was:
 * grid runs away -> cathode runs away -> plate saturates -> the split sends
 * almost everything to the SCREEN, ~78 W per valve against an 8 W screen limit.
 *
 * FORM: the one the project's triode uses (Dempwolf & Zölzer, Table 1 of the
 * paper, RSD-1):
 *     ig = GG * softplus(vg1k, CG)^XI + IG0
 *
 * CONSTANTS: CG, XI and IG0 are taken as they are — they describe the SHAPE
 * of the conduction knee, geometry common to any grid-cathode diode. GG is
 * scaled, as it carries the size: the grid-cathode diode is a Child-Langmuir
 * structure whose perveance goes with the emitting area, so it is scaled by
 * each valve's cathode perveance (the pentode's G_ against the triode's).
 * ⚠️ This is OUR INFERENCE, not a data-sheet figure: the EL34 data sheet
 * publishes no grid current. */
#define GG_TRIODE  6.177e-4          /* Table 1 (RSD-1), 12AX7 */
#define G_TRIODE   2.11116986e-03    /* the project triode's cathode perveance */
#define CG         9.901
#define XI         1.314
#define IG0        8.025e-8
#define GG         (GG_TRIODE * (G_ / G_TRIODE))

/* h(x) and h'(x) = |x|^n (|x|^n + (n+1) I0^n) / (|x|^n + I0^n)^2 — even, so smooth through 0 */
static inline double tail_h(double x, double* dh)
{
    const double ax = fabs(x);
    const double an = (ax > 0.0) ? pow(ax, N_TAIL) : 0.0;
    const double d = an + I0N_TAIL;
    if (dh) { *dh = an * (an + (N_TAIL + 1.0) * I0N_TAIL) / (d * d); }
    return x * an / d;
}

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

void bn_pentode_own_currents(const double* v, double* i)
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
    const double ikr = tail_h(ik - ig1, 0);
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}

/* ANALYTIC Jacobian, branch by branch. J[r*3+c] = d i[r] / d v[c].
 * Checked against centred differences in tools/bn_test_pentode_own.c (a wrong
 * derivative at a knee is a silent error). */
void bn_pentode_own(const double* v, double* i, double* J, void* user)
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
    double dh;
    const double ikr = tail_h(ik - ig1, &dh);

    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;

    /* d(ikr)/dv = h'(.) (d(ik)/dv minus the grid's share, which depends on vg1 only) */
    const double dikr_dvg1 = dh * (dik_dvg1 - dig1);
    J[0] = dig1; J[1] = 0.0; J[2] = 0.0;
    J[3] = dikr_dvg1 * (1.0 - s);
    J[4] = dh * dik_dvg2 * (1.0 - s);
    J[5] = dh * dik_dva * (1.0 - s) - ikr * ds_dva;
    J[6] = dikr_dvg1 * s;
    J[7] = dh * dik_dvg2 * s;
    J[8] = dh * dik_dva * s + ikr * ds_dva + GMIN;
}

/* ======================================================================
 * TABULATED VARIANT — the same model, without transcendentals.
 *
 * WHY: the closed form costs about five times the tabulated triode per
 * evaluation, and the output pentodes dominate the power stage's solve. Each
 * transcendental of the model is a function of ONE scalar, so it is tabulated
 * like the triode (bn_tube_own.c): four pairs of 1-D tables, linear
 * interpolation.
 *     L(z), S(z)       softplus and sigmoid of Koren's argument, z in [ZLO, ZHI]
 *     P(q), dP(q)      q^GAMMA and GAMMA*q^(GAMMA-1), q = E1 in [QLO, QHI]
 *     s(va), ds(va)    the split ramp and its derivative, va in [VA0, VA1]
 *     ig(vg1), dig     the grid current and its derivative, vg1 in [G0, G1]
 * Out of range it falls back to the closed form (abuse, not the operating
 * regime). tools/bn_test_pentode_own.c checks the table-vs-closed-form parity;
 * the closed form stays the reference.
 * ====================================================================== */
#define BN_PT_N 16384
/* q^GAMMA has infinite curvature at q = 0 (deep cutoff): 16k points left the
 * screen current 8e-7 A off there; 64k brings it under the 1e-4 relative bar. */
#define BN_PT_NQ 65536
/* the split ramp s(va) has its knee at va ~ 0 (TAU = 1 V) over a 1000 V range:
 * 16k points (0.064 V) left the plate 3e-6 A off at va = 0 and full current;
 * 64k (0.016 V) brings it under the 1e-6 A bar. */
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


#ifdef BN_PT_TAIL_TAB
/* The cutoff tail h(x) tabulated too (otherwise it puts one pow back into every evaluation). h is odd and h' even,
 * so the tables hold x >= 0: [0, 0.2] A at 1e-5 A (the knee is at I0_TAIL = 7.2 mA) and [0.2, 1.6] A at 1e-3 A,
 * linear interpolation of the value and of the slope, each computed by the closed form at init; above 1.6 A the
 * closed form. */
#define TT_N1 20001
#define TT_X1 0.2
#define TT_N2 1401
#define TT_X2 1.6
static double tt_h1[TT_N1], tt_d1[TT_N1], tt_h2[TT_N2], tt_d2[TT_N2];
static void tail_tab_init(void)
{
    for (int n = 0; n < TT_N1; ++n) { tt_h1[n] = tail_h(TT_X1 * n / (TT_N1 - 1), &tt_d1[n]); }
    for (int n = 0; n < TT_N2; ++n) { tt_h2[n] = tail_h(TT_X1 + (TT_X2 - TT_X1) * n / (TT_N2 - 1), &tt_d2[n]); }
}
static inline double tail_t(double x, double* dh)
{
    const double ax = fabs(x);
    const double* th; const double* td; double f; int nmax;
    if (ax < TT_X1)      { f = ax * ((TT_N1 - 1) / TT_X1); th = tt_h1; td = tt_d1; nmax = TT_N1 - 2; }
    else if (ax < TT_X2) { f = (ax - TT_X1) * ((TT_N2 - 1) / (TT_X2 - TT_X1)); th = tt_h2; td = tt_d2; nmax = TT_N2 - 2; }
    else                 { return tail_h(x, dh); }   /* (a NaN lands here too: the closed form propagates it) */
    int n = (int)f;
    if (n > nmax) { n = nmax; }
    const double a = f - n;
    if (dh) { *dh = td[n] + a * (td[n + 1] - td[n]); }
    const double h = th[n] + a * (th[n + 1] - th[n]);
    return x < 0.0 ? -h : h;
}
#define TAIL_TAB(x, dh) tail_t((x), (dh))
#else
#define TAIL_TAB(x, dh) tail_h((x), (dh))
#endif

void bn_pentode_own_tables_init(void)
{
    if (pt_ready) { return; }
#ifdef BN_PT_TAIL_TAB
    tail_tab_init();
#endif
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


#define FB_NOTE(k, z, va, gc) ((void)0)
#define FB_E1(k, e1) ((void)0)
void bn_pentode_own_tab(const double* v, double* i, double* J, void* user)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    /* below the vg1 table the grid current is IG0 (a clamp, as in the triode);
     * everything else out of range goes to the closed form */
    const double gc = (vg1 < G0) ? G0 : vg1;
    FB_NOTE(0, z, va, gc);
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        bn_pentode_own(v, i, J, user);
        return;
    }
    const double L = pt_lut(pt_L, ZLO, ZHI, z);
    const double S = pt_lut(pt_S, ZLO, ZHI, z);
    const double e1 = (g2 / KP) * L;
    FB_E1(0, e1);
    if (!(e1 < QHI - 1e-9)) { bn_pentode_own(v, i, J, user); return; }
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
    double dh;
    const double ikr = TAIL_TAB(ik - ig1, &dh);
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

void bn_pentode_own_tab_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double gc = (vg1 < G0) ? G0 : vg1;
    FB_NOTE(1, z, va, gc);
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        bn_pentode_own_currents(v, i);
        return;
    }
    const double e1 = (g2 / KP) * pt_lut(pt_L, ZLO, ZHI, z);
    FB_E1(1, e1);
    if (!(e1 < QHI - 1e-9)) { bn_pentode_own_currents(v, i); return; }
    const double ik = G_ * pt_lutq(pt_P, e1);
    const double s = pt_lutv(pt_s, va);
    const double ig1 = pt_lut(pt_ig, G0, G1, gc);
    const double ikr = TAIL_TAB(ik - ig1, 0);
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}

#ifdef BN_GRID_TAB
/* The control-grid law alone, for the send system (src/prepig/bn_prepig_device.c), which only needs ig1 and
 * dig1/dvg1. The same table lookup the table variant makes for ig1 (the same numbers wherever that variant is in
 * range); outside it the closed form, evaluated at a rest screen and plate. */
void bn_pentode_own_grid_tab(double vg1, double* ig, double* dig)
{
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(gc < G1 - 1e-9)) {
        const double v3[3] = { vg1, 400.0, 400.0 };
        double i3[3], J3[9];
        bn_pentode_own(v3, i3, J3, 0);
        *ig = i3[0];
        if (dig) { *dig = J3[0]; }
        return;
    }
    *ig = pt_lut(pt_ig, G0, G1, gc);
    if (dig) { *dig = pt_lut(pt_dig, G0, G1, gc); }
}
#endif
