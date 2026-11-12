/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_valve_el34.c @ d75dee0, sha256 d5ae9a9e871f25e4f9a24477c1913c743f62871f0915bf9194c4478f229fbb78 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* nl_valve_el34.c - the EL34 pentode: our own fit to the 1969 sheet (MK81 dsp/prototype/pentode_own.py --fit), a Koren-type 
cathode current Ik = G E1^gamma with E1 = (g2/KP) log1p(exp(KP (1/MU2 + w/g2))), w = vg1k + vak/MUA, g2 = max(vg2k, 1), the 
plate / screen split s = SINF (2/pi) atan(TAU softplus(vak/TAU)/KVB), the control-grid current in the triode's form scaled 
by the cathode perveance (an INFERENCE: the sheet gives none), and the cutoff tail h(x) = x|x|^n/(|x|^n + I0^n) on the 
cathode current. Ports: v[0] = vg1k, v[1] = vg2k, v[2] = vak (currents ig1, ig2, ia; J row-major 3x3). Tables: softplus / 
sigmoid, q^gamma, the split ramp, the grid law and the tail.
 * MOVED from MK81 (src/dsp/bn_*own*.c at 5ab77fa0, the product's build switches resolved: the tables on, the cutoff-tail and
 * grid-law tables on, no diagnostic hooks), by renaming only: the same operations in the same order, so test/valve holds
 * every current and Jacobian entry BIT-IDENTICAL to MK81's. The derivations and the measured reasons behind each guard
 * (the NaN-safe clamps, the knee smoothing, the table sizes) are commented in those MK81 files. Ours: nothing copied. */
#include "nl_dev_valve.h"
#ifndef NL_VALVE_PART_FALLBACK
#define NL_VALVE_PART_FALLBACK 0   /* POLICY (nlcore#14): off = MK81's tables bit for bit */
#endif
#include <math.h>
#define G_ 0.0098251541
#define KP 52.814461
#define GAMMA 1.2137747
#define MU2 13.98095
#define MUA 251.32573
#define SINF 0.89395901
#define KVB 9.2169857
#define GMIN 1e-10
#define I0_TAIL 7.153e-3
#define N_TAIL 2.814
#define I0N_TAIL 9.173390161821437e-07
#define TAU 1.0
#define TWO_OVER_PI 0.63661977236758134308
#define GG_TRIODE 6.177e-4
#define G_TRIODE 2.11116986e-03
#define CG 9.901
#define XI 1.314
#define IG0 8.025e-8
#define GG (GG_TRIODE * (G_ / G_TRIODE))
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
    if (z > 40.0) { return 1.0; }
    if (z < -40.0) { return 0.0; }
    return 1.0 / (1.0 + exp(-z));
}
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
void nl_valve_el34_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double e1 = (g2 / KP) * softplus_raw(z);
    const double ik = (e1 > 0.0) ? G_ * pow(e1, GAMMA) : 0.0;
    const double a = TAU * softplus_raw(va / TAU);
    const double s = SINF * TWO_OVER_PI * atan(a / KVB);
    const double hg = softplus_c(vg1, CG);
    const double ig1 = GG * pow(hg, XI) + IG0;
    const double ikr = tail_h(ik - ig1, 0);
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}
void nl_valve_el34(const double* v, double* i, double* J, void* user)
{
    (void)user;
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double L = softplus_raw(z);
    const double S = sigmoid_raw(z);
    const double e1 = (g2 / KP) * L;
    const double ik = (e1 > 0.0) ? G_ * pow(e1, GAMMA) : 0.0;
    const double dik_de1 = (e1 > 0.0) ? G_ * GAMMA * pow(e1, GAMMA - 1.0) : 0.0;
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
    const double dikr_dvg1 = dh * (dik_dvg1 - dig1);
    J[0] = dig1; J[1] = 0.0; J[2] = 0.0;
    J[3] = dikr_dvg1 * (1.0 - s);
    J[4] = dh * dik_dvg2 * (1.0 - s);
    J[5] = dh * dik_dva * (1.0 - s) - ikr * ds_dva;
    J[6] = dikr_dvg1 * s;
    J[7] = dh * dik_dvg2 * s;
    J[8] = dh * dik_dva * s + ikr * ds_dva + GMIN;
}
#define PT_N 16384
#define PT_NQ 65536
#define PT_NV 65536
static double pt_L[PT_N], pt_S[PT_N];
static double pt_P[PT_NQ], pt_dP[PT_NQ];
static double pt_s[PT_NV], pt_ds[PT_NV];
static double pt_ig[PT_N], pt_dig[PT_N];
static const double ZLO = -40.0, ZHI = 40.0;
static const double QLO = 0.0, QHI = 64.0;
static const double VA0 = -50.0, VA1 = 1000.0;
static const double G0 = -4.0, G1 = 12.0;
static int pt_ready = 0;
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
    if (ax < TT_X1) { f = ax * ((TT_N1 - 1) / TT_X1); th = tt_h1; td = tt_d1; nmax = TT_N1 - 2; }
    else if (ax < TT_X2) { f = (ax - TT_X1) * ((TT_N2 - 1) / (TT_X2 - TT_X1)); th = tt_h2; td = tt_d2; nmax = TT_N2 - 2; }
    else { return tail_h(x, dh); }
    int n = (int)f;
    if (n > nmax) { n = nmax; }
    const double a = f - n;
    if (dh) { *dh = td[n] + a * (td[n + 1] - td[n]); }
    const double h = th[n] + a * (th[n + 1] - th[n]);
    return x < 0.0 ? -h : h;
}
#define TAIL_TAB(x,dh) tail_t((x), (dh))
void nl_valve_el34_tables_init(void)
{
    if (pt_ready) { return; }
    tail_tab_init();
    for (int n = 0; n < PT_N; ++n) {
        const double z = ZLO + (ZHI - ZLO) * n / (PT_N - 1);
        pt_L[n] = softplus_raw(z);
        pt_S[n] = sigmoid_raw(z);
        const double g = G0 + (G1 - G0) * n / (PT_N - 1);
        const double hg = softplus_c(g, CG);
        const double sg = sigmoid_c(g, CG);
        pt_ig[n] = GG * pow(hg, XI) + IG0;
        pt_dig[n] = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    }
    for (int n = 0; n < PT_NV; ++n) {
        const double va = VA0 + (VA1 - VA0) * n / (PT_NV - 1);
        const double a = TAU * softplus_raw(va / TAU);
        const double t = a / KVB;
        pt_s[n] = SINF * TWO_OVER_PI * atan(t);
        pt_ds[n] = SINF * TWO_OVER_PI * sigmoid_raw(va / TAU) / (KVB * (1.0 + t * t));
    }
    for (int n = 0; n < PT_NQ; ++n) {
        const double q = QLO + (QHI - QLO) * n / (PT_NQ - 1);
        pt_P[n] = (q > 0.0) ? pow(q, GAMMA) : 0.0;
        pt_dP[n] = (q > 0.0) ? GAMMA * pow(q, GAMMA - 1.0) : 0.0;
    }
    pt_ready = 1;
}
static inline double pt_lutv(const double* t, double u)
{
    const double f = (u - VA0) * ((PT_NV - 1) / (VA1 - VA0));
    const int n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}
static inline double pt_lutq(const double* t, double u)
{
    const double f = (u - QLO) * ((PT_NQ - 1) / (QHI - QLO));
    const int n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}
static inline double pt_lut(const double* t, double u0, double u1, double u)
{
    const double f = (u - u0) * ((PT_N - 1) / (u1 - u0));
    const int n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}
#if NL_VALVE_PART_FALLBACK
/* POLICY NL_VALVE_PART_FALLBACK (nlcore#14, MK81#164): each of the law's four parts - softplus / sigmoid in z, the plate
 * ramp s(va), the control-grid law in vg1, e1^gamma - from its table where ITS argument is inside the table, from its own
 * closed form where it is not; the tables-off path above (whole closed form on any part out of range) cost MK81 ULTRA ~15 %
 * of its instructions in libm at the worst knob point, from Newton's trial iterates (plates seen at 61 kV, 2x) */
static inline void part_z(double z, double* L, double* S)
{
    if (z > ZLO && z < ZHI - 1e-9) { *L = pt_lut(pt_L, ZLO, ZHI, z); if (S) { *S = pt_lut(pt_S, ZLO, ZHI, z); } }
    else { *L = softplus_raw(z); if (S) { *S = sigmoid_raw(z); } }
}
static inline void part_e1(double e1, double* ik, double* dik)
{
    if (e1 < QHI - 1e-9) { *ik = G_ * pt_lutq(pt_P, e1); if (dik) { *dik = G_ * pt_lutq(pt_dP, e1); } }
    else { *ik = (e1 > 0.0) ? G_ * pow(e1, GAMMA) : 0.0; if (dik) { *dik = (e1 > 0.0) ? G_ * GAMMA * pow(e1, GAMMA - 1.0) : 0.0; } }
}
static inline void part_va(double va, double* s, double* ds)
{
    if (va > VA0 && va < VA1 - 1e-9) { *s = pt_lutv(pt_s, va); if (ds) { *ds = pt_lutv(pt_ds, va); } return; }
    const double a = TAU * softplus_raw(va / TAU);
    const double t = a / KVB;
    *s = SINF * TWO_OVER_PI * atan(t);
    if (ds) { *ds = SINF * TWO_OVER_PI * sigmoid_raw(va / TAU) / (KVB * (1.0 + t * t)); }
}
static inline void part_g(double vg1, double* ig, double* dig)
{
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (gc < G1 - 1e-9) { *ig = pt_lut(pt_ig, G0, G1, gc); if (dig) { *dig = pt_lut(pt_dig, G0, G1, gc); } return; }
    const double hg = softplus_c(vg1, CG);
    *ig = GG * pow(hg, XI) + IG0;
    if (dig) { const double sg = sigmoid_c(vg1, CG); *dig = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0; }
}
void nl_valve_el34_tab(const double* v, double* i, double* J, void* user)
{
    (void)user;
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    double L, S, ik, dik_de1, s, ds_dva, ig1, dig1;
    part_z(z, &L, &S);
    const double e1 = (g2 / KP) * L;
    part_e1(e1, &ik, &dik_de1);
    const double de1_dvg2 = g2_live ? (L / KP - S * w / g2) : 0.0;
    const double dik_dvg1 = dik_de1 * S;
    const double dik_dvg2 = dik_de1 * de1_dvg2;
    const double dik_dva = dik_de1 * S / MUA;
    part_va(va, &s, &ds_dva);
    part_g(vg1, &ig1, &dig1);
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
void nl_valve_el34_tab_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    double L, ik, s, ig1;
    part_z(z, &L, 0);
    part_e1((g2 / KP) * L, &ik, 0);
    part_va(va, &s, 0);
    part_g(vg1, &ig1, 0);
    const double ikr = TAIL_TAB(ik - ig1, 0);
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}
#else
void nl_valve_el34_tab(const double* v, double* i, double* J, void* user)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const int g2_live = (vg2 > 1.0);
    const double g2 = g2_live ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        nl_valve_el34(v, i, J, user);
        return;
    }
    const double L = pt_lut(pt_L, ZLO, ZHI, z);
    const double S = pt_lut(pt_S, ZLO, ZHI, z);
    const double e1 = (g2 / KP) * L;
    if (!(e1 < QHI - 1e-9)) { nl_valve_el34(v, i, J, user); return; }
    const double ik = G_ * pt_lutq(pt_P, e1);
    const double dik_de1 = G_ * pt_lutq(pt_dP, e1);
    const double de1_dvg2 = g2_live ? (L / KP - S * w / g2) : 0.0;
    const double dik_dvg1 = dik_de1 * S;
    const double dik_dvg2 = dik_de1 * de1_dvg2;
    const double dik_dva = dik_de1 * S / MUA;
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
void nl_valve_el34_tab_currents(const double* v, double* i)
{
    const double vg1 = v[0], vg2 = v[1], va = v[2];
    const double g2 = (vg2 > 1.0) ? vg2 : 1.0;
    const double w = vg1 + va / MUA;
    const double z = KP * (1.0 / MU2 + w / g2);
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(z > ZLO && z < ZHI - 1e-9) || !(va > VA0 && va < VA1 - 1e-9) || !(gc < G1 - 1e-9)) {
        nl_valve_el34_currents(v, i);
        return;
    }
    const double e1 = (g2 / KP) * pt_lut(pt_L, ZLO, ZHI, z);
    if (!(e1 < QHI - 1e-9)) { nl_valve_el34_currents(v, i); return; }
    const double ik = G_ * pt_lutq(pt_P, e1);
    const double s = pt_lutv(pt_s, va);
    const double ig1 = pt_lut(pt_ig, G0, G1, gc);
    const double ikr = TAIL_TAB(ik - ig1, 0);
    i[0] = ig1;
    i[1] = ikr * (1.0 - s);
    i[2] = ikr * s + GMIN * va;
}
#endif /* NL_VALVE_PART_FALLBACK */
void nl_valve_el34_grid_tab(double vg1, double* ig, double* dig)
{
    const double gc = (vg1 < G0) ? G0 : vg1;
    if (!(gc < G1 - 1e-9)) {
        const double v3[3] = { vg1, 400.0, 400.0 };
        double i3[3], J3[9];
        nl_valve_el34(v3, i3, J3, 0);
        *ig = i3[0];
        if (dig) { *dig = J3[0]; }
        return;
    }
    *ig = pt_lut(pt_ig, G0, G1, gc);
    if (dig) { *dig = pt_lut(pt_dig, G0, G1, gc); }
}
