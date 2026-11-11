/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_valve_12ax7.c @ c0e3536, sha256 259cd0c9b31d25816543b40fa4fc87fc70a35945398ee80c0917152d0be2e4f3 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* nl_valve_12ax7.c - the 12AX7 (ECC83) triode: Dempwolf & Zoelzer (DAFx-11) eqs. 10-12, softplus_C(x) = log1p(e^Cx)/C, Ik = G 
softplus_C(Va/mu + Vg)^gamma, Ig = Gg softplus_Cg(Vg)^xi + Ig0, Ia = (Ik - Ig) block(Va) + gmin Va; the cathode constants 
are OUR OWN fit to the manufacturer's sheet (MK81 dsp/prototype/tube_own.py --fit), the grid constants the paper's Table 1 (RSD-1); 
block(Va) a quintic smoothstep over 0..1 V that stops reverse plate conduction. Ports: v[0] = vgk (current ig), v[1] = vak 
(current ia). Tables: Ik(x) and Ig(vgk) with tabulated slopes, linear interpolation, clamped below the range (deep cut-off), 
the closed form above it.
 * MOVED from MK81 (src/dsp/bn_*own*.c at 5ab77fa0, the product's build switches resolved: the tables on, the cutoff-tail and
 * grid-law tables on, no diagnostic hooks), by renaming only: the same operations in the same order, so test/valve holds
 * every current and Jacobian entry BIT-IDENTICAL to MK81's. The derivations and the measured reasons behind each guard
 * (the NaN-safe clamps, the knee smoothing, the table sizes) are commented in those MK81 files. Ours: nothing copied. */
#include "nl_dev_valve.h"
#include <math.h>
static const double GG = 6.177e-4, CG = 9.901, XI = 1.314, IG0 = 8.025e-8;
static const double G_ = 2.11116986e-03, C_ = 2.06288891e+00;
static const double GAMMA_ = 1.26, MU = 1.005870e+02;
static const double GMIN = 1e-10;
static inline double softplus(double x, double c)
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
#define VAK_V0 1.0
static inline double block_g(double vak)
{
    if (vak >= VAK_V0) { return 1.0; }
    if (!(vak > 0.0)) { return 0.0; }
    const double t = vak / VAK_V0;
    return t * t * t * (10.0 + t * (6.0 * t - 15.0));
}
static inline double block_dg(double vak)
{
    if (vak >= VAK_V0 || !(vak > 0.0)) { return 0.0; }
    const double t = vak / VAK_V0;
    const double u = t * (1.0 - t);
    return 30.0 * u * u / VAK_V0;
}
void nl_valve_12ax7_currents(const double* v, double* i)
{
    const double vgk = v[0], vak = v[1];
    const double hk = softplus(vak / MU + vgk, C_);
    const double ik = G_ * pow(hk, GAMMA_);
    const double hg = softplus(vgk, CG);
    const double ig = GG * pow(hg, XI) + IG0;
    i[0] = ig;
    i[1] = (ik - ig) * block_g(vak) + GMIN * vak;
}
void nl_valve_12ax7(const double* v, double* i, double* J, void* user)
{
    (void)user;
    const double vgk = v[0], vak = v[1];
    const double x = vak / MU + vgk;
    const double hk = softplus(x, C_);
    const double sk = sigmoid_c(x, C_);
    const double ik = G_ * pow(hk, GAMMA_);
    const double dik = (hk > 0.0) ? G_ * GAMMA_ * pow(hk, GAMMA_ - 1.0) * sk : 0.0;
    const double hg = softplus(vgk, CG);
    const double sg = sigmoid_c(vgk, CG);
    const double ig = GG * pow(hg, XI) + IG0;
    const double dig = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    const double gb = block_g(vak), dgb = block_dg(vak);
    i[0] = ig;
    i[1] = (ik - ig) * gb + GMIN * vak;
    J[0] = dig; J[1] = 0.0;
    J[2] = (dik - dig) * gb; J[3] = (dik / MU) * gb + (ik - ig) * dgb + GMIN;
}

#define TRI_N 32768
static double tri_ik[TRI_N], tri_dik[TRI_N];
static double tri_ig[TRI_N], tri_dig[TRI_N];
static const double XK0 = -12.0, XK1 = 16.0;
static const double XG0 = -4.0, XG1 = 12.0;
static int tri_ready = 0;
void nl_valve_12ax7_tables_init(void)
{
    if (tri_ready) { return; }
    for (int n = 0; n < TRI_N; ++n) {
        const double x = XK0 + (XK1 - XK0) * n / (TRI_N - 1);
        const double h = softplus(x, C_);
        const double s = sigmoid_c(x, C_);
        tri_ik[n] = G_ * pow(h, GAMMA_);
        tri_dik[n] = (h > 0.0) ? G_ * GAMMA_ * pow(h, GAMMA_ - 1.0) * s : 0.0;
        const double g = XG0 + (XG1 - XG0) * n / (TRI_N - 1);
        const double hg = softplus(g, CG);
        const double sg = sigmoid_c(g, CG);
        tri_ig[n] = GG * pow(hg, XI) + IG0;
        tri_dig[n] = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    }
    tri_ready = 1;
}
static inline double lut1(const double* t, double u0, double u1, double u)
{
    const double f = (u - u0) * ((TRI_N - 1) / (u1 - u0));
    const int n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}
void nl_valve_12ax7_tab(const double* v, double* i, double* J, void* user)
{
    const double vgk = v[0], vak = v[1];
    const double x = vak / MU + vgk;
    if (x >= XK1 - 1e-9 || vgk >= XG1 - 1e-9) {
        nl_valve_12ax7(v, i, J, user);
        return;
    }
    const double xc = !(x > XK0) ? XK0 : x;
    const double gc = !(vgk > XG0) ? XG0 : vgk;
    const double ik = lut1(tri_ik, XK0, XK1, xc);
    const double dik = lut1(tri_dik, XK0, XK1, xc);
    const double ig = lut1(tri_ig, XG0, XG1, gc);
    const double dig = lut1(tri_dig, XG0, XG1, gc);
    const double gb = block_g(vak), dgb = block_dg(vak);
    i[0] = ig;
    i[1] = (ik - ig) * gb + GMIN * vak;
    J[0] = dig; J[1] = 0.0;
    J[2] = (dik - dig) * gb; J[3] = (dik / MU) * gb + (ik - ig) * dgb + GMIN;
}
void nl_valve_12ax7_tab_currents(const double* v, double* i)
{
    const double vgk = v[0], vak = v[1];
    const double x = vak / MU + vgk;
    if (x >= XK1 - 1e-9 || vgk >= XG1 - 1e-9) {
        nl_valve_12ax7_currents(v, i);
        return;
    }
    const double xc = !(x > XK0) ? XK0 : x;
    const double gc = !(vgk > XG0) ? XG0 : vgk;
    const double ik = lut1(tri_ik, XK0, XK1, xc);
    const double ig = lut1(tri_ig, XG0, XG1, gc);
    i[0] = ig;
    i[1] = (ik - ig) * block_g(vak) + GMIN * vak;
}
