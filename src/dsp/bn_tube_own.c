/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_tube_own.c — the project's own triode (12AX7). EXACT port of
 * dsp/prototype/tube_own.py: if the two disagree, the port is wrong
 * (tools/bn_test_tube_own.c checks this).
 *
 * Model (Dempwolf & Zölzer, DAFx-11, eqs. 10-12), softplus_C(x)=log1p(e^Cx)/C:
 *   Ik = G · softplus_C(Va/µ + Vg)^γ
 *   Ig = Gg · softplus_Cg(Vg)^ξ + Ig0
 *   Ia = Ik − Ig + gmin·Va
 * Cathode constants: OUR OWN FIT to the RCA datasheet (tube_own.py --fit).
 * Grid constants: Table 1 of the paper (RSD-1), citable literature.
 */
#include "bn_tube_own.h"

#include <stdio.h>
#include <stdlib.h>

#include <math.h>

/* grid: Table 1 of the paper (RSD-1) */
static const double GG = 6.177e-4, CG = 9.901, XI = 1.314, IG0 = 8.025e-8;
/* cathode: our own fit (dsp/prototype/tube_own.py --fit) */
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
    if (z > 40.0)  { return 1.0; }
    if (z < -40.0) { return 0.0; }
    return 1.0 / (1.0 + exp(-z));
}

#define VAK_PROBE(vak, ia) ((void)0)
#define BN_VAK_SLOT(t) ((void)0)

/* REVERSE-CONDUCTION BLOCK.
 *
 * The converged state rarely reaches vak < 0 (a few samples in millions, on
 * the 50 W head at full drive), but Newton's trial points do, where the
 * ungated model would put up to ~200 mA of impossible current. The block is
 * solver robustness, not a change of sound: with and without it the output
 * nulls below -110 dB, at no measurable cost.
 *
 * PHYSICS: an anode below the cathode does NOT collect electrons, so the PLATE
 * current vanishes. The grid current does not: what the grid draws is still
 * carried by the grid (charge conservation). Since here ia = ik - ig and ig is
 * computed separately from vgk, zeroing ia is enough: the cathode current
 * becomes ig alone, which is exactly the real split.
 *
 * SHAPE: a smoothstep gate on vak, with two deliberately chosen properties
 * that make the validation CHECKABLE:
 *   - vak >= V0  ->  g == 1.0 EXACTLY => multiplying by 1.0 is exact in IEEE
 *     and the model is BIT-IDENTICAL over the whole normal region.
 *   - vak <= 0   ->  g == 0 and g' == 0 => valve cut off = open circuit.
 *     The GMIN*vak term stays OUTSIDE the gate: it is the numerical leakage
 *     that keeps the matrix non-singular, and without it cut-off leaves a row
 *     of zeros. Same reason GMIN exists.
 * ⚠️ QUINTIC, not cubic: with the cubic (3t^2-2t^3) the gate is C1 but NOT
 * C2, and `make test` catches it — not through the audio but through the
 * Jacobian. `bn_test_tube_own` takes a centred difference exactly at vak=0,
 * where the function is 0 on the left and O(h^2) on the right: that leaves a
 * first-order residual that the test turns into 2.85e-4 relative after
 * dividing by its scale floor (the ABSOLUTE error is 2.8e-10, i.e. nothing).
 * The quintic also zeroes g'' at both ends => the term becomes O(h^4) and
 * vanishes. The GATE is fixed, the test is not loosened.
 * It is C2 over the whole travel: a hard cut would put a kink in the
 * derivative, and a kink generates broadband content (aliasing) besides
 * punishing Newton — which is exactly why this model uses softplus and not
 * max().
 *
 * V0 = 1 V is not a free fit: below ~1 V the plate is space-charge limited,
 * so the ungated model is not valid there anyway. It is the width of the
 * smoothing, not a third-party constant.
 *
 * -DBN_TUBE_NO_BLOCK restores the ungated behaviour, for the A/B. */
#define BN_VAK_V0 1.0
static inline double bn_block_g(double vak)
{
    if (vak >= BN_VAK_V0) { return 1.0; }
    if (!(vak > 0.0))     { return 0.0; }   /* negated comparison: NaN -> 0 */
    const double t = vak / BN_VAK_V0;
    return t * t * t * (10.0 + t * (6.0 * t - 15.0));   /* QUINTIC smoothstep */
}
static inline double bn_block_dg(double vak)
{
    if (vak >= BN_VAK_V0 || !(vak > 0.0)) { return 0.0; }
    const double t = vak / BN_VAK_V0;
    const double u = t * (1.0 - t);
    return 30.0 * u * u / BN_VAK_V0;                    /* g' = 30 t^2 (1-t)^2 */
}

void bn_tube_own_currents(const double* v, double* i)
{
    const double vgk = v[0], vak = v[1];
    const double hk = softplus(vak / MU + vgk, C_);
    const double ik = G_ * pow(hk, GAMMA_);
    const double hg = softplus(vgk, CG);
    const double ig = GG * pow(hg, XI) + IG0;
    i[0] = ig;
    i[1] = (ik - ig) * bn_block_g(vak) + GMIN * vak;
    VAK_PROBE(vak, i[1]);
}

void bn_tube_own(const double* v, double* i, double* J, void* user)
{
    (void)user;
    const double vgk = v[0], vak = v[1];
    const double x  = vak / MU + vgk;
    const double hk = softplus(x, C_);
    const double sk = sigmoid_c(x, C_);
    const double ik  = G_ * pow(hk, GAMMA_);
    const double dik = (hk > 0.0) ? G_ * GAMMA_ * pow(hk, GAMMA_ - 1.0) * sk : 0.0;

    const double hg = softplus(vgk, CG);
    const double sg = sigmoid_c(vgk, CG);
    const double ig  = GG * pow(hg, XI) + IG0;
    const double dig = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;

    const double gb = bn_block_g(vak), dgb = bn_block_dg(vak);
    i[0] = ig;
    i[1] = (ik - ig) * gb + GMIN * vak;
    VAK_PROBE(vak, i[1]);
    J[0] = dig;               J[1] = 0.0;
    J[2] = (dik - dig) * gb;  J[3] = (dik / MU) * gb + (ik - ig) * dgb + GMIN;
}

/* Per-stage current trims (lab build only, -DBN_TRIMS_LEVER: the BN_TRIMS
 * environment variable), a spread between stages within the 12AX7 datasheet's
 * ±20 % of gm. trims[t] = 1.0 by default => BIT-EXACT with the untrimmed path.
 * Declared above every device function so the tabulated and the closed-form
 * paths both read it. */
static double bn_own_trims[4] = { 1.0, 1.0, 1.0, 1.0 };
static int    bn_own_trims_read = 0;

static void bn_own_trims_init(void)
{
    if (bn_own_trims_read) { return; }
    bn_own_trims_read = 1;
}

int bn_own_trims_puestos(void)
{
    bn_own_trims_init();
    for (int t = 0; t < 4; ++t) {
        if (bn_own_trims[t] != 1.0) { return 1; }
    }
    return 0;
}

/* Four triodes over 8 ports. It does NOT re-zero the whole 8x8: it only
 * writes its 2x2 blocks (the rest is already 0 and does not change). */
void bn_tube_own_x4(const double* v, double* i, double* J, void* user)
{
    (void)user;
    bn_own_trims_init();
    for (int t = 0; t < 4; ++t) {
        double Jt[4];
        BN_VAK_SLOT(t);
        bn_tube_own(v + 2 * t, i + 2 * t, Jt, 0);
        const double g = bn_own_trims[t];
        if (g != 1.0) {
            i[2 * t] *= g; i[2 * t + 1] *= g;
            Jt[0] *= g; Jt[1] *= g; Jt[2] *= g; Jt[3] *= g;
        }
        J[(2 * t + 0) * 8 + (2 * t + 0)] = Jt[0];
        J[(2 * t + 0) * 8 + (2 * t + 1)] = Jt[1];
        J[(2 * t + 1) * 8 + (2 * t + 0)] = Jt[2];
        J[(2 * t + 1) * 8 + (2 * t + 1)] = Jt[3];
    }
}

void bn_tube_own_currents_x4(const double* v, double* i, void* user)
{
    (void)user;
    bn_own_trims_init();
    for (int t = 0; t < 4; ++t) {
        BN_VAK_SLOT(t);
        bn_tube_own_currents(v + 2 * t, i + 2 * t);
        const double g = bn_own_trims[t];
        if (g != 1.0) { i[2 * t] *= g; i[2 * t + 1] *= g; }
    }
}

/* ------------------------------------------------- TABULATED variant ------
 * The closed form costs ~4 transcendentals per triode and evaluation. The
 * model is 1-D twice: Ik(x) with x = Va/µ + Vg, and Ig(vgk). Two LUTs with
 * tabulated derivative, linear interpolation, and the closed form out of
 * range. tools/bn_test_tube_own.c checks the table-vs-closed-form parity. */

#define BN_OWN_N 32768   /* 8k gives 6e-5 at the knee; 32k brings it to 3e-6 */
static double bn_own_ik[BN_OWN_N], bn_own_dik[BN_OWN_N];
static double bn_own_ig[BN_OWN_N], bn_own_dig[BN_OWN_N];
static const double XK0 = -12.0, XK1 = 16.0;    /* range of x = Va/µ + Vg  */
static const double XG0 = -4.0,  XG1 = 12.0;    /* range of vgk            */
static int bn_own_ready = 0;

void bn_tube_own_tables_init(void)
{
    if (bn_own_ready) { return; }
    for (int n = 0; n < BN_OWN_N; ++n) {
        const double x = XK0 + (XK1 - XK0) * n / (BN_OWN_N - 1);
        const double h = softplus(x, C_);
        const double s = sigmoid_c(x, C_);
        bn_own_ik[n]  = G_ * pow(h, GAMMA_);
        bn_own_dik[n] = (h > 0.0) ? G_ * GAMMA_ * pow(h, GAMMA_ - 1.0) * s : 0.0;
        const double g = XG0 + (XG1 - XG0) * n / (BN_OWN_N - 1);
        const double hg = softplus(g, CG);
        const double sg = sigmoid_c(g, CG);
        bn_own_ig[n]  = GG * pow(hg, XI) + IG0;
        bn_own_dig[n] = (hg > 0.0) ? GG * XI * pow(hg, XI - 1.0) * sg : 0.0;
    }
    bn_own_ready = 1;
}

static inline double lut1(const double* t, double u0, double u1, double u)
{
    const double f = (u - u0) * ((BN_OWN_N - 1) / (u1 - u0));
    const int    n = (int)f;
    const double a = f - n;
    return t[n] + a * (t[n + 1] - t[n]);
}

/* BELOW the range the currents are negligible constants (softplus in
 * underflow: ik(−12)≈2e-11 A, ig(−4)≈IG0): the input is CLAMPED to the table
 * edge instead of falling back to the closed form, since deep cut-off (grids
 * at −20..−60 V, normal in the preamp) is common and would eat the table's
 * gain. The clamp error stays under the parity test's bar (bn_test_tube_own).
 * ABOVE the range (x>16, vgk>12: only extreme abuse) the closed form is used. */
void bn_tube_own_tab(const double* v, double* i, double* J, void* user)
{
    const double vgk = v[0], vak = v[1];
    const double x = vak / MU + vgk;
    if (x >= XK1 - 1e-9 || vgk >= XG1 - 1e-9) {
        bn_tube_own(v, i, J, user);       /* above the table: closed form */
        return;
    }
    /* clamp with the NEGATED comparison: with NaN the direct form lets the
     * value through and (int)NaN = INT_MIN blows up the LUT index. This way a
     * NaN lands on the table edge instead of out of bounds. */
    const double xc = !(x > XK0) ? XK0 : x;
    const double gc = !(vgk > XG0) ? XG0 : vgk;
    const double ik  = lut1(bn_own_ik, XK0, XK1, xc);
    const double dik = lut1(bn_own_dik, XK0, XK1, xc);
    const double ig  = lut1(bn_own_ig, XG0, XG1, gc);
    const double dig = lut1(bn_own_dig, XG0, XG1, gc);
    const double gb = bn_block_g(vak), dgb = bn_block_dg(vak);
    i[0] = ig;
    i[1] = (ik - ig) * gb + GMIN * vak;
    J[0] = dig;               J[1] = 0.0;
    J[2] = (dik - dig) * gb;  J[3] = (dik / MU) * gb + (ik - ig) * dgb + GMIN;
}

void bn_tube_own_tab_currents(const double* v, double* i)
{
    const double vgk = v[0], vak = v[1];
    const double x = vak / MU + vgk;
    if (x >= XK1 - 1e-9 || vgk >= XG1 - 1e-9) {
        bn_tube_own_currents(v, i);
        return;
    }
    /* clamp with the NEGATED comparison: with NaN the direct form lets the
     * value through and (int)NaN = INT_MIN blows up the LUT index. This way a
     * NaN lands on the table edge instead of out of bounds. */
    const double xc = !(x > XK0) ? XK0 : x;
    const double gc = !(vgk > XG0) ? XG0 : vgk;
    const double ik = lut1(bn_own_ik, XK0, XK1, xc);
    const double ig = lut1(bn_own_ig, XG0, XG1, gc);
    i[0] = ig;
    i[1] = (ik - ig) * bn_block_g(vak) + GMIN * vak;
    VAK_PROBE(vak, i[1]);
}

/* VARIANTS WITHOUT TRIMS. Checking the trims would be paid on the device's
 * HOTTEST path (a call and four comparisons per evaluation, several
 * evaluations per oversampled sample), so the pointer is chosen ONCE when the
 * circuit is built (bn_tube_own_pick_*). With the trims unset these variants
 * are output-identical by construction: they only drop code that executes
 * nothing. Whoever switches the device must switch tools/bn_bench.c too. */
void bn_tube_own_tab_x4_notrim(const double* v, double* i, double* J, void* user)
{
    (void)user;
    if (!bn_own_ready) { bn_tube_own_tables_init(); }
    for (int t = 0; t < 4; ++t) {
        double Jt[4];
        BN_VAK_SLOT(t);
        bn_tube_own_tab(v + 2 * t, i + 2 * t, Jt, 0);
        J[(2 * t + 0) * 8 + (2 * t + 0)] = Jt[0];
        J[(2 * t + 0) * 8 + (2 * t + 1)] = Jt[1];
        J[(2 * t + 1) * 8 + (2 * t + 0)] = Jt[2];
        J[(2 * t + 1) * 8 + (2 * t + 1)] = Jt[3];
    }
}

void bn_tube_own_tab_currents_x4_notrim(const double* v, double* i, void* user)
{
    (void)user;
    /* same lazy init its `_x4` twin has: these read the tables too, and without
     * it a call before bn_tube_own_tables_init() returns currents off zeroed
     * tables, silently. The twins already decided this branch is affordable. */
    if (!bn_own_ready) { bn_tube_own_tables_init(); }
    for (int t = 0; t < 4; ++t) { BN_VAK_SLOT(t); bn_tube_own_tab_currents(v + 2 * t, i + 2 * t); }
}

/* These pick the device ONCE, depending on whether trims are set. */
BnDeviceFn bn_tube_own_pick_tab_x4(void)
{
    bn_tube_own_tables_init();
    return bn_own_trims_puestos() ? bn_tube_own_tab_x4 : bn_tube_own_tab_x4_notrim;
}

BnDeviceCurrFn bn_tube_own_pick_tab_currents_x4(void)
{
    bn_tube_own_tables_init();
    return bn_own_trims_puestos() ? bn_tube_own_tab_currents_x4
                                  : bn_tube_own_tab_currents_x4_notrim;
}

void bn_tube_own_tab_x4(const double* v, double* i, double* J, void* user)
{
    (void)user;
    if (!bn_own_ready) { bn_tube_own_tables_init(); }
    bn_own_trims_init();
    for (int t = 0; t < 4; ++t) {
        double Jt[4];
        BN_VAK_SLOT(t);
        bn_tube_own_tab(v + 2 * t, i + 2 * t, Jt, 0);
        const double g = bn_own_trims[t];
        if (g != 1.0) {
            i[2 * t] *= g; i[2 * t + 1] *= g;
            Jt[0] *= g; Jt[1] *= g; Jt[2] *= g; Jt[3] *= g;
        }
        J[(2 * t + 0) * 8 + (2 * t + 0)] = Jt[0];
        J[(2 * t + 0) * 8 + (2 * t + 1)] = Jt[1];
        J[(2 * t + 1) * 8 + (2 * t + 0)] = Jt[2];
        J[(2 * t + 1) * 8 + (2 * t + 1)] = Jt[3];
    }
}

void bn_tube_own_tab_currents_x4(const double* v, double* i, void* user)
{
    (void)user;
    /* same lazy init its `_x4` twin has: these read the tables too, and without
     * it a call before bn_tube_own_tables_init() returns currents off zeroed
     * tables, silently. The twins already decided this branch is affordable. */
    if (!bn_own_ready) { bn_tube_own_tables_init(); }
    bn_own_trims_init();
    for (int t = 0; t < 4; ++t) {
        BN_VAK_SLOT(t);
        bn_tube_own_tab_currents(v + 2 * t, i + 2 * t);
        const double g = bn_own_trims[t];
        if (g != 1.0) { i[2 * t] *= g; i[2 * t + 1] *= g; }
    }
}
