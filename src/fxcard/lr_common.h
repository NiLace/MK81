/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* lr_common.h - what the effects card's linear return (c/) and linear send (send/) share: the matrix exponential of
 * the discretisation and the back-to-back 12 V zener pair as a port table (nlcore nl_dev_zener.h's law, the series pair solved
 * at instantiate; its current from the node's open-circuit value v0 through the port's self term R11). Static functions:
 * each including file gets its own copy. Needs LR_VT and the LR_Z_* law constants defined before it. */
#ifndef LR_COMMON_H
#define LR_COMMON_H
#include <math.h>
#include <string.h>
#include "nl_dev_zener.h"

#define LR_NZ 4001            /* the zener pair's sweep, asinh-spaced in current */
#ifndef LR_ZU_H
#define LR_ZU_H 0.004         /* the uniform table on |v0| (the pair is symmetric): step in volts */
#endif
#define LR_ZU_N 10001         /* covers |v0| < 40 V; beyond it, the sweep's binary search */

typedef struct {
    double R11;                                          /* the port's self term: v = v0 + R11 i_into_node */
    double zv0[LR_NZ], zi[LR_NZ];                        /* the pair: open-circuit v0 of the node -> the pair current node -> VREF */
    double zu[LR_ZU_N];                                  /* the pair current on a uniform |v0| grid (direct index) */
} LrZener;

/* e^M for a small dense matrix (n <= 12): scaling and squaring with a 24-term Taylor series (instantiate only) */
static inline void expm(int n, const double* M, double* E)
{
    double nrm = 0.0;
    for (int i = 0; i < n; i++) { double r = 0.0; for (int j = 0; j < n; j++) r += fabs(M[i * n + j]); if (r > nrm) nrm = r; }
    int s = 0; while (nrm > 0.125) { nrm *= 0.5; s++; }
    const double sc = ldexp(1.0, -s);
    double A[144], T[144], P[144];
    for (int i = 0; i < n * n; i++) { A[i] = M[i] * sc; E[i] = 0.0; T[i] = 0.0; }
    for (int i = 0; i < n; i++) { E[i * n + i] = 1.0; T[i * n + i] = 1.0; }
    for (int k = 1; k <= 24; k++) {
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double a = 0.0; for (int l = 0; l < n; l++) a += T[i * n + l] * A[l * n + j]; P[i * n + j] = a / k;
        }
        memcpy(T, P, sizeof(double) * n * n);
        for (int i = 0; i < n * n; i++) E[i] += T[i];
    }
    for (int q = 0; q < s; q++) {
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double a = 0.0; for (int l = 0; l < n; l++) a += E[i * n + l] * E[l * n + j]; P[i * n + j] = a;
        }
        memcpy(E, P, sizeof(double) * n * n);
    }
}

static inline void zener_law(NlZener* z)
{
    memset(z, 0, sizeof *z);
    z->vt = LR_VT;
    z->f_is = LR_Z_F_IS; z->f_n = LR_Z_F_N; z->f_rs = LR_Z_F_RS;
    z->r_is = LR_Z_R_IS; z->r_n = LR_Z_R_N;
    z->b_is = LR_Z_B_IS; z->b_n = LR_Z_B_N; z->b_rs = LR_Z_B_RS; z->voff = LR_Z_VOFF;
    nl_zener_init(z);
}

/* the zener's voltage at current i (anode -> cathode): a bracketed Newton from v (instantiate only) */
static inline double zener_inv(const NlZener* z, double i, double v)
{
    double lo = -40.0, hi = 5.0;
    for (int k = 0; k < 200; k++) {
        double ii, g; nl_zener_iv(z, v, &ii, &g);
        const double f = ii - i;
        if (f > 0) hi = v; else lo = v;
        double vn = v - f / g;
        if (!(vn > lo && vn < hi)) vn = 0.5 * (lo + hi);
        if (fabs(vn - v) < 1e-13) return vn;
        v = vn;
    }
    return v;
}

static inline void zener_pair_table(LrZener* r)
{
    NlZener z; zener_law(&z);
    /* the pair d -> VREF: D3 anode d, cathode m2; D4 anode VREF, cathode m2. Current I from d into the pair:
     * v_d - v_m2 = inv(I), v_VREF - v_m2 = inv(-I)  =>  v_d - v_VREF = inv(I) - inv(-I); v_d = v0 + R11 iz, iz = -I */
    const double I0 = 1e-13, imax = 0.7, amax = asinh(imax / I0);
    double v3 = 0.0, v4 = 0.0;
    for (int k = 0; k < LR_NZ; k++) {
        const double I = I0 * sinh(amax * (2.0 * k / (LR_NZ - 1) - 1.0));
        v3 = zener_inv(&z, I, v3); v4 = zener_inv(&z, -I, v4);
        r->zi[k] = I; r->zv0[k] = (v3 - v4) + r->R11 * I;
    }
}

/* the pair current from v0 by the sweep (binary search, linear between its points) */
static inline double zener_sweep(const LrZener* r, double v0)
{
    int lo = 0, hi = LR_NZ - 1;
    if (v0 <= r->zv0[0]) hi = 1; else if (v0 >= r->zv0[LR_NZ - 1]) lo = LR_NZ - 2;
    else while (hi - lo > 1) { const int m = (lo + hi) >> 1; if (r->zv0[m] > v0) hi = m; else lo = m; }
    const double t = (v0 - r->zv0[lo]) / (r->zv0[lo + 1] - r->zv0[lo]);
    return r->zi[lo] + t * (r->zi[lo + 1] - r->zi[lo]);
}

static inline void zener_uniform(LrZener* r)
{
    for (int k = 0; k < LR_ZU_N; k++) r->zu[k] = zener_sweep(r, k * LR_ZU_H);
}

/* the pair current from v0: the uniform table (odd symmetry), the sweep beyond it */
static inline double zener_i(const LrZener* r, double v0)
{
    const double a = fabs(v0) * (1.0 / LR_ZU_H);
    if (!(a < LR_ZU_N - 1)) return zener_sweep(r, v0);
    const int k = (int)a; const double t = a - k;
    const double i = r->zu[k] + t * (r->zu[k + 1] - r->zu[k]);
    return v0 < 0 ? -i : i;
}

#endif
