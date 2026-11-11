/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_dev_zener.h @ c0e3536, sha256 72f3b4cdddf5f5c2708a98b3595b10e5de7baa1bafe12ba55944ccb803d37370 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* nl_dev_zener.h - a zener diode as ONE port (nlcore#25, MK81#165). Header only, no policy: nothing in the
 * core includes it; a circuit that has a zener includes it in its device code.
 *
 * The law is the structure of Vishay's BZX55 macro-model (all seven members Vishay publishes share it), DC part:
 *   DFOR  anode -> cathode   forward junction IS N with series RS
 *   DREV  cathode -> anode   the soft pre-breakdown conduction: IS N, no RS
 *   DBRE  breakdown junction IS N with series RS, behind the offset VOFF = V0 + 1 V (Vishay's EB POLY(1) d 0 V0 1 with
 *         IB 1 mA into RB 1 kohm: 1 V at TNOM; at another temperature the caller passes its own VOFF)
 * At the port voltage v = v_anode - v_cathode (forward positive), with vr = -v:
 *   i(v) = I_F(v) - I_DREV(vr) - I_DBRE(vr - VOFF)
 * Each junction with a series resistance is solved EXACTLY: v = nVt ln(1 + i / IS) + i RS has i + IS = (nVt / RS) u,
 * with u the Wright omega of x = (v + IS RS) / nVt - ln(nVt / (IS RS)), i.e. u + ln u = x (Newton from a close start;
 * no table). Its slope is u / (1 + u) / RS. A junction without RS continues as its tangent past NL_ZENER_IMAX, so a
 * Newton probe far outside cannot overflow (no circuit here comes near 1 A).
 * GMIN is not part of the law (ngspice adds 1e-12 S across each junction; across DBRE, which VOFF holds reverse-biased,
 * that is a current that does not exist: the workshop note gmin-across-a-junction-behind-an-offset-source-
 * charges-a-floating-node.md).
 *
 * Charge: DFOR's depletion charge, ngspice's dioload law with the FC linearisation, across the port (DFOR's RS is not
 * between them: 0.4 ohm x 52 pF = 20 ps for the BZX55B12); TT is 0 in the whole family. q(v) and c(v) = dq / dv; the
 * caller integrates it (trapezoidal: i = 2 (q - q_prev) / T - i_prev).
 */
#ifndef NL_DEV_ZENER_H
#define NL_DEV_ZENER_H

#include <math.h>

#ifndef NL_ZENER_IMAX
#define NL_ZENER_IMAX 1.0          /* A: an RS-less junction past this current continues as its tangent */
#endif

typedef struct {
    double vt;                          /* kT/q at the device temperature */
    double f_is, f_n, f_rs;             /* DFOR */
    double r_is, r_n;                   /* DREV (r_is = 0: no soft branch) */
    double b_is, b_n, b_rs, voff;       /* DBRE behind VOFF */
    double cjo, vj, m, fc;              /* DFOR's depletion charge (cjo = 0: none) */
    /* derived by nl_zener_init */
    double f_nvt, b_nvt, r_nvt, f_lx, b_lx, r_vmax, r_imax_g;
    double q_f1, q_f2, q_f3;
} NlZener;

static inline void nl_zener_init(NlZener* z)
{
    z->f_nvt = z->f_n * z->vt;
    z->b_nvt = z->b_n * z->vt;
    z->r_nvt = z->r_n * z->vt;
    z->f_lx = z->f_rs > 0.0 ? log(z->f_nvt / (z->f_is * z->f_rs)) : 0.0;
    z->b_lx = z->b_rs > 0.0 ? log(z->b_nvt / (z->b_is * z->b_rs)) : 0.0;
    if (z->r_is > 0.0) {
        z->r_vmax = z->r_nvt * log(NL_ZENER_IMAX / z->r_is + 1.0);
        z->r_imax_g = (NL_ZENER_IMAX + z->r_is) / z->r_nvt;
    }
    if (z->cjo > 0.0) {
        z->q_f1 = z->vj * (1.0 - pow(1.0 - z->fc, 1.0 - z->m)) / (1.0 - z->m);
        z->q_f2 = pow(1.0 - z->fc, 1.0 + z->m);
        z->q_f3 = 1.0 - z->fc * (1.0 + z->m);
    }
}

/* Wright omega: the u > 0 with u + ln u = x */
static inline double nl_zener_omega(double x)
{
    if (x < -30.0) { return exp(x); }                      /* u = e^x (1 - e^x + ...): exact to e^-30 relative */
    double u = x < 1.0 ? exp(x) : x - log(x);
    if (x >= 1.0 && x < 3.0) { u = 0.5 * (u + exp(x / 3.0)); }
    for (int k = 0; k < 20; ++k) {
        const double f = u + log(u) - x;
        const double du = f * u / (1.0 + u);
        u -= du;
        if (u <= 0.0) { u = 1e-300; }
        if (fabs(du) <= 1e-15 * u) { break; }
    }
    return u;
}

/* a junction IS N with series RS at total voltage v: current and slope */
static inline void nl_zener_rs_branch(double v, double is, double nvt, double rs, double lx, double* i, double* g)
{
    const double u = nl_zener_omega((v + is * rs) / nvt - lx);
    *i = nvt / rs * u - is;
    *g = u / (1.0 + u) / rs;
}

static inline void nl_zener_iv(const NlZener* z, double v, double* i, double* g)
{
    double ii, gg, ib, gb;
    if (z->f_rs > 0.0) { nl_zener_rs_branch(v, z->f_is, z->f_nvt, z->f_rs, z->f_lx, &ii, &gg); }
    else {
        const double e = exp(fmin(v / z->f_nvt, 700.0));
        ii = z->f_is * (e - 1.0); gg = z->f_is * e / z->f_nvt;
    }
    const double vr = -v;
    if (z->r_is > 0.0) {
        if (vr < z->r_vmax) {
            const double e = exp(vr / z->r_nvt);
            ii -= z->r_is * (e - 1.0); gg += z->r_is * e / z->r_nvt;
        } else {
            ii -= NL_ZENER_IMAX + (vr - z->r_vmax) * z->r_imax_g; gg += z->r_imax_g;
        }
    }
    nl_zener_rs_branch(vr - z->voff, z->b_is, z->b_nvt, z->b_rs, z->b_lx, &ib, &gb);
    *i = ii - ib;
    *g = gg + gb;
}

/* DFOR's depletion charge and its capacitance at the port voltage v (ngspice dioload.c) */
static inline void nl_zener_qc(const NlZener* z, double v, double* q, double* c)
{
    if (z->cjo <= 0.0) { *q = 0.0; *c = 0.0; return; }
    const double vj = z->vj, m = z->m, fcv = z->fc * vj;
    if (v < fcv) {
        const double a = 1.0 - v / vj, s = pow(a, -m);
        *q = z->cjo * vj * (1.0 - a * s) / (1.0 - m);
        *c = z->cjo * s;
        return;
    }
    *q = z->cjo * z->q_f1 + z->cjo / z->q_f2 * (z->q_f3 * (v - fcv) + m / (2.0 * vj) * (v * v - fcv * fcv));
    *c = z->cjo / z->q_f2 * (z->q_f3 + m * v / vj);
}

#endif
