/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_fx_comp.c — see bn_fx_comp.h. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "bn_fx_comp.h"

#define RHO_ON   0.90
#define RHO_OFF  0.80
#define E_FLOOR  1e-12     /* per-sample mean square below which a window is silence and decides nothing */

static size_t pow2_at_least(size_t n) { size_t p = 1; while (p < n) { p <<= 1; } return p; }

static void start_search(BnFxComp* x)
{
    x->state = 0; x->engaged = 0;
    x->win_left = (long)(0.5 * x->fs_host / BN_FXC_DEC);
    memset(x->acc_c, 0, sizeof x->acc_c); x->er_c = 0.0; x->es_c = 0.0;
}

int bn_fxc_init(BnFxComp* x, int factor, size_t max_block, double fs_host)
{
    memset(x, 0, sizeof *x);
    x->factor = factor > 0 ? factor : 1; x->fs_host = fs_host;
    if (bn_os_init(&x->os, x->factor, max_block) != 0) { return -1; }
    x->lo = (double*)calloc(max_block, sizeof(double));
    x->m  = (double*)calloc(max_block * (size_t)x->factor + 8, sizeof(double));
    /* the pair's round trip, measured on an impulse: the group delay that puts the complement above the band */
    {
        const size_t nh = 512 * (size_t)x->factor;
        double* imp = (double*)calloc(nh, sizeof(double));
        if (!imp || !x->lo || !x->m) { free(imp); bn_fxc_free(x); return -1; }
        imp[0] = 1.0;
        bn_os_down(&x->os, imp, 512, x->lo);
        bn_os_up(&x->os, x->lo, 512, x->m);
        size_t best = 0;
        for (size_t k = 1; k < nh; ++k) { if (fabs(x->m[k]) > fabs(x->m[best])) { best = k; } }
        x->L = (int)best;
        free(imp);
        bn_os_reset(&x->os);
    }
    x->sd_mask = pow2_at_least((size_t)x->L + max_block * (size_t)x->factor + 1) - 1;
    x->c_mask  = pow2_at_least((size_t)BN_FXC_DMAX * (size_t)x->factor + max_block * (size_t)x->factor + 1) - 1;
    x->s_mask  = pow2_at_least((size_t)BN_FXC_DMAX + BN_FXC_NFINE + max_block + 1) - 1;
    x->sdec_mask = pow2_at_least((size_t)BN_FXC_NCOARSE + 2) - 1;
    x->sd   = (double*)calloc(x->sd_mask + 1, sizeof(double));
    x->c    = (double*)calloc(x->c_mask + 1, sizeof(double));
    x->s    = (double*)calloc(x->s_mask + 1, sizeof(double));
    x->sdec = (double*)calloc(x->sdec_mask + 1, sizeof(double));
    if (!x->sd || !x->c || !x->s || !x->sdec) { bn_fxc_free(x); return -1; }
    x->a_mon = exp(-1.0 / (0.5 * fs_host));
    x->gain_step = 1.0 / (0.010 * fs_host * x->factor);
    bn_fxc_reset(x);
    return 0;
}

void bn_fxc_free(BnFxComp* x)
{
    bn_os_free(&x->os);
    free(x->lo); free(x->m); free(x->sd); free(x->c); free(x->s); free(x->sdec);
    x->lo = x->m = x->sd = x->c = x->s = x->sdec = NULL;
}

void bn_fxc_reset(BnFxComp* x)
{
    bn_os_reset(&x->os);
    if (x->sd)   { memset(x->sd, 0, (x->sd_mask + 1) * sizeof(double)); }
    if (x->c)    { memset(x->c, 0, (x->c_mask + 1) * sizeof(double)); }
    if (x->s)    { memset(x->s, 0, (x->s_mask + 1) * sizeof(double)); }
    if (x->sdec) { memset(x->sdec, 0, (x->sdec_mask + 1) * sizeof(double)); }
    x->sd_w = x->c_w = x->s_w = x->dec_w = x->base = 0;
    x->dec_s = x->dec_r = 0.0; x->dec_k = 0;
    x->D = 1; x->g = 0.0; x->srs = x->srr = x->sss = 0.0; x->rho = 0.0; x->gain_now = 0.0;
    start_search(x);
}

/* one host-rate sample of the estimator: r is the return at absolute host time u = s_w - 1 (the send at u is
 * already in the history) */
static void estimate(BnFxComp* x, double r)
{
    const uint64_t u = x->s_w - 1;
    if (x->state == 2) {
        const double s = (u >= (uint64_t)x->D) ? x->s[(u - (uint64_t)x->D) & x->s_mask] : 0.0;
        const double a = x->a_mon;
        x->srs = a * x->srs + r * s; x->srr = a * x->srr + r * r; x->sss = a * x->sss + s * s;
        return;
    }
    if (x->state == 1) {
        x->er_f += r * r;
        for (int j = 0; j < BN_FXC_NFINE; ++j) {
            const uint64_t lag = (uint64_t)(x->d0 + j);
            const double s = (u >= lag) ? x->s[(u - lag) & x->s_mask] : 0.0;
            x->acc_f[j] += r * s; x->es_f[j] += s * s;
        }
        if (--x->win_left > 0) { return; }
        const double nwin = 0.25 * x->fs_host;
        int best = -1; double brho = 0.0;
        for (int j = 0; j < BN_FXC_NFINE; ++j) {
            if (x->es_f[j] < E_FLOOR * nwin || x->er_f < E_FLOOR * nwin) { continue; }
            const double rho = x->acc_f[j] / sqrt(x->er_f * x->es_f[j]);
            if (rho > brho) { brho = rho; best = j; }
        }
        if (best < 0 || brho < RHO_ON) { start_search(x); return; }
        x->D = x->d0 + best;
        x->g = x->acc_f[best] / x->es_f[best];
        if (x->g < 0.0) { x->g = 0.0; } if (x->g > 4.0) { x->g = 4.0; }
        x->srs = x->acc_f[best]; x->srr = x->er_f; x->sss = x->es_f[best];   /* seed the monitor with the window */
        x->rho = brho; x->state = 2; x->engaged = 1;
        return;
    }
    /* coarse: 16x boxcar of r and s, then every decimated sample correlates against the decimated history */
    x->dec_r += r; x->dec_s += x->s[u & x->s_mask];
    if (++x->dec_k < BN_FXC_DEC) { return; }
    const double rd = x->dec_r / BN_FXC_DEC, sdv = x->dec_s / BN_FXC_DEC;
    x->dec_r = x->dec_s = 0.0; x->dec_k = 0;
    x->sdec[x->dec_w & x->sdec_mask] = sdv; ++x->dec_w;
    x->er_c += rd * rd; x->es_c += sdv * sdv;
    const uint64_t m = x->dec_w - 1;
    for (int k = 0; k < BN_FXC_NCOARSE; ++k) {
        if (m < (uint64_t)k) { break; }
        x->acc_c[k] += rd * x->sdec[(m - (uint64_t)k) & x->sdec_mask];
    }
    if (--x->win_left > 0) { return; }
    const double nwin = 0.5 * x->fs_host / BN_FXC_DEC;
    if (x->er_c < E_FLOOR * nwin || x->es_c < E_FLOOR * nwin) { start_search(x); return; }
    int kb = 0;
    for (int k = 1; k < BN_FXC_NCOARSE; ++k) { if (x->acc_c[k] > x->acc_c[kb]) { kb = k; } }
    if (x->acc_c[kb] / sqrt(x->er_c * x->es_c) < 0.5) { start_search(x); return; }
    x->d0 = kb * BN_FXC_DEC - BN_FXC_DEC; if (x->d0 < 1) { x->d0 = 1; }
    memset(x->acc_f, 0, sizeof x->acc_f); memset(x->es_f, 0, sizeof x->es_f); x->er_f = 0.0;
    x->win_left = (long)(0.25 * x->fs_host); x->state = 1;
}

void bn_fxc_end(BnFxComp* x, const double* send_hi, const double* send_lo, const double* ret_lo, size_t n)
{
    const size_t nh = n * (size_t)x->factor;
    /* the complement of this chunk: send delayed by L, minus the model of the host round trip */
    bn_os_down(&x->os, send_hi, n, x->lo);
    bn_os_up(&x->os, x->lo, n, x->m);
    for (size_t i = 0; i < nh; ++i) {
        x->sd[x->sd_w & x->sd_mask] = send_hi[i]; ++x->sd_w;
        const double sdel = (x->sd_w > (uint64_t)x->L) ? x->sd[(x->sd_w - 1 - (uint64_t)x->L) & x->sd_mask] : 0.0;
        x->c[x->c_w & x->c_mask] = sdel - x->m[i]; ++x->c_w;
    }
    /* the estimator, at the host rate */
    for (size_t j = 0; j < n; ++j) {
        x->s[x->s_w & x->s_mask] = send_lo[j]; ++x->s_w;
        estimate(x, ret_lo[j]);
    }
    if (x->state == 2) {
        const double den = sqrt(x->srr * x->sss);
        if (x->sss > E_FLOOR / (1.0 - x->a_mon) && x->srr > E_FLOOR / (1.0 - x->a_mon) && den > 0.0) {
            x->rho = x->srs / den;
            if (x->rho < RHO_OFF) { start_search(x); return; }
            double g = x->srs / x->sss; if (g < 0.0) { g = 0.0; } if (g > 4.0) { g = 4.0; }
            x->g = g;
        }   /* silence: hold the lock, decide nothing */
    }
}
