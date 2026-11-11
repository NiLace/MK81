/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_fx_pair.c — see bn_fx_pair.h.
 *
 * The histories are LINEAR and written twice (bn_oversample.c, BN_UP_HIST, explains why): a branch read backwards
 * from h[pos + RING] is contiguous and never crosses the edge. The dot products run in four partial accumulators in
 * ascending tap order, the tap of index i (from the branch's first) into accumulator i % 4, reduced as
 * (b0 + b1) + (b2 + b3); the decimator's odd branch goes in first, into b0. On arm 0 (the 95-tap halfband) that is
 * bn_oversample's arithmetic, operation for operation: tools/bn_test_fx_pair.c gates it byte-identical. */
#include "bn_fx_pair.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bn_fx_pair_coeffs.h"

#define RING BN_FXP_RING
#define MASK (BN_FXP_RING - 1)

static const BnFxpStage* stage_of(const BnFxPair* p, int s) { return &BN_FXP_ARMS[p->arm].st[s == 0 ? 0 : 1]; }

static double chain_latency(int arm, int stages)
{
    double rt = 0.0;
    for (int s = 0; s < stages; ++s) { rt += BN_FXP_ARMS[arm].st[s == 0 ? 0 : 1].delay / (double)(1 << s); }
    return rt;
}

static int stages_of(int factor)
{
    switch (factor) { case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3; default: return -1; }
}

int bn_fxp_init_arm(BnFxPair* p, int arm, int factor, size_t max_block)
{
    memset(p, 0, sizeof *p);
    const int stages = stages_of(factor);
    if (stages < 0 || arm < 0 || arm >= BN_FXP_N_ARMS) { return -1; }
    p->arm = arm; p->factor = factor; p->n_stages = stages; p->max_block = max_block;
    {
        const double rt = chain_latency(arm, stages);
        p->pad = (int)lround((ceil(rt - 1e-9) - rt) * factor);
        if (p->pad > 8) { return -1; }
    }
    for (int s = 0; s < stages; ++s) {
        const BnFxpStage* st = stage_of(p, s);
        if (st->e0 + st->ne > RING || st->o0 + st->no > RING) { return -1; }
        p->up_h[s] = (double*)calloc(2 * RING, sizeof(double));
        p->dn_e[s] = (double*)calloc(2 * RING, sizeof(double));
        p->dn_o[s] = (double*)calloc(2 * RING, sizeof(double));
        if (!p->up_h[s] || !p->dn_e[s] || !p->dn_o[s]) { bn_fxp_free(p); return -1; }
    }
    p->scratch_a = (double*)calloc(max_block * (size_t)factor + 8, sizeof(double));
    p->scratch_b = (double*)calloc(max_block * (size_t)factor + 8, sizeof(double));
    if (!p->scratch_a || !p->scratch_b) { bn_fxp_free(p); return -1; }
    return 0;
}

int bn_fxp_init(BnFxPair* p, int factor, size_t max_block) { return bn_fxp_init_arm(p, BN_FX_PAIR, factor, max_block); }

void bn_fxp_free(BnFxPair* p)
{
    for (int s = 0; s < BN_FXP_MAX_STAGES; ++s) { free(p->up_h[s]); free(p->dn_e[s]); free(p->dn_o[s]); }
    free(p->scratch_a); free(p->scratch_b);
    memset(p, 0, sizeof *p);
}

void bn_fxp_reset(BnFxPair* p)
{
    for (int s = 0; s < p->n_stages; ++s) {
        memset(p->up_h[s], 0, 2 * RING * sizeof(double));
        memset(p->dn_e[s], 0, 2 * RING * sizeof(double));
        memset(p->dn_o[s], 0, 2 * RING * sizeof(double));
        p->up_pos[s] = p->dn_pos[s] = 0;
    }
    memset(p->pad_ring, 0, sizeof p->pad_ring);
    p->pad_pos = 0;
}

/* sum_i c[i] * h[-(j0 + i)], four partial accumulators added onto b[0..3] */
static inline void dot4(const double* c, int n, const double* h, double b[4])
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        b[0] += c[i]     * h[-i];
        b[1] += c[i + 1] * h[-i - 1];
        b[2] += c[i + 2] * h[-i - 2];
        b[3] += c[i + 3] * h[-i - 3];
    }
    for (; i < n; ++i) { b[i & 3] += c[i] * h[-i]; }
}

void bn_fxp_up(BnFxPair* p, const double* in, size_t n, double* out)
{
    if (p->n_stages == 0) { memcpy(out, in, n * sizeof(double)); return; }
    const double* src = in;
    size_t len = n;
    double* dst = p->scratch_a;
    for (int s = 0; s < p->n_stages; ++s) {
        const BnFxpStage* st = stage_of(p, s);
        double* target = (s == p->n_stages - 1) ? out : dst;
        double* hist = p->up_h[s];
        int pos = p->up_pos[s];
        for (size_t i = 0; i < len; ++i) {
            pos = (pos + 1) & MASK;
            hist[pos] = src[i];
            hist[pos + RING] = src[i];
            const double* h = hist + pos + RING;
            double b[4] = { 0.0, 0.0, 0.0, 0.0 };
            dot4(st->e, st->ne, h - st->e0, b);
            target[2 * i] = 2.0 * ((b[0] + b[1]) + (b[2] + b[3]));
            double c[4] = { 0.0, 0.0, 0.0, 0.0 };
            dot4(st->o, st->no, h - st->o0, c);
            target[2 * i + 1] = 2.0 * ((c[0] + c[1]) + (c[2] + c[3]));
        }
        p->up_pos[s] = pos;
        src = target;
        len *= 2;
        dst = (dst == p->scratch_a) ? p->scratch_b : p->scratch_a;
    }
    if (p->pad > 0) {
        int q = p->pad_pos;
        for (size_t i = 0; i < len; ++i) {
            const double d = p->pad_ring[q];
            p->pad_ring[q] = out[i];
            out[i] = d;
            if (++q == p->pad) { q = 0; }
        }
        p->pad_pos = q;
    }
}

void bn_fxp_down(BnFxPair* p, const double* in, size_t n, double* out)
{
    if (p->n_stages == 0) { memcpy(out, in, n * sizeof(double)); return; }
    const double* src = in;
    size_t len = n * (size_t)p->factor;
    double* dst = p->scratch_a;
    for (int s = p->n_stages - 1; s >= 0; --s) {
        const BnFxpStage* st = stage_of(p, s);
        double* target = (s == 0) ? out : dst;
        double* he = p->dn_e[s];
        double* ho = p->dn_o[s];
        int pos = p->dn_pos[s];
        for (size_t i = 0; i < len / 2; ++i) {
            /* the even sample of this pair enters; the odd history still ends at the PREVIOUS pair's odd sample,
             * v[2i-1], which is the odd branch's j = 0 */
            const int pe = (pos + 1) & MASK;
            he[pe] = src[2 * i];
            he[pe + RING] = src[2 * i];
            double b[4] = { 0.0, 0.0, 0.0, 0.0 };
            {
                double c[4] = { 0.0, 0.0, 0.0, 0.0 };
                dot4(st->o, st->no, ho + pos + RING - st->o0, c);
                b[0] = (c[0] + c[1]) + (c[2] + c[3]);
            }
            dot4(st->e, st->ne, he + pe + RING - st->e0, b);
            target[i] = (b[0] + b[1]) + (b[2] + b[3]);
            ho[pe] = src[2 * i + 1];
            ho[pe + RING] = src[2 * i + 1];
            pos = pe;
        }
        p->dn_pos[s] = pos;
        src = target;
        len /= 2;
        dst = (dst == p->scratch_a) ? p->scratch_b : p->scratch_a;
    }
}

double bn_fxp_latency_exact(const BnFxPair* p)
{
    return chain_latency(p->arm, p->n_stages) + (double)p->pad / (double)(p->factor > 0 ? p->factor : 1);
}

int bn_fxp_latency(const BnFxPair* p) { return (int)(bn_fxp_latency_exact(p) + 0.5); }

int bn_fxp_roundtrip(int factor)
{
    const int s = stages_of(factor);
    return s < 0 ? -1 : (int)ceil(chain_latency(BN_FX_PAIR, s) - 1e-9);
}

const char* bn_fxp_arm_name(void) { return BN_FXP_ARMS[BN_FX_PAIR].name; }
