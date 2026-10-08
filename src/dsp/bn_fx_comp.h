/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_fx_comp.h — the FX loop's complementary path: what the host cannot carry comes back inside the plugin.
 *
 * The effects loop runs at the host's rate, so the part of the seam signal above the host's Nyquist never returns
 * (measured on the shipped preamp: -44.6 dB of w2's energy, and it moves the neural power stage by
 * -42.7 dB of null). This module keeps a model of the host round trip — the same
 * halfband pair, down then up, with no effect in between — and the complement
 *
 *     c[t] = send[t - L] - up(down(send))[t]
 *
 * which, added to the real return delayed by the host's own loop delay D, gives the seam back exactly when the
 * effect is transparent: return + c = send delayed by L + D, for ANY L (L only decides where c's energy sits; the
 * measured group delay of the pair puts it above the host's band).
 *
 * D is not known to a plugin, and a misaligned complement is worse than none (+2.7..+3.9 ANMR against +0.2), so D is
 * ESTIMATED from the host-rate send and return: a coarse normalised cross-correlation on a 16x boxcar decimation
 * (lags 0..BN_FXC_DMAX), then a fine one at the host rate around the coarse peak. The complement is engaged only
 * while the return correlates with the send at D (rho >= 0.9: the effect is close to linear), scaled by the fitted
 * gain, and faded out (10 ms) the moment rho drops below 0.8, when the search starts again. With a heavy effect
 * (distortion, reverb) it stays off and the loop is the plain loop.
 *
 * Real-time safety: everything is allocated in bn_fxc_init; the audio-thread calls do no allocation and no I/O. */
#ifndef BN_FX_COMP_H
#define BN_FX_COMP_H
#include <stddef.h>
#include <stdint.h>
#include "bn_oversample.h"

#define BN_FXC_DMAX    4096     /* longest host loop delay searched, in host samples (85 ms at 48 kHz) */
#define BN_FXC_DEC     16       /* coarse search decimation */
#define BN_FXC_NCOARSE (BN_FXC_DMAX / BN_FXC_DEC)
#define BN_FXC_NFINE   (2 * BN_FXC_DEC + 1)

typedef struct {
    int           factor, L;            /* oversampling factor; the pair's round trip at fs_hi (measured) */
    BnOversampler os;                   /* the model of the host round trip */
    double       *lo, *m;               /* scratch: down(send) (n), up(down(send)) (n*factor) */
    double       *sd;  size_t sd_mask;  uint64_t sd_w;   /* send at fs_hi, to delay it by L */
    double       *c;   size_t c_mask;   uint64_t c_w;    /* the complement at fs_hi, absolute index c_w */
    double       *s;   size_t s_mask;   uint64_t s_w;    /* host-rate send history */
    double       *sdec; size_t sdec_mask; uint64_t dec_w; /* decimated send history */
    double        dec_s, dec_r; int dec_k;               /* boxcar accumulators */
    int           state;                /* 0 coarse search · 1 fine search · 2 locked */
    long          win_left;             /* samples left in the current search window */
    double        acc_c[BN_FXC_NCOARSE], er_c, es_c;
    double        acc_f[BN_FXC_NFINE], es_f[BN_FXC_NFINE], er_f; int d0;
    int           D;                    /* locked host delay (host samples) */
    double        g;                    /* fitted return/send gain at D */
    double        srs, srr, sss, a_mon; /* monitor: exponential sums at D */
    double        rho;                  /* last monitored correlation */
    int           engaged;
    double        gain_now, gain_step;  /* the complement's fade, per fs_hi sample */
    double        fs_host;
    uint64_t      base;                 /* c_w at the start of the current chunk */
} BnFxComp;

int  bn_fxc_init(BnFxComp* x, int factor, size_t max_block, double fs_host);
void bn_fxc_free(BnFxComp* x);
void bn_fxc_reset(BnFxComp* x);          /* not RT-heavy: clears state and rings (memset) */

/* Call once per chunk BEFORE the per-sample loop: marks where this chunk starts in the complement ring. */
static inline void bn_fxc_begin(BnFxComp* x) { x->base = x->c_w; }

/* Per fs_hi sample i of the current chunk: the complement to add to the up-sampled return (send units),
 * already faded and scaled; 0 when disengaged or when the sample it needs is not written yet (D below the chunk). */
static inline double bn_fxc_sample(BnFxComp* x, size_t i)
{
    const double tgt = x->engaged ? x->g : 0.0;
    if (x->gain_now < tgt) { x->gain_now += x->gain_step; if (x->gain_now > tgt) { x->gain_now = tgt; } }
    else if (x->gain_now > tgt) { x->gain_now -= x->gain_step; if (x->gain_now < tgt) { x->gain_now = tgt; } }
    if (x->gain_now == 0.0) { return 0.0; }
    const uint64_t back = (uint64_t)x->D * (uint64_t)x->factor;
    const uint64_t t = x->base + i;
    if (t < back || t - back >= x->base) { return 0.0; }
    return x->gain_now * x->c[(t - back) & x->c_mask];
}

/* Call once per chunk AFTER the per-sample loop, with this chunk's send at fs_hi (n*factor), the same send brought
 * down to the host rate (n, what the host received) and the host's return for this chunk (n, sanitised). */
void bn_fxc_end(BnFxComp* x, const double* send_hi, const double* send_lo, const double* ret_lo, size_t n);

#endif
