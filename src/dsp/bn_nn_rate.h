/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_nn_rate.h — runs the neural power stage at the rate it was trained at (96 kHz) inside a loop
 * that ticks at any fs_hi. A recurrent net's time constants are its sample period, so running it at
 * another rate is running another model: at fs_hi = 192 kHz it scores no better than the cascade.
 *
 * Two band-limited interpolators around the engine: fs_hi -> 96 kHz on the way in, 96 kHz -> fs_hi
 * on the way out. Each is a Kaiser-windowed sinc sampled into a fixed table (linear interpolation
 * between table points) and normalised per evaluation to unit DC gain. The pass band is the audio
 * band plus a margin (24 kHz, 20 kHz at 44.1); each stop band starts where content would fold into
 * the pass band (bn_nn_rate.c says where it is stricter). The rail is quasi-static and is
 * interpolated linearly.
 *
 * The output lags the input by `delay` fs_hi samples (an integer, fixed per fs_hi): the caller
 * aligns everything else to it. When fs_hi equals the model rate the converter is inactive, delay
 * is 0, and the caller runs the engine directly (bit-identical to the engine alone).
 * RT-safe after bn_nn_rate_init() (called outside the audio thread). */
#ifndef BN_NN_RATE_H
#define BN_NN_RATE_H
#include "bn_nn_lstm.h"

#define BN_NN_MODEL_RATE 96000.0      /* the default: a BNN2 net's rate; a BNN3 net carries its own */
#define BN_NN_RATE_RING  2048          /* power of two; holds the longest kernel span (a 48 kHz net at 768 kHz) */
#define BN_NN_RATE_TAB   4096          /* table points per kernel, over its whole span */

typedef struct {
    int    L;                          /* half-span in samples of the kernel's own spacing */
    double step;                       /* table points per sample of spacing */
    float  tab[BN_NN_RATE_TAB + 2];
} BnNnKernel;

typedef struct {
    int        active;                 /* 0 = fs_hi is the model rate: no conversion */
    int        delay;                  /* fs_hi samples from input to output */
    double     rho;                    /* fs_hi samples per model sample */
    double     model_fs;               /* the rate the engine runs at (the net's training rate) */
    double     fpass;                  /* Hz: the band the conversion is transparent in */
    BnNnKernel kin, kout;              /* in: spacing 1/fs_hi · out: spacing 1/96 kHz */
    float      xin[BN_NN_RATE_RING];   /* w2 history at fs_hi */
    float      rin[BN_NN_RATE_RING];   /* rail history at fs_hi */
    float      yout[BN_NN_RATE_RING];  /* the engine's output history at 96 kHz */
    int        pin, pout;              /* next write positions */
    double     tin;                    /* fs_hi samples until the next model sample is due */
    double     tm_rel;                 /* time of the newest model sample relative to the newest input */
    long       n_in, n_out;            /* samples written so far (warm-up guard) */
} BnNnRate;

/* build for fs_hi and the net's own rate (outside the audio thread); returns the delay in fs_hi samples */
int   bn_nn_rate_init(BnNnRate* r, double fs_hi, double model_fs);
void  bn_nn_rate_reset(BnNnRate* r);
/* one fs_hi tick: push w2 and the rail, run the engine for every 96 kHz sample now due, return the
 * engine's output at fs_hi, `delay` samples late. Knobs as bn_nn_lstm_stage. */
float bn_nn_rate_tick(BnNnRate* r, BnNnLstm* g, float w2, float rail,
                      float pres, float bias, float cab, float head, float us);
/* the same tick around any per-sample engine (the calibration test drives an identity through it) */
typedef float (*BnNnEngineFn)(void* ctx, float w2, float rail);
float bn_nn_rate_tick_fn(BnNnRate* r, BnNnEngineFn fn, void* ctx, float w2, float rail);
#endif
