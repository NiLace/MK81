/* bn_nn_lstm.c — see bn_nn_lstm.h. PyTorch LSTM (gate order i, f, g, o):
 *   i = s(W_ii x + b_ii + W_hi h + b_hi)   f = s(...)   g = tanh(...)   o = s(...)
 *   c' = f * c + i * g ;  h' = o * tanh(c')
 * The hidden mat-vec uses four explicit partial sums (the form SSE2 vectorises; the model's
 * truth is PyTorch to a gate, not a bit pattern: the parity gate says what the association
 * costs). Encodings are the trainer's (nn_potencia.py feat()): w2 / g_in [after the frame's
 * division by the rail], rail (r - 0.9) / 0.1, knobs 2k - 1. */
#include "bn_nn_lstm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BN_NN_GATE_BLOCK 32      /* gates accumulated per pass: 8 SSE registers */
/* PROBE fixed point (-DBN_NN_INT16), NOT shipped. Measured on the four arms: parity vs PyTorch -56..-64 dB
 * (max error 1.66 V on 20 V rms; the float path is -101..-110) for ~10 % of step time (2 200 -> 1 960 ns).
 * The recurrent loop amplifies the quantisation and 13 bits is the ceiling before the int32 sums overflow.
 * It stays off; it is only worth enabling with a premise that changes that (wider exact accumulation).
 * Weights and h on 13 bits. |q_w q_h| <= 8191^2 = 6.7e7, a pmaddwd pair <= 1.34e8,
 * 8 pairs <= 1.07e9 < 2^31: the int32 accumulators are flushed to float every BN_NN_QGRP pairs. */
#define BN_NN_QW   8191.0f
#define BN_NN_QH   8191.0f
#define BN_NN_QGRP 8
#define LUT_N   4096
#define LUT_LIM 9.0f
static float lut_tanh[LUT_N + 2];
static int lut_ready = 0;
void bn_nn_lstm_lut_init(void)
{
    for (int i = 0; i <= LUT_N + 1; ++i) {
        const float x = -LUT_LIM + (2.0f * LUT_LIM) * (float)i / (float)LUT_N;
        lut_tanh[i] = tanhf(x);
    }
    lut_ready = 1;
}
static inline float ltanh(float x)
{
    if (x >= LUT_LIM) return 1.0f;
    if (x <= -LUT_LIM) return -1.0f;
    const float u = (x + LUT_LIM) * ((float)LUT_N / (2.0f * LUT_LIM));
    const int i = (int)u;
    const float f = u - (float)i;
    return lut_tanh[i] + f * (lut_tanh[i + 1] - lut_tanh[i]);
}
static inline float sigm(float x) { return 0.5f + 0.5f * ltanh(0.5f * x); }

#if defined(__SSE2__)
#include <emmintrin.h>
/* 2^f on [0,1), degree 6, lowest order first: tools/gen_exp2_poly.py (max rel err 8.5e-8 in float32) */
static const float BN_NN_EXP2_P[7] = { 1.0f, 0.693147004f, 0.240229800f, 0.0554835238f,
                                       0.00967847276f, 0.00124430819f, 0.000216906090f };
/* dst[k] = tanh(s * src[k]) four at a time, n a multiple of 4. tanh(x) = sign(x) (1 - 2 / (e^{2|x|} + 1)),
 * |x| clamped at 9 (tanh(9) = 1 - 3e-8); e^{2|x|} = 2^y, y = 2|x| log2(e): the integer part of y goes
 * straight into the float's exponent, the fraction through the polynomial. Absolute error ~1e-7, against
 * ~2e-6 for the 4096-point table it replaces, and no branches, no gathers. */
static void tanh_v(float* dst, const float* src, float s, int n)
{
    const __m128 vs = _mm_set1_ps(s), lim = _mm_set1_ps(9.0f), k2l = _mm_set1_ps(2.0f * 1.44269504088896341f);
    const __m128 one = _mm_set1_ps(1.0f), two = _mm_set1_ps(2.0f);
    const __m128 sign = _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000u));
    for (int k = 0; k < n; k += 4) {
        const __m128 x = _mm_mul_ps(_mm_loadu_ps(src + k), vs);
        const __m128 ax = _mm_min_ps(_mm_andnot_ps(sign, x), lim);
        const __m128 y = _mm_mul_ps(ax, k2l);
        const __m128i ni = _mm_cvttps_epi32(y);                         /* y >= 0: truncation = floor */
        const __m128 f = _mm_sub_ps(y, _mm_cvtepi32_ps(ni));
        __m128 p = _mm_set1_ps(BN_NN_EXP2_P[6]);
        for (int j = 5; j >= 0; --j) p = _mm_add_ps(_mm_mul_ps(p, f), _mm_set1_ps(BN_NN_EXP2_P[j]));
        const __m128 e = _mm_mul_ps(p, _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ni, _mm_set1_epi32(127)), 23)));
        const __m128 t = _mm_sub_ps(one, _mm_div_ps(two, _mm_add_ps(e, one)));
        _mm_storeu_ps(dst + k, _mm_or_ps(t, _mm_and_ps(sign, x)));
    }
}
#endif

int bn_nn_lstm_bind(BnNnLstm* g, const void* blob, size_t bytes)
{
    const unsigned char* p = (const unsigned char*)blob;
    size_t need = 4 + 4 + 4 + 4 + 4 + 4 + 4;
    /* BNN2: trained at 96 kHz. BNN3: the same layout with the training rate appended as one float. */
    const int v3 = bytes >= 4 && memcmp(p, "BNN3", 4) == 0;
    if (bytes < need || (memcmp(p, "BNN2", 4) != 0 && !v3)) return -2;
    int H, C, flags, ncond; float gin, gout;
    memcpy(&H, p + 4, 4); memcpy(&C, p + 8, 4); memcpy(&gin, p + 12, 4); memcpy(&gout, p + 16, 4);
    memcpy(&flags, p + 20, 4); memcpy(&ncond, p + 24, 4);
    if (H < 4 || H > BN_NN_LSTM_MAX_H || (H & 3) || C < 1 || C > BN_NN_LSTM_MAX_IN || ncond < 0 || ncond > BN_NN_LSTM_MAX_IN) return -3;
    if (1 + ((flags & BN_NN_FLAG_RAIL) ? 1 : 0) + ncond != C) return -3;
    size_t off = 28;
    need = off + 4 * (size_t)ncond + 4 * (4 * (size_t)H * C + 4 * (size_t)H * H + 8 * (size_t)H + H + 1);
    if (bytes < need) return -4;
    memset(g->h, 0, sizeof g->h); memset(g->c, 0, sizeof g->c);
    g->has_rest = 0; g->y_rest = 0.0f;   /* a new blob: its rest state is not the previous one's */
    g->H = H; g->C = C; g->flags = flags; g->n_cond = ncond; g->g_in = gin; g->g_out = gout;
    for (int k = 0; k < ncond; ++k) { memcpy(&g->cond[k], p + off, 4); off += 4; }
    g->w_ih = (const float*)(p + off); off += 4 * (size_t)4 * H * C;
    g->w_hh = (const float*)(p + off); off += 4 * (size_t)4 * H * H;
    g->b_ih = (const float*)(p + off); off += 4 * (size_t)4 * H;
    g->b_hh = (const float*)(p + off); off += 4 * (size_t)4 * H;
    g->w_out = (const float*)(p + off); off += 4 * (size_t)H;
    memcpy(&g->b_out, p + off, 4);
    g->fs = 96000.0f;
    if (v3) {
        if (bytes < need + 4) return -4;
        memcpy(&g->fs, p + off + 4, 4);
        if (!(g->fs >= 8000.0f && g->fs <= 768000.0f)) return -3;
    }
    if (!lut_ready) bn_nn_lstm_lut_init();
    g->wt = NULL; g->bsum = NULL;
    if ((4 * H) % BN_NN_GATE_BLOCK == 0) {
        const int G = 4 * H, K = H + C;
        g->wt = malloc(sizeof(float) * (size_t)G * (size_t)K);
        g->bsum = malloc(sizeof(float) * (size_t)G);
        if (!g->wt || !g->bsum) { free(g->wt); free(g->bsum); g->wt = g->bsum = NULL; return -5; }
        for (int i = 0; i < G; ++i) {
            for (int k = 0; k < H; ++k) g->wt[(size_t)k * G + i] = g->w_hh[(size_t)i * H + k];
            for (int k = 0; k < C; ++k) g->wt[(size_t)(H + k) * G + i] = g->w_ih[(size_t)i * C + k];
            g->bsum[i] = g->b_ih[i] + g->b_hh[i];
        }
    }
    g->wq = NULL; g->sw = NULL;
    return 0;
}

int bn_nn_lstm_load(BnNnLstm* g, const char* path)
{
    memset(g, 0, sizeof *g);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return -1; }
    void* blob = malloc((size_t)n);
    if (!blob || fread(blob, 1, (size_t)n, f) != (size_t)n) { free(blob); fclose(f); return -1; }
    fclose(f);
    const int rc = bn_nn_lstm_bind(g, blob, (size_t)n);
    if (rc) { free(blob); return rc; }
    g->owned = blob;
    return 0;
}

void bn_nn_lstm_free(BnNnLstm* g)
{
    free(g->owned); g->owned = NULL; g->w_ih = g->w_hh = g->b_ih = g->b_hh = g->w_out = NULL;
    free(g->wt); free(g->bsum); g->wt = g->bsum = NULL;
    free(g->wq); free(g->sw); g->wq = NULL; g->sw = NULL;
}
void bn_nn_lstm_reset(BnNnLstm* g) { memset(g->h, 0, sizeof g->h); memset(g->c, 0, sizeof g->c); }

float bn_nn_lstm_settle(BnNnLstm* g, int n, float pres, float bias, float cab, float head, float us)
{
    bn_nn_lstm_reset(g);
    float y = 0.0f;
    for (int i = 0; i < n; ++i) { y = bn_nn_lstm_stage(g, 0.0f, 1.0f, pres, bias, cab, head, us); }
    memcpy(g->h_rest, g->h, sizeof g->h); memcpy(g->c_rest, g->c, sizeof g->c);
    g->y_rest = y; g->has_rest = 1;
    return y;
}

void bn_nn_lstm_rest(BnNnLstm* g)
{
    if (!g->has_rest) { bn_nn_lstm_reset(g); return; }
    memcpy(g->h, g->h_rest, sizeof g->h); memcpy(g->c, g->c_rest, sizeof g->c);
}

float bn_nn_lstm_step(BnNnLstm* g, const float* x)
{
    const int H = g->H, C = g->C;
    float* gi = g->gi; float* gh = g->gh; float* h = g->h; float* c = g->c;
    if (g->wt) {
        /* the packed path: all gates' pre-activations z = bsum + W^T [h; x], BN_NN_GATE_BLOCK gates
         * at a time with the accumulators in registers; each input broadcasts once per block. */
        const int G = 4 * H;
        for (int ib = 0; ib < G; ib += BN_NN_GATE_BLOCK) {
            float acc[BN_NN_GATE_BLOCK];
            for (int t = 0; t < BN_NN_GATE_BLOCK; ++t) acc[t] = g->bsum[ib + t];
            const float* w = g->wt + ib;
            for (int k = 0; k < H; ++k, w += G) {
                const float hk = h[k];
                for (int t = 0; t < BN_NN_GATE_BLOCK; ++t) acc[t] += w[t] * hk;
            }
            for (int k = 0; k < C; ++k, w += G) {
                const float xk = x[k];
                for (int t = 0; t < BN_NN_GATE_BLOCK; ++t) acc[t] += w[t] * xk;
            }
            for (int t = 0; t < BN_NN_GATE_BLOCK; ++t) gi[ib + t] = acc[t];
        }
        gh = NULL;                                   /* gi already holds the whole pre-activation */
    } else {
    for (int i = 0; i < 4 * H; ++i) {
        float s = g->b_ih[i];
        for (int k = 0; k < C; ++k) s += g->w_ih[i * C + k] * x[k];
        gi[i] = s;
    }
    for (int i = 0; i < 4 * H; ++i) {
        const float* row = g->w_hh + (size_t)i * H;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int j = 0; j < H; j += 4) {
            s0 += row[j] * h[j];         s1 += row[j + 1] * h[j + 1];
            s2 += row[j + 2] * h[j + 2]; s3 += row[j + 3] * h[j + 3];
        }
        gh[i] = g->b_hh[i] + ((s0 + s1) + (s2 + s3));
    }
    }
    float y = g->b_out;
    if (gh) { for (int i = 0; i < 4 * H; ++i) gi[i] += gh[i]; }
#if defined(__SSE2__)
    if ((H & 3) == 0) {
        /* the vector activations: sigmoid(z) = 1/2 + tanh(z/2)/2, the four gates then tanh(c) */
        float* a = g->gh;                            /* scratch: gi holds the pre-activations */
        tanh_v(a,         gi,         0.5f, H);      /* input gate */
        tanh_v(a + H,     gi + H,     0.5f, H);      /* forget gate */
        tanh_v(a + 2 * H, gi + 2 * H, 1.0f, H);      /* cell candidate */
        tanh_v(a + 3 * H, gi + 3 * H, 0.5f, H);      /* output gate */
        for (int i = 0; i < H; ++i) {
            const float ig = 0.5f + 0.5f * a[i];
            const float fg = 0.5f + 0.5f * a[H + i];
            c[i] = fg * c[i] + ig * a[2 * H + i];
        }
        tanh_v(a, c, 1.0f, H);
        /* `y` is the one declared above: it already holds g->b_out. */
        for (int i = 0; i < H; ++i) {
            h[i] = (0.5f + 0.5f * a[3 * H + i]) * a[i];
            y += g->w_out[i] * h[i];
        }
        return y;
    }
#endif
    for (int i = 0; i < H; ++i) {
        const float ig = sigm(gi[i]);
        const float fg = sigm(gi[H + i]);
        const float gg = ltanh(gi[2 * H + i]);
        const float og = sigm(gi[3 * H + i]);
        c[i] = fg * c[i] + ig * gg;
        h[i] = og * ltanh(c[i]);
        y += g->w_out[i] * h[i];
    }
    return y;
}

/* ⛔ FOR THE TEST GATE ONLY, kept out of the shipped `.so`. Its two `static float[96000]` are
 * **768 KB of `.bss`** that the product never touches: the only caller is
 * `tools/bn_test_nn_lstm.c`, which is built with -DBN_NN_DIAG.
 * Compiled in, it would cost zero CPU and 768 KB of memory. */

float bn_nn_lstm_stage(BnNnLstm* g, float w2, float rail, float pres, float bias, float cab, float head, float us)
{
    float x[BN_NN_LSTM_MAX_IN];
    int k = 0;
    const float r = (g->flags & BN_NN_FLAG_RAIL_SCALE) ? rail : 1.0f;
    x[k++] = (w2 / r) / g->g_in;
    if (g->flags & BN_NN_FLAG_RAIL) x[k++] = (rail - 0.9f) * 10.0f;
    for (int q = 0; q < g->n_cond; ++q) {
        float v = 0.0f;
        switch (g->cond[q]) {
            case BN_NN_COND_PRES: v = pres; break;  case BN_NN_COND_BIAS: v = bias; break;
            case BN_NN_COND_CAB:  v = cab;  break;  case BN_NN_COND_HEAD: v = head; break;
            case BN_NN_COND_US:   v = us;   break;
        }
        x[k++] = 2.0f * v - 1.0f;
    }
    return bn_nn_lstm_step(g, x) * g->g_out * r;
}
