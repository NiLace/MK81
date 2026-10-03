/* bn_nn_lstm.h — the neural power stage of MK 81 as it ships: an LSTM
 * (PyTorch layout, gates i, f, g, o) on five inputs — w2, the rail factor, presence, bias,
 * cabinet feel — in the RAIL'S FRAME: y = r · net(w2 / r, r, knobs). The frame is the cascade's
 * own s·f(u/s); the net learns the deviation. One engine, four weight sets
 * (100 W / 50 W × UK / US), each trained from the exact circuit of its arm at 16×
 * and decimated to 96 kHz: the engine's sample rate is the pairs' (96 kHz). At any other fs_hi
 * the plugin runs it through bn_nn_rate (a resampler on each side), never directly: run at another
 * rate its time constants scale and it is another model.
 *
 * Activation: tanh by a 4096-point table with linear interpolation (the GRU engine's, gated at
 * -92.9 dB against PyTorch); sigmoid derived from it. Where SSE2 is built in, the vector path
 * uses tanh_v (a polynomial exp2) instead. RT-safe after bn_nn_lstm_lut_init() (called from
 * instantiate). No allocation in process(). */
#ifndef BN_NN_LSTM_H
#define BN_NN_LSTM_H
#include <stddef.h>

#define BN_NN_LSTM_MAX_H  128
#define BN_NN_LSTM_MAX_IN 8

enum { BN_NN_COND_PRES = 0, BN_NN_COND_BIAS = 1, BN_NN_COND_CAB = 2, BN_NN_COND_HEAD = 3, BN_NN_COND_US = 4 };
enum { BN_NN_FLAG_RAIL = 1, BN_NN_FLAG_RAIL_SCALE = 2 };

typedef struct {
    int H, C;                     /* hidden size (multiple of 4), input channels */
    int flags;                    /* BN_NN_FLAG_* */
    int n_cond;                   /* conditioning channels after w2 [and rail] */
    int cond[BN_NN_LSTM_MAX_IN];  /* BN_NN_COND_* per conditioning channel, in input order */
    float g_in, g_out;            /* the trainer's scales */
    float fs;                     /* the rate the net was trained at: 96 kHz for a BNN2 blob, stored in a BNN3 */
    const float* w_ih;            /* 4H x C */
    const float* w_hh;            /* 4H x H */
    const float* b_ih;            /* 4H */
    const float* b_hh;            /* 4H */
    const float* w_out;           /* H */
    float b_out;
    float h[BN_NN_LSTM_MAX_H], c[BN_NN_LSTM_MAX_H];         /* state */
    float h_rest[BN_NN_LSTM_MAX_H], c_rest[BN_NN_LSTM_MAX_H]; /* the state the net settles to on silence at rest (bn_nn_lstm_settle) */
    float y_rest; int has_rest;                              /* its output there; 0 until settled */
    float gi[4 * BN_NN_LSTM_MAX_H], gh[4 * BN_NN_LSTM_MAX_H]; /* scratch */
    void* owned;                  /* the blob when loaded from a file (free'd), NULL when static */
    /* the two gate matrices packed TRANSPOSED and concatenated, [w_hh | w_ih]^T: row k is the
     * weight of input k (h[0..H-1], then x[0..C-1]) on all 4H gates, so the gates accumulate in
     * parallel (independent lanes) instead of as 4H horizontal dot products. Built at bind
     * (outside the audio thread), freed by bn_nn_lstm_free. NULL: the plain path runs. */
    float* wt;
    float* bsum;                  /* b_ih + b_hh */
    /* PROBE (-DBN_NN_INT16, not shipped): the recurrent weights in fixed point for SSE2 pmaddwd */
    short* wq;                    /* per 16-gate block, per k pair: 4 groups x (4 gates x 2 k) int16 */
    float* sw;                    /* per gate: weight scale (float weight = q * sw) */
} BnNnLstm;

/* bind to a weight blob in the .bnn2 layout (see nn_export.py); the blob must outlive the engine */
int  bn_nn_lstm_bind(BnNnLstm* g, const void* blob, size_t bytes);
int  bn_nn_lstm_load(BnNnLstm* g, const char* path);   /* file -> owned blob -> bind */
void bn_nn_lstm_free(BnNnLstm* g);                    /* also for a bound (static) blob: frees the packed copy */
void bn_nn_lstm_reset(BnNnLstm* g);                    /* zero state (what PyTorch starts from: the parity test uses it) */
/* The zero state is NOT the net at rest: from it the net emits a start-up transient on silence (0.32 peak at the
 * plugin's output 1.2 ms after activate, a tail to ~300 ms). settle runs n steps of silence at the rest rail with
 * the given knobs from zero and stores the state it lands on; rest restores it (zero if never settled). Both are
 * for OUTSIDE the audio thread or cheap copies: settle costs n steps, rest a memcpy. */
float bn_nn_lstm_settle(BnNnLstm* g, int n, float pres, float bias, float cab, float head, float us);
void  bn_nn_lstm_rest(BnNnLstm* g);
void bn_nn_lstm_lut_init(void);
/* one sample: w2 in volts (the preamp's w2 at fs_hi), rail = vpi / RAIL_NOM (1 = rest), knobs 0..1;
 * returns t8 in volts. The rail's frame and the input encodings are applied inside. */
float bn_nn_lstm_stage(BnNnLstm* g, float w2, float rail, float pres, float bias, float cab, float head, float us);
/* raw step on an already-encoded input vector x[C] (the parity test drives it this way too) */
float bn_nn_lstm_step(BnNnLstm* g, const float* x);
/* the gate's check of the engine's tanh against double precision over [-12, 12]: max abs error (the vector
 * path where SSE2 is built in, the table otherwise) */
/* ⛔ only with -DBN_NN_DIAG: 768 KB of `.bss` the product does not use (see the .c) */
#endif
