/* bn_oversample.h — a chain of halfband filters to oversample the nonlinear part.
 *
 * A halfband has half its coefficients exactly zero, so it costs half a generic FIR of the same
 * length. Chaining stages reaches 2x, 4x or 8x, each doubling at little cost.
 *
 * Coefficients in bn_halfband_coeffs.h, generated and checked by the generator named there
 * (passband ripple 0.0002 dB, stopband -99.6 dB).
 *
 * No allocation on the audio path: the buffers are sized at construction for the largest block
 * the caller declares (`max_block`).
 */
#ifndef BN_OVERSAMPLE_H
#define BN_OVERSAMPLE_H

#include <stddef.h>

#define BN_OS_MAX_STAGES 3          /* up to 8x */

typedef struct {
    int    factor;                  /* 1, 2, 4 or 8 */
    int    n_stages;
    size_t max_block;
    /* POLYPHASE state per stage. up: the input history (64 entries, written twice, see
     * BN_UP_HIST). down: the even-sample history (64, written twice) and a ring of odd
     * samples (32, masked). */
    double* up_state[BN_OS_MAX_STAGES];
    double* down_state[BN_OS_MAX_STAGES];
    int     up_pos[BN_OS_MAX_STAGES];
    int     down_pos_e[BN_OS_MAX_STAGES];
    int     down_pos_o[BN_OS_MAX_STAGES];
    double* scratch_a;
    double* scratch_b;
    /* the halfband chain's round trip is FRACTIONAL with more than one stage (70.5 base
     * samples at 4x, 82.25 at 8x) and a host only takes an integer latency: a pure delay of
     * `pad` samples at the high rate, after the way up, makes it whole (2 at 4x, 6 at 8x) */
    int     pad, pad_pos;
    double  pad_ring[8];
} BnOversampler;

int  bn_os_init(BnOversampler* os, int factor, size_t max_block);
void bn_os_free(BnOversampler* os);
void bn_os_reset(BnOversampler* os);

/* Up-sample `n` samples to n*factor in `out` (which must have room). */
void bn_os_up(BnOversampler* os, const double* in, size_t n, double* out);
/* Down-sample n*factor samples to `n` in `out`. */
void bn_os_down(BnOversampler* os, const double* in, size_t n, double* out);

/* Total latency in base-rate samples, up and down. The halfband chain alone is FRACTIONAL with
 * more than one stage (70.5 at 4x, 82.25 at 8x); the `pad` delay makes it WHOLE: 47 at 2x, 71 at
 * 4x, 83 at 8x. */
double bn_os_latency_exact(const BnOversampler* os);
int    bn_os_latency(const BnOversampler* os);   /* rounded, for the host */

#endif /* BN_OVERSAMPLE_H */
