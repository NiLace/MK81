/* bn_oversample.c — the halfband chain in POLYPHASE form.
 *
 * The halfband (BN_HB_TAPS = 95, centre 47) decomposes as:
 *   - every tap k != 47 with k ODD is zero -> only the centre and the 48 even taps P[j] = BN_HB[2j] remain;
 *   - interpolator: even output = 2·Σ P[j]·x[i−j]; odd output = 2·c47·x[i−23];
 *   - decimator:    y[i] = c47·v_odd[2i−47] + Σ P[j]·v_even[2i−2j].
 * The polyphase form does per output sample only the products that are not zero, with no memmove.
 *
 * The histories that feed a FIR loop are LINEAR, 128 entries, each sample written twice (see
 * BN_UP_HIST); the decimator's odd branch is read at ONE index, so it stays a masked ring of 32.
 * The sums run in four partial accumulators: that reassociates them with respect to a single
 * accumulator, well below float32's floor on a LINEAR filter, and breaks the floating-point
 * latency chain that bounded the loop.
 */
#include "bn_oversample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bn_halfband_coeffs.h"

#define BN_HB_DELAY  ((BN_HB_CENTER - 1) / 2)        /* 23: the pure-delay branch */
#define BN_UP_RING   64
#define BN_UP_MASK   (BN_UP_RING - 1)
#define BN_DN_RING_E 64
#define BN_DN_MASK_E (BN_DN_RING_E - 1)
#define BN_DN_RING_O 32
#define BN_DN_MASK_O (BN_DN_RING_O - 1)

/* A LINEAR HISTORY WRITTEN TWICE.
 *
 * A 48-tap loop over a masked ring needs a mask and a serially dependent index update every few
 * products, and the compiler cannot vectorise it (-fopt-info-vec: "latch block not empty"): pure
 * addressing cost. So the buffer is twice the ring (128) and every sample is written twice, at
 * h[pos] and h[pos+64]; the 48-tap window read backwards from h[pos+64] is then CONTIGUOUS and
 * never crosses the edge, and the inner loop has no mask.
 *   Why it works: the sample of instant t-k lives at p = (pos-k) mod 64, with copies at p and p+64.
 *   As k <= 47, the linear index pos+64-k always falls in [17,127] and hits exactly one of the two
 *   copies: the high one if pos-k >= 0, the low one otherwise.
 * The accumulators, the tap-sample pairing, the ascending order of j and the final reduction
 * (b0+b1)+(b2+b3) are those of a masked ring: only the address changes, so the output is
 * byte-identical to it. Cost: one more store per input sample and 512 B more state per stage. */
#define BN_UP_HIST   (2 * BN_UP_RING)      /* 128 */
#define BN_DN_HIST_E (2 * BN_DN_RING_E)    /* 128 */

/* the halfband chain alone, up and down, in base-rate samples: each stage adds (TAPS-1)/2
 * samples of its own rate each way, 2·D·(1 − 2^−S) in all */
static double bn_os_chain_latency(int stages)
{
    const double D = (BN_HB_TAPS - 1) / 2.0;
    if (stages == 0) { return 0.0; }
    return 2.0 * D * (1.0 - pow(2.0, -(double)stages));
}

int bn_os_init(BnOversampler* os, int factor, size_t max_block)
{
    memset(os, 0, sizeof(*os));
    int stages = 0;
    switch (factor) {
    case 1: stages = 0; break;
    case 2: stages = 1; break;
    case 4: stages = 2; break;
    case 8: stages = 3; break;
    default: return -1;
    }
    os->factor = factor;
    os->n_stages = stages;
    os->max_block = max_block;
    {
        const double chain = bn_os_chain_latency(stages);
        os->pad = (int)lround((ceil(chain - 1e-9) - chain) * factor);
    }
    for (int s = 0; s < stages; ++s) {
        os->up_state[s]   = (double*)calloc(BN_UP_HIST, sizeof(double));
        os->down_state[s] = (double*)calloc(BN_DN_HIST_E + BN_DN_RING_O,
                                            sizeof(double));
        if (!os->up_state[s] || !os->down_state[s]) { return -1; }
    }
    os->scratch_a = (double*)calloc(max_block * (size_t)factor + 8, sizeof(double));
    os->scratch_b = (double*)calloc(max_block * (size_t)factor + 8, sizeof(double));
    return (os->scratch_a && os->scratch_b) ? 0 : -1;
}

void bn_os_free(BnOversampler* os)
{
    for (int s = 0; s < BN_OS_MAX_STAGES; ++s) {
        free(os->up_state[s]);
        free(os->down_state[s]);
    }
    free(os->scratch_a);
    free(os->scratch_b);
    memset(os, 0, sizeof(*os));
}

void bn_os_reset(BnOversampler* os)
{
    for (int s = 0; s < os->n_stages; ++s) {
        memset(os->up_state[s], 0, BN_UP_HIST * sizeof(double));
        memset(os->down_state[s], 0,
               (BN_DN_HIST_E + BN_DN_RING_O) * sizeof(double));
        os->up_pos[s] = 0;
        os->down_pos_e[s] = 0;
        os->down_pos_o[s] = 0;
    }
    memset(os->pad_ring, 0, sizeof os->pad_ring);
    os->pad_pos = 0;
}

void bn_os_up(BnOversampler* os, const double* in, size_t n, double* out)
{
    if (os->n_stages == 0) {
        memcpy(out, in, n * sizeof(double));
        return;
    }
    const double* src = in;
    size_t len = n;
    double* dst = os->scratch_a;
    for (int s = 0; s < os->n_stages; ++s) {
        double* target = (s == os->n_stages - 1) ? out : dst;
        double* ring = os->up_state[s];
        int pos = os->up_pos[s];
        const double c47 = BN_HB[BN_HB_CENTER];
        for (size_t i = 0; i < len; ++i) {
            pos = (pos + 1) & BN_UP_MASK;
            ring[pos] = src[i];
            ring[pos + BN_UP_RING] = src[i];   /* the high copy: see BN_UP_HIST */
            /* even branch: the even taps in ascending k, in four partial accumulators (see the file
             * header: they reassociate the sum, far below float32's floor, on a linear filter) */
            double b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            /* a contiguous window: no mask inside the loop */
            const double* h = ring + pos + BN_UP_RING;
            for (int j = 0; j < BN_HB_NPOLY; j += 4) {
                b0 += BN_HB_P[j]     * h[-j];
                b1 += BN_HB_P[j + 1] * h[-j - 1];
                b2 += BN_HB_P[j + 2] * h[-j - 2];
                b3 += BN_HB_P[j + 3] * h[-j - 3];
            }
            const double acc = (b0 + b1) + (b2 + b3);
            target[2 * i] = 2.0 * acc;
            /* odd branch: the centre alone, on x[i−23] */
            target[2 * i + 1] = 2.0 * (c47 * h[-BN_HB_DELAY]);
        }
        os->up_pos[s] = pos;
        src = target;
        len *= 2;
        dst = (dst == os->scratch_a) ? os->scratch_b : os->scratch_a;
    }
    if (os->pad > 0) {                        /* the whole-sample remainder, see BnOversampler.pad */
        int p = os->pad_pos;
        for (size_t i = 0; i < len; ++i) {
            const double d = os->pad_ring[p];
            os->pad_ring[p] = out[i];
            out[i] = d;
            if (++p == os->pad) { p = 0; }
        }
        os->pad_pos = p;
    }
}

void bn_os_down(BnOversampler* os, const double* in, size_t n, double* out)
{
    if (os->n_stages == 0) {
        memcpy(out, in, n * sizeof(double));
        return;
    }
    const double* src = in;
    size_t len = n * (size_t)os->factor;
    double* dst = os->scratch_a;
    for (int s = 0; s < os->n_stages; ++s) {
        double* target = (s == os->n_stages - 1) ? out : dst;
        double* ring_e = os->down_state[s];                    /* even   v[2i]  */
        double* ring_o = os->down_state[s] + BN_DN_HIST_E;     /* odd    v[2i+1] */
        int pe = os->down_pos_e[s];
        int po = os->down_pos_o[s];
        const double c47 = BN_HB[BN_HB_CENTER];
        for (size_t i = 0; i < len / 2; ++i) {
            pe = (pe + 1) & BN_DN_MASK_E;
            ring_e[pe] = src[2 * i];
            ring_e[pe + BN_DN_RING_E] = src[2 * i];   /* the high copy */
            /* the centre FIRST (v[2i−47] is the odd sample of 23 steps ago), then the even taps in
             * ascending k, in four partial accumulators (see the file header) */
            double b0 = c47 * ring_o[(po - BN_HB_DELAY) & BN_DN_MASK_O];
            double b1 = 0.0, b2 = 0.0, b3 = 0.0;
            /* a contiguous window, no mask: see BN_DN_HIST_E */
            const double* h = ring_e + pe + BN_DN_RING_E;
            for (int j = 0; j < BN_HB_NPOLY; j += 4) {
                b0 += BN_HB_P[j]     * h[-j];
                b1 += BN_HB_P[j + 1] * h[-j - 1];
                b2 += BN_HB_P[j + 2] * h[-j - 2];
                b3 += BN_HB_P[j + 3] * h[-j - 3];
            }
            const double acc = (b0 + b1) + (b2 + b3);
            target[i] = acc;
            po = (po + 1) & BN_DN_MASK_O;
            ring_o[po] = src[2 * i + 1];
        }
        os->down_pos_e[s] = pe;
        os->down_pos_o[s] = po;
        src = target;
        len /= 2;
        dst = (dst == os->scratch_a) ? os->scratch_b : os->scratch_a;
    }
}

double bn_os_latency_exact(const BnOversampler* os)
{
    /* Each stage adds D = (TAPS-1)/2 samples OF ITS OWN RATE each way; in base-rate samples, up and
     * down, that is 2·D·(1 − 2^−S). It is FRACTIONAL as soon as there is more than one stage — 47
     * samples at 2x, but 70.5 at 4x and 82.25 at 8x: a linear-phase halfband always has an odd D
     * (TAPS ≡ 3 mod 4), so the half sample belongs to the chain. A host only takes an integer
     * latency, so the `pad` delay (2 high-rate samples at 4x, 6 at 8x, after the way up) completes
     * the fraction, and the latency declared is the one that happens (a rounded declaration would
     * misalign a bypass crossfade by half a sample at 4x). */
    return bn_os_chain_latency(os->n_stages) + (double)os->pad / (double)(os->factor > 0 ? os->factor : 1);
}

int bn_os_latency(const BnOversampler* os)
{
    return (int)(bn_os_latency_exact(os) + 0.5);
}
