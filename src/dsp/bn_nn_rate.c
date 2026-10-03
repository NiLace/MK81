/* bn_nn_rate.c — see bn_nn_rate.h.
 *
 * Kernel: h(s) = 2fc sinc(2fc s) · w(s/L), s in samples of the kernel's spacing, fc in cycles per
 * sample, w the Kaiser window. Length from the Kaiser rule N = (A - 7.95) / (14.36 Δf), window
 * shape β = 0.1102 (A - 8.7), with A the stop-band attenuation in dB and Δf the transition width
 * in cycles per sample (Kaiser, "Nonrecursive digital filter design using the I0-sinh window
 * function", 1974).
 *
 * Time bookkeeping is relative to the newest fs_hi input (position 0, older inputs negative):
 * `tin` is how far ahead the next model sample falls due, `tm_rel` where the newest model sample
 * sits. A model sample due at time t is computed at t - L_in, so every input tap it needs exists;
 * the output at fs_hi is read at -delay, which leaves every model tap it needs in the past. */
#include "bn_nn_rate.h"
#include <math.h>
#include <string.h>

#define BN_NN_RATE_ATT_DB 80.0
#define BN_NN_RATE_PI     3.14159265358979323846

static double bessel_i0(double x)
{
    double s = 1.0, t = 1.0;
    for (int k = 1; k < 64; ++k) {
        t *= (x / (2.0 * k)) * (x / (2.0 * k));
        s += t;
        if (t < 1e-17 * s) break;
    }
    return s;
}

/* fpass, fstop in Hz; fs the kernel's spacing in Hz */
static void kernel_build(BnNnKernel* k, double fs, double fpass, double fstop)
{
    const double df = (fstop - fpass) / fs;
    const double fc = 0.5 * (fpass + fstop) / fs;
    const double beta = 0.1102 * (BN_NN_RATE_ATT_DB - 8.7);
    const int n = (int)ceil((BN_NN_RATE_ATT_DB - 7.95) / (14.36 * df));
    k->L = (n + 1) / 2;
    k->step = (double)BN_NN_RATE_TAB / (2.0 * k->L);
    const double i0b = bessel_i0(beta);
    for (int j = 0; j <= BN_NN_RATE_TAB; ++j) {
        const double s = -k->L + (double)j / k->step;
        const double a = 2.0 * fc * s;
        const double sinc = (fabs(a) < 1e-12) ? 1.0 : sin(BN_NN_RATE_PI * a) / (BN_NN_RATE_PI * a);
        const double q = s / k->L;
        const double win = (fabs(q) >= 1.0) ? 0.0 : bessel_i0(beta * sqrt(1.0 - q * q)) / i0b;
        k->tab[j] = (float)(2.0 * fc * sinc * win);
    }
    k->tab[BN_NN_RATE_TAB + 1] = 0.0f;
}

static inline float kernel_at(const BnNnKernel* k, double s)
{
    const double u = (s + k->L) * k->step;
    if (u < 0.0 || u >= (double)BN_NN_RATE_TAB) return 0.0f;
    const int i = (int)u;
    const float f = (float)(u - i);
    return k->tab[i] + f * (k->tab[i + 1] - k->tab[i]);
}

int bn_nn_rate_init(BnNnRate* r, double fs_hi, double model_fs)
{
    memset(r, 0, sizeof *r);
    r->model_fs = model_fs;
    if (fabs(fs_hi - r->model_fs) < 0.5) { r->active = 0; r->delay = 0; return 0; }
    r->active = 1;
    r->rho = fs_hi / r->model_fs;
    /* pass band: the audio band with a margin, 24 kHz, or 0.4536 of the lower rate below 53 kHz
     * (20 kHz at 44.1). Stop bands: whatever could fold INTO the pass band is held off, and no more
     * (the classic converter bound, stop = lower rate - pass), so the kernels stay short enough for
     * the delay to fit inside the power stage's own declared pre-delay at the usual rates. The one
     * exception is the input when fs_hi is above the model rate: there the net would see the
     * preamp's content between 48 kHz and the fold as spurious input, so its stop band is 48 kHz. */
    const double fmin = fs_hi < r->model_fs ? fs_hi : r->model_fs;
    double fpass = fmin * 0.4536 < 24000.0 ? fmin * 0.4536 : 24000.0;   /* 20.004 kHz at 44.1 */
    /* and never closer to the MODEL's Nyquist than 0.4 of its rate: a net trained at 48 kHz (19.2 kHz) would
     * otherwise need an input kernel hundreds of taps long above its rate. At 96 kHz this changes nothing. */
    if (fpass > 0.4 * r->model_fs) { fpass = 0.4 * r->model_fs; }
    r->fpass = fpass;
    double in_pass = fpass, in_stop = fs_hi > r->model_fs ? 0.5 * r->model_fs : fs_hi - fpass;
    kernel_build(&r->kin, fs_hi, in_pass, in_stop);
    kernel_build(&r->kout, r->model_fs, fpass, fmin - fpass);
    r->delay = (int)ceil(r->kin.L + r->rho * (r->kout.L + 1));
    bn_nn_rate_reset(r);
    return r->delay;
}

void bn_nn_rate_reset(BnNnRate* r)
{
    memset(r->xin, 0, sizeof r->xin);
    memset(r->rin, 0, sizeof r->rin);
    memset(r->yout, 0, sizeof r->yout);
    r->pin = r->pout = 0;
    r->tin = 0.0;
    r->tm_rel = -(double)r->kin.L;
    r->n_in = r->n_out = 0;
}

typedef struct { BnNnLstm* g; float pres, bias, cab, head, us; } LstmCtx;
static float lstm_engine(void* ctx, float w2, float rail)
{
    const LstmCtx* c = (const LstmCtx*)ctx;
    return bn_nn_lstm_stage(c->g, w2, rail, c->pres, c->bias, c->cab, c->head, c->us);
}

float bn_nn_rate_tick(BnNnRate* r, BnNnLstm* g, float w2, float rail,
                      float pres, float bias, float cab, float head, float us)
{
    LstmCtx c = { g, pres, bias, cab, head, us };
    return bn_nn_rate_tick_fn(r, lstm_engine, &c, w2, rail);
}

float bn_nn_rate_tick_fn(BnNnRate* r, BnNnEngineFn fn, void* ctx, float w2, float rail)
{
    const int M = BN_NN_RATE_RING - 1;
    if (r->n_in == 0) {
        /* first tick after a reset: the history is the present held still (the rail at 0 would
         * divide the frame by zero; w2 at 0 would be a step the engine never sees directly) */
        for (int i = 0; i < BN_NN_RATE_RING; ++i) { r->xin[i] = w2; r->rin[i] = rail; }
    }
    r->xin[r->pin] = w2;
    r->rin[r->pin] = rail;
    const int newest = r->pin;
    r->pin = (r->pin + 1) & M;
    ++r->n_in;
    r->tin -= 1.0;
    r->tm_rel -= 1.0;
    /* every model sample now due: w2 through the in-kernel, the rail linearly */
    while (r->tin <= 0.0) {
        const double tau = r->tin - r->kin.L;          /* <= -L_in: all taps are in the past */
        const int k0 = (int)ceil(tau - r->kin.L), k1 = (int)floor(tau + r->kin.L);
        float acc = 0.0f, wsum = 0.0f;
        for (int k = k0; k <= k1; ++k) {
            const float w = kernel_at(&r->kin, tau - k);
            acc += w * r->xin[(newest + k) & M];
            wsum += w;
        }
        const float w2m = acc / wsum;
        const int kf = (int)floor(tau);
        const float fr = (float)(tau - kf);
        const float railm = r->rin[(newest + kf) & M] + fr * (r->rin[(newest + kf + 1) & M] - r->rin[(newest + kf) & M]);
        r->yout[r->pout] = fn(ctx, w2m, railm);
        r->pout = (r->pout + 1) & M;
        ++r->n_out;
        r->tm_rel = tau;
        r->tin += r->rho;
    }
    /* the output at -delay, in model samples back from the newest one */
    const double q = (r->tm_rel + r->delay) / r->rho;
    const int j0 = (int)ceil(q - r->kout.L), j1 = (int)floor(q + r->kout.L);
    const int last = (r->pout - 1) & M;
    float acc = 0.0f, wsum = 0.0f;
    for (int j = j0; j <= j1; ++j) {
        const float w = kernel_at(&r->kout, j - q);
        acc += w * r->yout[(last - j) & M];
        wsum += w;
    }
    return acc / wsum;
}
