/* mk81.c — LV2 plugin written against the bare API, no frameworks.
 *
 * The preamp of the 100 W head is solved with the DK method (netlist ->
 * discretisation -> C -> LV2), and the power stage is a neural LSTM.
 *
 * Audio-path rules: no memory allocation, no system calls, no denormals.
 * Everything expensive (building and solving the nodal system, allocating
 * buffers) happens in instantiate.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <lv2/core/lv2.h>

#include "bn_dk.h"
#include "bn_oversample.h"
#include "bn_tube_own.h"
#include "bn_preamp2203.h"
#include "bn_psu.h"
#if defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#define BN_HAVE_MXCSR 1
#endif
#include "bn_nn_lstm.h"       /* the neural power stage */
#include "bn_nn_rest.h"       /* each arm's settled rest state, precomputed (see below) */
#include "bn_nn_rate.h"       /* the net runs at its training rate at any fs_hi */
#include "bn_fx_comp.h"       /* the FX loop's complementary path */
#include "bn_nn_weights.h"    /* generated: the four shipped weight sets, embedded */




#define BN_URI "https://nylarea.com/plugins/mk81"

/* The FULL preamp saturates with ~60 V peaks at the master wiper (measured):
 * this compensates so the ceiling sits near 0 dBFS and the Output knob is a
 * fine trim, not a rescue. */
#define BN_STAGE_GAIN 60.0
/* The neural power stage's input guard. The nets were trained on the
 * preamp's w2, which the preamp's own saturation bounds at |w2| <= 57,8 V; the FX-loop return is the only input
 * with no such bound (1,0 FS x fx_return_lvl 2 x BN_STAGE_GAIN = 120 V), and outside its training range an LSTM
 * does not saturate like the circuit (measured: at +6 dB of return the LSTM-32 already played 1,5 dB louder than the
 * cascade with 1,51 peaks; at 120 V, 1,5-1,8x louder). This is the recovery triode's grid of a real loop: identity
 * up to BN_NN_GUARD_KNEE, then a tanh approach to BN_NN_GUARD_LIM, the trained maximum. It touches nothing the
 * preamp can produce (the reference DI peaks at 50,4 V); the FX send and the power bypass carry the unguarded w2. */
#define BN_NN_GUARD_KNEE 54.0
#define BN_NN_GUARD_LIM  58.0
/* A net trained on w2x (the preamp fed the exact node X; `xlive` arms) saw a lower maximum,
 * because the preamp sags: 56,27 V over the 100 W UK head's 132 sets (57,78 on w2). Its guard ends at ITS trained maximum,
 * with the same 4 V knee span; still above everything the preamp produces on the reference DI (50,4 V). */
#define BN_NN_GUARD_LIM_XLIVE  56.27
#define BN_NN_GUARD_KNEE_XLIVE (BN_NN_GUARD_LIM_XLIVE - (BN_NN_GUARD_LIM - BN_NN_GUARD_KNEE))
static inline double bn_nn_guard(double x, int xlive)
{
    const double knee = xlive ? BN_NN_GUARD_KNEE_XLIVE : BN_NN_GUARD_KNEE;
    const double lim  = xlive ? BN_NN_GUARD_LIM_XLIVE  : BN_NN_GUARD_LIM;
    const double ax = fabs(x);
    if (ax <= knee) { return x; }
    const double span = lim - knee;
    const double y = knee + span * tanh((ax - knee) / span);
    return x < 0.0 ? -y : y;
}
/* The power stage clips with ~53 V peaks at the 8 ohm tap (measured on the exact full amp): the output is
 * normalised to it, the same ceiling philosophy as BN_STAGE_GAIN. */
#define BN_POWER_GAIN 53.0

/* DC blocker: the corner, ANCHORED to a frequency.
 * It is expressed as "the pole A_REF at the rate FS_REF" instead of as an fc in
 * hertz, and that is deliberate: FS_REF = 96 kHz is exactly the fs_hi of the
 * factory configuration (2x on a 48k host), where the pole is the literal
 * 0.9995. So the shipped case is reproduced BYTE for BYTE and the other rates
 * are derived from it. The implied fc is
 *     fc = -ln(0.9995) * 96000 / (2*pi) = 7.6403 Hz */
#define BN_DCBLOCK_A_REF  0.9995
#define BN_DCBLOCK_FS_REF 96000.0
/* Blocker of the w2 SEAM (FX loop send and the power bypass's dry path).
 * SUB-HERTZ corner on purpose: w2 rests at ~1e-9 V (the wiper hangs after the
 * coupling capacitor), so there is no DC to remove here, only the guarantee
 * that an audio port never carries it. With the output blocker's corner
 * (7,64 Hz) the send rotated 2 degrees at 220 Hz and a loop closed through a
 * TRANSPARENT effect gave the amp back only -29 dB from the original
 * (measured; the whole residual lived below 1 kHz). At 0,05 Hz the phase at
 * low E (82 Hz) is 0,035 degrees: -64 dB, below the -60 dB porting bar. */
#define BN_SEAM_HP_HZ 0.05

#define BN_PI           3.14159265358979323846

/* Thresholds of the silence bypass (the quiet freeze).
 * These absolute constants are the fallback: by default the thresholds are TIED
 * to `dk.tol` (see the quiet block in run()), and the freeze is taken on the
 * AVERAGED state, because what spoiled it was the Newton's jitter.
 * `BN_QUIET=0` restores the absolute constants below, so the before and after
 * can be measured on the same machine.
 * MOVEMENT: how much the output may move from sample to sample and still count
 * as still. 1e-9 (-180 dBFS) does not work: measured, with a zero input the
 * output keeps trembling ~7.4e-9 indefinitely — numerical noise of the solver,
 * not signal — so the bypass would never engage. 1e-7 is -140 dBFS: below the
 * floor of a 24-bit converter (-144 dBFS) and of anything audible.
 * ⚠️ That 7.4e-9 figure no longer describes the engine: with the absolute
 * constants the measured rest floor is max|y| = 3.97e-7 (-128 dBFS) with a
 * sample-to-sample max|dy| = 2.90e-7, ~40x ABOVE this 1e-7 threshold. Its cause
 * is NOT rounding: the FFT of the rest state shows a sustained LIMIT CYCLE at
 * 3265,75 Hz + a 3rd harmonic at 9797,25 Hz that does not decay.
 * MEASURED with the thresholds tied to tol (tolerance 3e-5): the bypass engages
 * on 53,9 % of a silent run at os 2 (factory) and os 4, floor -109,6 / -111,6
 * dBFS; a stopped track drops from 3,84 % of a core to 1,57 % (-57 %) and an
 * 8-track session with one track playing goes from 33,3 % to 17,2 %. ANMR of
 * the freeze -31,33 dB, 21,33 dB of margin.
 * ⛔ At os 1 it does NOT engage: the `0,116 x tol` rule holds from os >= 2 up,
 * and the os 1 floor has a cause of its own, still open.
 * ⛔ Raising EPS is NOT a fix (measured: with loose thresholds the note after
 * the silence moved to -63 dB).
 * LEVEL: the output must also BE in the tail (|y| < 1e-5, -100 dBFS), so the
 * engine never freezes on something that only looks flat. */
#ifndef BN_QUIET_EPS
#define BN_QUIET_EPS   1e-7
#endif
#ifndef BN_QUIET_LEVEL
#define BN_QUIET_LEVEL 1e-5
#endif
/* STATE: watching the output is NOT enough. Measured: with the output already
 * at -140 dBFS the circuit is still recovering inside (the coupling capacitors
 * recharge to hundreds of volts and that barely shows at the output, between
 * the gain staging and the DC blocker). Freezing on the output alone, the note
 * that came AFTER 30 s of silence came out -82.8 dB different. So the DK state
 * itself must also be still, in terms RELATIVE to its magnitude. */
/* The factor of the STATE threshold, relative to dk.tol. The only thing it decides is the bypass's
 * ENGAGE TIME: the stricter it is, the longer the circuit state takes to drop below it.
 * MEASURED SWEEP (20 s of zeros for engaging; ANMR of the freeze against the same build without it):
 *     0,1 -> 7,98 s to engage · ANMR -45,97 dB     (calibrated to tol 1e-4)
 *   ⭐0,3 -> 4,50 s           · ANMR -25,01 dB     <- CHOSEN: -44 % engage time, 15 dB of margin
 *     1,0 -> 1,10 s           · ANMR -11,34 dB     ⛔ passes by only 1,34 dB: a threshold AT THE EDGE gives no warning
 *     3,0 -> same as 1,0 (saturates)
 * ⇒ It matters when the transport stops and starts often (while mixing): 3,5 s less of full-load engine
 * PER STOPPED TRACK and per stop. With a track stopped for minutes it makes no difference. */
#ifndef BN_QUIET_XEPS_K
#define BN_QUIET_XEPS_K 0.3
#endif
#ifndef BN_QUIET_XEPS
#define BN_QUIET_XEPS 1e-10
#endif

#define BN_MAX_BLOCK 8192

/* Port indices and symbols are public API (lv2/mk81.ttl, lv2/frozen_symbols.txt): a host saves sessions and
 * automation by them, so new ports are APPENDED, never inserted or renumbered. */
typedef enum {
    /* There is no `drive` port (a virtual pre-gain the real amp does not have): index 2 is the output trim. */
    BN_IN        = 0,
    BN_OUT       = 1,
    BN_TRIM      = 2,   /* output level, in dB                */
    BN_OVERSAMPLE = 3,  /* one-value port: the engine runs at 2x whatever it says (the symbol is public API and stays) */
    BN_VOL       = 4,   /* the panel knobs: 0..1              */
    BN_TREBLE    = 5,
    BN_MIDDLE    = 6,
    BN_BASS      = 7,
    BN_MASTER    = 8,
    BN_LATENCY   = 9,  /* output: latency in samples         */
    BN_BIAS      = 10,  /* power bias 0..1 (0.5 = factory): the BIAS knob on the back */
    BN_US        = 11,  /* the export (6550) version 0/1: the model display's list on the front */
    BN_LOW       = 12,  /* ON THE PANEL: the HIGH/LOW jacks */
    BN_PRESENCE  = 13,  /* ON THE PANEL: the PRESENCE knob */
    BN_CAB_LOAD  = 14,  /* the Cabinet Feel knob on the back: CONTINUOUS "Cabinet feel" 0..1 —
                         * 0 = the factory 8 ohm
                         * (test bench), 1 = the real impedance of the
                         * 4x12, and every position in between is a REAL CIRCUIT
                         * (the resonance and the voice-coil inductance are
                         * scaled). */
    /* The two bypass-style switches. They are NOT two names for the same thing:
     *   BYPASS  = pass-through of the signal WITHOUT switching the plugin off. The
     *             DSP keeps running with its warm state, so coming back gives no
     *             transient or click. It costs the usual CPU.
     *   POWER   = bypass of the WHOLE plugin. The DSP is skipped, so the CPU
     *             cost drops to zero. It is the real "off". */
    BN_BYPASS    = 15,  /* 0 = active, 1 = pass-through (warm DSP) */
    BN_POWER     = 16,  /* 1 = on (default), 0 = whole plugin out */
    BN_HUM       = 17,  /* mains hum, 0..1. DEFAULT 0 — see bn_hum() */
    BN_HISS      = 18,  /* Johnson hiss, 0..1. DEFAULT 0 */
    BN_ERA       = 19,  /* 0 = new way (>=86, ours), 1 = old */
    BN_HEAD      = 20,  /* 0 = the 100 W head, 1 = the 50 W head */
    BN_BYPASS_PREAMP = 21,  /* 1 = skip the preamp; the input feeds w2 (the master wiper)
                             * directly and the power stage still runs. 0 = default. Crossfaded. */
    /* The five FX LOOP ports and the POWER bypass cut at the SAME w2 seam as
     * port 21. Their symbols are public API. */
    BN_BYPASS_POWER  = 22,  /* 1 = skip the power stage; w2 (DC-free, aligned
                             * with the power stage's latency) goes out at preamp level. */
    BN_FX_LOOP       = 23,  /* 1 = loop active: live send and return mixed into w2 */
    BN_FX_SEND       = 24,  /* audio OUT: DC-free w2, at preamp level (/BN_STAGE_GAIN) */
    BN_FX_RETURN     = 25,  /* audio IN: what comes back from the effect, enters at C9 / the PI's grid */
    BN_FX_MIX        = 26,  /* 0..100 %: 0 = dry (the loop only SENDS), 100 = series (return only) */
    BN_FX_RETURN_LVL = 27,
    /* engine port: fixed at 1 (the neural LSTM-32 power stage, the only one the plugin carries). Kept so the
     * port indices and symbols do not move; run() ignores its value. */
    BN_ENGINE        = 28,
    BN_N_PORTS   = 29
} BnPortIndex;

/* The ring of the power bypass's DRY path, at fs_hi. The power stage's latency (bn_pw_lat: the net's rate
 * converter delay) is part of the declared latency; w2 is delayed the same so the power <-> w2 crossfade neither
 * combs nor leads. Bounded when read, like BN_DRY_RING. */
/* ⛔ This ring must be at least as long as the rate converter's ring.
 * `bn_nn_rate` is sized "for a 48 kHz net at 768 kHz" (BN_NN_RATE_RING 2048) and can return a
 * `delay` above 1023; with a shorter ring `bn_pw_lat` clips it to BN_PB_RING-1, so
 * `nn_delay = pb_delay - nnr.delay` comes out NEGATIVE and the read index `rn = nn_pos - nn_delay`
 * runs off the TOP of the ring — the wrap in the power stage only covers the bottom side. It does not
 * crash: it reads fields of the struct itself and plays them as audio, with nothing to give it away.
 * It is not reachable with the four shipped arms (all 96 kHz), but a 48 kHz arm would reach it.
 * The `_Static_assert` ties the two constants so they cannot drift apart silently. It costs 128 KB
 * across the four configurations. */
#define BN_PB_RING 2048
_Static_assert(BN_PB_RING >= BN_NN_RATE_RING,
               "BN_PB_RING must carry any delay bn_nn_rate can produce, or nn_delay goes negative");

/* Full-scale time of the ramps that carry the net's conditioning knobs (presence, bias, cabinet feel). */
#ifndef BN_COND_RAMP_S
#define BN_COND_RAMP_S 0.010
#endif

/* The five preamp pots (volume, treble, middle, bass, master) move through the circuit's matrices, rebuilt once
 * per chunk while a pot moves, so a moving pot is a staircase: each rebuild is a small step at the output, and its
 * size is the pot's travel per rebuild = chunk / (slew time x rate). A full-scale step at the 2x default, input
 * 0.0025 (the click test: 2nd difference against the steady signal, bar 5x), measured on master,
 * the worst pot: 32 samples / 30 ms 68.9x · 8 / 30 ms ~18x · 1 / 30 ms 2.7x · 8 / 240 ms 2.9x · 4 / 120 ms 3.3x.
 * The travel per rebuild sets the click, the chunk sets the cost while moving: a rebuild every 1/6000 s
 * (8 samples at 48 kHz, the same TIME at every rate) and a full-scale slew of 240 ms give 1/1440 of full scale
 * per rebuild. Cost measured on a low-power 4-core x86 CPU (2x, 128-sample blocks): held 61 % of real time, full-scale
 * jumps once a second 70 % mean and a worst block of 100 % of its budget (78 % held); a 1-sample chunk at
 * 30 ms gave the same click at a 271 % worst block. */
#ifndef BN_KNOB_CHUNK_S
#define BN_KNOB_CHUNK_S (1.0 / 6000.0)
#endif
#ifndef BN_KNOB_SLEW_S
#define BN_KNOB_SLEW_S 0.240f
#endif

/* A complete set of state that depends on the oversampling factor. */
typedef struct {
    BnOversampler os;
    BnDK          dk;
    BnDKVar       dkvar;
    /* DC blocker: the plate rests at ~255 V and that is not audio */
    double        dc_x1, dc_y1;
    /* The blocker's pole, DERIVED from fs_hi. A fixed literal 0.9995 inside the
     * oversampled loop would make its corner NOT a frequency: it would scale
     * with the oversampling factor and with the host rate (7.6 Hz at factory,
     * but 30.6 at 8x@48k and 61 at 8x@96k). In audio terms raising the QUALITY
     * would cut real bass: −0.57 dB at low E at 8x@48k and −1.92 at 8x@96k,
     * plus the phase rotation, and it would contaminate any A/B between
     * factors. So fc is fixed. */
    double        dc_a;
    /* SNAPSHOT of the operating point, taken after the DC init and the 256
     * settling iterations. It is the RECOVERY path: if the DK state x is
     * poisoned (a NaN coming in from anywhere), activate() restores it; without
     * it the only way out would be to re-instantiate.
     * ⚠️ Setting x to ZERO, which is what bn_dk_reset does, does NOT work: zero
     * is the COLD amp (discharged capacitors), not its operating point — there
     * would be a huge transient on every activate(). The snapshot is restored. */
    double*       x_dc;      /* physical state: capacitors and inductors */
    double*       v_dc;      /* valve voltages at the DC point           */
    int           n_x_dc, n_v_dc;
    /* --- THE w2 SEAM (preamp/power bypass, FX loop) ----------------------
     * A DC blocker of its OWN on w2 (corner BN_SEAM_HP_HZ, primed at w2's rest
     * value so it starts settled): the send and the power bypass's dry path
     * come out of it. The output one does NOT serve: it sees t8, which is
     * already DC-free through the power stage's HP, and its 7,64 Hz corner is
     * too high for a send (see BN_SEAM_HP_HZ). */
    double        w2_dc;               /* w2's rest value after the DC init (V) */
    double        seam_a;              /* the seam blocker's pole, from fs_hi */
    double        seam_dc_x1, seam_dc_y1;
    double        pb_ring[BN_PB_RING]; /* w2 (AC) delayed by the power stage's latency (bn_pw_lat), at fs_hi */
    double        nn_ring[BN_PB_RING]; /* the net's output delayed to the power stage's latency (bn_pw_lat) */
    int           nn_pos;
    int           pb_pos;
    /* The net runs at its training rate (96 kHz) at any fs_hi, through a converter whose
     * delay is fixed per fs_hi. The power stage's latency is then ONE number, bn_pw_lat(): the
     * converter's delay rounded up to a whole host sample; the nn ring delays the net's output by
     * the rounding. At fs_hi = 96 kHz the converter is inactive and the power stage adds no latency. */
    BnNnRate      nnr;
    /* The FX loop's resampling pair: the send goes DOWN from fs_hi to the host
     * rate and the return goes UP. It is a second set of halfbands with its own
     * state (the plugin's set is already busy with the input and the output).
     * It only runs with the loop active; it is zeroed when the loop engages. */
    BnOversampler fxos;
    BnFxComp      fxc;            /* the loop's complement: what the host cannot carry, with the host delay estimated */
    int           ready;           /* 1 = already built */
} BnCfg;

typedef struct {
    const float* in;
    float*       out;
    const float* trim;
    const float* oversample;
    const float* vol;
    const float* treble;
    const float* middle;
    const float* bass;
    const float* master;
    float*       latency;
    const float* bias;
    const float* us_spec;
    const float* low_input;
    const float* presence;
    const float* cab_load;
    const float* bypass;
    const float* power;
    const float* hum;
    const float* hiss;
    const float* era;
    const float* head;
    const float* bypass_preamp;    /* skip the preamp, input -> w2 */
    const float* bypass_power;     /* skip the power stage, w2 -> out */
    const float* fx_loop;          /* FX loop: switch */
    float*       fx_send;          /* FX loop: audio OUT (may be NULL) */
    const float* fx_return;        /* FX loop: audio IN (may be NULL) */
    const float* fx_mix;           /* FX loop: 0 dry .. 100 series (%) */
    const float* fx_return_lvl;    /* FX loop: return gain 0..2 */
    const float* engine;           /* connected and never read: the port is fixed at 1 in the TTL */
    BnNnLstm     nn[BN_NN_N_ARMS]; /* one LSTM-32 per (head, us) variant, bound at instantiate to the embedded blobs */
    double       nn_fs;            /* the rate the nets were trained at (all four the same) */
    int          nn_arm;           /* the slot whose state is live (head + 2*us) */
    int          nn_arm_want;      /* the slot the head and US ports ask for; differs from nn_arm during a variant switch */
    int          nn_hold;          /* fs_hi samples the power stage stays silent after a switch, so the new net's first output emerges from its ring and converter before the fade-in */
    int          dry_from_ports;   /* first run() after activate(): take dry_mix from the ports, which are only valid there */
    double       nn_gain;          /* the power stage's output gain: 1 in steady state, 10 ms ramps at fs_hi through 0 on a variant switch */
    /* the net's conditioning knobs (presence, bias, cabinet feel) as applied: linear ramps at fs_hi toward the ports,
     * full scale in BN_COND_RAMP_S. The first run() after activate() takes them, and hum_now, from the ports
     * (cond_from_ports). */
    double       nn_pres_now, nn_bias_now, nn_cab_now;
    double       hum_now;          /* the hum's level as applied: a 10 ms ramp at the host rate toward the port */
    int          cond_from_ports;
    double       byp_pow_mix;      /* 0 = power stage active, 1 = bypassed; 10 ms ramp at fs_hi */
    double       fx_mix_now;       /* FX: applied mix (10 ms ramp at fs_hi); 0 with the loop off */
    double       fx_lvl_now;       /* FX: applied return level (ramp) */
    double       fx_send_gain;     /* FX: fade of the send at the base rate, 0..1 */
    int          fx_running;       /* FX: 1 = the resampling pair is warm */
    double*      fx_ret_lo;        /* FX: sanitised return at the base rate / downsampled send (n) */
    double*      fx_ret_hi;        /* FX: return upsampled to fs_hi (n*factor) */
    double*      fx_send_hi;       /* FX: w2 (AC, at preamp level) at fs_hi (n*factor) */
    double*      fx_rcap;          /* FX: the host's return of this chunk, kept for the delay estimator (n) */
    int          low_state;        /* -1 forces the first apply */
    int          us_state;         /* -1 forces the first apply; the US preamp uses 250k/25k */
    double       byp_pre_mix;      /* 0 = preamp active, 1 = bypassed; 10 ms ramp in the loop */
    /* 10 ms ramp to enter and leave bypass/power without a click: a hard jump
     * from wet to dry is a step in the signal, and that is audible. */
    double       dry_mix;          /* 0 = plugin, 1 = pass-through */
    /* The Newton's relative tolerance is an additive noise floor: at 3e-5 a silent input gives a stationary -63 dBFS
     * tick train (the power stage multiplies the preamp's jitter x15) and the quiet freeze never engages; at 1e-6 the
     * floor is -95 dBFS. The tolerance follows the SOLVER INPUT (after drive and hiss) with hysteresis: fine (1e-6)
     * below 1e-3 peak, normal above 3.16e-3. Keyed on the solver input on purpose: the ticks scale with the tolerance,
     * not with the jack level, so a low drive setting simply spends more time in the fine mode (cost, not fidelity). */
    double       tol_norm, tol_fine;
    int          tol_fine_on;
    double       tol_env, tol_env_decay;   /* the input peak follower at fs_hi and its 50 ms release factor */
    /* The DRY path is delayed by the same latency the plugin reports.
     * Without this, on bypass the signal would come out EARLY relative to the
     * other tracks (the host is already compensating the latency we declare),
     * and the crossfade would mix two misaligned signals -> comb. 512 is
     * plenty: the maximum is ~90 samples (82 from the oversampler at 8x + the
     * power stage's latency / factor). */
#define BN_DRY_RING 512
    double       dry_ring[BN_DRY_RING];
    int          dry_pos;
    int          dry_delay;        /* samples; refreshed with the latency */

    double sample_rate;

    /* EVERYTHING that depends on the oversampling factor lives together, and
     * there are FOUR sets, one per factor. Building one on the fly from run()
     * when the knob moves would be a textbook real-time violation: memory
     * allocation, plus a DC solve, plus 256 settling iterations, all on the
     * audio thread (measured peak 7.8 ms). The four are built in instantiate
     * and changing factor is moving a pointer. */
    BnCfg          cfg[4];
    BnCfg*         c;              /* the ACTIVE configuration */
    BnPsu          psu;            /* the supply sag, at block rate */
    BnPsu          psu_idle;       /* the same supply stepped with the IDLE currents: node X at rest for this head/edition/era */
    double         vx_now;
    /* The supply ticks on a FIXED grid of cells aligned to
     * the stream (256 samples at 48 kHz, scaled with the rate), not on the host's blocks. Chunks
     * are cut at the cell edges; the PSU, the dynamic bias and the rail's interpolation advance
     * once per cell, fed with the follower's maximum over the whole previous cell. */
    uint32_t       psu_grid;       /* cell length, base samples */
    uint32_t       psu_phase;      /* base samples already processed in the current cell */
    double         pk_cell;        /* the follower's max so far in the current cell */
    double         vx_cell0, dvx_hi;   /* the preamp rail at the cell's start and its slope per fs_hi sample */
    /* The NET's rail: the rail at the netlist's `htraw` equivalent — plates (vc23)
     * plus RSAG times the rectifier's current — normalised to its idle value. That is where the truth the nets
     * learn from takes its `railfile=` (above RSAG, the choke and the 20k); the PI rail the former cascade scaled by
     * is that rail already filtered by 20k x 100 uF (tau 2 s), and fed to the net it made the power stage sag
     * slowly and deep. Against the consistent truth: worst frame +8,63/+12,26/+4,96 -> +6,12/+12,37/+3,16,
     * the net alone's reach; no retraining (the pairs' rail tracks are applied at htraw). */
    double         rq_now, rq_cell0, drq_hi;
    double         w2_peak_fast;   /* the master-wiper envelope the PSU map and the bias read (release 5 ms) */
    double         env_fast;       /* its per-sample peak follower, at fs_hi */
    float          knobs_last[5];
    int            os_factor;
    /* Ramp around an oversampling change. Without it the switch pops: a
     * sample-to-sample step of 1.02-1.35 against 0.268 in steady state (x3.8
     * to x5.0) with a sustained low E (measured). The switch happens in silence:
     *   phase 1 = ramping down (with the OLD config)
     *   phase 2 = at zero, rebuild on the next block
     *   phase 3 = ramping up (with the NEW one)
     * It is not a crossfade —that would need both chains running at once, which
     * is exactly the CPU there is not— but a short gap: the quality knob is not
     * moved while playing. */
    int            os_want;
    int            os_phase;
    int            os_hold;          /* phase 4: samples of silence left while the new factor settles */
    float          os_gain, os_step;
    /* The FIRST block is not ramped: on opening the plugin there is nothing to
     * smooth, and if the host instantiates it with a factor other than the
     * factory one the ramp would put a 10 ms fade into the start (and break
     * the byte-for-byte identity of the measurements at 4x). */
    int            os_primero;

    double* buf_in;      /* block at the base rate, in volts */
    double* buf_hi;      /* oversampled block                */
    double* buf_out;

    float  trim_lin, smooth_coef;
    uint32_t knob_chunk;           /* the chunk while a preamp pot moves, in samples (BN_KNOB_CHUNK_S) */
    /* silence bypass (see the block in run()) */
    int    quiet_run;           /* consecutive samples with a still output */
    int    frozen;              /* 1 = engine frozen */
    double quiet_last;          /* last output sample seen */
    double hum_ph;              /* hum: phase of the fundamental, rad */
    double hum_vpp;             /* hum: Vpp of the block's ripple */
    uint32_t hiss_rng;          /* hiss: state of the hiss generator */
    double psu_i_total;         /* the supply's total current this cell, A (the hum's ripple reads it) */
    int    era_now;             /* era applied to the PSU */
    int    head_now;            /* head applied (0=100W, 1=50W) */
    double x_snap[64];          /* snapshot of the circuit state (nx=18) */
    /* Moving average of the state during the still blocks. The state of a SINGLE sample carries
     * the Newton's jitter (measured: 0,116 x tol), which on unfreezing shifts the operating point
     * PERSISTENTLY — it is not a click. Averaged over ~10 blocks that jitter cancels: ANMR of the
     * worst frame -8,62 -> -10,66 dB. The silence bypass is active by default: see BN_QUIET_EPS. */
    double x_avg[64], v_avg[16];
    int    avg_ok;
    int    quiet_rel;           /* 1 = silence thresholds TIED to dk.tol (BN_QUIET lever) */
    double frozen_out;          /* value emitted while frozen */
} BritishNoise;

/* EXPERIMENT hooks (not documented in the ttl). They are read ONCE per
 * process. BN_MAXIT<=0 is ignored (otherwise max_iter=0 and the Newton would
 * never iterate).
 * These getenv calls live in bn_build_cfg, which only instantiate calls;
 * rebuild_oversampler, reachable from run(), only moves the self->c pointer.
 * The caching is still right for another reason: an experiment lever must not
 * be able to change mid-session. */
static int bn_env_maxit(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = 48;
        const char* e = getenv("BN_MAXIT");
        if (e) { const int v = atoi(e); if (v > 0) { cached = v; } }
    }
    return cached;
}

static double bn_env_tol(void)
{
    static double cached = -1.0;
    if (cached < 0.0) {
        cached = 0.0;
        const char* e = getenv("BN_TOL");
        if (e) { const double v = atof(e); if (v > 0.0) { cached = v; } }
    }
    return cached;
}

/* index of the configuration for each legal factor (1,2,4,8) */
static int bn_osi(int factor)
{
    switch (factor) {
    case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3;
    default: return -1;
    }
}

/* BUILDS a configuration. This is the expensive part: memory allocation,
 * building the nodal system, DC solve and 256 settling iterations. It is only
 * called from instantiate — NEVER from the audio thread. */
static void bn_build_cfg(BritishNoise* self, int factor)
{
    const int ci = bn_osi(factor);
    if (ci < 0) { return; }
    self->c = &self->cfg[ci];
    bn_os_free(&self->c->os);
    if (bn_os_init(&self->c->os, factor, BN_MAX_BLOCK) != 0) {
        /* The config is abandoned, NOT degraded to 1x: a 1x oversampler with
         * `ready = 1` in the slot of the REQUESTED factor (e.g. 8x) would make
         * run(), which computes n_hi = n * os_factor with the requested factor,
         * process n*8 samples over a buffer of n. The config stays at
         * ready = 0 and the guard in rebuild_oversampler and the safety net in
         * instantiate take care of it. */
        bn_os_free(&self->c->os);
        return;
    }
    bn_os_free(&self->c->fxos);
    if (bn_os_init(&self->c->fxos, factor, BN_MAX_BLOCK) != 0) {
        bn_os_free(&self->c->fxos);
        bn_os_free(&self->c->os);
        return;
    }
    bn_fxc_free(&self->c->fxc);
    if (bn_fxc_init(&self->c->fxc, factor, BN_MAX_BLOCK, self->sample_rate) != 0) {
        bn_fxc_free(&self->c->fxc);
        bn_os_free(&self->c->fxos);
        bn_os_free(&self->c->os);
        return;
    }
    if (self->c->dkvar.dk) { bn_dkvar_free(&self->c->dkvar); }
    bn_dk_free(&self->c->dk);
    const double fs_hi = self->sample_rate * factor;
    /* The return value is checked: on OOM or with a singular mutual-inductance
     * matrix, carrying on would mark ready = 1 on a half-built system. */
    if (bn_dk_init(&self->c->dk, &BN_PREAMP2203_CIRCUIT, fs_hi) != 0) { return; }
    /* Valve: by default the plugin's OWN model (bn_tube_own, the shipped one).
     * The ORACLE build with a third-party reference valve model (for nulls against ngspice) is
     * compiled with -DBN_TUBE_AYUMI_ORACLE and adds src/dsp/bn_tube.c to the
     * link (make plugin-oracle). */
    bn_tube_own_tables_init();
    /* The selector picks ONCE between the variant with and without the trims
     * check. With BN_TRIMS unset —the normal case— the check would do nothing
     * and would be paid on the device's hottest path. */
    self->c->dk.device          = bn_tube_own_pick_tab_x4();
    self->c->dk.device_currents = bn_tube_own_pick_tab_currents_x4();
    /* The OWN triode sets J[1] = 0.0 literally (bn_tube_own.c): the grid
     * current of the D&Z model depends only on vgk. That leaves 32 of the 128
     * madds of the Jg assembly multiplying by an exact zero — see
     * jdev_ig_indep_va in bn_dk.h. Byte-identical, not approximate. */
    self->c->dk.jdev_ig_indep_va = 1;
    self->c->dk.jdev_blockdiag  = 1;
    /* Real-time tolerance: 3e-5 in the plugin, chosen together with the
     * silence bypass (see BN_QUIET_*). The ORACLE/bench paths stay at 1e-8:
     * this is the plugin only.
     *
     * Loosening the tolerance only trims Newton iterations; the
     * FACTORISATIONS do not drop (1.035/sample in every case), and iterations
     * are ~64% of the time. Measured with hard material (trebles with a dry
     * attack) at drive +12 dB and 4x, over 1.54 M samples:
     *   tol 1e-7 -> 5.173 iterations/sample   (reference)
     *   tol 1e-6 -> 4.636 iterations/sample   (-10.4%), output at -125.3 dB
     *   tol 1e-5 -> 4.130 iterations/sample   (-20.2%), output at  -89.4 dB
     *   tol 1e-4 -> 3.593 iterations/sample   (-30.5%), output at  -69.8 dB
     * Scored on ANMR instead of dB re peak: the preamp at 1e-4 against the
     * preamp at 1e-6, over 6 s of a real DI, nulls at -88,1 dB with a worst
     * ANMR frame of -29,4 dB (acceptable below -10), for 2,29 iterations per
     * sample instead of 3,00 (-24 %) and -11 % of the preamp's time (five
     * interleaved pairs, low-power 4-core x86 CPU). 1e-5 gave -129 dB / -72 for only -8 % of
     * iterations. The BN_TOL lever is still read once, to measure again. */
    self->c->dk.tol             = 3e-5;
    /* Real-time guard of the Newton (calibrated against the plugin's REAL
     * profile, which uses chord): guitar at drive +12 dB needs up to ~48
     * iterations at isolated samples — a cap of 48 gives IDENTICAL output
     * (−600 dB) and still protects against a ceiling of 100 on pathological
     * blocks. NOTE: a cap of 16, calibrated with the histogram WITHOUT chord,
     * changed the output at −26 dB: the guard is calibrated against the path
     * that actually runs. */
    /* the env reads are cached in bn_env_*(): this function runs at instantiate only, never from run() */
    self->c->dk.max_iter        = bn_env_maxit();
    { const double t = bn_env_tol(); if (t > 0.0) { self->c->dk.tol = t; } }
    self->tol_norm = self->c->dk.tol;
    { const char* e = getenv("BN_TOL_FINE"); const double tf = (e && *e) ? atof(e) : 1e-6;
      self->tol_fine = (tf > 0.0 && tf < self->tol_norm) ? tf : self->tol_norm; }
    self->tol_fine_on = 0; self->tol_env = 0.0;
    /* Persistent chord + explicit inverse, both ON in the product.
     * Persistent chord: the LU of the last factorisation survives across samples and the
     * adaptive refresh (bn_dk_kernel.h, ratio 0,5) refactors only when an iteration fails to halve
     * the residual. Explicit inverse: the inverse of Jg is built once per factorisation and
     * every Newton step is an 8x8 matvec instead of the serial triangular solve. Measured on 6 s
     * of DI at os 2 (idle, four alternating cycles): 490-531 ms -> 364-381 (-26 % of the plugin);
     * against the engine at tol 1e-6 the preamp lands at -89,6 dB / -37,8 worst frame where the
     * old start read -88,1 / -29,4; the explicit inverse's output is the plain chord's to -152,6 dB
     * re peak. Product gates (block invariance, fs/os invariance, RT safety) pass with both on.
     * The levers switch them OFF: BN_CHORD=0 and BN_CHORD_INV=0 (read once). The predictor
     * (BN_PRED) stays off: it is exclusive with the chord. */
    { static int chord = -1, inv = -1;
      if (chord < 0) { const char* e = getenv("BN_CHORD"); chord = (e && e[0] == '0') ? 0 : 1; }
      if (inv < 0)   { const char* e = getenv("BN_CHORD_INV"); inv = (e && e[0] == '0') ? 0 : chord; }
      self->c->dk.chord_persist = chord;
      self->c->dk.solve_inverse = inv; }
    /* Extrapolating v helps MOST below 4x, because that is where the warm
     * start is worst (iterations rise as the rate drops: 5.86 at 192k, 6.69 at
     * 96k, 7.91 at 48k).
     *
     * Measured on the preamp, bench with hard material:
     *      96 kHz  33.3% -> 23.2% of a core   (iter 6.69 -> 4.56)
     *     192 kHz  57.7% -> 43.4%             (iter 5.86 -> 3.94)
     * and on the full plugin at 2x, own work: 31.9% -> 23.9% on average.
     *
     * And it does NOT change the sound: null against the path without it
     * -145.4 dB rms / -106.4 peak at 2x, and -141.9 / -103.4 in the abuse
     * corner (1.5 V, steps, silence-to-full, full-scale noise, knobs at max),
     * with finite output. At 4x it is BIT-EXACT.
     *
     * (The flag is poison for the FULL-AMP circuit at the push-pull's
     * cut-off-to-conduction transition. That is another circuit: the preamp
     * does not switch like that.) */
    self->c->dk.extrapolate_v   = 1;
    /* variable parts: 9 pot sections + 4 contacts of the HIGH/LOW jack
     * (switched by Woodbury, without re-init and with the state x intact) */
    {
        int var_idx[BN_PREAMP2203_N_POTS + BN_PREAMP2203_N_JACK];
        for (int k = 0; k < BN_PREAMP2203_N_POTS; ++k) {
            var_idx[k] = BN_PREAMP2203_POT_IDX[k];
        }
        for (int k = 0; k < BN_PREAMP2203_N_JACK; ++k) {
            var_idx[BN_PREAMP2203_N_POTS + k] = BN_PREAMP2203_JACK_IDX[k];
        }
        /* The return value MUST be checked. It returns -1 if nv exceeds
         * BN_DK_NV_MAX, if the dimensions do not match or if there are mutual
         * inductances, and leaves the struct zeroed or with nv set and the
         * pointers NULL. The first run() would go through bn_apply_vars ->
         * bn_dkvar_set and dereference NULL: SIGSEGV on the audio thread. It
         * cannot fire with the current netlist (13 variable parts, no
         * mutuals), but regenerating the netlist is the realistic trigger. */
        if (bn_dkvar_init(&self->c->dkvar, &self->c->dk, &BN_PREAMP2203_CIRCUIT, fs_hi,
                          var_idx, BN_PREAMP2203_N_POTS + BN_PREAMP2203_N_JACK) != 0) { return; }
    }
    self->low_state = -1;
    self->us_state = -1;
    bn_psu_init(&self->psu);
    bn_psu_init(&self->psu_idle);
    self->vx_now = BN_PREAMP2203_BPLUS_V;
    self->rq_now = 1.0; self->rq_cell0 = 1.0; self->drq_hi = 0.0;
    self->w2_peak_fast = 0.0;
    self->env_fast = 0.0;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }  /* forces apply */

    /* Operating point by static DC (bn_dk_dc_init): with C21 (tau=0.5 s) a
     * brute-force warm-up would be seconds of audio in instantiate. */
    double u[2] = { 0.0, BN_PREAMP2203_BPLUS_V };
    double y[16];
    if (bn_dk_dc_init(&self->c->dk, &BN_PREAMP2203_CIRCUIT, fs_hi, u) != 0) { return; }
    for (int k = 0; k < 256; ++k) {          /* settle the numerical residual */
        bn_dk_process(&self->c->dk, u, y);
    }
    /* SNAPSHOT of the operating point — HERE, not earlier.
     * ⚠️ Taking it next to `ready = 1` would capture the state AT ZERO,
     * because the DC init and these 256 iterations happen AFTER it. Measured:
     * the output level sank by 81.8 % at the lowest input — the exact
     * signature of an amp starting COLD.
     * ⚠️ And `v` goes into the snapshot just like `x`: they are the valves'
     * grid and plate voltages at their operating point (vgk ~ -2 V,
     * vak ~ 250 V), not a numerical residual. */
    self->c->n_x_dc = self->c->dk.n_states;
    self->c->n_v_dc = self->c->dk.n_nl;
    self->c->x_dc = (double*)calloc((size_t)self->c->n_x_dc, sizeof(double));
    self->c->v_dc = (double*)calloc((size_t)self->c->n_v_dc, sizeof(double));
    if (self->c->x_dc && self->c->v_dc) {
        memcpy(self->c->x_dc, self->c->dk.x, sizeof(double) * (size_t)self->c->n_x_dc);
        memcpy(self->c->v_dc, self->c->dk.v, sizeof(double) * (size_t)self->c->n_v_dc);
    } else {
        free(self->c->x_dc); free(self->c->v_dc);
        self->c->x_dc = NULL; self->c->v_dc = NULL;   /* activate() honours this */
    }
    self->c->dc_x1 = 0.0;      /* activate() starts the blocker on the net's rest output */
    self->c->dc_y1 = 0.0;
    /* the seam: the w2 blocker starts settled at its rest value, and the dry
     * ring carries the AC rest value, which is zero */
    self->c->w2_dc = y[0];
    self->c->seam_a = exp(-2.0 * BN_PI * BN_SEAM_HP_HZ / fs_hi);
    self->c->seam_dc_x1 = y[0];
    self->c->seam_dc_y1 = 0.0;
    memset(self->c->pb_ring, 0, sizeof self->c->pb_ring);
    self->c->pb_pos = 0;
    bn_nn_rate_init(&self->c->nnr, fs_hi, self->nn_fs);

    /* DC blocker pole at a FIXED fc (7.64 Hz, the factory configuration's).
     *
     * For a one-pole filter the time constant is preserved across rates with
     * a2 = a1^(fs1/fs2) — which is exactly what this pow does. It is written
     * this way, and NOT as exp(-2*pi*fc/fs_hi), for a specific reason: at the
     * factory setting (2x at 48k => fs_hi = 96000) the exponent is exactly
     * 1.0, and pow(x, 1.0) returns x WITHOUT error, so the shipped case is
     * BYTE-IDENTICAL to the literal 0.9995. The exponential form would give
     * 0.99949980… instead of 0.9995 and would move the factory sound by
     * rounding. Verified by measurement: exact null at 2x/48k. */
    self->c->dc_a = pow(BN_DCBLOCK_A_REF, BN_DCBLOCK_FS_REF / fs_hi);

    /* `ready = 1` must be the LAST thing, after bn_dkvar_init and bn_dk_dc_init
     * (the same pattern as the early returns after bn_os_init and bn_dk_init).
     * bn_dk_dc_init returns -2 when the static Newton falls back (not only on
     * OOM): with ready already set, the config would be marked good with
     * dc_a = 0.0, and the output blocker would degenerate into y = x - x1, a
     * DIFFERENTIATOR (+6 dB/oct across the whole band), plus seam_a = 0,
     * x_dc = NULL (activate() could never restore the operating point) and
     * the reduced stage without init or prime. The plugin would instantiate
     * fine and sound wrong, SILENTLY.
     * ⇒ The flag is the LAST thing: reaching here means the config is whole. */

    /* And "whole" includes FINITE. Nothing between bn_dk_dc_init and here
     * checks that the operating point is a number: the static Newton can come back with a NaN
     * in `v` without returning an error, the 256 settling iterations then spread it through
     * `x`, and the snapshot copies it. With ready = 1 the config is used, activate() restores
     * that NaN as the working point on every transport stop, and the plugin outputs silence or
     * NaN for the rest of the session — with no error anywhere, which is the expensive part.
     * ⚠️ x_dc/v_dc being NULL is a DIFFERENT case and stays legal: it means the calloc failed
     * and activate() knows to skip the restore. Only a snapshot that EXISTS and is not finite
     * is a broken config. */
    if (self->c->x_dc) {
        for (int k = 0; k < self->c->n_x_dc; ++k) {
            if (!isfinite(self->c->x_dc[k])) { return; }
        }
    }
    if (self->c->v_dc) {
        for (int k = 0; k < self->c->n_v_dc; ++k) {
            if (!isfinite(self->c->v_dc[k])) { return; }
        }
    }
    self->c->ready = 1;
}

/* The whole teardown of an instance, IN ONE PLACE.
 *
 * It frees every configuration regardless of `ready`: `bn_build_cfg` has SEVERAL early returns
 * AFTER allocating that leave a half-built config with `ready = 0`, and skipping those would leak
 * that memory forever. The failure paths of `instantiate` also come through here, so the DK/dkvar/os
 * of the configs and the weights of the four arms are freed too; otherwise a host that retries the
 * load in a loop would lose megabytes per attempt.
 * ⇒ Freeing a half-built config is SAFE: `self` comes from `calloc`, so the structs start at zero,
 * and bn_dk_free / bn_dkvar_free / bn_os_free / bn_nn_lstm_free are all free(NULL) + memset on a
 * zeroed struct. That is why no `ready` condition is needed. */
static void
bn_teardown(BritishNoise* self)
{
    if (!self) { return; }
    for (int ci = 0; ci < 4; ++ci) {
        free(self->cfg[ci].x_dc);
        free(self->cfg[ci].v_dc);
        bn_dk_free(&self->cfg[ci].dk);
        bn_dkvar_free(&self->cfg[ci].dkvar);
        bn_os_free(&self->cfg[ci].os);
        bn_os_free(&self->cfg[ci].fxos);
        bn_fxc_free(&self->cfg[ci].fxc);
    }
    for (int k = 0; k < BN_NN_N_ARMS; ++k) { bn_nn_lstm_free(&self->nn[k]); }   /* the packed weights */
    free(self->buf_in);
    free(self->buf_hi);
    free(self->buf_out);
    free(self->fx_ret_lo);
    free(self->fx_ret_hi);
    free(self->fx_send_hi);
    free(self->fx_rcap);
    free(self);
}

static LV2_Handle
instantiate(const LV2_Descriptor* descriptor, double rate,
            const char* bundle_path, const LV2_Feature* const* features)
{
    (void)descriptor; (void)bundle_path; (void)features;
    BritishNoise* self = (BritishNoise*)calloc(1, sizeof(BritishNoise));
    if (!self) {
        return NULL;
    }
    self->sample_rate = rate;
    self->tol_env_decay = exp(-1.0 / (0.050 * rate * 2.0));   /* the tolerance follower's 50 ms release at fs_hi (2x) */
    self->trim_lin    = 1.0f;
    self->smooth_coef = (float)(1.0 - exp(-1.0 / (0.005 * rate)));
    {   /* the chunk while a pot moves: BN_KNOB_CHUNK_S in samples at this rate, at least 1 */
        const double kc = floor(BN_KNOB_CHUNK_S * rate + 0.5);
        self->knob_chunk = kc < 1.0 ? 1u : kc > (double)BN_MAX_BLOCK ? (uint32_t)BN_MAX_BLOCK : (uint32_t)kc;
    }

    self->buf_in  = (double*)calloc(BN_MAX_BLOCK, sizeof(double));
    self->buf_out = (double*)calloc(BN_MAX_BLOCK, sizeof(double));
    self->buf_hi  = (double*)calloc((size_t)BN_MAX_BLOCK * 8 + 8, sizeof(double));
    /* FX loop: the return at the base rate (n) and the two buffers at fs_hi */
    self->fx_ret_lo  = (double*)calloc(BN_MAX_BLOCK, sizeof(double));
    self->fx_ret_hi  = (double*)calloc((size_t)BN_MAX_BLOCK * 8 + 8, sizeof(double));
    self->fx_send_hi = (double*)calloc((size_t)BN_MAX_BLOCK * 8 + 8, sizeof(double));
    self->fx_rcap    = (double*)calloc(BN_MAX_BLOCK, sizeof(double));
    if (!self->buf_in || !self->buf_out || !self->buf_hi
        || !self->fx_ret_lo || !self->fx_ret_hi || !self->fx_send_hi || !self->fx_rcap) {
        bn_teardown(self);       /* the buffers AND the configs AND the arms: see bn_teardown */
        return NULL;
    }
    self->fx_lvl_now = 1.0;   /* the port's default; the ramp starts from here */

    /* The configurations are built HERE, outside the audio thread. Changing
     * factor live is just moving a pointer. */
    /* The neural power stage — the four embedded weight sets, bound (no I/O, no copy),
     * the activation table built here and never in run(). The net IS the power stage: if any arm
     * fails to bind, or the arms were trained at different rates, the instance is not created and
     * the host reports a failed load. Bound BEFORE the configurations: each one's rate converter
     * needs the nets' rate. */
    bn_nn_lstm_lut_init();
    for (int k = 0; k < BN_NN_N_ARMS; ++k) {
        if (bn_nn_lstm_bind(&self->nn[k], bn_nn_arms[k].blob, bn_nn_arms[k].bytes) != 0
            || self->nn[k].fs != self->nn[0].fs) {
            bn_teardown(self);
            return NULL;
        }
    }
    self->nn_fs = (double)self->nn[0].fs;
    /* The zero state is NOT the net at rest: from it the net emitted a start-up transient —
     * 0,32 peak at the output 1,2 ms after activate, tail to ~300 ms, at ANY knob setting, volume 0 included —
     * and then a DC its blocker had to drain. Each arm is settled HERE (outside the audio thread) on 0,5 s of
     * silence at the rest rail and the netlist's default knobs; `activate` restores that state, the net's ring
     * carries its rest output and the DC blocker starts settled on it. Measured rest outputs: UK −0,297 V ·
     * 50 W −0,433 · US −0,060 · 50 W US +0,276 (they take 0,3-0,7 s of silence to converge). */
    /* THE NET'S REST STATE IS PRECOMPUTED, NOT RECOMPUTED ON EVERY INSTANCE.
     * Settling the FOUR arms for 0,5 s each here would be 4 x 48000 steps = 3910 M instructions
     * = 0,40 s of a 0,417 s instantiation (low-power 4-core x86 CPU, `performance` governor), i.e. 3,5 TIMES what
     * processing TWO SECONDS of audio costs. And it would run ALWAYS, also in the preamp-only
     * build, where the net is compiled OUT of run().
     * The settled state is DETERMINISTIC given the blob, i.e. a generated constant like any
     * other: `tools/gen_nn_rest.c` emits it by calling the SAME bn_nn_lstm_settle with the same
     * arguments, so the table is BIT-IDENTICAL to what settling here would produce. Its gate
     * (`make mk81-nn-rest`) reruns the settle and demands equality, so the table cannot go stale
     * when a weight set is retrained.
     * ⛔ It is NOT deferred to first use: that would put 0,1 s INSIDE the audio thread on a variant
     * switch. It is precomputed, not postponed. */
    for (int k = 0; k < BN_NN_N_ARMS; ++k) {
        memcpy(self->nn[k].h_rest, BN_NN_REST_H_TAB[k], sizeof self->nn[k].h_rest);
        memcpy(self->nn[k].c_rest, BN_NN_REST_C_TAB[k], sizeof self->nn[k].c_rest);
        self->nn[k].y_rest   = BN_NN_REST_Y_TAB[k];
        self->nn[k].has_rest = 1;
    }
    self->os_factor = 0;
    /* the silence thresholds lever, read ONCE (never on the audio thread). */
    { const char* e = getenv("BN_QUIET"); self->quiet_rel = (e && e[0] == '0') ? 0 : 1; }
    self->psu_grid = (uint32_t)lround(256.0 * rate / 48000.0);
    if (self->psu_grid < 1) { self->psu_grid = 1; }
    /* Only the factor the product offers is built: 2x. Each configuration is a whole engine (DK init,
     * rings, oversamplers), so building one is the instantiate time and memory of one. Every loop over
     * cfg[] guards on `ready`. */
    bn_build_cfg(self, 2);
    /* FACTORY = 2x: 23.9% of a core on average against 40.8% at 4x, within
     * the target. See the port's comment in the .ttl. 2x is the only factor
     * built and the only one the port can select. */
    if (!self->cfg[bn_osi(2)].ready) {   /* should not happen; safety net */
        bn_teardown(self);       /* the buffers AND the configs AND the arms: see bn_teardown */
        return NULL;
    }
    self->c = &self->cfg[bn_osi(2)];
    self->os_factor = 2;
    /* Ramp of the oversampling change: 5 ms down + 5 ms up. At rest
     * os_gain = 1 and bn_os_ramp does nothing. */
    self->os_phase = 0;
    self->os_want  = 2;
    self->os_primero = 1;
    self->os_gain  = 1.0f;
    self->os_step  = (float)(1.0 / (0.005 * rate));
    self->nn_arm = 0; self->nn_arm_want = 0; self->nn_hold = 0; self->nn_gain = 1.0;
    self->cond_from_ports = 1;
    return (LV2_Handle)self;
}

static void
connect_port(LV2_Handle instance, uint32_t port, void* data)
{
    BritishNoise* self = (BritishNoise*)instance;
    switch ((BnPortIndex)port) {
    case BN_IN:         self->in         = (const float*)data; break;
    case BN_OUT:        self->out        = (float*)data;       break;
    case BN_TRIM:       self->trim       = (const float*)data; break;
    case BN_OVERSAMPLE: self->oversample = (const float*)data; break;
    case BN_VOL:    self->vol    = (const float*)data; break;
    case BN_TREBLE: self->treble = (const float*)data; break;
    case BN_MIDDLE: self->middle = (const float*)data; break;
    case BN_BASS:   self->bass   = (const float*)data; break;
    case BN_MASTER: self->master = (const float*)data; break;
    case BN_LATENCY:    self->latency    = (float*)data;       break;
    case BN_BIAS:       self->bias       = (const float*)data; break;
    case BN_US:         self->us_spec    = (const float*)data; break;
    case BN_LOW:        self->low_input  = (const float*)data; break;
    case BN_PRESENCE:   self->presence   = (const float*)data; break;
    case BN_CAB_LOAD:   self->cab_load   = (const float*)data; break;
    case BN_BYPASS:     self->bypass     = (const float*)data; break;
    case BN_POWER:      self->power      = (const float*)data; break;
    case BN_HUM:        self->hum        = (const float*)data; break;
    case BN_HISS:       self->hiss       = (const float*)data; break;
    case BN_ERA:        self->era        = (const float*)data; break;
    case BN_HEAD:       self->head       = (const float*)data; break;
    case BN_BYPASS_PREAMP: self->bypass_preamp = (const float*)data; break;
    case BN_BYPASS_POWER:  self->bypass_power  = (const float*)data; break;
    case BN_FX_LOOP:       self->fx_loop       = (const float*)data; break;
    case BN_FX_SEND:       self->fx_send       = (float*)data;       break;
    case BN_FX_RETURN:     self->fx_return     = (const float*)data; break;
    case BN_FX_MIX:        self->fx_mix        = (const float*)data; break;
    case BN_FX_RETURN_LVL: self->fx_return_lvl = (const float*)data; break;
    case BN_ENGINE:        self->engine        = (const float*)data; break;
    default: break;
    }
}

/* Real-time factor CHANGE: move a pointer and nothing else. Every
 * configuration is already built in instantiate, so there is no memory
 * allocation and no solve here (rebuilding from run() measured a 7.8 ms peak
 * on the audio thread). */
/* the power stage's latency in fs_hi samples (see BnCfg.nnr): the net's rate-converter delay rounded up to a
 * whole host sample; what the declared latency, the power bypass's dry path and the net all align to */
static int bn_pw_lat(const BnCfg* c, int factor)
{
    const int f = factor > 0 ? factor : 1;
    int lat = c->nnr.delay > 0 ? ((c->nnr.delay + f - 1) / f) * f : 0;
    if (lat > BN_PB_RING - 1) { lat = BN_PB_RING - 1; }
    return lat;
}

static void rebuild_oversampler(BritishNoise* self, int factor)
{
    if (factor == self->os_factor) { return; }
    const int ci = bn_osi(factor);
    if (ci < 0 || !self->cfg[ci].ready) { return; }
    self->c = &self->cfg[ci];
    /* a cell in progress keeps its rail ramp; its slope is per fs_hi sample */
    if (self->os_factor > 0) {
        self->dvx_hi  *= (double)self->os_factor / (double)factor;
        self->drq_hi  *= (double)self->os_factor / (double)factor;
    }
    self->os_factor = factor;
    /* the incoming configuration carries the knobs from the last time it was
     * used: force them to be reapplied (Woodbury, cheap) */
    self->low_state = -1;
    self->us_state = -1;
    self->byp_pre_mix = 0.0;   /* starts with the preamp ACTIVE; the ramp leads to the target */
    self->byp_pow_mix = 0.0;   /* likewise the power stage */
    self->fx_mix_now  = 0.0;   /* FX: the loop re-engages (and resets its pair) */
    self->fx_running  = 0;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }
    /* ⚠️ The PSU is NOT reset here (`bn_psu_init` + `vx_now = BPLUS`): the
     * supply is NOT state that depends on the oversampling — it is the amp's
     * rail, and changing a simulation's factor has no reason to switch it off.
     * Resetting it makes the rail jump to B+ and the amp momentarily louder
     * until the sag pulls it down again: measured, an OVERSHOOT of 1.31
     * against 1.14 for the control during ~150 ms after each change. The rail
     * is carried as is.
     * `w2_peak_fast` IS reset: it is a per-block peak, not a physical state. */
    self->w2_peak_fast = 0.0;
    self->env_fast = 0.0;
    /* the incoming config's rings hold what it carried when it was last left: clear them (the os
     * ramp is at its bottom here). The net's state is at its own rate in every config: kept. */
    bn_nn_rate_reset(&self->c->nnr);
    memset(self->c->nn_ring, 0, sizeof self->c->nn_ring);
    self->c->nn_pos = 0;
}

/* Advances the oversampling-change ramp and returns the gain to apply.
 * At rest (phase 0) it returns 1.0f and costs nothing. See the os_phase block. */
#ifndef BN_OS_SETTLE_S
#define BN_OS_SETTLE_S 0.050
#endif
static inline float bn_os_ramp(BritishNoise* self)
{
    if (self->os_phase == 1) {
        self->os_gain -= self->os_step;
        if (self->os_gain <= 0.f) { self->os_gain = 0.f; self->os_phase = 2; }
    } else if (self->os_phase == 4) {
        /* the new configuration runs muted for BN_OS_SETTLE_S before the fade-in */
        if (--self->os_hold <= 0) { self->os_phase = 3; }
    } else if (self->os_phase == 3) {
        self->os_gain += self->os_step;
        if (self->os_gain >= 1.f) { self->os_gain = 1.f; self->os_phase = 0; }
    }
    return self->os_gain;
}

/* Reads a port in dB. NaN must be caught explicitly (with the plain form
 * `if (db < lo)` a NaN escapes both ifs and poisons the smoother forever).
 *
 * ⚠️ NaN and "out of range below" are TWO distinct cases and are kept apart.
 * A single negated guard `if (!(db >= lo)) return 0` would merge them: a
 * finite, legitimate value just below the .ttl minimum (e.g. -30 dB against a
 * -24 minimum) would JUMP to the 0 dB default — a +24 dB leap, exactly the
 * opposite of what a clamp should do. So: NaN -> default, out of range ->
 * CLAMP. (The default for NaN is 0 dB and not the minimum, because -24 dB is
 * almost silence and would be indistinguishable from the fault itself.) */
static inline float bn_clamp_db(float db, float lo, float hi)
{
    if (db != db) { return 0.0f; }        /* NaN -> default */
    if (db < lo)  { return lo; }          /* out below -> clamp */
    if (db > hi)  { return hi; }          /* out above -> clamp */
    return db;
}

/* Pushes a sample into the dry delay line and returns the one coming out,
 * already aligned with the latency the plugin declares. */
static inline double bn_dry_push(BritishNoise* self, double x)
{
    self->dry_ring[self->dry_pos] = x;
    int r = self->dry_pos - self->dry_delay;
    if (r < 0) { r += BN_DRY_RING; }
    self->dry_pos = (self->dry_pos + 1) & (BN_DRY_RING - 1);
    return self->dry_ring[r];
}

/* One step of the wet<->dry ramp of BYPASS/POWER. Linear, 10 ms: all it has
 * to do is keep the change from being a step. */
static inline void bn_dry_ramp(BritishNoise* self, double target, double step)
{
    if (self->dry_mix < target) {
        self->dry_mix += step;
        if (self->dry_mix > target) { self->dry_mix = target; }
    } else if (self->dry_mix > target) {
        self->dry_mix -= step;
        if (self->dry_mix < target) { self->dry_mix = target; }
    }
}

/* The same linear ramp, for any state (the mixes of the w2 seam). */
static inline void bn_ramp_to(double* v, double target, double step)
{
    if (*v < target) {
        *v += step;
        if (*v > target) { *v = target; }
    } else if (*v > target) {
        *v -= step;
        if (*v < target) { *v = target; }
    }
}

/* Reads a CONTINUOUS control port with its default and range: NaN -> default,
 * out of range -> clamp (negated comparisons, so a NaN cannot slip through). */
static inline double bn_ctl(const float* port, double def, double lo, double hi)
{
    if (!port) { return def; }
    const double v = (double)*port;
    if (!(v >= lo)) { return (v != v) ? def : lo; }
    if (!(v <= hi)) { return hi; }
    return v;
}

/* THE SILENCE BYPASS THRESHOLDS, TIED TO THE TOLERANCE.
 *
 * WHY. The rest floor is NOT fixed noise: it is the Newton's residual, and it scales LINEARLY with
 * `dk.tol` — measured, `max|dy| = 0,116 x tol`, constant over three decades (tol 1e-4 -> 1,164e-5 ·
 * 1e-5 -> 1,163e-6 · 1e-6 -> 1,164e-7 · 1e-8 -> 5,66e-8). The absolute thresholds were calibrated
 * against a floor of 7,4e-9, so raising the tolerance from 1e-6 to 1e-4 lifts the floor 100x and makes
 * them unreachable: the bypass never engages and a stopped track costs as much as a playing one. An
 * absolute tolerance is not a tolerance; tying the thresholds to `tol` keeps them from going stale
 * whenever the tolerance moves again.
 *
 * ⭐ ON BY DEFAULT (together with tol 3e-5). The factors are the MEASURED ones. With `BN_QUIET=0` the
 * absolute constants come back, so the before and after can be measured on the same machine. */
/* Every read of a 0..1 control port goes through here: NULL, NaN and Inf read as 0 (a +Inf on `hiss` once poisoned
 * the DK's state for good), the rest is clamped to the TTL range. */
static inline double bn_port01(const float* p)
{
    if (!p) { return 0.0; }
    const float v = *p;
    if (!isfinite(v)) { return 0.0; }
    return v < 0.0f ? 0.0 : (v > 1.0f ? 1.0 : (double)v);
}
static inline double bn_quiet_eps(const BritishNoise* self)
{   return self->quiet_rel ? 0.5 * self->c->dk.tol : (double)BN_QUIET_EPS;   }
static inline double bn_quiet_level(const BritishNoise* self)
{   return self->quiet_rel ? 1.0 * self->c->dk.tol : (double)BN_QUIET_LEVEL; }
static inline double bn_quiet_xeps(const BritishNoise* self)
{   return self->quiet_rel ? BN_QUIET_XEPS_K * self->c->dk.tol : (double)BN_QUIET_XEPS;  }

static void bn_apply_vars(BritishNoise* self, const float* kn, int low)
{
    double pv[BN_PREAMP2203_N_POTS + BN_PREAMP2203_N_JACK];
    const int us = (self->us_spec && *self->us_spec > 0.5f) ? 1 : 0;   /* US preamp pots */
    bn_preamp2203_pot_values(kn[0], kn[1], kn[2], kn[3], kn[4], pv, us);
    const double* jv = low ? BN_PREAMP2203_JACK_LOW : BN_PREAMP2203_JACK_HIGH;
    for (int k = 0; k < BN_PREAMP2203_N_JACK; ++k) {
        pv[BN_PREAMP2203_N_POTS + k] = jv[k];
    }
    /* a refused move (singular I + Q·dG, counted in dkvar.n_singular) leaves the states as they were, so the
     * next block tries again instead of treating the knob as applied */
    if (bn_dkvar_set(&self->c->dkvar, pv) != 0) { return; }
    self->low_state = low;
    self->us_state = us;
}

static void
activate(LV2_Handle instance)
{
    BritishNoise* self = (BritishNoise*)instance;
    self->quiet_run = 0; self->frozen = 0;
    /* the supply grid restarts with the stream */
    self->psu_phase = 0; self->pk_cell = 0.0; self->dvx_hi = 0.0; self->drq_hi = 0.0; self->rq_cell0 = 1.0;
    self->hiss_rng = 0x9E3779B9u;   /* fixed seed: reproducible */
    self->psu_i_total = 0.213;      /* idle current */
    /* The two envelopes are state too. Without this, after a transport stop the first block
     * reads the map (and the bias' grid current) with the PRE-STOP peak — full drive on an amp
     * this function has just put back at rest. */
    self->w2_peak_fast = 0.0;
    self->env_fast = 0.0;
    /* OLD WAY: it must MATCH the .ttl's lv2:default (1.0), or a host that
     * does not connect the port would sound different from one that connects
     * it with its default. */
    self->era_now = 1;              /* old way, the catalogue 100 W head */
    /* ⛔ And it MUST BE APPLIED TO THE PSU HERE; declaring it is not enough.
     * `bn_psu_init()` leaves `old_way = 0`, and the only place that calls
     * `bn_psu_set_era()` is behind `if (era_q != self->era_now)`. With the
     * default at 1, that guard NEVER fires on instantiation ⇒ the plugin would
     * say old way while the PSU ran new way, and the control would change
     * NOTHING in the audio (only the hum, which takes another path).
     * Measured without this: with the hum off, era=0 and era=1 gave
     * BIT-IDENTICAL output; with the hum on they did differ (x2,000, its known
     * figure) — so the port arrived and the default was what lied. */
    /* LIFECYCLE: the supply, the rails, the hum's phase and the power stage are STATE; if they survived a
     * deactivate/activate, a host that stopped and restarted the transport would get a different render
     * (measured on 4 s of DI: 0,45-0,56 apart half a second in, both engines). LV2's activate() is where the
     * instance returns to its initial state. ⛔ This is NOT the reset `rebuild_oversampler` must not do: there the
     * rail is carried across on purpose (a factor change is not an amp switched off). */
    bn_psu_init(&self->psu); bn_psu_init(&self->psu_idle);
    bn_psu_set_era(&self->psu, self->era_now);
    bn_psu_set_era(&self->psu_idle, self->era_now);
    self->head_now = 0;             /* the 100 W head */
    bn_psu_set_head(&self->psu, self->head_now); bn_psu_set_head(&self->psu_idle, self->head_now);
    self->vx_now = BN_PREAMP2203_BPLUS_V; self->vx_cell0 = self->vx_now;
    self->rq_now = 1.0;
    self->hum_ph = 0.0;
    memset(self->x_snap, 0, sizeof(self->x_snap));
    self->quiet_last = 0.0; self->frozen_out = 0.0;
    self->trim_lin  = 1.0f;
    bn_os_reset(&self->c->os);
    /* The dry ring is cleared: if it survived deactivate/activate, up to
     * `dry_delay` samples of audio from BEFORE the pause would come out after
     * resuming. It is a pure signal buffer, not physical state (unlike the
     * PSU rail, which is physical and is kept on purpose in
     * rebuild_oversampler). */
    memset(self->dry_ring, 0, sizeof self->dry_ring);
    self->dry_pos = 0;
    /* RECOVERY: activate() returns EVERY configuration to its operating point,
     * so a poisoned DK state is cured by deactivate/activate instead of
     * needing a re-instantiation. It is not a reset to zero: it is the
     * snapshot instantiate left after the DC init, so the amp comes back warm
     * and without a transient. It goes here and not in run() because
     * activate() is not the audio thread. */
    for (int ci = 0; ci < 4; ++ci) {
        BnCfg* c = &self->cfg[ci];
        if (!c->ready || !c->x_dc) { continue; }
        /* The engine's own numeric reset, not a hand-kept list of its flags: resetting only
         * have_lu would leave pred_ok and inv_ok set across a transport stop, and with
         * BN_PRED=1 the first sample after a restart would take a tangent step from the
         * pre-stop p. `x` is untouched by bn_dk_reset_estado; v/v_prev are restored right after. */
        bn_dk_reset_estado(&c->dk);
        memcpy(c->dk.x, c->x_dc, sizeof(double) * (size_t)c->n_x_dc);
        memcpy(c->dk.v, c->v_dc, sizeof(double) * (size_t)c->n_v_dc);
        if (c->dk.v_prev) {
            memcpy(c->dk.v_prev, c->v_dc, sizeof(double) * (size_t)c->n_v_dc);
        }
        /* the w2 seam: since the DK returns to its rest point, its blocker
         * and the dry ring return to theirs; the FX loop's pair goes cold
         * (it re-engages by itself if the loop is still active) */
        c->seam_dc_x1 = c->w2_dc;
        c->seam_dc_y1 = 0.0;
        c->dc_x1 = 0.0;             /* the active configuration's blocker is started on the net's rest below */
        c->dc_y1 = 0.0;
        memset(c->pb_ring, 0, sizeof c->pb_ring);
        c->pb_pos = 0;
        memset(c->nn_ring, 0, sizeof c->nn_ring);
        c->nn_pos = 0;
        bn_nn_rate_reset(&c->nnr);
        bn_os_reset(&c->os);   /* every configuration, not just the one in use: the host may switch factor after a restart */
        bn_os_reset(&c->fxos);
        bn_fxc_reset(&c->fxc);
    }
    self->fx_running = 0;
    self->fx_send_gain = 0.0;
    self->nn_hold = 0;
    self->tol_env = 0.0; self->tol_fine_on = 0;   /* the tolerance follower restarts where a fresh instance does */
    /* the port ramps and the "last applied" caches are state too: a restart must start them where a fresh
     * instance does, or the first blocks ramp from wherever the previous run left them */
    self->byp_pre_mix = 0.0; self->byp_pow_mix = 0.0;
    self->fx_mix_now = 0.0; self->fx_lvl_now = 1.0;
    /* `dry_mix` and the oversampling ramp are state too: a restart must start them where the PORTS say, or the first
     * take begins with 10 ms of the wrong path fading out (raw DI after a stop in bypass; processed signal on an
     * instance created in bypass). Control port contents are not guaranteed meaningful until run() (lv2.h), so the
     * value read here is a best guess and the first run() overrides it from its own target.
     * `self->c` and the PSU rail are NOT reset here: that is physical state, and its non-reset is measured. */
    self->dry_mix = ((self->power && *self->power < 0.5f) || (self->bypass && *self->bypass > 0.5f)) ? 1.0 : 0.0;
    self->dry_from_ports = 1;
    self->cond_from_ports = 1;
    self->os_phase = 0; self->os_gain = 1.0f; self->os_want = self->os_factor;
    self->low_state = -1; self->us_state = -1;
    /* x_avg/v_avg/avg_ok must be reset here too — the mechanical failure of this list is
     * forgetting it when a field is added. `avg_ok` is only cleared in run() when the input stops
     * being zero, so if the transport stops DURING silence it stays at 1 with the average of the
     * previous operating point; after activate the first chunk is all zeros again, the weight is
     * 0,10 instead of 1,0 and the new silence's average starts seeded with the pre-stop one —
     * which may be from another head, other knobs or another os configuration.
     * ⚠️ HONEST MAGNITUDE, and that is why it is not claimed audible: chunks are cut at the PSU
     * cell edge (256 base samples), so half a second of silence is ~94 updates and the stale
     * seed decays to 0,9^94 ~ 5e-5. What this fixes is the INVARIANT; the margin that saves it
     * otherwise is incidental, not a decision of this code. */
    memset(self->x_avg, 0, sizeof self->x_avg);
    memset(self->v_avg, 0, sizeof self->v_avg);
    self->avg_ok = 0;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }
    for (int k = 0; k < BN_NN_N_ARMS; ++k) { bn_nn_lstm_rest(&self->nn[k]); }   /* the net starts at ITS rest, not at zero state */
    self->nn_gain = 1.0;
    /* The ARM is derived here from the PORTS, the same way run() does it. Priming with `nn_arm`
     * as it happened to be — 0 from instantiate, or the previous session's — while run() picks by
     * (head, us) would, on the first activation of a session with the head at 50 W, start the ring
     * with the UK arm's rest while the net emitted the 50 W one: a step that the 7,64 Hz blocker
     * turns into a decaying thump at the start of every session.
     * ⛔ `head_now` does NOT serve here: activate forces it to 0 a few lines above. */
    self->nn_arm = (self->head && *self->head > 0.5f ? 1 : 0)
                 + 2 * (self->us_spec && *self->us_spec > 0.5f ? 1 : 0);
    self->nn_arm_want = self->nn_arm;
    if (self->c) {
        /* the net from the first sample: its ring and its DC blocker start ON the net's rest output, so
         * nothing sees a step */
        const double yr = (double)self->nn[self->nn_arm].y_rest;
        for (size_t k = 0; k < BN_PB_RING; ++k) { self->c->nn_ring[k] = yr; }
        self->c->dc_x1 = yr; self->c->dc_y1 = 0.0;
    }
}





/* ============================================================================
 * JOHNSON HISS (thermal noise of the input network).
 *
 * WHAT IT IS: the electrons in any resistor are agitated by temperature, and
 * that produces a noise voltage even with no current flowing. v = sqrt(4kTRB).
 * Pure physics, not a defect. In audio it is the background "shhhh" — bright
 * and constant, unlike the hum, which is low and tonal.
 *
 * WHERE IT ENTERS, and why this CAN go through the real input (unlike the
 * hum): the noise is generated in the input network, i.e. BEFORE the whole
 * circuit. So it is added to buf_in —which is in VOLTS— and the DK solver
 * amplifies and filters it by itself, with the amp's real response. There is
 * no need to derive the output level: it emerges.
 *
 * THE NETWORK, from the schematic: pickup -> jH, R2 1M to ground, R3 68k to
 * the grid. The grid is high impedance, so both noise sources arrive whole and
 * their powers add:  R_eq = (Rpickup ∥ 1M) + 68k
 *
 * ⭐ And this is the good part, because it removes the arbitrariness: the 68k
 * STOPPER contributes 90% of the noise power, more than the pickup itself. So
 * the choice of pickup barely matters:
 *      single coil  6k -> 5.42 uV      humbucker    8k -> 5.49 uV
 *      humbucker   16k -> 5.77 uV
 * i.e. 5.4-5.8 uV, about ±3%. The hiss is mostly THE AMP'S, not the guitar's.
 * An 8k humbucker (typical for this kind of amp) is assumed.
 *
 * IT SCALES WITH THE RATE because the band is fs/2: 5.27 uV at 44.1k, 5.49 at
 * 48k, 7.77 at 96k. That is CORRECT — the spectral density is what is
 * preserved.
 *
 * ⚠️ WHAT IT IS NOT: the noise of the VALVE itself (shot and flicker), which
 * in a real preamp usually DOMINATES over the resistors'. It is not modelled
 * because its magnitude depends on the individual valve and is on no
 * schematic. So this hiss is a LOWER BOUND of a real 100 W head's.
 * ========================================================================== */
#define BN_HISS_K       1.380649e-23   /* Boltzmann */
#define BN_HISS_T       300.0          /* ambient temperature, K */
#define BN_HISS_RPICK   8000.0         /* typical humbucker */
#define BN_HISS_RLEAK   1.0e6          /* R2, grid leak */
#define BN_HISS_RSTOP   68000.0        /* R3, grid stopper — the dominant one */

/* noise rms per sample at the given rate */
static double
bn_hiss_rms(double fs)
{
    const double rpar = BN_HISS_RPICK * BN_HISS_RLEAK
                      / (BN_HISS_RPICK + BN_HISS_RLEAK);
    const double req  = rpar + BN_HISS_RSTOP;
    return sqrt(4.0 * BN_HISS_K * BN_HISS_T * req * (fs * 0.5));
}

/* xorshift32: cheap, period 2^32-1, flat spectrum. UNIFORM rather than
 * Gaussian on purpose: thermal noise IS Gaussian, but at 5 uV against a
 * circuit that clips in volts the regime is perfectly linear, so only the
 * SPECTRUM matters —flat in both cases— and not the amplitude distribution.
 * A uniform one costs half. */
static double
bn_hiss_sample(uint32_t* st)
{
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    /* uniform in [-1,1); its rms is 1/sqrt(3), so it is scaled by sqrt(3)
     * here for the final rms to be the requested one */
    return ((double)x * (2.0 / 4294967296.0) - 1.0) * 1.7320508075688772;
}


/* ============================================================================
 * MAINS HUM. Everything here is MEASURED, nothing chosen by eye.
 *
 * WHY THIS WAY, and not by injecting into the preamp (the obvious first
 * choice): the ripple does NOT reach the preamp. Propagated through the
 * schematic's filter chain (choke 3H/105 + C22, RDRP 20k + C21c, R29 10k +
 * C21b, R8 10k + C21a) the 41 Vpp of the plate rail end up as NANOVOLTS on
 * V1's rail. The three cascaded RCs are there precisely for that.
 *
 * WHERE IT REALLY ENTERS: the plate ripple is COMMON to both halves of the
 * push-pull and cancels in the OT's primary — but only as well as the
 * balance, and the imbalance is modelled: eps = -0.07, measured.
 *
 * THE THREE PIECES, none of them invented:
 *   · ripple amplitude: MEASURED on the PSU model (`make psu`), 15.7 Vpp at
 *     0.213 A and 41.1 Vpp at 0.633 A -> line Vpp = 2.81 + 60.5*I
 *   · push-pull imbalance: eps = 0.07 (the pair's fit)
 *   · OT ratio: 14.588 plate-to-plate : 8 ohm, from the REAL WINDING of the
 *     100 W head's output transformer (620+620 : 85 turns).
 *
 * THE WAVEFORM is not a sine either: a bridge with a capacitor gives a
 * sawtooth. Harmonics MEASURED on the model itself (FFT of the steady rail):
 *   100 Hz 1.0000 · 200 0.3820 · 300 0.1305 · 400 0.0220 · 500 0.0167 · 600 0.0147
 * and the fundamental is 0.3745 of the total Vpp.
 *
 * ⚠️ WHAT THIS IS NOT: the reduced power model is scaled by the INVERTER's
 * rail (srail), so it cannot SEE the plate rail's ripple. This is injected,
 * derived analytically, not emerging from the circuit — so it does NOT
 * intermodulate with the signal as it really would. Declared, not hidden.
 *
 * ⚠️ And the HEATER hum (50 Hz through cathode-heater leakage), the other
 * half of a real amp's hum, is not modelled: it is not on the schematic and
 * would require inventing a parameter.
 *
 * WHERE it is added: AFTER the silence detector, on purpose. If it came in
 * before, the output would never stay still and the silence freeze —which
 * takes the CPU cost to zero— would never engage.
 * ========================================================================== */
/* M_PI does not exist under -std=c11 without _GNU_SOURCE; same convention as
 * BN_SQRT2. */
/* THE MAINS FREQUENCY IS NOT A KNOB: it is a property of the REGION, and this
 * plugin already has a region axis — the `us_spec` port (US edition, 6550).
 * So the frequency goes WITH it, and no extra control is added:
 *      us_spec = 0  ->  UK / Europe   50 Hz mains  ->  100 Hz ripple
 *      us_spec = 1  ->  USA / Canada  60 Hz mains  ->  120 Hz ripple
 * (the ripple is at TWICE the mains frequency because the 100 W head has a
 * full bridge; the BIAS chain's would be at the mains frequency, half wave —
 * but that one was ruled out by measurement.)
 *
 * ⭐ AND IT IS NOT ONLY THE PITCH. With faster mains the reservoir discharges
 * for less time between peaks, so the ripple is SMALLER. MEASURED on the PSU
 * model:
 *      50 Hz  16.3 Vpp idle / 42.5 at full load
 *      60 Hz  13.6 Vpp      / 35.6
 *      ratio 0.832 idle and 0.838 under load
 * and reservoir theory gives 1/f = 50/60 = 0.833. They agree to the third
 * digit, so the factor is NOT chosen: it is derived.
 * ⇒ a US-mains 100 W head hums 1.6 dB LESS and at a higher pitch. */
#define BN_HUM_F0_UK    100.0      /* 2 x 50 Hz */
#define BN_HUM_F0_US    120.0      /* 2 x 60 Hz */
#define BN_HUM_US_K     (50.0/60.0)  /* 1/f ripple — measured 0.832, theory 0.833 */
#define BN_HUM_EPS      0.07       /* push-pull imbalance, measured */
/* ⭐ Plate-to-plate : 8 ohm ratio. The 14,588 is from the REAL winding of the
 * 100 W head's OT (Zaa 1708 -> sqrt(1708/8) = 14,61). The 50 W head has ~TWICE
 * the Zaa (3,2 k; the netlist models it with Laa 14,1 H against 7,5)
 * ⇒ N = sqrt(3200/8) = 20,00, +37 %. Scaling the 100 W figure gives the same:
 * 14,588 x sqrt(3200/1702,5) = 20,00.
 * It divides the hum, so a wrong OT gives the wrong hum LEVEL on the 50 W
 * head. */
#define BN_HUM_OT_N     14.588     /* plate-to-plate : 8 ohm, 100 W head's OT winding */
#define BN_HUM_OT_N_50W 20.000     /* 50 W head: double Zaa */
#define BN_HUM_FUND     0.3745     /* fundamental peak / total Vpp */
static const double BN_HUM_HARM[6] = { 1.0, 0.3820, 0.1305, 0.0220, 0.0167, 0.0147 };

/* Ripple Vpp for a given total current (a line through the two measured
 * points). It saturates from below at the idle value: the amp at rest ALREADY
 * draws 0.213 A, not zero. */
static double
bn_hum_ripple_vpp(double i_total, int w50)
{
    /* Each head has its own line: the 50 W head's supply is not the 100 W's
     * (different secondary, RWIND and reservoir). Measured with the SAME deck
     * (psu_checks.py --head 50w), the 50 W head ripples 7,3 Vpp idle and 26,4
     * flat out, where the 100 W line would give 9,4 and 22,5 — i.e. +2,2 dB of
     * hum idle and −1,4 flat out. A line through each head's two points, with
     * the current on the plugin's scale (I_REST/I_FULL, and ×K_I_50W on the
     * 50 W head: 0,1096 / 0,3257 A). The minimum is the idle value.
     * ⭐ And the SHAPE (harmonics) has no head axis: measured in the four
     * head×era cases it agrees within 2 % at equal load — it depends on the
     * conduction angle, i.e. on the LOAD (H2/H1 0,40 idle, 0,32 flat out), and
     * the single shipped table sits between the two. The old way's ×2 also
     * holds on the 50 W head (×2,01 idle, ×1,98 flat out). */
    if (w50) {
        const double v = -2.39 + 88.4 * i_total;
        return v < 7.3 ? 7.3 : v;
    }
    const double v = 2.81 + 60.5 * i_total;
    return v < 2.81 ? 2.81 : v;
}

/* One hum sample, in the same units as buf_out (i.e. volts at the 8 ohm tap
 * divided by BN_POWER_GAIN). `ph` advances in radians of the fundamental and
 * the caller wraps it. */
static double
bn_hum_sample(double vpp, double ph, int w50)
{
    double s = 0.0;
    for (int k = 0; k < 6; ++k) {
        s += BN_HUM_HARM[k] * sin((double)(k + 1) * ph);
    }
    /* vpp -> fundamental peak -> differential that does NOT cancel -> secondary */
    /* The OT is the HEAD's own. The 100 W one on the 50 W head would give
     * +37 % too much hum, because it divides. */
    const double otn = w50 ? BN_HUM_OT_N_50W : BN_HUM_OT_N;
    return s * vpp * BN_HUM_FUND * BN_HUM_EPS / otn / BN_POWER_GAIN;
}


static void
run(LV2_Handle instance, uint32_t n_samples)
{
    /* FTZ+DAZ while the DSP runs: without -ffast-math (parity), the DK/FIR
     * states can fall into denormals in silent tails and a denormal costs
     * ~50x — measured: peaks of 258%% of the budget in silence.
     * The MXCSR is restored on exit (the host owns its own mode). */
    BritishNoise* self = (BritishNoise*)instance;
    const float* const in  = self->in;
    float* const       out = self->out;
    /* The pointer check goes BEFORE touching the MXCSR: in the other order
     * this early return would skip the restore and leave FTZ+DAZ SET on the
     * host's audio thread — measured, 0x1fa0 -> 0x9fe0. From then on every
     * plugin in the chain silently flushes denormals: a bug that shows up in
     * ANOTHER plugin and is hellish to track down. */
    if (!in || !out) {
        return;
    }
#ifdef BN_HAVE_MXCSR
    const unsigned int bn_csr_save = _mm_getcsr();
    _mm_setcsr(bn_csr_save | 0x8040u);   /* FTZ (bit 15) + DAZ (bit 6) */
#endif

    /* The oversampling factor can change: it switches outside the sample loop
     * and only when it really changes. */
    if (self->os_phase == 2) {
        /* the ramp reached zero on the previous block: switch HERE, outside
         * the sample loop and with the output silent */
        rebuild_oversampler(self, self->os_want);
        /* The incoming configuration carries the state it had when it was left
         * (or its primed rest, the first time), so its first ~50 ms depart from a configuration that ran all along:
         * measured 15-50 ms after the switch -20,5 dB on first use, -36 dB on reuse. The output stays muted while it
         * settles, then fades in as before. */
        self->os_hold = (int)(BN_OS_SETTLE_S * self->sample_rate);
        self->os_phase = self->os_hold > 0 ? 4 : 3;
    } else if (self->oversample) {
        /* One legal factor: 2x (not on the panel). Whatever the port carries -- a session saved at 4x, 1, 8,
         * NaN, Inf -- lands on 2 without being parsed. ⛔ No table to index (a table lookup can read past its
         * end when the table shrinks). */
        const int want = 2;
        if (want != self->os_factor && self->os_phase == 0) {
            if (self->os_primero) {
                rebuild_oversampler(self, want);   /* start-up: no ramp */
            } else {
                self->os_want  = want;
                self->os_phase = 1;      /* start ramping down; do NOT switch yet */
            }
        }
    }
    /* Cleared HERE, where it is consumed, and not at the end of run(): there
     * are two early `return`s (unconnected ports, and POWER off with the ramp
     * finished) that would skip it and leave the first real change without a
     * ramp. */
    self->os_primero = 0;
    /* The dry delay line is aligned with what we declare: if the oversampling
     * factor changes, the latency changes and must be followed. */
    {
        int lat = bn_os_latency(&self->c->os)
            + bn_pw_lat(self->c, self->os_factor)
              / (self->os_factor > 0 ? self->os_factor : 1)
            ;
        if (lat < 0) { lat = 0; }
        if (lat > BN_DRY_RING - 1) { lat = BN_DRY_RING - 1; }
        self->dry_delay = lat;
    }
    if (self->latency) {
        *self->latency = (float)bn_os_latency(&self->c->os)
            /* the power stage's latency (bn_pw_lat), in base-rate samples */
            + (float)bn_pw_lat(self->c, self->os_factor)
              / (float)(self->os_factor > 0 ? self->os_factor : 1)
            ;
    }

    /* Panel knobs: the TARGET of this cycle; the actual application is slewed
     * inside the chunk loop (no zipper — a full-scale automation jump is
     * spread over BN_KNOB_SLEW_S, 240 ms; the Woodbury update costs 0,02 ms per
     * move — measured on a low-power 4-core x86 CPU with the dense transformation inside,
     * `bn_test_var` — so re-applying per chunk while moving is cheap). The
     * state x survives every step; the chord's LU does not (bn_dkvar_set
     * invalidates it). */
    float kn_tgt[5];
    kn_tgt[0] = self->vol    ? *self->vol    : 0.7f;
    kn_tgt[1] = self->treble ? *self->treble : 0.5f;
    kn_tgt[2] = self->bass   ? *self->bass   : 0.5f;
    kn_tgt[3] = self->middle ? *self->middle : 0.5f;
    kn_tgt[4] = self->master ? *self->master : 1.0f;
    /* NEGATED comparisons: with the plain form a NaN on a knob survives both
     * ifs, and on the FIRST run() it reaches bn_apply_vars -> dG NaN -> LU
     * NaN -> every matrix A..N NaN -> SIGSEGV. Verified. With the negated
     * form a NaN falls to the knob's default (KN_DEF). */
    static const float KN_DEF[5] = { 0.7f, 0.5f, 0.5f, 0.5f, 1.0f };
    for (int k = 0; k < 5; ++k) {
        if (!(kn_tgt[k] >= 0.0f))      { kn_tgt[k] = KN_DEF[k]; }
        else if (!(kn_tgt[k] <= 1.0f)) { kn_tgt[k] = 1.0f; }
    }
    const int low_tgt = (self->low_input && *self->low_input > 0.5f) ? 1 : 0;
    const int us_tgt  = (self->us_spec && *self->us_spec > 0.5f) ? 1 : 0;
    if (self->knobs_last[0] < 0.0f) {          /* first run: no ramp */
        for (int k = 0; k < 5; ++k) { self->knobs_last[k] = kn_tgt[k]; }
        bn_apply_vars(self, kn_tgt, low_tgt);
    } else if (low_tgt != self->low_state || us_tgt != self->us_state) {
        /* the jack and the US edition switch INSTANTLY (Woodbury, 0,06 ms; a
         * real replug clicks too, and a version change is a different amp) */
        bn_apply_vars(self, self->knobs_last, low_tgt);
    }

    /* --- BYPASS and POWER. They are not the same:
     *   BYPASS  -> pass-through, but the DSP KEEPS RUNNING. The circuit state
     *              stays warm, so coming back gives no start-up transient or
     *              click. It costs the usual CPU, and that is intended.
     *   POWER    -> the whole plugin out. The DSP is skipped and the cost
     *              drops to ZERO, which is what is expected of an "off".
     * In both cases entry and exit use a 10 ms ramp: a hard jump from wet to
     * dry is a step in the signal, and a step is audible. */
    const int power_off = (self->power  && *self->power  < 0.5f);
    const int bypass_on = (self->bypass && *self->bypass > 0.5f);
    const double dry_target = (power_off || bypass_on) ? 1.0 : 0.0;
    if (self->dry_from_ports) { self->dry_mix = dry_target; self->dry_from_ports = 0; }
    /* Continuous controls applied through per-sample ramps, so a step on the port never reaches the audio as a
     * step: the net's conditioning knobs (a step moves the net's operating point at once, a jump at the output
     * even in silence; linear ramps at fs_hi, BN_COND_RAMP_S) and the hum's level (a step cuts the hum waveform
     * mid-cycle; 10 ms at the host rate). Held still, each ramp sits exactly on its port's value. The first
     * run() after activate() starts them on the ports. */
    const double nn_pres_tgt = bn_ctl(self->presence, 0.0, 0.0, 1.0);
    const double nn_bias_tgt = bn_ctl(self->bias, 0.5, 0.0, 1.0);
    const double nn_cab_tgt  = bn_ctl(self->cab_load, 0.0, 0.0, 1.0);
    const double hum_tgt     = bn_port01(self->hum);
    if (self->cond_from_ports) {
        self->nn_pres_now = nn_pres_tgt; self->nn_bias_now = nn_bias_tgt; self->nn_cab_now = nn_cab_tgt;
        self->hum_now = hum_tgt;
        self->cond_from_ports = 0;
    }
    const double dry_step = 1.0 / (0.010 * self->sample_rate);
    /* --- THE w2 SEAM: power bypass and FX loop. Read ONCE per run.
     * Precedence: POWER > whole BYPASS > half bypasses; the top two act at
     * the base rate on whatever comes out of here, so there is nothing to
     * arbitrate inside the loop. */
    const double byp_pow_tgt = (self->bypass_power && *self->bypass_power > 0.5f) ? 1.0 : 0.0;
    const int    fx_on       = (self->fx_loop && *self->fx_loop > 0.5f);
    const double fx_mix_tgt  = fx_on ? bn_ctl(self->fx_mix, 100.0, 0.0, 100.0) * 0.01 : 0.0;
    const double fx_lvl_tgt  = bn_ctl(self->fx_return_lvl, 1.0, 0.0, 2.0);
    float* const fx_send = self->fx_send;

    /* POWER off and the ramp finished: exit here without touching the DSP.
     * NOTE: it comes BEFORE the processing — and it must restore the MXCSR,
     * or FTZ+DAZ would stay set on the host's audio thread. */
    if (power_off && self->dry_mix >= 1.0) {
        for (uint32_t i = 0; i < n_samples; ++i) {
            const double s = (double)in[i];
            out[i] = (float)bn_dry_push(self, isfinite(s) ? s : 0.0);
        }
        if (fx_send) { memset(fx_send, 0, n_samples * sizeof(float)); }
        self->fx_send_gain = 0.0;
        self->fx_running = 0;
#ifdef BN_HAVE_MXCSR
        _mm_setcsr(bn_csr_save);
#endif
        return;
    }

    /* The dB port MUST go through the NaN guard. It is not cosmetic: the
     * smoother is a recurrence `lin += k*(target-lin)`, and NaN is absorbing
     * in it — a single block with NaN on Output would leave the plugin mute
     * FOREVER, until the host calls activate() again. Clamped to the .ttl
     * ranges; a NaN falls to the DEFAULT (0 dB), not to the minimum, because
     * here the minimum (-24 dB) is almost silence and would be
     * indistinguishable from the fault itself. */
    const float trim_db  = bn_clamp_db(self->trim  ? *self->trim  : 0.0f,
                                       -24.0f, 12.0f);
    const float trim_target  = (float)pow(10.0, (double)trim_db  / 20.0);
    const float k = self->smooth_coef;

    uint32_t done = 0;
    int knobs_moving = 0;
    while (done < n_samples) {
        uint32_t n = n_samples - done;
        if (n > BN_MAX_BLOCK) { n = BN_MAX_BLOCK; }
        /* while a pot moves the chunk drops to knob_chunk samples (BN_KNOB_CHUNK_S): the matrices are
         * rebuilt once per chunk, so the chunk is the staircase's step (see BN_KNOB_CHUNK_S) */
        {
            int moving_pre = 0;
            /* `km` and not `k`: out there `k` is the smoother's coefficient
             * (checked with -Wshadow). Three different uses of the same name
             * in the same function is exactly how a bug slips in while
             * editing. */
            for (int km = 0; km < 5; ++km) {
                if (fabsf(kn_tgt[km] - self->knobs_last[km]) > 1e-4f) { moving_pre = 1; }
            }
            if (moving_pre && n > self->knob_chunk) { n = self->knob_chunk; }
        }
        /* a chunk never crosses a supply cell's edge */
        uint32_t cell_len = self->psu_grid;
        int grid = 1;
        if (grid) {
            const uint32_t edge = cell_len - self->psu_phase;
            if (n > edge) { n = edge; }
        } else {
            self->psu_phase = 0;
            cell_len = n;
        }

        /* pot slew: full scale in BN_KNOB_SLEW_S (the only source of that time), matrices rebuilt once per
         * chunk while a pot moves */
        {
            const float step = (float)n / (BN_KNOB_SLEW_S * (float)self->sample_rate);
            int moving = 0;
            for (int km = 0; km < 5; ++km) {
                const float d2 = kn_tgt[km] - self->knobs_last[km];
                if (fabsf(d2) > 1e-4f) {
                    moving = 1;
                    if (d2 >  step)      { self->knobs_last[km] += step; }
                    else if (d2 < -step) { self->knobs_last[km] -= step; }
                    else                 { self->knobs_last[km] = kn_tgt[km]; }
                }
            }
            if (moving) {
                bn_apply_vars(self, self->knobs_last, self->low_state);
            }
            knobs_moving = moving;
        }

        /* --- SILENCE BYPASS --------------------------------------------------
         * MEASURED: without it, with an input of exact zeros the plugin used
         * 26.1% of the budget at 4x — 62% of what it costs while PLAYING. That
         * is the engine solving the circuit to produce nothing, on every
         * stopped track of the session.
         *
         * As soon as anything sounds this does not help (a guitar always
         * brings background noise), so it is NOT the answer to "the plugin
         * costs a lot while playing": it makes a track at rest stop costing.
         *
         * Discipline so the audio does not change: seeing zeros at the input
         * is not enough, because the amp's tails keep sounding. It only
         * freezes when (a) the input is EXACT zeros, (b) no knob moves
         * —moving a knob reconfigures the circuit— and (c) the output has not
         * moved more than BN_QUIET_EPS for half a second. By then the state is
         * at its rest point and freezing it is indistinguishable from
         * computing on. Any non-zero sample unfreezes at once, and the
         * internal state is intact because it was never touched. */
        int all_zero = 1;
        for (uint32_t i = 0; i < n; ++i) {
            if (in[done + i] != 0.0f) { all_zero = 0; break; }
        }
        /* with the loop active the return can carry signal even when the
         * input is zero: no freeze */
        if (!all_zero || knobs_moving || fx_on) {
            self->quiet_run = 0;
            self->frozen = 0;
            self->avg_ok = 0;
        }
        if (self->frozen) {
            /* the trim keeps ramping: the user can move Output during silence */
            /* And the HUM keeps going too. Without this the hum would cut out
             * after half a second of silence — i.e. RIGHT when it is most
             * audible, the worst possible failure for this feature. A real amp
             * hums precisely when you are not playing.
             * ⭐ This path is live: the freeze engages on 53,9 % of a silent
             * run at os 2 and os 4, so the hum is really heard through here. */
            const int hum_us_f = (self->us_spec && *self->us_spec > 0.5f);
            const double hum_k_f = hum_us_f ? BN_HUM_US_K : 1.0;
            const double hum_dph_f = 2.0 * BN_PI
                * (hum_us_f ? BN_HUM_F0_US : BN_HUM_F0_UK) / self->sample_rate;
            for (uint32_t i = 0; i < n; ++i) {
                self->trim_lin += k * (trim_target - self->trim_lin);
                double y_f = self->frozen_out;
                bn_ramp_to(&self->hum_now, hum_tgt, dry_step);
                if (self->hum_now > 0.0) {
                    y_f += self->hum_now * bn_hum_sample(self->hum_vpp * hum_k_f, self->hum_ph,
                                               self->head_now);
                    self->hum_ph += hum_dph_f;
                    if (self->hum_ph > 2.0 * BN_PI) { self->hum_ph -= 2.0 * BN_PI; }
                }
                const double wet = y_f * (double)self->trim_lin;
                /* Sanitise BEFORE pushing into the ring. With dry_mix=0.0
                 * exactly it is not enough for the wet path to be clean:
                 * 0.0*(NaN-wet) = NaN in IEEE-754, so a NaN from the bus would
                 * come out of out[] even with the mix at 100% wet, and would
                 * also stay in the ring. Same pattern as the power-off line. */
                const double s = (double)in[done + i];
                const double dry = bn_dry_push(self, isfinite(s) ? s : 0.0);
                bn_dry_ramp(self, dry_target, dry_step);
                out[done + i] = (float)(wet + self->dry_mix * (dry - wet))
                                * bn_os_ramp(self);
            }
            if (fx_send) { memset(fx_send + done, 0, n * sizeof(float)); }
            self->psu_phase += n;                       /* the grid stays aligned to the stream */
            if (self->psu_phase >= cell_len) { self->psu_phase = 0; }
            done += n;
            continue;
        }

        /* Hiss amplitude. Out of the path when the control is at 0 (the
         * default), so the default sound is unchanged. */
        const double hiss_v = bn_port01(self->hiss);
        const double hiss_a = (hiss_v > 0.0) ? hiss_v * bn_hiss_rms(self->sample_rate) : 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            /* SANITISED: a NaN/Inf on the bus —put there by any badly behaved
             * upstream plugin, a corrupt file or a device change— would reach
             * the index of a LUT: (int)NaN = INT_MIN and SIGSEGV on the audio
             * thread, i.e. the whole DAW session down. Verified. One compare
             * per sample at the BASE rate (not oversampled) is unnoticeable. */
            const double sin_ = (double)in[done + i];
            /* the hiss is added at the input: thermal noise is generated in the amp's input network. */
            self->buf_in[i]  = (isfinite(sin_) ? sin_ : 0.0)
                             + hiss_a * bn_hiss_sample(&self->hiss_rng);
        }

        bn_os_up(&self->c->os, self->buf_in, n, self->buf_hi);

        const size_t n_hi = (size_t)n * (size_t)self->os_factor;

        /* FX LOOP: the resampling pair runs while the loop is active or still
         * fading (mix or send with a ramp half way). It is zeroed when the
         * loop engages: the 10 ms fade covers the halfbands' start-up (47 taps
         * at fs_hi, < 1 ms). The return is SANITISED on reading, like the
         * input: a NaN from the bus cannot enter the DK. */
        const int fx_run = fx_on || self->fx_mix_now > 0.0 || self->fx_send_gain > 0.0;
        if (fx_run) {
            if (!self->fx_running) {
                bn_os_reset(&self->c->fxos);
                bn_fxc_reset(&self->c->fxc);
                self->fx_running = 1;
            }
            const float* ret = self->fx_return;
            for (uint32_t i = 0; i < n; ++i) {
                const double r = ret ? (double)ret[done + i] : 0.0;
                self->fx_ret_lo[i] = isfinite(r) ? r : 0.0;
                self->fx_rcap[i] = self->fx_ret_lo[i];
            }
            bn_os_up(&self->c->fxos, self->fx_ret_lo, n, self->fx_ret_hi);
            bn_fxc_begin(&self->c->fxc);
        } else {
            self->fx_running = 0;
        }

        /* --- The supply sag. The power stage's circuit is not solved here,
         * but its CONSUMPTION is modelled: a map measured on the oracle
         * (master level -> mean currents) feeds the PSU at block rate, and the
         * preamp's B+ (u[1]) breathes accordingly. */
        /* PSU CURRENT MAP. Generated with tools/gen_psu_map.py from
         * `bn_bench psumap 192000 full col own`, i.e. with the plugin's own
         * valve models (the oracle's reference model passes ~12% less current). It is only
         * reliable with grid current in the pentode: without it the screen
         * ran away to 0.68 A (~78 W/valve, against an 8 W limit) because
         * nothing held the grid back. See bn_pentode_own.c. */
        /* ⭐ ONE MAP PER HEAD AND PER VALVE FAMILY.
         * Generated with `bn_bench psumap 192000 <variant> vbias=<its own>`.
         *
         * ⚠️ `vbias=` IS NOT OPTIONAL: without it, `psumap` uses the header's
         * default (-40,19 V, the EL34 calibration) for the 6550 too, and then
         * the 6550 draws 137,9 mA/valve against 41,0 — 3,4x over, some 64 W of
         * plate against a limit of 42, with spurious differences of 74 %.
         *
         * Per HEAD: the 50W/100W ratio is NOT constant — 0,488 idle and 0,462
         * flat out. A single scale factor cannot reproduce that; a map can.
         *
         * Per FAMILY: with each valve set at ITS bias (40 mA/valve), UK and US
         * draw the same current at the two ENDS of the travel — 0,719 against
         * 0,720 A flat out on the 100 W head, 0,332 against 0,331 on the 50,
         * and idle too (0,1643 / 0,1644) — but not in between. Measured over
         * the WHOLE grid, each variant at its bias:
         *
         *   w2      100W UK   100W US    50W UK    50W US
         *    0       0.1643    0.1644    0.0800    0.0801   <- identical
         *    2       0.3507    0.3217    0.1948    0.1720   <- 8,3 % and 11,7 %
         *    5       0.6037    0.6214    0.2885    0.2980   <- and the sign FLIPS
         *   30       0.7186    0.7199    0.3318    0.3312   <- identical
         *
         * ⇒ It is not an offset (a factor would fix that): it is a curve of another SHAPE.
         * ⭐ And the SCREEN current separates even more: at w2=2, 0,051 against 0,021 — a
         * factor of 2,4. That is not visible looking only at the plates.
         *
         * In audio terms: the sag is `rchg × (I − I_idle)`, so at low-to-mid level the 6550 amp
         * sags differently from the EL34 one. It is FEEL —how it responds to a light touch—, not
         * timbre. Small, but not zero.
         *
         * ⭐ CONTROL: the four maps are generated by the same tools/gen_psu_map.py pipeline, and
         * the two UK maps match the UK-only generation exactly ⇒ the UK path stays BIT-IDENTICAL. */
        static const double MAP_W2[7]  = { 0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0 };
        static const double MAP_IPL_100_UK[7] = { 0.164, 0.351, 0.604, 0.690,
                                                  0.709, 0.715, 0.719 };
        static const double MAP_ISC_100_UK[7] = { 0.022, 0.051, 0.118, 0.141,
                                                  0.146, 0.148, 0.150 };
        static const double MAP_IPL_100_US[7] = { 0.164, 0.322, 0.621, 0.702,
                                                  0.714, 0.718, 0.720 };
        static const double MAP_ISC_100_US[7] = { 0.009, 0.021, 0.099, 0.123,
                                                  0.126, 0.128, 0.129 };
        static const double MAP_IPL_50_UK[7]  = { 0.080, 0.195, 0.288, 0.322,
                                                  0.328, 0.330, 0.332 };
        static const double MAP_ISC_50_UK[7]  = { 0.011, 0.031, 0.058, 0.067,
                                                  0.069, 0.070, 0.070 };
        static const double MAP_IPL_50_US[7]  = { 0.080, 0.172, 0.298, 0.325,
                                                  0.329, 0.330, 0.331 };
        static const double MAP_ISC_50_US[7]  = { 0.004, 0.013, 0.049, 0.057,
                                                  0.058, 0.059, 0.059 };
        /* same expression as us_q below; it is not in scope here yet */
        const int map_us = (self->us_spec && *self->us_spec > 0.5f) ? 1 : 0;
        const double* MAP_IPL = self->head_now ? (map_us ? MAP_IPL_50_US  : MAP_IPL_50_UK)
                                               : (map_us ? MAP_IPL_100_US : MAP_IPL_100_UK);
        const double* MAP_ISC = self->head_now ? (map_us ? MAP_ISC_50_US  : MAP_ISC_50_UK)
                                               : (map_us ? MAP_ISC_100_US : MAP_ISC_100_UK);
        /* Why the map carries the head axis:
         *  · asking the 50 W head for the 100 W head's current gives a sag of
         *    118,9 V where ~53,6 is right — more than DOUBLE. And it looks like
         *    "the expected result" for the smaller head, which is exactly how a
         *    flattering figure slips in.
         *  · an axis whose effect lives in a table that does NOT carry that axis
         *    simply does not exist. That is why the map has both dimensions and
         *    not a scale factor: the 50W/100W ratio is NOT constant (0,488 idle,
         *    0,462 flat out). */
        /* the supply ticks once per CELL (see psu_grid in BritishNoise), at its first chunk */
        const int tick = (self->psu_phase == 0);
        const size_t phase_hi = (size_t)self->psu_phase * (size_t)(self->os_factor > 0 ? self->os_factor : 1);
        if (tick) {
        double ipl = MAP_IPL[6], isc = MAP_ISC[6];
        {
            /* The map is read with the FAST envelope. The valves' current
             * follows the drive within a cycle: measured on the exact amp, after an abrupt
             * cut the total current is back at rest within 12 ms (then 25 ms slightly
             * below it, the grid-blocking choke, and a 40 ms rebound). Fed through this
             * same bn_psu, those currents put the PI rail's minimum 28 ms after the cut;
             * the 80 ms envelope kept the map's current up and put it at 200 ms, 1,4 V
             * deeper (rms error 2,37 V over the 500 ms after the cut; with 5 ms, 0,34 V;
             * the block peak with no hold, 0,28 V). */
            const double w = self->w2_peak_fast;
            for (int k2 = 1; k2 < 7; ++k2) {
                if (w <= MAP_W2[k2]) {
                    const double f2 = (w - MAP_W2[k2 - 1])
                                      / (MAP_W2[k2] - MAP_W2[k2 - 1]);
                    ipl = MAP_IPL[k2 - 1] + f2 * (MAP_IPL[k2] - MAP_IPL[k2 - 1]);
                    isc = MAP_ISC[k2 - 1] + f2 * (MAP_ISC[k2] - MAP_ISC[k2 - 1]);
                    break;
                }
            }
        }
        const double vx_prev = self->vx_now;
        /* DC draws, from the FACTORY TABLE of the 100 W master-volume head
         * (issue 1, 1981): inverter 2.5 mA from ITS rail · V2 (V2a 1.22 +
         * follower 1.67) 2.9 mA from X · V1a+V1b 0.91 mA from XP. */
        bn_psu_step(&self->psu, (double)cell_len / self->sample_rate,
                    ipl, isc, 0.0025, 0.0029, 0.00091);
        bn_psu_step(&self->psu_idle, (double)cell_len / self->sample_rate,
                    MAP_IPL[0], MAP_ISC[0], 0.0025, 0.0029, 0.00091);
        /* The ripple implied by THAT current. Kept per block; the hum is
         * added below, after the silence detector. */
        /* The era changes the reservoir capacitor. It is only re-applied if
         * the control has moved — it costs nothing per block. */
        {
            const int era_q = (self->era && *self->era > 0.5f);
            if (era_q != self->era_now) {
                bn_psu_set_era(&self->psu, era_q);
                bn_psu_set_era(&self->psu_idle, era_q);
                self->era_now = era_q;
            }
        }
        self->psu_i_total = ipl + isc + 0.0025 + 0.0029 + 0.00091;
        /* With half the reservoir the RIPPLE doubles — measured 16.3 -> 34.5
         * Vpp idle and 42.5 -> 84.5 at full load, i.e. x2.0. The averaged PSU
         * model does not represent the ripple (it is out of its scope on
         * purpose), so the factor is applied here. ⇒ the "old way" hums 6 dB
         * MORE, at no cost. */
        self->hum_vpp = bn_hum_ripple_vpp(self->psu_i_total, self->head_now)
                      * (self->era_now ? 2.0 : 1.0);
        /* The neural power stage applies the PREAMP's sag itself: it was trained on w2 from the preamp at
         * rest supply (the bench's `preamp`) against the full amp under the rail, whose preamp sags through
         * the netlist's droppers — so the net learned that sag from its rail input. Fed a w2 that had
         * already sagged here it applied it twice: 4,5-10 dB of worst-frame ANMR on the product gate,
         * growing with level. So node X is held at REST: the same supply stepped with the idle currents
         * (psu_idle). The loaded supply still runs: its rectifier current and plates make the net's rail
         * (rq below) and its current the hum's ripple. The preamp-only build holds it the same way, so its
         * output is the w2 the product's power stage, FX send and power bypass see.
         * An arm trained on w2 WITH the exact node X (`xlive` in bn_nn_weights.h, see
         * gen_nn_blob.py) gets the loaded supply's X instead: its net never learned the preamp's sag. */
        self->vx_now = bn_nn_arms[self->nn_arm].xlive ? self->psu.vx : self->psu_idle.vx;
        const size_t cell_hi = (size_t)cell_len * (size_t)(self->os_factor > 0 ? self->os_factor : 1);
        self->vx_cell0 = vx_prev;
        self->dvx_hi = (self->vx_now - vx_prev) / (double)(cell_hi ? cell_hi : 1);
        {
            const double rsag = self->head_now ? 48.6 : 25.0;   /* the OLD supply's RSAG: what the shipped nets were trained with; power_*.cir carry the research values of an alternative mains transformer (33.0 / 64.2) */
            const double ich  = self->psu.vsrc > self->psu.vc23 ? (self->psu.vsrc - self->psu.vc23) / self->psu.rchg : 0.0;
            const double ichi = self->psu_idle.vsrc > self->psu_idle.vc23 ? (self->psu_idle.vsrc - self->psu_idle.vc23) / self->psu_idle.rchg : 0.0;
            const double rq_prev = self->rq_now;
            self->rq_now = (self->psu.vc23 + rsag * ich) / (self->psu_idle.vc23 + rsag * ichi);
            self->rq_cell0 = rq_prev;
            self->drq_hi = (self->rq_now - rq_prev) / (double)(cell_hi ? cell_hi : 1);
        }
        }   /* tick */

        /* The 100 W / 50 W head. The supply follows the head too (switching only the power stage would make
         * the 50 W sag like a 100 W — and the sag is what tells the two apart when playing). */
        {
            const int head_q = (self->head && *self->head > 0.5f);
            if (head_q != self->head_now) {
                bn_psu_set_head(&self->psu, head_q);
                bn_psu_set_head(&self->psu_idle, head_q);
                self->head_now = head_q;
            }
        }
        /* Which net. The slot follows head and US spec; when it changes, the sample loop fades the
         * live net out, switches at silence and fades the new one in (see nn_gain). */
        self->nn_arm_want = (self->head_now ? 1 : 0) + 2 * ((self->us_spec && *self->us_spec > 0.5f) ? 1 : 0);
        /* PREAMP bypass — crossfade w2 <-> input, 10 ms ramp at fs_hi.
         * The preamp KEEPS running (its state stays warm: no click on return). */
        const double byp_pre_tgt = (self->bypass_preamp && *self->bypass_preamp > 0.5f) ? 1.0 : 0.0;
        const double byp_pre_step = 1.0 / (0.010 * self->sample_rate
                                    * (double)(self->os_factor > 0 ? self->os_factor : 1));
        const double seam_step = byp_pre_step;   /* same 10 ms ramp at fs_hi */
        const double cond_step = 1.0 / (BN_COND_RAMP_S * self->sample_rate
                                        * (double)(self->os_factor > 0 ? self->os_factor : 1));
        /* the power stage's latency (the net's converter delay, rounded up to a host sample: bn_pw_lat): the
         * power bypass's dry path is delayed by it, the net's output by what its converter does not already
         * carry */
        const int pb_delay = bn_pw_lat(self->c, self->os_factor);
        int nn_delay  = pb_delay - self->c->nnr.delay;
        if (nn_delay < 0) { nn_delay = 0; }   /* belt for the _Static_assert above: a negative
                                               * delay would read past the ring, and the fold at
                                               * the read site only covers the low side */
        /* Zero-initialised on purpose: both components are assigned on entering the loop (u[0]
         * from buf_hi, u[1] with `phase_hi + i + 1`). An initialiser like
         * `vx_cell0 + dvx_hi * phase_hi` would be computed and never used, and would look so much
         * like the right one that whoever reordered the loop could take it for the correct
         * initialiser — the difference (`+ i + 1`) is exactly the rail cell's index. */
        double u[2] = { 0.0, 0.0 };
        double y[16];
        /* The fast envelope is a PER-SAMPLE peak follower (decay 5 ms at fs_hi), so its DECAY
         * does not depend on the host's block size. The supply reads its maximum over a fixed
         * CELL of the stream, never over the host's block — the block's maximum of a 5 ms
         * follower falls below the note's peak when the block is shorter than a period, and a
         * supply ticking per block would hold its current and ramp its rail at the host's rate.
         * Both would make a sustained note's sag depend on the buffer. */
        const double env_k = exp(-1.0 / (0.005 * self->sample_rate
                                          * (double)(self->os_factor > 0 ? self->os_factor : 1)));
        double env_fast = self->env_fast, pk_fast = 0.0;
        for (size_t i = 0; i < n_hi; ++i) {
            u[0] = self->buf_hi[i];
            u[1] = self->vx_cell0 + self->dvx_hi * (double)(phase_hi + i + 1);
            {   /* the tolerance follows the SOLVER INPUT level per sample (peak follower at fs_hi, 50 ms release,
                 * hysteresis 1e-3 / 3.16e-3). Per sample, not per host block, so the output cannot depend on the block size. */
                const double a = u[0] < 0.0 ? -u[0] : u[0];
                self->tol_env = a > self->tol_env ? a : self->tol_env * self->tol_env_decay;
                if (self->tol_env < 1.0e-3) { self->tol_fine_on = 1; } else if (self->tol_env > 3.16e-3) { self->tol_fine_on = 0; }
                self->c->dk.tol = self->tol_fine_on ? self->tol_fine : self->tol_norm;
            }
            bn_dk_process(&self->c->dk, u, y);
            if (self->byp_pre_mix < byp_pre_tgt) {
                self->byp_pre_mix += byp_pre_step;
                if (self->byp_pre_mix > byp_pre_tgt) { self->byp_pre_mix = byp_pre_tgt; }
            } else if (self->byp_pre_mix > byp_pre_tgt) {
                self->byp_pre_mix -= byp_pre_step;
                if (self->byp_pre_mix < byp_pre_tgt) { self->byp_pre_mix = byp_pre_tgt; }
            }
            /* w2 as used by the power stage: preamp (y[0]) faded towards the input (u[0]).
             * mix=0 ⇒ w2v = y[0] EXACTLY (bit-identical with the bypass off).
             * The input enters the seam with the SAME level convention as
             * the FX loop return (x BN_STAGE_GAIN: 1,0 from the host = 60 V
             * at the wiper): with the preamp AND the power stage bypassed
             * the output is the input, like the whole-plugin bypass, and with
             * the same declared latency. */
            const double w2v = y[0] + self->byp_pre_mix * (u[0] * BN_STAGE_GAIN - y[0]);
            /* --- THE w2 SEAM -----------------------------------------------
             * w2ac = DC-free w2 (the seam's own blocker): it is what GOES OUT
             * through the send and the power bypass's dry path, as in the
             * preamp-only build. What ENTERS the power stage with the loop in
             * series is the return plus w2's CONSTANT rest value (w2_dc,
             * ~1e-9 V: the wiper hangs after the coupling capacitor), so that
             * its input HPs, primed at that rest value, do not see a step.
             * ⛔ "What the blocker removes" is NOT let through: measured, a
             * pole at 7,6 Hz passes the signal 24 dB down at 100 Hz, and in
             * series with a zero return the output still ran at rms 0,19
             * against 0,71 — the loop was not an insert loop.
             * ⚠️ The mixes are written as `x + m*(a - b)`: with m = 0 exactly
             * they give `x + 0.0 = x` bit for bit, which is what keeps the
             * shipped path unchanged with the ports at their defaults
             * (verified: byte-identical). */
            const double w2ac = w2v - self->c->seam_dc_x1 + self->c->seam_a * self->c->seam_dc_y1;
            self->c->seam_dc_x1 = w2v;
            self->c->seam_dc_y1 = w2ac;
            double w2p = w2v, w2p_ac = w2ac;
            if (fx_run) {
                bn_ramp_to(&self->fx_mix_now, fx_mix_tgt, seam_step);
                bn_ramp_to(&self->fx_lvl_now, fx_lvl_tgt, 2.0 * seam_step);
                /* send and return are at PREAMP LEVEL (w2 / BN_STAGE_GAIN),
                 * like the output of the preamp-only build: the return is
                 * put back into w2 volts on entry */
                /* the return plus what the host could not carry (bn_fx_comp.h): 0 until the estimator
                 * locks on a near-linear effect, exact at unity once it does. The complement takes the
                 * user's return level like the return itself: it is the part of the RETURN the host dropped,
                 * so a return level of 0 must leave the power stage with nothing of the preamp, complement
                 * included. */
                const double ret_c = self->fx_ret_hi[i] + bn_fxc_sample(&self->c->fxc, i);
                const double ret_g = ret_c * self->fx_lvl_now * BN_STAGE_GAIN;
                w2p    = w2v  + self->fx_mix_now * ((ret_g + self->c->w2_dc) - w2v);
                w2p_ac = w2ac + self->fx_mix_now * (ret_g - w2ac);
                self->fx_send_hi[i] = w2ac * (1.0 / BN_STAGE_GAIN);
            }
            /* the master envelope (PSU map and bias) follows what REALLY
             * drives the power stage: with the loop, the return */
            env_fast *= env_k;
            if (env_fast < 1e-9) { env_fast = 0.0; }          /* no denormal tail in silence */
            if (fabs(w2p) > env_fast) { env_fast = fabs(w2p); }
            if (env_fast > pk_fast) { pk_fast = env_fast; }
            /* The neural power stage, w2 -> t8. The frame y = r·net(w2/r, r, knobs) lives inside
             * bn_nn_lstm_stage; r is the rail at the netlist's htraw equivalent (rq, per fs_hi sample).
             *
             * A variant switch (head or US spec) runs through silence, sample by sample so the result
             * does not depend on the host's blocks: the live net's output fades to 0 over 10 ms, the slot
             * switches there and the new net starts from its rest state; the gain then holds at 0 for
             * the power stage's latency (pb_delay), which is how long the new net's first output takes to
             * come out of its ring and converter, and fades back in over 10 ms. If the ports return to
             * the live slot before the fade-out ends, the gain simply ramps back up. In steady state the
             * gain is exactly 1 and none of this touches the sample. */
            if (self->nn_arm_want != self->nn_arm) {
                bn_ramp_to(&self->nn_gain, 0.0, seam_step);
                if (self->nn_gain <= 0.0) {
                    self->nn_arm = self->nn_arm_want;
                    bn_nn_lstm_rest(&self->nn[self->nn_arm]);
                    bn_nn_rate_reset(&self->c->nnr);
                    self->nn_hold = pb_delay;
                }
            } else if (self->nn_hold > 0) {
                --self->nn_hold;
            } else if (self->nn_gain < 1.0) {
                bn_ramp_to(&self->nn_gain, 1.0, seam_step);
            }
            const double nn_rail = self->rq_cell0 + self->drq_hi * (double)(phase_hi + i + 1);
            bn_ramp_to(&self->nn_pres_now, nn_pres_tgt, cond_step);
            bn_ramp_to(&self->nn_bias_now, nn_bias_tgt, cond_step);
            bn_ramp_to(&self->nn_cab_now,  nn_cab_tgt,  cond_step);
            const float nn_pres = (float)self->nn_pres_now;
            const float nn_bias = (float)self->nn_bias_now;
            const float nn_cab  = (float)self->nn_cab_now;
            const double w2n = bn_nn_guard(w2p, bn_nn_arms[self->nn_arm].xlive);   /* the net never sees more than it was trained on */
            const double xn_now = self->c->nnr.active
                ? (double)bn_nn_rate_tick(&self->c->nnr, &self->nn[self->nn_arm], (float)w2n,
                                    (float)nn_rail, nn_pres, nn_bias, nn_cab,
                                    (float)(self->nn_arm & 1), (float)((self->nn_arm >> 1) & 1))
                : (double)bn_nn_lstm_stage(&self->nn[self->nn_arm], (float)w2n,
                                    (float)nn_rail, nn_pres, nn_bias, nn_cab,
                                    (float)(self->nn_arm & 1), (float)((self->nn_arm >> 1) & 1));
            /* the net's output delayed to the power stage's latency (the converter carries its own part) */
            self->c->nn_ring[self->c->nn_pos] = xn_now;
            int rn = self->c->nn_pos - nn_delay;
            if (rn < 0) { rn += BN_PB_RING; }
            self->c->nn_pos = (self->c->nn_pos + 1) & (BN_PB_RING - 1);
            const double x0 = self->nn_gain * self->c->nn_ring[rn];
            const double y0 = x0 - self->c->dc_x1 + self->c->dc_a * self->c->dc_y1;
            self->c->dc_x1 = x0;
            self->c->dc_y1 = y0;
            const double wet_hi = y0 * (1.0 / BN_POWER_GAIN);
            /* POWER bypass. The power stage KEEPS running (warm state, no
             * click on return); at the output t8 is faded with w2 delayed by
             * the power stage's latency and at preamp level. The declared
             * latency does not change: the delay is inside the dry path. */
            self->c->pb_ring[self->c->pb_pos] = w2p_ac;
            {
                int r = self->c->pb_pos - pb_delay;
                if (r < 0) { r += BN_PB_RING; }
                self->c->pb_pos = (self->c->pb_pos + 1) & (BN_PB_RING - 1);
                const double dry_hi = self->c->pb_ring[r] * (1.0 / BN_STAGE_GAIN);
                bn_ramp_to(&self->byp_pow_mix, byp_pow_tgt, seam_step);
                self->buf_hi[i] = wet_hi + self->byp_pow_mix * (dry_hi - wet_hi);
            }
        }
        /* FX LOOP: the send goes down to the host rate and out with its fade */
        if (fx_run) {
            bn_os_down(&self->c->fxos, self->fx_send_hi, n, self->fx_ret_lo);
            bn_fxc_end(&self->c->fxc, self->fx_send_hi, self->fx_ret_lo, self->fx_rcap, n);
        }

        /* The master-wiper envelope the PSU's current map and the bias' grid-current
         * map read — the per-sample follower above (release 5 ms: shorter than a low-E
         * period, and the PI rail's dropper RC, 2 s, hides any ripple at the note's rate). */
        self->env_fast = env_fast;
        /* the block's MAXIMUM of the follower, not its value at the block's end: on a
         * sustained tone that is the tone's peak whatever the block. ⚠️ A per-block hold
         * (80 ms release) is NOT: it decays whenever the new block's peak is a hair below
         * the held one, so on a steady 220 Hz sine in 4096-sample blocks at 48 kHz it reads
         * 8,8 / 3,1 / 8,8 / 3,0 V on alternate blocks (exp(-85/80) = 0,34 per decay) and
         * the map's current flaps 0,67 / 0,45 A with it. */
        /* the cell's maximum; the supply reads it when the cell closes */
        if (pk_fast > self->pk_cell) { self->pk_cell = pk_fast; }
        self->psu_phase += n;
        if (self->psu_phase >= cell_len) {
            self->w2_peak_fast = self->pk_cell;
            self->pk_cell = 0.0;
            self->psu_phase = 0;
        }

        bn_os_down(&self->c->os, self->buf_hi, n, self->buf_out);

        /* has the output gone still? (see the silence bypass above) */
        if (all_zero) {
            double dmax = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const double d = fabs(self->buf_out[i] - self->quiet_last);
                if (d > dmax) { dmax = d; }
                const double m = fabs(self->buf_out[i]);
                /* The whole condition is written out; it is bit-identical to
                 * `if (m > dmax) { dmax = m > BN_QUIET_LEVEL ? 1e300 : dmax; }`.
                 * ⛔ Do NOT simplify it to a bare `if (m > BN_QUIET_LEVEL)`: that is NOT
                 * equivalent — with dmax already above BN_QUIET_LEVEL (the `d` line above can
                 * leave it there) this form does not freeze and the bare one does. */
                if (m > dmax && m > bn_quiet_level(self)) { dmax = 1e300; }
            }
            /* and the circuit state? (see BN_QUIET_XEPS) */
            {
                const int nx = self->c->dk.n_states < 64 ? self->c->dk.n_states : 64;
                double dxmax = 0.0, xmax = 1.0;
                for (int q = 0; q < nx; ++q) {
                    const double xq = self->c->dk.x[q];
                    const double d2 = fabs(xq - self->x_snap[q]);
                    if (d2 > dxmax) { dxmax = d2; }
                    if (fabs(xq) > xmax) { xmax = fabs(xq); }
                    self->x_snap[q] = xq;
                }
                if (dxmax > bn_quiet_xeps(self) * xmax) { dmax = 1e300; }
            }
            if (dmax < bn_quiet_eps(self)) {
                { const int nxq = self->c->dk.n_states < 64 ? self->c->dk.n_states : 64;
                  const int nvq = self->c->dk.n_nl < 16 ? self->c->dk.n_nl : 16;
                  const double w = self->avg_ok ? 0.10 : 1.0;   /* ~10 blocks; measured: more buys nothing more */
                  for (int q = 0; q < nxq; ++q) { self->x_avg[q] += w * (self->c->dk.x[q] - self->x_avg[q]); }
                  for (int q = 0; q < nvq; ++q) { self->v_avg[q] += w * (self->c->dk.v[q] - self->v_avg[q]); }
                  self->avg_ok = 1; }
                self->quiet_run += (int)n;
                if (self->quiet_run >= (int)(0.5 * self->sample_rate)) {
                    self->frozen = 1;
                    if (self->avg_ok) {      /* freeze on the AVERAGE, not on the last sample */
                        const int nxq = self->c->dk.n_states < 64 ? self->c->dk.n_states : 64;
                        const int nvq = self->c->dk.n_nl < 16 ? self->c->dk.n_nl : 16;
                        for (int q = 0; q < nxq; ++q) { self->c->dk.x[q] = self->x_avg[q]; }
                        for (int q = 0; q < nvq; ++q) { self->c->dk.v[q] = self->v_avg[q]; }
                    }
                    { double acc = 0.0; const uint32_t m0 = n > 64 ? n - 64 : 0; uint32_t cnt = 0;
                      for (uint32_t q = m0; q < n; ++q) { acc += self->buf_out[q]; ++cnt; }
                      self->frozen_out = cnt ? acc / (double)cnt : self->buf_out[n - 1]; }
                }
            } else {
                self->quiet_run = 0;
            }
        }
        self->quiet_last = self->buf_out[n - 1];

        /* Mains hum. Out of the path when the control is at 0, the default —
         * so the default sound does NOT change. */
        /* the mains follows the edition: UK 50 Hz, USA 60 (see BN_HUM_F0_UK) */
        const int hum_us = (self->us_spec && *self->us_spec > 0.5f);
        const double hum_f0 = hum_us ? BN_HUM_F0_US : BN_HUM_F0_UK;
        const double hum_k  = hum_us ? BN_HUM_US_K  : 1.0;
        const double hum_dph = 2.0 * BN_PI * hum_f0 / self->sample_rate;
        for (uint32_t i = 0; i < n; ++i) {
            self->trim_lin += k * (trim_target - self->trim_lin);
            double y_amp = self->buf_out[i];
            bn_ramp_to(&self->hum_now, hum_tgt, dry_step);
            if (self->hum_now > 0.0) {
                y_amp += self->hum_now * bn_hum_sample(self->hum_vpp * hum_k, self->hum_ph,
                                           self->head_now);
                self->hum_ph += hum_dph;
                if (self->hum_ph > 2.0 * BN_PI) { self->hum_ph -= 2.0 * BN_PI; }
            }
            const double wet = y_amp * (double)self->trim_lin;
            /* see the note in the frozen loop — 0.0*NaN = NaN, so the dry
             * path must sanitise even when dry_mix is 0. */
            const double s = (double)in[done + i];
            const double dry = bn_dry_push(self, isfinite(s) ? s : 0.0);
            bn_dry_ramp(self, dry_target, dry_step);
            out[done + i] = (float)(wet + self->dry_mix * (dry - wet))
                            * bn_os_ramp(self);
        }
        if (fx_send) {
            const double send_tgt = fx_on ? 1.0 : 0.0;
            if (fx_run) {
                for (uint32_t i = 0; i < n; ++i) {
                    bn_ramp_to(&self->fx_send_gain, send_tgt, dry_step);
                    fx_send[done + i] = (float)(self->fx_ret_lo[i] * self->fx_send_gain);
                }
            } else {
                memset(fx_send + done, 0, n * sizeof(float));
            }
        } else if (fx_run) {
            for (uint32_t i = 0; i < n; ++i) {
                bn_ramp_to(&self->fx_send_gain, fx_on ? 1.0 : 0.0, dry_step);
            }
        }
        done += n;
    }
#ifdef BN_HAVE_MXCSR
    _mm_setcsr(bn_csr_save);
#endif
}

static void
deactivate(LV2_Handle instance)
{
    (void)instance;
}

static void
cleanup(LV2_Handle instance)
{
    /* ALL the configurations, not only the active one, and also those that never reached
     * `ready`: all of that lives in bn_teardown, the ONLY place where teardown happens. */
    bn_teardown((BritishNoise*)instance);
}

static const void*
extension_data(const char* uri)
{
    (void)uri;
    return NULL;
}

static const LV2_Descriptor descriptor = {
    BN_URI,
    instantiate,
    connect_port,
    activate,
    run,
    deactivate,
    cleanup,
    extension_data
};

LV2_SYMBOL_EXPORT const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : NULL;
}
