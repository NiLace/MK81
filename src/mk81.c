/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* mk81.c — LV2 plugin written against the bare API, no frameworks.
 *
 * The whole amp (preamp, phase inverter, power stage, output transformer and
 * speaker load) is one nodal system solved at the oversampled rate (ULTRA,
 * src/ultra/); the supply around it, the w2 seam (bypasses, FX loop), the
 * hum, the hiss and the output trim live here.
 *
 * Audio-path rules: no memory allocation, no system calls, no denormals.
 * Allocation happens in instantiate. The engine's DC solves (a start) run in
 * activate(), and in run() once per switch: a factor change before the
 * oversampled loop, a variant change or the effects card's start at the
 * sample where its fade reaches zero.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <lv2/core/lv2.h>

#include "bn_oversample.h"
#include "bn_psu.h"
#if defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#define BN_HAVE_MXCSR 1
#endif
#include "bn_fx_comp.h"       /* the FX loop's complementary path */

/* Measurement build only (-DBN_CENSUS): reference cycles (rdtsc) spent in each component of the run, printed at
 * unload. Read as ratios: rdtsc counts reference cycles. */
#define CZ_T0(v) ((void)0)
#define CZ_ADD(acc, v) ((void)0)
#include "bn_fx_pair.h"       /* the FX loop's resampler pair */
#include "bn_ultra.h"         /* ULTRA: the whole amp on the nodal engine */
#include "bn_ultra_bias.h"    /* its bias-reservoir law (generated table) */
#include "prepig/bn_prepig.h" /* the send system the preamp bypass runs (preamp + phase inverter) */
#include "bn_card.h"          /* the reissue effects card's state hand-over (the loop ON / OFF) */
#ifndef BN_CARD_XF
#define BN_CARD_XF 0.00125   /* s: the loop ON / OFF crossfade, raised cosine. Short because both engines run during it;
                              * against 5 ms it adds no energy above 6 kHz */
#endif
#include "ret_nodal_api.h"    /* the card's return half on nlcore's nodal solver */
#define CARD_RET_TICK(r, j) cr_ret_tick((r), (j))
#ifdef BN_SEND_TWIN
#ifndef BN_SEND_K
#define BN_SEND_K(fi) 1   /* sub-steps per engine sample: 1, the engine's own rate (the twin's nodal Newton, not
                           * sub-stepping, is what makes its send right) */
#endif
#include "send_nodal_api.h"   /* the card's send half alone, the send TWIN (see the card block in run()) */
#endif



#define BN_URI "https://nylarea.com/plugins/mk81"

/* Level conventions. The preamp saturates with ~60 V peaks at the master wiper (w2): the FX send, the return and
 * the preamp-bypass input use 1.0 = 60 V there. The power stage clips with ~53 V peaks at the 8 ohm tap: the
 * output is divided by it, so the ceiling sits near 0 dBFS and the Output knob is a trim, not a rescue. */
#define BN_STAGE_GAIN 60.0
#define BN_POWER_GAIN 53.0

/* Blocker of the w2 SEAM (FX send and the power bypass's dry path). SUB-HERTZ on purpose: w2 rests at ~0 V (the
 * wiper hangs after the coupling capacitor), so it only guarantees an audio port never carries DC. A 7.6 Hz corner
 * rotated the send enough that a loop through a transparent effect came back only 29 dB below the original; at
 * 0.05 Hz the phase at low E (82 Hz) is 0.035 degrees, a -64 dB residual. */
#define BN_SEAM_HP_HZ 0.05

#define BN_PI           3.14159265358979323846

/* The silence freeze's STATE threshold, as a fraction of BN_QUIET_TOL relative to the state's magnitude (see
 * bn_quiet_xeps and the freeze in run()). Watching the output alone is not enough: with the output already near
 * -140 dBFS the coupling capacitors are still recharging inside. The factor only sets how long the freeze takes to
 * engage (stricter = later); 0.3 keeps a wide margin on the error the freeze introduces, where 1.0 sat at the edge. */
#ifndef BN_QUIET_XEPS_K
#define BN_QUIET_XEPS_K 0.3
#endif

#define BN_MAX_BLOCK 8192

/* The silence freeze watches this many of the engine's capacitor histories: every one of them (the systems carry
 * 67-80); an engine with more never freezes. -DBN_QUIET_NX=64 is the gate's arm (the first 64 only, as before). */
#ifndef BN_QUIET_NX
#define BN_QUIET_NX 96
#endif
/* how close to its idle twin the supply must be before the freeze, as a multiple of BN_QUIET_TOL of each state */
#ifndef BN_QUIET_PSU_K
#define BN_QUIET_PSU_K 1.0
#endif

/* Port indices and symbols are public API (lv2/mk81.ttl, lv2/frozen_symbols.txt): a host saves sessions and
 * automation by them, so new ports are APPENDED, never inserted or renumbered. */
typedef enum {
    /* There is no `drive` port (a virtual pre-gain the real amp does not have): index 2 is the output trim. */
    BN_IN        = 0,
    BN_OUT       = 1,
    BN_TRIM      = 2,   /* output level, in dB                */
    BN_OVERSAMPLE = 3,  /* ignored (the factor is `hq`'s); kept because the symbol is public API */
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
    BN_CAB_LOAD  = 14,  /* Cabinet Feel 0..1 (the back panel): 0 = a resistive 8 ohm load, 1 = the 4x12's
                         * impedance; in between, the speaker's R, L and C scaled (a real circuit at every point) */
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
    BN_ERA       = 19,  /* the 100 W heads' supply: 0 = new way, 1 = old way (half the reservoir) */
    BN_HEAD      = 20,  /* 0 = the 100 W head, 1 = the 50 W head, 2 = the reissue (bn_head_port) */
    BN_BYPASS_PREAMP = 21,  /* 1 = skip the preamp; the input feeds w2 (the master wiper)
                             * directly and the power stage still runs. 0 = default. Crossfaded. */
    /* The FX loop ports and the POWER bypass cut at the same w2 seam as port 21. */
    BN_BYPASS_POWER  = 22,  /* 1 = skip the power stage; the DC-free w2 goes out at preamp level */
    BN_FX_LOOP       = 23,  /* 1 = loop on: the reissue card's loop, in series */
    BN_FX_SEND       = 24,  /* audio OUT: the card's send, at preamp level (/BN_STAGE_GAIN) */
    BN_FX_RETURN     = 25,  /* audio IN: what comes back from the effect, into the card's return stage */
    BN_FX_MIX        = 26,  /* not read: the card's loop is series only; kept so saved sessions load */
    BN_FX_RETURN_LVL = 27,  /* return gain 0..2, applied before the return stage */
    BN_ENGINE        = 28,  /* ignored: one engine; kept so saved sessions load */
    BN_N_PORTS   = 29,  /* the ports of lv2/mk81.ttl before tools/ttl_ultra.py appends `hq` and `fx_level` */
    /* the oversampling factor: 0 = 2x (default), 1 = 4x. Symbol `hq`, the same switch's name on the other plugins */
    BN_HQ            = 29,
    /* the reissue effects card's LEVEL switch: 0 = Line (the hot send, about a third of the master signal; the
     * return x3.1), 1 = Pedal (the guitar-level send, about 1/15; the return x15.5) */
    BN_FX_LEVEL      = 30,
    BN_N_PORTS_ULTRA = 31
} BnPortIndex;


/* Full-scale time of the bias knob's ramp at fs_hi (bias_now). Presence and Cabinet Feel follow the pots'
 * slew instead (BN_KNOB_SLEW_S), since a move of either rebuilds the engine's matrices. */
#ifndef BN_COND_RAMP_S
#define BN_COND_RAMP_S 0.020
#endif

/* The pots (volume, treble, middle, bass, master, presence, Cabinet Feel) live in the circuit's matrices, rebuilt
 * once per chunk while a pot moves, so a moving pot is a staircase whose step is its travel per rebuild =
 * chunk / (slew time x rate). The travel per rebuild sets the click, the chunk sets the cost while moving: a rebuild
 * every 1/6000 s (8 samples at 48 kHz, the same TIME at every rate) with a 240 ms full-scale slew moves 1/1440 of
 * full scale per rebuild, under the click bar (2nd difference within 5x the steady signal) at a bounded cost; a
 * 30 ms slew needed a 1-sample chunk for the same click. */
#ifndef BN_KNOB_CHUNK_S
#define BN_KNOB_CHUNK_S (1.0 / 6000.0)
#endif
#ifndef BN_KNOB_SLEW_S
#define BN_KNOB_SLEW_S 0.240f
#endif

/* A complete set of state that depends on the oversampling factor. */
typedef struct {
    BnOversampler os;
    /* --- THE w2 SEAM (preamp/power bypass, FX loop) ----------------------
     * A DC blocker of its own on w2 (corner BN_SEAM_HP_HZ, primed at w2's rest
     * value so it starts settled): the power bypass's dry path, and the send
     * while the card is not running, come out of it. */
    double        w2_dc;               /* w2's rest value (V): 0, the wiper after its coupling capacitor */
    double        seam_a;              /* the seam blocker's pole, from fs_hi */
    double        seam_dc_x1, seam_dc_y1;
    /* The FX loop's resampling pair: the send goes DOWN from fs_hi to the host
     * rate and the return goes UP. Its own FIRs (bn_fx_pair.h), not the main
     * oversampler's: its round trip is the loop's own delay. It only runs with
     * the loop active; it is zeroed when the loop engages. */
    BnFxPair      fxos;
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
    const float* fx_mix;           /* FX loop mix port: not read (the card is series only) */
    const float* fx_return_lvl;    /* FX loop: return gain 0..2 */
    const float* engine;           /* connected and never read (port kept for saved sessions) */
    const float* hq;               /* port 29: the factor, 1 = 4x, 0 = 2x (bn_os_target) */
    const float* fx_level;         /* port 30: the reissue card's LEVEL, 0 = Line, 1 = Pedal */
    /* The latency PAD: the plugin reports the LARGEST latency of its configurations at every factor and delays the
     * wet output of the others by the difference, so the hq switch never moves the declared latency.
     * pad_len = lat_max - the running configuration's own latency. Base-rate samples. */
#define BN_PAD_RING 512
    double       pad_ring[BN_PAD_RING];
    int          pad_pos, pad_len, lat_max;
    /* The declared latency also carries the FX loop's own round trip (bn_fxp_latency), whether the loop is on or
     * not, so switching the loop never moves it. pad_len is the loop-OFF pad; pad_len_on = pad_len - the running
     * pair's round trip, the loop-ON pad (the loop itself supplies the round trip). fx_eng_now crossfades the two
     * taps in 10 ms toward the DOMINANT path: the return when the loop is on with mix >= 50 %, else the direct one
     * (the loop on at mix 0 stays the loop off, sample for sample). Never weighted by the mix itself: a
     * mix-weighted blend of two taps would comb. */
    int          pad_len_on;
    double       fx_eng_now;
    int          dry_from_ports;   /* first run() after activate(): take dry_mix from the ports, which are only valid there */
    /* the bias as a linear ramp at fs_hi toward its port, full scale in BN_COND_RAMP_S (the engine's bias source;
     * presence and Cabinet Feel reach the engine through ultra_pres_now / ultra_cab_now). The first run() after
     * activate() takes it, and hum_now, from the ports (cond_from_ports). */
    double       bias_now;
    double       hum_now;          /* the hum's level as applied: a 10 ms ramp at the host rate toward the port */
    int          cond_from_ports;
    double       byp_pow_mix;      /* 0 = power stage active, 1 = bypassed; 10 ms ramp at fs_hi */
    double       fx_mix_now;       /* FX: applied mix (10 ms ramp at fs_hi); 0 with the loop off */
    double       fx_lvl_now;       /* FX: applied return level (ramp) */
    double       fx_send_gain;     /* FX: fade of the send at the base rate, 0..1 */
    int          fx_running;       /* FX: 1 = the resampling pair is warm */
    double*      fx_ret_lo;        /* FX: sanitised return at the base rate, then the downsampled send (n) */
    double*      fx_ret_hi;        /* FX: return upsampled to fs_hi (n*factor) */
    double*      fx_send_hi;       /* FX: w2 (AC, at preamp level) at fs_hi (n*factor) */
    double*      fx_rcap;          /* FX: the host's return of this chunk, kept for the delay estimator (n) */
    int          low_state;        /* -1 forces the first apply */
    double       byp_pre_mix;      /* 0 = preamp active, 1 = bypassed; 10 ms ramp in the loop */
    /* 10 ms ramp to enter and leave bypass/power without a click: a hard jump
     * from wet to dry is a step in the signal, and that is audible. */
    double       dry_mix;          /* 0 = plugin, 1 = pass-through */
    /* The DRY path is delayed by the same latency the plugin reports.
     * Without this, on bypass the signal would come out EARLY relative to the
     * other tracks (the host is already compensating the latency we declare),
     * and the crossfade would mix two misaligned signals -> comb. 512 holds
     * the declared latency (lat_max) with room to spare. */
#define BN_DRY_RING 512
    double       dry_ring[BN_DRY_RING];
    int          dry_pos;
    int          dry_delay;        /* samples; refreshed with the latency */

    double sample_rate;

    /* EVERYTHING that depends on the oversampling factor lives together, one
     * set per factor, indexed by bn_osi (1, 2, 4, 8x; the product builds 2x
     * and 4x). They are built in instantiate, since building one from run()
     * would allocate on the audio thread; changing factor moves a pointer. */
    BnCfg          cfg[4];
    BnCfg*         c;              /* the ACTIVE configuration */
    BnPsu          psu;            /* the supply sag, at block rate */
    BnPsu          psu_idle;       /* the same supply stepped with the IDLE currents: the rest reference for this head and era */
    /* The supply ticks on a FIXED grid of cells aligned to
     * the stream (256 samples at 48 kHz, scaled with the rate), not on the host's blocks. Chunks
     * are cut at the cell edges; the supply, the bias reservoir and the B+ ramp advance once per
     * cell, fed with the previous cell's load. */
    uint32_t       psu_grid;       /* cell length, base samples */
    uint32_t       psu_phase;      /* base samples already processed in the current cell */
    double         pk_cell;        /* the follower's max so far in the current cell */
    /* rq: the rail at the netlist's `htraw` point (plates vc23 + RSAG x the rectifier current), normalised to its
     * idle value, per cell and ramped per fs_hi sample (~1 at rest). Computed in the product too; read only by the
     * measurement builds' old B+ / bias laws (oldb, oldlaw) and as a start's rail. */
    double         rq_now, rq_cell0, drq_hi;
    /* The engine's bias source: the circuit's bias-reservoir law, z -> 1 - g (ich - ich_rest) with a lag tau;
     * zss per supply cell, z per engine sample (bn_ultra_bias.h). Its REST reference psu_urest is the same supply
     * stepped with the engine's OWN rest currents (its output valves at the DC operating point of its last start). */
    BnPsu          psu_urest;
    double         urest_ipl, urest_isc;
    int            urest_ok, oldrest;
    double         bht_zss, bht_z, bht_a;
    double         q_psu[7];       /* the supply's states and the bias reservoir at the start of the freeze's still run */
    /* The engine's B+ source (htraw): the supply's rectifier source, per supply cell, ramped per engine sample */
    double         u1_now, u1_cell0, du1_hi;
    /* measurement builds only (-DBN_DUMP_SLOW; 0 in the product): alternative B+ / bias laws (oldlaw, oldb, bmode,
     * oldrest) and fx_ident, the loop's return := its own send of the same sample (an ideal host) */
    int            oldlaw, oldb, bmode;
    int            fx_ident;
    double         w2_peak_fast;   /* the last closed cell's peak of the master-wiper envelope: the PSU map reads it */
    double         env_fast;       /* its per-sample peak follower, at fs_hi */
    float          knobs_last[5];
    int            os_factor;
    /* Ramp around an oversampling change, which happens in silence (without it
     * the switch pops):
     *   phase 1 = ramping down (with the OLD config)
     *   phase 2 = at zero, switch on the next block
     *   phase 4 = muted while the new config settles (BN_OS_SETTLE_S)
     *   phase 3 = ramping up (with the NEW one)
     * Not a crossfade, which would run two engines at once: a short gap, as
     * the quality switch is not moved while playing. */
    int            os_want;
    int            os_phase;
    int            os_hold;          /* phase 4: samples of silence left while the new factor settles */
    float          os_gain, os_step;
    /* The FIRST block after instantiate is not ramped: there is nothing to
     * smooth, and a host whose first run() asks for 4x would otherwise get a
     * fade at the start. */
    int            os_primero;

    double* buf_in;      /* block at the base rate, in volts */
    double* buf_hi;      /* oversampled block                */
    double* buf_out;

    float  trim_lin, smooth_coef;
    uint32_t knob_chunk;           /* the chunk while a preamp pot moves, in samples (BN_KNOB_CHUNK_S) */
    uint32_t knob_phase;           /* samples into the current knob cell: the pots' grid, aligned to the stream (MK81#202) */
    /* the silence freeze (see the block in run()) */
    int    quiet_run;           /* consecutive samples with a still output */
    int    frozen;              /* 1 = engine frozen */
    double quiet_last;          /* last output sample seen */
    double hum_ph;              /* hum: phase of the fundamental, rad */
    double hum_vpp;             /* hum: Vpp of the block's ripple */
    uint32_t hiss_rng;          /* hiss: state of the hiss generator */
    double psu_i_total;         /* the supply's total current this cell, A (the hum's ripple reads it) */
    int    era_now;             /* era applied to the PSU */
    int    head_now;            /* head applied (0=100W, 1=50W) */
    int    ri_now;              /* the reissue applied (head port 2: a 100 W head, head_now 0) */
    double x_snap[BN_QUIET_NX]; /* the engine's capacitor histories at the last still block (see the freeze) */
    /* ULTRA: the whole amp solved at fs_hi */
    void*        ultra;            /* the engine's memory (bn_ultra_size()), one variant at a time */
    int          ultra_v;          /* the variant initialised in it, bn_ultra_want's numbering (-1 = none) */
    int          ultra_failed;     /* this run()'s start found no operating point: not retried until the next run() */
    double       ultra_gain;       /* the output's gain through a head / edition switch (fade out, re-start, fade in) */
    int          ultra_hold;       /* engine samples the gain holds at 0 after a re-start */
    BnUltraKnobs ultra_kn;         /* the knobs the engine's matrices were built for */
    /* The loop ON runs the reissue card's system: the card variant BN_ULTRA_FX(v, level), seeded from the heard
     * engine through bn_card_seed_on, and its return half (IC1b, the zeners) on nlcore's nodal solver, one per
     * LEVEL x factor (built at instantiate: nl_nodal_init allocates) */
    void*  ultra_card;
    void*  card_ret[2][2];         /* [level][factor index: 0 = 2x, 1 = 4x] */
#ifdef BN_SEND_TWIN
    void*  card_snd[2][2];         /* the send TWIN, [level][factor], at BN_SEND_K(fi) x the engine rate */
    double card_w2p, card_w2pp;    /* w2 one and two engine samples back: the twin's parabola through t-2, t-1, t */
#endif
    int    card_on, card_lv;
    double card_send;              /* the card's send node (IC1a follows it) after the last engine sample */
    double card_fxr;               /* its return half's output (what drives the PI through C1) at that sample */
    /* The ON / OFF switch: the output crossfades heard <-> card while both engines run. At ON the heard (loop-free)
     * engine keeps playing for card_wait engine samples (the chunk the return arrives late, the send's 10 ms ramp
     * and the loop's resamplers), then fades to the card over BN_CARD_XF; at OFF the heard engine, seeded from the
     * card, fades in while the card still runs on the return the host is handing back */
    double card_g;                 /* the card's share of the output, 0..1 */
    long   card_wait;              /* engine samples left before the fade to the card starts */
    long   card_wait0;             /* the wait's length at ON (read by -DBN_CARD_SPREAD builds) */
    int    card_fading_out;        /* 1 while the card fades out after OFF */
    double       ultra_pres_now;   /* presence as the engine applies it: slewed like the pots, rebuilt per chunk */
    double       ultra_cab_now;    /* Cabinet Feel as the engine applies it (the 4x12's R, L, C), slewed like the pots */
    double       ultra_ipl_acc, ultra_isc_acc;   /* the output valves' plate / screen current summed over the cell */
    long         ultra_ncur;
    double       ultra_ipl, ultra_isc;   /* the last closed cell's mean: the supply's load */
    int          ultra_cur_ok;     /* 1 once a cell has closed since the last start: ultra_ipl / ultra_isc replace the map */
    /* The SEND system (src/prepig/: preamp + phase inverter with the output valves' grids conducting, its rail and
     * NFB read from the heard engine one sample late) gives the loop-free wiper while the preamp bypass acts; the
     * heard engine then has C9's top DETACHED and driven by the bypass's mix. It is seeded from the heard engine the
     * sample the bypass starts acting; idle again, the heard engine is re-attached. send_kn: the knobs its matrices
     * were last built for. */
    void*        ultra_send;
    int          send_on;
    BnUltraKnobs send_kn;
    double frozen_out;          /* value emitted while frozen */
} BritishNoise;

static int bn_cfg_lat(const BnCfg* c, int factor);

/* index of the configuration for each legal factor (1,2,4,8) */
static int bn_osi(int factor)
{
    switch (factor) {
    case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3;
    default: return -1;
    }
}

/* BUILDS a configuration: its oversampler, the FX loop's resampler pair and complement (all allocate), the seam
 * blocker at rest, and the supply's and knobs' starting state. Called from instantiate only, NEVER from the audio
 * thread. */
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
    bn_fxp_free(&self->c->fxos);
    if (bn_fxp_init(&self->c->fxos, factor, BN_MAX_BLOCK) != 0) {
        bn_fxp_free(&self->c->fxos);
        bn_os_free(&self->c->os);
        return;
    }
    bn_fxc_free(&self->c->fxc);
    if (bn_fxc_init(&self->c->fxc, factor, BN_MAX_BLOCK, self->sample_rate) != 0) {
        bn_fxc_free(&self->c->fxc);
        bn_fxp_free(&self->c->fxos);
        bn_os_free(&self->c->os);
        return;
    }
    const double fs_hi = self->sample_rate * factor;
    self->low_state = -1;
    bn_psu_init(&self->psu);
    bn_psu_init(&self->psu_idle);
    self->rq_now = 1.0; self->rq_cell0 = 1.0; self->drq_hi = 0.0;
    self->bht_zss = 1.0; self->bht_z = 1.0; self->bht_a = 0.0; self->u1_now = 0.0;
    self->w2_peak_fast = 0.0;
    self->env_fast = 0.0;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }  /* forces the first application */
    /* the seam's w2 blocker starts at rest (0 V: the wiper after its coupling capacitor), the dry ring empty */
    self->c->w2_dc = 0.0;
    self->c->seam_a = exp(-2.0 * BN_PI * BN_SEAM_HP_HZ / fs_hi);
    self->c->seam_dc_x1 = 0.0;
    self->c->seam_dc_y1 = 0.0;

    self->c->ready = 1;
}

/* The whole teardown of an instance, IN ONE PLACE.
 *
 * It frees every configuration regardless of `ready`: `bn_build_cfg` has early returns AFTER
 * allocating that leave a half-built config with `ready = 0`, and skipping those would leak.
 * The failure paths of `instantiate` come through here too, so a host that retries the load in a
 * loop loses nothing. Freeing a half-built or never-built config is safe: `self` comes from
 * `calloc`, and bn_os_free / bn_fxp_free / bn_fxc_free accept a zeroed struct. */
static void
bn_teardown(BritishNoise* self)
{
    if (!self) { return; }
    for (int ci = 0; ci < 4; ++ci) {
        bn_os_free(&self->cfg[ci].os);
        bn_fxp_free(&self->cfg[ci].fxos);
        bn_fxc_free(&self->cfg[ci].fxc);
    }
    free(self->buf_in);
    free(self->buf_hi);
    free(self->buf_out);
    free(self->fx_ret_lo);
    free(self->fx_ret_hi);
    free(self->fx_send_hi);
    free(self->ultra_card);
    for (int lv = 0; lv < 2; ++lv) { for (int fi = 0; fi < 2; ++fi) { free(self->card_ret[lv][fi]); } }
#ifdef BN_SEND_TWIN
    for (int lv = 0; lv < 2; ++lv) { for (int fi = 0; fi < 2; ++fi) { free(self->card_snd[lv][fi]); } }
#endif
    free(self->fx_rcap);
    free(self->ultra);
    free(self->ultra_send);
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
        bn_teardown(self);       /* the buffers AND the configs: see bn_teardown */
        return NULL;
    }
    self->fx_lvl_now = 1.0;   /* the port's default; the ramp starts from here */

    /* The configurations are built HERE, outside the audio thread. Changing factor live is just moving a pointer. */
    self->os_factor = 0;
    self->psu_grid = (uint32_t)lround(256.0 * rate / 48000.0);
    if (self->psu_grid < 1) { self->psu_grid = 1; }
    /* the two factors the product offers, 2x and 4x (hq): each configuration is its oversampler, the loop's
     * resampler pair and complement; every loop over cfg[] guards on `ready` */
    bn_build_cfg(self, 2);
    /* built here, never from run(): the hq switch only moves self->c. Guarded so a measuring copy whose line above
     * builds 4x builds it once */
    if (!self->cfg[bn_osi(4)].ready) { bn_build_cfg(self, 4); }
    if (!self->cfg[bn_osi(2)].ready) {   /* 2x, the default, must exist */
        bn_teardown(self);       /* the buffers AND the configs: see bn_teardown */
        return NULL;
    }
    self->c = &self->cfg[bn_osi(2)];
    self->os_factor = 2;
    /* the latency every factor reports: the largest of the configurations built (a 4x that failed to build is never
     * selected, so it does not count) */
    self->lat_max = 0;
    for (int f = 2; f <= 4; f *= 2) {
        const BnCfg* cf = &self->cfg[bn_osi(f)];
        if (cf->ready) { const int l = bn_cfg_lat(cf, f) + bn_fxp_latency(&cf->fxos); if (l > self->lat_max) { self->lat_max = l; } }
    }
    if (self->lat_max > BN_PAD_RING - 1) { self->lat_max = BN_PAD_RING - 1; }
    self->pad_len = 0; self->pad_pos = 0; self->pad_len_on = 0; self->fx_eng_now = 0.0;
    /* Ramp of the oversampling change: 5 ms down, BN_OS_SETTLE_S muted, 5 ms up. At rest
     * os_gain = 1 and bn_os_ramp does nothing. */
    self->os_phase = 0;
    self->os_want  = 2;
    self->os_primero = 1;
    self->os_gain  = 1.0f;
    self->os_step  = (float)(1.0 / (0.005 * rate));
    self->cond_from_ports = 1;
    bn_ultra_tables_init();
    self->ultra = calloc(1, bn_ultra_size());
    bn_prepig_tables_init();
    self->ultra_send = calloc(1, bn_prepig_size());
    self->ultra_card = calloc(1, bn_ultra_size());
    if (!self->ultra_card) { bn_teardown(self); return NULL; }
    for (int lv = 0; lv < 2; ++lv) {
        for (int fi = 0; fi < 2; ++fi) {
            self->card_ret[lv][fi] = cr_ret_new(rate * (fi ? 4.0 : 2.0), lv);
            if (!self->card_ret[lv][fi]) { bn_teardown(self); return NULL; }
#ifdef BN_SEND_TWIN
            self->card_snd[lv][fi] = cs_send_new(rate * (fi ? 4.0 : 2.0) * BN_SEND_K(fi), lv);
            if (!self->card_snd[lv][fi]) { bn_teardown(self); return NULL; }
#endif
        }
    }
#ifdef BN_FXZ_WARM
    {   /* every card system initialised once here (not real-time), so the tables its init builds once
         * (MKS_FXZ_TAB) are warm before the loop is first switched on in run() */
        BnUltraKnobs kw; memset(&kw, 0, sizeof kw); kw.vol = kw.treb = kw.bass = kw.mid = kw.master = kw.pres = 0.5;
        for (int lv = 0; lv < 2; ++lv) for (int vv = 0; vv < BN_ULTRA_NV; ++vv)
            (void)bn_ultra_init(self->ultra_card, BN_ULTRA_FX(vv, lv), rate * 2.0, &kw, bn_ultra_vht(vv), bn_ultra_vbias(vv, 0.5));
    }
#endif
    self->card_on = 0; self->card_lv = 1; self->card_send = 0.0; self->card_fxr = 0.0;
    self->card_g = 0.0; self->card_wait = 0; self->card_fading_out = 0;
    if (!self->ultra || !self->ultra_send) { bn_teardown(self); return NULL; }
    self->ultra_v = -1; self->ultra_gain = 1.0;
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
    case BN_HQ:            self->hq            = (const float*)data; break;
    case BN_FX_LEVEL:      self->fx_level      = (const float*)data; break;
    default: break;
    }
}

/* a configuration's own latency at the base rate: its oversampler's */
static int bn_cfg_lat(const BnCfg* c, int factor)
{
    (void)factor;
    const int lat = bn_os_latency(&c->os);
    return lat < 0 ? 0 : lat;
}

/* Real-time factor CHANGE: move a pointer and reset the per-configuration
 * caches. Every configuration is already built in instantiate, so there is no
 * allocation here; the engine's re-start at the new rate is bn_os_switch's. */
static void rebuild_oversampler(BritishNoise* self, int factor)
{
    if (factor == self->os_factor) { return; }
    const int ci = bn_osi(factor);
    if (ci < 0 || !self->cfg[ci].ready) { return; }
    self->c = &self->cfg[ci];
    /* a cell in progress keeps its rail ramp; its slope is per fs_hi sample */
    if (self->os_factor > 0) {
        self->drq_hi  *= (double)self->os_factor / (double)factor;
    }
    self->os_factor = factor;
    /* force the knobs to be reapplied on the next block */
    self->low_state = -1;
    self->byp_pre_mix = 0.0;   /* starts with the preamp ACTIVE; the ramp leads to the target */
    self->byp_pow_mix = 0.0;   /* likewise the power stage */
    self->fx_mix_now  = 0.0;   /* FX: the loop re-engages (and resets its pair) */
    self->fx_running  = 0;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }
    /* ⚠️ The PSU is NOT reset here: the supply does not depend on the
     * oversampling, and resetting it makes the rail jump to B+ and the amp
     * momentarily louder until the sag pulls it down again (an overshoot for
     * ~150 ms after each change). The rail is carried as is.
     * `w2_peak_fast` IS reset: it is a per-cell peak, not a physical state. */
    self->w2_peak_fast = 0.0;
    self->env_fast = 0.0;
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

/* Every read of a 0..1 control port goes through here: NULL, NaN and Inf read as 0 (a +Inf on `hiss` would
 * poison the circuit's state for good), the rest is clamped to the TTL range. */
static inline double bn_port01(const float* p)
{
    if (!p) { return 0.0; }
    const float v = *p;
    if (!isfinite(v)) { return 0.0; }
    return v < 0.0f ? 0.0 : (v > 1.0f ? 1.0 : (double)v);
}
/* The silence freeze's thresholds, as multiples of one tolerance (BN_QUIET_TOL, volts): the rest floor is the
 * solver's residual, so thresholds tied to a tolerance follow it where absolute ones would go stale.
 *   eps   (0.5 tol): the most the output may move per sample and still count as still
 *   level (1 tol):   the output must also be that small (the freeze never takes a flat but loud signal)
 *   xeps  (BN_QUIET_XEPS_K tol, relative to the state's magnitude): the most the circuit's state may move */
#define BN_QUIET_TOL 3e-5
static inline double bn_quiet_eps(const BritishNoise* self)   { (void)self; return 0.5 * BN_QUIET_TOL; }
static inline double bn_quiet_level(const BritishNoise* self) { (void)self; return 1.0 * BN_QUIET_TOL; }
static inline double bn_quiet_xeps(const BritishNoise* self)  { (void)self; return BN_QUIET_XEPS_K * BN_QUIET_TOL; }



/* The head port, read only through here: 0 = the 100 W head, 1 = the 50 W head, 2 = the reissue (a 100 W EL34 head
 * with its own pots, no C2 and its own supply); anything else (NaN included) reads 0. */
static int bn_head_port(const BritishNoise* self)
{
    if (!self->head) { return 0; }
    const float h = *self->head;
    return h > 1.5f ? 2 : h > 0.5f ? 1 : 0;
}
/* The engine's variant from the ports, bn_ultra.h's order: 0 = 100 W UK, 1 = 100 W US, 2 = 50 W UK, 3 = 50 W US
 * (2 x head + us), 4 = the reissue (EL34 whatever the US switch says) */
static int bn_ultra_want(const BritishNoise* self)
{
    if (bn_head_port(self) == 2) { return 4; }
    return 2 * (bn_head_port(self) == 1 ? 1 : 0) + ((self->us_spec && *self->us_spec > 0.5f) ? 1 : 0);
}

/* the knobs as applied now: the pots' slewed values, presence and Cabinet Feel, the jack, the rectifier's era */
static BnUltraKnobs bn_ultra_knobs_now(const BritishNoise* self)
{
    BnUltraKnobs k;
    k.vol = self->knobs_last[0]; k.treb = self->knobs_last[1]; k.bass = self->knobs_last[2];
    k.mid = self->knobs_last[3]; k.master = self->knobs_last[4];
    k.pres = self->ultra_pres_now;
    k.cab = self->ultra_cab_now;
    k.lowin = self->low_state > 0 ? 1.0 : 0.0;
    k.era = (self->era_now || self->ri_now) ? 1.0 : 0.0;   /* the rectifier's reservoir inside the engine (the reissue: the old way) */
    return k;
}

/* Measurement build only (-DBN_DUMP_SLOW, env BN_ULTRA_BMODE): 7 drives the engine from an IDEAL supply (B+ the
 * netlist's htraw source value, the bias at rest); 8 the B+ from the rectifier source vsrc; 9 = 8 with the bias at
 * rest. In the product the three tests are 0. */
#define BN_B7(self) 0
#define BN_B8(self) 0
#define BN_B9(self) 0
/* The engine's B+ source from the supply as it is now: the rectifier's source vsrc (the systems carry the averaged
 * rectifier and its reservoir themselves). Used per cell; a start uses it only in the measurement builds
 * (bmode 5, BN_ULTRA_U1_REPLICA), the product's start takes the rest point (bn_ultra_u1_start). */
static double bn_ultra_u1_psu(const BritishNoise* self, int v)
{
    const double ich = self->psu.vsrc > self->psu.vc23 ? (self->psu.vsrc - self->psu.vc23) / self->psu.rchg : 0.0;
    /* the systems carry the averaged rectifier (htraw-plates = rchg, C23 on the Era knob), so htraw is its source vsrc;
     * the replica supply here still serves the bias law and the hum */
    (void)ich; (void)v;
    return self->psu.vsrc;
}
/* The engine's B+ for its DC solve: the supply's, once the supply runs THIS head; before that (activate() starts the
 * supply on the 100 W head and the head port is applied on the first supply cell) the head's own pinned rest point
 * (a 50 W start solved at the 100 W supply's voltage would start far from its operating point) */
static double bn_ultra_u1_start(const BritishNoise* self, int v)
{
    /* The supply runs inside the system, and a DC solve puts its reservoir at REST; a start while the amp plays (the
     * hq switch, a variant change) would then begin with the sag undone. Solved at vsrc minus the replica's present
     * sag (idle twin minus loaded), the reservoir starts where the supply is now, and the run continues on vsrc */
    if (self->psu.w50 == ((v & 2) ? 1 : 0)) { return self->psu.vsrc - (self->psu_idle.vc23 - self->psu.vc23); }
    const double vr = (v & 2) ? BN_PSU_V_REST * BN_PSU_K_V_50W : BN_PSU_V_REST;
    const double ir = (v & 2) ? BN_PSU_I_REST * BN_PSU_K_I_50W : BN_PSU_I_REST;
    /* that head's vsrc: bn_psu_refresh's pinned rest point vr + rchg ir, with the head's (and era's) rchg */
    const double rc = (v & 2) ? BN_PSU_RCHG_NEW * BN_PSU_K_RCHG_50W
                    : (self->era_now || v == 4) ? BN_PSU_RCHG_OLD : BN_PSU_RCHG_NEW;
    return vr + rc * ir;
}
/* the send system's knobs: the engine's without the Cabinet Feel (the send system has no output stage) */
static BnPrepigKnobs bn_prepig_knobs(const BnUltraKnobs* k)
{
    const BnPrepigKnobs p = { k->vol, k->treb, k->bass, k->mid, k->master, k->pres, k->lowin };
    return p;
}
static int bn_prepig_knobs_differ(const BnUltraKnobs* a, const BnUltraKnobs* b)
{
    return a->vol != b->vol || a->treb != b->treb || a->bass != b->bass || a->mid != b->mid || a->master != b->master
        || a->pres != b->pres || a->lowin != b->lowin;
}
/* The engine's knob set brought up to date (the pots, presence, Cabinet Feel, and the era the supply applies): rebuilt
 * only when it changed. Called by the pots' slew at a knob cell's edge and by the supply's tick right after it moves the
 * era (MK81#202: the era reached the engine at the NEXT chunk, whose start was the host block's - -23.5 dBFS at blocks of
 * 1 / 7 with the era flipped after activate()). */
static void bn_ultra_sync_knobs(BritishNoise* self)
{
    const BnUltraKnobs kq = bn_ultra_knobs_now(self);
    if (self->ultra_v >= 0 && memcmp(&kq, &self->ultra_kn, sizeof kq) != 0) {
        bn_ultra_set_knobs(self->ultra, self->ultra_v, &kq);
        if (self->send_on && bn_prepig_knobs_differ(&kq, &self->send_kn)) {
            const BnPrepigKnobs pk = bn_prepig_knobs(&kq);
            bn_prepig_set_knobs(self->ultra_send, self->ultra_v, &pk);
            self->send_kn = kq;
        }
        self->ultra_kn = kq;
    }
}
/* The engine starts on variant v from ITS operating point at the present knobs, bias and supply (a DC solve, no
 * settling: the engine settles in pseudo-time inside its own start), and the send system is built at the same rate and
 * knobs. Called from activate(), and from run() once
 * per switch: on a factor change before the oversampled loop, on a variant change at the sample its fade reaches zero. Returns 0, or -1 if a solve failed (ultra_failed set,
 * ultra_v = -1: the amp stays muted until a later run() retries). */
static int bn_ultra_start(BritishNoise* self, int v, int settle)
{
    const double rail = self->rq_now;
    const BnUltraKnobs k = bn_ultra_knobs_now(self);
    const double u1 = BN_B8(self) ? self->psu.vsrc : BN_B7(self) ? bn_ultra_vht(v) : (self->oldb || self->bmode == 2) ? bn_ultra_vht(v) * rail : bn_ultra_u1_start(self, v);
    const double u2 = (BN_B7(self) || BN_B9(self)) ? bn_ultra_vbias(v, self->bias_now)
                    : self->oldlaw ? bn_ultra_vbias(v, self->bias_now) * (1.0 - ((v & 2) ? 1.0 : 11.35 / 32.43) * (1.0 - rail))
                                   : bn_ultra_vbias(v, self->bias_now) * self->bht_z;   /* the bias reservoir where it is now */
    const double fs_e = self->sample_rate * (double)(self->os_factor > 0 ? self->os_factor : 1);
    if (bn_ultra_init(self->ultra, v, fs_e, &k, u1, u2) != 0) {
        self->ultra_failed = 1; self->ultra_v = -1;
        return -1;
    }
    (void)settle;   /* every start settles inside the engine now (bn_ultra_engine.c, nlcore's mks_mna_settle) */
    {   /* the send system at this rate and these knobs. Its own operating point here only builds it: run() seeds it
         * from the heard engine the sample the preamp bypass starts acting (bn_prepig_seed) */
        const BnPrepigKnobs pk = bn_prepig_knobs(&k);
        if (bn_prepig_init(self->ultra_send, v, fs_e, &pk, u2, bn_ultra_nodes(self->ultra, v)) != 0) {
            self->ultra_failed = 1; self->ultra_v = -1;
            return -1;
        }
        self->send_kn = k;
    }
    self->ultra_kn = k;
    self->ultra_v = v;
    bn_ultra_currents(self->ultra, v, &self->urest_ipl, &self->urest_isc);   /* its rest: the bias law's reference */
    self->psu_urest = self->psu;
    self->urest_ok = 1;
    self->send_on = 0;   /* the engine was re-initialised attached; the send system is re-seeded from it if still needed */
    self->ultra_ipl_acc = 0.0; self->ultra_isc_acc = 0.0; self->ultra_ncur = 0; self->ultra_cur_ok = 0;
    return 0;
}

#if defined(BN_SEND_TWIN)
/* The card's send from its TWIN: the send half alone at BN_SEND_K x the engine rate (the zeners' clamp pulses charge
 * C8 in less than an engine sample). w2 inside the sample on the parabola through the last three samples; the last
 * sub-step is the value at this sample. */
static inline void bn_card_send_twin(BritishNoise* self, int fi, double w2)
{
    void* sw = self->card_snd[self->card_lv][fi]; const int k = BN_SEND_K(fi); double y = 0.0;
    const double w0 = self->card_w2p, d1 = 0.5 * (w2 - self->card_w2pp), d2 = 0.5 * (w2 - 2.0 * w0 + self->card_w2pp);
    for (int q = 1; q <= k; ++q) { const double u = (double)q / k; y = cs_send_tick(sw, w0 + u * d1 + u * u * d2); }
    self->card_w2pp = w0; self->card_w2p = w2; self->card_send = y;
}
#endif


/* The effects card off and forgotten: its system, its return / send halves and the switch's ramps are re-made by the
 * loop-ON path the next time the loop is on. Called from activate() and on a variant switch (the card system belongs
 * to the old variant). */
static void bn_card_drop(BritishNoise* self)
{
    self->card_on = 0; self->card_g = 0.0; self->card_fading_out = 0; self->card_wait = 0; self->card_wait0 = 0;
    self->card_send = 0.0; self->card_fxr = 0.0;
#ifdef BN_SEND_TWIN
    self->card_w2p = self->card_w2pp = 0.0;
#endif
}

/* Reads no port (the engine port is ignored): 2 while the engine can start, 1 after a start failed in this run(),
 * which holds bn_os_target at 2x until the next run() retries */
static int bn_engine_target(const BritishNoise* self)
{
    return !self->ultra_failed ? 2 : 1;
}

/* THE FACTOR the ports ask for: 4x with hq on (>= 0.5), 2x otherwise. A NaN or an unconnected port reads 2x, the
 * port's default. */
static int bn_os_target(const BritishNoise* self, int eng)
{
    const int hq_on = self->hq && *self->hq >= 0.5f;   /* NaN compares false: 2x, the default */
    const int f = (eng == 2 && hq_on) ? 4 : 2;
    /* a configuration that was not built (out of memory at instantiate, or a measuring copy forced to one factor) is
     * never asked for: the plugin stays where it is */
    return self->cfg[bn_osi(f)].ready ? f : self->os_factor;
}

/* A factor change, applied only where the output is SILENT (the os ramp at zero, or the first block of a stream):
 * the configuration pointer moves (rebuild_oversampler: no allocation) and the engine re-starts at once at the new
 * rate from its operating point at the present knobs (bn_ultra_start, a DC solve). Always returns 0. */
static int bn_os_switch(BritishNoise* self, int factor)
{
    rebuild_oversampler(self, factor);
    self->ultra_hold = 0; self->ultra_gain = 1.0;
    {
        /* the knobs as activate() starts the engine: the pots' slewed values, or the ports before the first block
         * has applied any; the jack from its port (rebuild_oversampler has just invalidated low_state) */
        static const float KD[5] = { 0.7f, 0.5f, 0.5f, 0.5f, 1.0f };
        const float* const P[5] = { self->vol, self->treble, self->bass, self->middle, self->master };
        float kl[5];
        for (int k = 0; k < 5; ++k) {
            kl[k] = self->knobs_last[k];
            if (kl[k] < 0.0f) {
                const float x = P[k] ? *P[k] : KD[k];
                self->knobs_last[k] = !(x >= 0.0f) ? KD[k] : !(x <= 1.0f) ? 1.0f : x;
            }
        }
        self->low_state = (self->low_input && *self->low_input > 0.5f) ? 1 : 0;
        (void)bn_ultra_start(self, bn_ultra_want(self), 1);   /* a failure leaves ultra_v < 0: muted, retried by run() */
        for (int k = 0; k < 5; ++k) { self->knobs_last[k] = kl[k]; }
        self->low_state = -1;
    }
    return 0;   /* a start that failed leaves the amp muted until run() retries it */
}

/* the latency pad's delay line (see pad_len): the wet output delayed by pad_len base-rate samples with the FX loop
 * off (or under 50 % mix), by pad_len_on with the return dominant, crossfaded in 10 ms by fx_eng_now; with the loop
 * off and settled the result is the pad_len tap bit for bit (x + 0.0 * (y - x) = x) */
static inline double bn_pad_push(BritishNoise* self, double x, int fx_wet, double eng_step)
{
    self->pad_ring[self->pad_pos] = x;
    int r = self->pad_pos - self->pad_len;
    if (r < 0) { r += BN_PAD_RING; }
    int r_on = self->pad_pos - self->pad_len_on;
    if (r_on < 0) { r_on += BN_PAD_RING; }
    self->pad_pos = (self->pad_pos + 1) & (BN_PAD_RING - 1);
    bn_ramp_to(&self->fx_eng_now, fx_wet ? 1.0 : 0.0, eng_step);
    const double a = self->pad_ring[r];
    return a + self->fx_eng_now * (self->pad_ring[r_on] - a);
}

static void
activate(LV2_Handle instance)
{
    BritishNoise* self = (BritishNoise*)instance;
    /* the factor from the ports FIRST, so everything below (the oversampler reset, the engine's start at fs_hi)
     * acts on the configuration that will run. A best guess (lv2.h: control ports are meaningful from run()); a
     * different answer in the first run() is switched there without a ramp (os_primero). */
    { const int f = bn_os_target(self, bn_engine_target(self)); if (f != self->os_factor) { rebuild_oversampler(self, f); } }
    memset(self->pad_ring, 0, sizeof self->pad_ring);
    self->pad_pos = 0;
    self->fx_eng_now = 0.0;   /* the loop-pad crossfade starts where the loop does (off, ramping in) */
    self->quiet_run = 0; self->frozen = 0;
    /* the supply grid restarts with the stream */
    self->psu_phase = 0; self->pk_cell = 0.0; self->drq_hi = 0.0; self->rq_cell0 = 1.0;
    self->knob_phase = 0;   /* and the pots' grid (MK81#202) */
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
    /* the head and the era as run() will read them on its first block (MK81#198): a start on the default head / era
     * while the ports say another made the first block switch the supply and the engine's rectifier mid-start, at a
     * point that depends on the host's block size (-DBN_START_ON_DEFAULTS: the defaults as before; it was the bloque gate's
     * arm until 11-Nov-2026, when the era started reaching the engine at the supply's tick (MK81#202) and that start became
     * block-independent too - kept as a measurement switch, no gate builds it) */
    const int head_a = bn_head_port(self);
    self->era_now = head_a == 0 ? (self->era ? *self->era > 0.5f : 1) : 1;   /* run()'s era_q */
    /* ⛔ And it MUST BE APPLIED TO THE PSU HERE: `bn_psu_init()` leaves the new way, and run() only calls
     * `bn_psu_set_era()` when the port differs from era_now, so with the default at 1 the supply would run the
     * new way while the plugin said old. */
    /* LIFECYCLE: the supply, the rails, the hum's phase, the effects card and the engine are STATE; if they survived
     * a deactivate/activate, a host that stopped and restarted the transport would get a different render. LV2's
     * activate() is where the instance returns to its initial state. ⛔ This is NOT the reset `rebuild_oversampler`
     * must not do: there the rail is carried across on purpose (a factor change is not an amp switched off). */
    bn_psu_init(&self->psu); bn_psu_init(&self->psu_idle);
    bn_card_drop(self);   /* the effects card restarts with the loop, as on a fresh instance */
    bn_psu_set_era(&self->psu, self->era_now);
    bn_psu_set_era(&self->psu_idle, self->era_now);
    self->head_now = head_a == 1;   /* the 50 W head's supply (run()'s head_q) */
    self->ri_now = head_a == 2;     /* the reissue's supply (run()'s ri_q) */
    bn_psu_set_head(&self->psu, self->head_now); bn_psu_set_head(&self->psu_idle, self->head_now);
    if (self->ri_now) { bn_psu_set_ri(&self->psu, 1); bn_psu_set_ri(&self->psu_idle, 1); }
    self->rq_now = 1.0;
    self->bht_zss = 1.0; self->bht_z = 1.0; self->bht_a = 0.0;   /* the bias reservoir at rest */
    self->urest_ok = 0; self->u1_now = 0.0;
    self->hum_ph = 0.0;
    memset(self->x_snap, 0, sizeof(self->x_snap)); memset(self->q_psu, 0, sizeof(self->q_psu));
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
    /* every configuration, not just the one in use (the host may switch factor after a restart): its oversamplers,
     * the FX loop's pair and complement, and the w2 seam's blocker back to rest */
    for (int ci = 0; ci < 4; ++ci) {
        BnCfg* c = &self->cfg[ci];
        if (!c->ready) { continue; }
        c->seam_dc_x1 = c->w2_dc;
        c->seam_dc_y1 = 0.0;
        bn_os_reset(&c->os);
        bn_fxp_reset(&c->fxos);
        bn_fxc_reset(&c->fxc);
    }
    self->fx_running = 0;
    self->fx_send_gain = 0.0;
    /* the port ramps and the "last applied" caches are state too: a restart must start them where a fresh
     * instance does, or the first blocks ramp from wherever the previous run left them */
    self->byp_pre_mix = 0.0; self->byp_pow_mix = 0.0;
    self->fx_mix_now = 0.0; self->fx_lvl_now = 1.0;
    /* `dry_mix` and the oversampling ramp are state too: a restart must start them where the PORTS say, or the first
     * take begins with 10 ms of the wrong path fading out (raw DI after a stop in bypass; processed signal on an
     * instance created in bypass). Control port contents are not guaranteed meaningful until run() (lv2.h), so the
     * value read here is a best guess and the first run() overrides it from its own target. */
    self->dry_mix = ((self->power && *self->power < 0.5f) || (self->bypass && *self->bypass > 0.5f)) ? 1.0 : 0.0;
    self->dry_from_ports = 1;
    self->cond_from_ports = 1;
    self->os_phase = 0; self->os_gain = 1.0f; self->os_want = self->os_factor;
#ifndef BN_OS_PRIMERO_ONCE   /* the gate's negative arm: the start-up switch only after instantiate, as before */
    /* MK81#202: a restart is a start - the first run() after activate() switches a factor the ports changed WITHOUT the
     * silent ramp, as the comment at the top of activate() promises; os_primero was set only at instantiate, so a
     * re-activated stream with hq flipped took the ramp and the 50 ms settle mute (Roadcase's kit: -8.0 dBFS against the
     * first render, at every block size) */
    self->os_primero = 1;
#endif
    self->low_state = -1;
    for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }
    /* the engine starts from its operating point at the ports' knobs, with no fade. Control ports are a best guess
     * here (lv2.h); a different value in the first run() is an ordinary knob move or switch. */
    self->ultra_gain = 1.0; self->ultra_hold = 0; self->ultra_v = -1; self->send_on = 0;
    self->ultra_pres_now = bn_ctl(self->presence, 0.0, 0.0, 1.0);
    self->ultra_cab_now = bn_ctl(self->cab_load, 0.0, 0.0, 1.0);
    if (!self->ultra_failed) {
        static const float KD[5] = { 0.7f, 0.5f, 0.5f, 0.5f, 1.0f };
        const float* const P[5] = { self->vol, self->treble, self->bass, self->middle, self->master };
        for (int k = 0; k < 5; ++k) {
            const float x = P[k] ? *P[k] : KD[k];
            self->knobs_last[k] = !(x >= 0.0f) ? KD[k] : !(x <= 1.0f) ? 1.0f : x;
        }
        self->low_state = (self->low_input && *self->low_input > 0.5f) ? 1 : 0;
        self->bias_now = bn_ctl(self->bias, 0.5, 0.0, 1.0);
        if (bn_ultra_start(self, bn_ultra_want(self), 1) == 0) {
        }
        for (int k = 0; k < 5; ++k) { self->knobs_last[k] = -1.0f; }   /* run()'s first block takes the ports */
        self->low_state = -1;
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
 * circuit. So it is added to buf_in —which is in VOLTS— and the engine
 * amplifies and filters it by itself, with the amp's real response. There is
 * no need to derive the output level: it emerges.
 *
 * THE NETWORK, from the schematic: pickup -> jH, R2 1M to ground, R3 68k to
 * the grid. The grid is high impedance, so both noise sources arrive whole and
 * their powers add:  R_eq = (Rpickup ∥ 1M) + 68k
 *
 * And this removes the arbitrariness: the 68k
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
 *   · ripple amplitude: MEASURED on the PSU model, 15.7 Vpp at
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
 * ⚠️ WHAT THIS IS NOT: the engine's supply is an averaged model with no
 * ripple, so the hum is injected at the output, derived analytically, not
 * emerging from the circuit — it does NOT intermodulate with the signal as it
 * really would.
 *
 * ⚠️ And the HEATER hum (50 Hz through cathode-heater leakage), the other
 * half of a real amp's hum, is not modelled: it is not on the schematic and
 * would require inventing a parameter.
 *
 * WHERE it is added: AFTER the silence detector, on purpose. If it came in
 * before, the output would never stay still and the silence freeze —which
 * takes the engine's cost to zero— would never engage.
 * ========================================================================== */
/* THE MAINS FREQUENCY IS NOT A KNOB: it is a property of the REGION, and this
 * plugin already has a region axis — the `us_spec` port (US edition, 6550).
 * So the frequency goes WITH it, and no extra control is added:
 *      us_spec = 0  ->  UK / Europe   50 Hz mains  ->  100 Hz ripple
 *      us_spec = 1  ->  USA / Canada  60 Hz mains  ->  120 Hz ripple
 * (the ripple is at TWICE the mains frequency because the 100 W head has a
 * full bridge; the BIAS chain's would be at the mains frequency, half wave —
 * but that one was ruled out by measurement.)
 *
 * AND IT IS NOT ONLY THE PITCH. With faster mains the reservoir discharges
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
/* Plate-to-plate : 8 ohm turns ratio, which divides the hum. 100 W head: from
 * the OT's winding, 14.588 = sqrt(1702.5 / 8) (Zaa 1702.5 ohm). The 50 W
 * head's OT has about twice the Zaa (3.2 k) ⇒ N = sqrt(3200/8) = 20.0. */
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
    /* Each head has its own line (the 50 W head's supply has its own
     * secondary, winding resistance and reservoir): 7.3 Vpp idle and 26.4 flat
     * out on the 50 W head. A line through each head's two points, with the
     * current on the plugin's scale (I_REST / I_FULL, x K_I_50W on the 50 W
     * head: 0.1096 / 0.3257 A); the minimum is the idle value. The harmonic
     * SHAPE has no head axis: it depends on the load, and the one table
     * (BN_HUM_HARM) sits between its idle and flat-out shapes. */
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
    /* FTZ+DAZ while the DSP runs: without -ffast-math the engine's and the
     * FIRs' states can fall into denormals in silent tails, and a denormal
     * costs ~50x. The MXCSR is restored on exit (the host owns its own mode). */
    BritishNoise* self = (BritishNoise*)instance;
    const float* const in  = self->in;
    float* const       out = self->out;
    /* The pointer check goes BEFORE touching the MXCSR: in the other order
     * this early return would skip the restore and leave FTZ+DAZ SET on the
     * host's audio thread, and every plugin after it in the chain would
     * silently flush denormals. */
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
        (void)bn_os_switch(self, self->os_want);
        /* The incoming configuration carries the state it had when it was left (or its primed rest, the first
         * time), so its first ~50 ms depart from a configuration that ran all along: the output stays muted for
         * BN_OS_SETTLE_S while it settles, then fades in. */
        self->os_hold = (int)(BN_OS_SETTLE_S * self->sample_rate);
        self->os_phase = self->os_hold > 0 ? 4 : 3;
    }
    else {
        /* the factor follows the hq port (bn_os_target), through the same silent ramp */
        const int want = bn_os_target(self, bn_engine_target(self));
        if (want != self->os_factor && self->os_phase == 0) {
            if (self->os_primero) {
                (void)bn_os_switch(self, want);   /* start-up: no ramp */
            } else {
                self->os_want  = want;
                self->os_phase = 1;
            }
        }
    }
    /* Cleared HERE, where it is consumed, and not at the end of run(): there
     * are two early `return`s (unconnected ports, and POWER off with the ramp
     * finished) that would skip it and leave the first real change without a
     * ramp. */
    self->os_primero = 0;
    /* One latency at every factor (the largest): the running configuration's wet output is padded to it, and the
     * dry delay line is aligned with it */
    {
        int pad = self->lat_max - bn_cfg_lat(self->c, self->os_factor);
        if (pad < 0) { pad = 0; }
        if (pad > BN_PAD_RING - 1) { pad = BN_PAD_RING - 1; }
        self->pad_len = pad;
        const int pad_on = pad - bn_fxp_latency(&self->c->fxos);   /* the loop supplies its own round trip */
        self->pad_len_on = pad_on < 0 ? 0 : pad_on;
        self->dry_delay = self->lat_max < BN_DRY_RING - 1 ? self->lat_max : BN_DRY_RING - 1;
        if (self->latency) { *self->latency = (float)self->lat_max; }
    }

    /* Panel knobs: the TARGET of this cycle; the actual application is slewed
     * inside the chunk loop (a full-scale jump is spread over BN_KNOB_SLEW_S)
     * and the engine's matrices are rebuilt once per chunk while a pot moves.
     * The circuit's state survives every rebuild. */
    float kn_tgt[5];
    kn_tgt[0] = self->vol    ? *self->vol    : 0.7f;
    kn_tgt[1] = self->treble ? *self->treble : 0.5f;
    kn_tgt[2] = self->bass   ? *self->bass   : 0.5f;
    kn_tgt[3] = self->middle ? *self->middle : 0.5f;
    kn_tgt[4] = self->master ? *self->master : 1.0f;
    /* NEGATED comparisons: with the plain form a NaN on a knob survives both
     * ifs and reaches the engine's matrices. With the negated form a NaN falls
     * to the knob's default (KN_DEF). */
    static const float KN_DEF[5] = { 0.7f, 0.5f, 0.5f, 0.5f, 1.0f };
    for (int k = 0; k < 5; ++k) {
        if (!(kn_tgt[k] >= 0.0f))      { kn_tgt[k] = KN_DEF[k]; }
        else if (!(kn_tgt[k] <= 1.0f)) { kn_tgt[k] = 1.0f; }
    }
    const int low_tgt = (self->low_input && *self->low_input > 0.5f) ? 1 : 0;
    if (self->knobs_last[0] < 0.0f) {          /* first run: no ramp */
        for (int k = 0; k < 5; ++k) { self->knobs_last[k] = kn_tgt[k]; }
    }
    self->low_state = low_tgt;                 /* the jack moves at once, as a real replug */

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
     * step: the bias (a step moves the operating point at once; linear at fs_hi, BN_COND_RAMP_S) and the hum's
     * level (a step cuts the hum waveform mid-cycle; 10 ms at the host rate). Held still, each ramp sits exactly
     * on its port's value. The first run() after activate() starts them on the ports. */
    const double bias_tgt    = bn_ctl(self->bias, 0.5, 0.0, 1.0);
    const double hum_tgt     = bn_port01(self->hum);
    if (self->cond_from_ports) {
        self->bias_now = bias_tgt;
        self->hum_now = hum_tgt;
        self->cond_from_ports = 0;
    }
    const double dry_step = 1.0 / (0.010 * self->sample_rate);
    /* a start that found no operating point is retried once per run() call (a DC solve: never per sample) */
    self->ultra_failed = 0;
    /* presence follows its port at the pots' slew (a presence move rebuilds the engine's matrices, per chunk) */
    const double ultra_pres_tgt = bn_ctl(self->presence, 0.0, 0.0, 1.0);
    /* and Cabinet Feel the same way: a move rebuilds the 4x12's R, L and C, per chunk */
    const double ultra_cab_tgt = bn_ctl(self->cab_load, 0.0, 0.0, 1.0);
    /* --- THE w2 SEAM: power bypass and FX loop. Read ONCE per run.
     * Precedence: POWER > whole BYPASS > half bypasses; the top two act at
     * the base rate on whatever comes out of here, so there is nothing to
     * arbitrate inside the loop. */
    const double byp_pow_tgt = (self->bypass_power && *self->bypass_power > 0.5f) ? 1.0 : 0.0;
    const int    fx_on       = (self->fx_loop && *self->fx_loop > 0.5f);
    /* the card's loop is series only: the mix port is not read (kept so saved sessions load). Reading it put a mix
     * below 50 % on the loop-off latency tap while the card ran in series (-DBN_FX_MIX_PORT: the gate's arm) */
    const double fx_mix_tgt  = fx_on ? 1.0 : 0.0;
    const double fx_lvl_tgt  = bn_ctl(self->fx_return_lvl, 1.0, 0.0, 2.0);
    float* const fx_send = self->fx_send;
    const double eng_step = 100.0 / self->sample_rate;   /* = 1 / (10 ms x fs): the latency pad's tap crossfade per sample */
    const int    fx_wet   = fx_mix_tgt >= 0.5;               /* the return dominates (0 with the loop off) */

    /* POWER off and the ramp finished: exit here without touching the DSP.
     * NOTE: it comes BEFORE the processing — and it must restore the MXCSR,
     * or FTZ+DAZ would stay set on the host's audio thread. */
    if (power_off && self->dry_mix >= 1.0) {
        for (uint32_t i = 0; i < n_samples; ++i) {
            const double s = (double)in[i];
            out[i] = (float)bn_dry_push(self, isfinite(s) ? s : 0.0);
            (void)bn_pad_push(self, 0.0, fx_wet, eng_step);   /* the wet is silent while the plugin is off: the pad carries silence */
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
            if (fabs(ultra_pres_tgt - self->ultra_pres_now) > 1e-4 || fabs(ultra_cab_tgt - self->ultra_cab_now) > 1e-4) { moving_pre = 1; }
            /* MK81#202: the chunk ends on the pots' grid (knob_chunk cells counted from activate), not knob_chunk
             * samples from wherever the host block began - at host blocks of 1 / 7 / random sizes the staircase's
             * steps fell elsewhere (Roadcase's kit: -64 dBFS against blocks of 8192) */
            if (moving_pre && n > self->knob_chunk - self->knob_phase) { n = self->knob_chunk - self->knob_phase; }
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
            /* MK81#202: the pots step once per knob CELL, at its first sample, by a whole cell's slew, so their
             * trajectory is the same at every host block size */
            const int knob_edge = (self->knob_phase == 0);
            self->knob_phase += n;
            if (self->knob_phase >= self->knob_chunk) { self->knob_phase %= self->knob_chunk; }
#ifdef BN_NO_KNOB_GRID   /* the gate's negative arm only: every chunk starts a cell, wherever the host block cut it */
            self->knob_phase = 0;
#endif
            const float step = knob_edge ? (float)self->knob_chunk / (BN_KNOB_SLEW_S * (float)self->sample_rate) : 0.0f;
            int moving = 0;
            for (int km = 0; km < 5; ++km) {
                const float d2 = kn_tgt[km] - self->knobs_last[km];
                if (fabsf(d2) > 1e-4f) {
                    moving = 1;
                    if (!knob_edge) { continue; }
                    if (d2 >  step)      { self->knobs_last[km] += step; }
                    else if (d2 < -step) { self->knobs_last[km] -= step; }
                    else                 { self->knobs_last[km] = kn_tgt[km]; }
                }
            }
            {
                const double d3 = ultra_pres_tgt - self->ultra_pres_now;
                if (fabs(d3) > 1e-4) {
                    moving = 1;
                    self->ultra_pres_now = d3 > step ? self->ultra_pres_now + step
                                         : d3 < -step ? self->ultra_pres_now - step : ultra_pres_tgt;
                }
                const double d4 = ultra_cab_tgt - self->ultra_cab_now;
                if (fabs(d4) > 1e-4) {
                    moving = 1;
                    self->ultra_cab_now = d4 > step ? self->ultra_cab_now + step
                                        : d4 < -step ? self->ultra_cab_now - step : ultra_cab_tgt;
                }
                bn_ultra_sync_knobs(self);
            }
            knobs_moving = moving;
        }

        /* --- SILENCE FREEZE ---------------------------------------------------
         * With an input of exact zeros the engine would keep solving the
         * circuit to produce nothing, on every stopped track of a session.
         * It does not help while playing (a guitar always brings noise): it
         * makes a track at rest stop costing.
         *
         * Seeing zeros at the input is not enough, because the amp's tails
         * keep sounding. It only freezes when (a) the input is EXACT zeros,
         * (b) no knob moves and the loop is off, and (c) for half a second the
         * output and the circuit's state have stayed within the thresholds of
         * bn_quiet_eps / bn_quiet_level / bn_quiet_xeps. By then the state is
         * at its rest point and the frozen output (the mean of the chunk's
         * last min(n, 64) samples) is indistinguishable from computing on. Any non-zero sample
         * unfreezes at once; the engine's state was never touched. */
        int all_zero = 1;
        for (uint32_t i = 0; i < n; ++i) {
            if (in[done + i] != 0.0f) { all_zero = 0; break; }
        }
        /* with the loop active the return can carry signal even when the
         * input is zero: no freeze */
        if (!all_zero || knobs_moving || fx_on) {
            self->quiet_run = 0;
            self->frozen = 0;
        }
        if (self->frozen) {
            /* the trim keeps ramping: the user can move Output during silence */
            /* And the HUM keeps going too: without this it would cut out after
             * half a second of silence, exactly when a real amp's hum is heard. */
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
                const double wet = bn_pad_push(self, y_f * (double)self->trim_lin, fx_wet, eng_step);   /* the latency pad (see the main loop) */
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
            /* SANITISED: a NaN/Inf on the bus (a badly behaved upstream plugin,
             * a corrupt file, a device change) would reach the engine's table
             * lookups as an index: (int)NaN = INT_MIN, a crash on the audio
             * thread. One compare per sample at the base rate. */
            const double sin_ = (double)in[done + i];
            /* the hiss is added at the input: thermal noise is generated in the amp's input network. */
            self->buf_in[i]  = (isfinite(sin_) ? sin_ : 0.0)
                             + hiss_a * bn_hiss_sample(&self->hiss_rng);
        }

        { CZ_T0(cz0); bn_os_up(&self->c->os, self->buf_in, n, self->buf_hi); CZ_ADD(cz_up, cz0); }

        const size_t n_hi = (size_t)n * (size_t)self->os_factor;

        /* FX LOOP: the resampling pair runs while the loop is active or still
         * fading (mix or send with a ramp half way). It is zeroed when the
         * loop engages: the 10 ms fade covers the pair's start-up (its round trip
         * at fs_hi, < 1 ms). The return is SANITISED on reading, like the
         * input: a NaN from the bus cannot enter the engine. */
        const int fx_run = fx_on || self->fx_mix_now > 0.0 || self->fx_send_gain > 0.0;
        if (fx_run) {
            if (!self->fx_running) {
                bn_fxp_reset(&self->c->fxos);
#ifdef BN_FXC_KEEP   /* the complement keeps its delay estimate across loop switches */
                bn_fxc_rearm(&self->c->fxc);
#else
                bn_fxc_reset(&self->c->fxc);
#endif
                self->fx_running = 1;
            }
            const float* ret = self->fx_return;
            for (uint32_t i = 0; i < n; ++i) {
                const double r = ret ? (double)ret[done + i] : 0.0;
                self->fx_ret_lo[i] = isfinite(r) ? r : 0.0;
                self->fx_rcap[i] = self->fx_ret_lo[i];
            }
            bn_fxp_up(&self->c->fxos, self->fx_ret_lo, n, self->fx_ret_hi);
            bn_fxc_begin(&self->c->fxc);
        } else {
            self->fx_running = 0;
        }

        /* --- THE SUPPLY. A replica of the amp's supply (bn_psu) steps once per cell, loaded with the output
         * valves' plate and screen currents: the engine's own, the last closed cell's mean, once one has closed
         * since its start (ultra_cur_ok); before that, a MAP of those currents against the master-wiper envelope.
         * Its rectifier source drives the engine's B+ and its rectifier current the bias reservoir; the hum reads
         * the total DC load (psu_i_total: the output valves' currents plus the preamp's fixed draws).
         * PSU CURRENT MAP: mean plate (IPL) and screen (ISC) currents in A against w2's peak in V, measured on the
         * plugin's own valve models, one map per head and valve family, each family at its own bias (40 mA per
         * valve): the 50 W / 100 W ratio is not constant over w2, and the EL34 and 6550 curves differ in SHAPE
         * between their identical ends, so neither can be a scale factor. Row 0 is the idle load (psu_idle). */
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
        /* the US edition: the 6550 maps; the reissue is EL34 whatever the US switch says (its engine variant 4), so
         * it reads the EL34 rows (-DBN_RI_US_MAP: the gate's arm, the 6550 rows as before) */
        const int map_us = bn_ultra_want(self) & 1;   /* 1, 3 = US; 4 = the reissue */
        const double* MAP_IPL = self->head_now ? (map_us ? MAP_IPL_50_US  : MAP_IPL_50_UK)
                                               : (map_us ? MAP_IPL_100_US : MAP_IPL_100_UK);
        const double* MAP_ISC = self->head_now ? (map_us ? MAP_ISC_50_US  : MAP_ISC_50_UK)
                                               : (map_us ? MAP_ISC_100_US : MAP_ISC_100_UK);
        /* the supply ticks once per CELL (see psu_grid in BritishNoise), at its first chunk */
        const int tick = (self->psu_phase == 0);
        const size_t phase_hi = (size_t)self->psu_phase * (size_t)(self->os_factor > 0 ? self->os_factor : 1);
        if (tick) {
        double ipl = MAP_IPL[6], isc = MAP_ISC[6];
        {
            /* The map is read with the FAST (5 ms) envelope: the valves' current follows the drive within a
             * cycle and is back at rest ~12 ms after an abrupt cut, where a slow envelope would hold the map's
             * current, and the sag, up for far longer. Replaced below by the engine's own currents. */
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
        /* the supply's load is the engine's own output valves, the last closed cell's mean, once one has closed */
        if (self->ultra_cur_ok) { ipl = self->ultra_ipl; isc = self->ultra_isc; }
        /* The preamp's DC draws, from the factory voltage table: inverter 2.5 mA
         * from ITS rail · V2 (V2a 1.22 + follower 1.67) 2.9 mA from X · V1a+V1b
         * 0.91 mA from XP. */
        CZ_T0(czp);
        bn_psu_step(&self->psu, (double)cell_len / self->sample_rate,
                    ipl, isc, 0.0025, 0.0029, 0.00091);
        bn_psu_step(&self->psu_idle, (double)cell_len / self->sample_rate,
                    MAP_IPL[0], MAP_ISC[0], 0.0025, 0.0029, 0.00091);
        if (self->urest_ok) {
            bn_psu_step(&self->psu_urest, (double)cell_len / self->sample_rate,
                        self->urest_ipl, self->urest_isc, 0.0025, 0.0029, 0.00091);
        }
        CZ_ADD(cz_psu, czp);
        /* The era changes the reservoir capacitor; re-applied only when the port has moved. */
        {
            /* the era is the 100 W originals' (their reservoir); the 50 W head and the reissue have one supply, held
             * at the default (1, the old way) whatever the port says. An unconnected port reads its default too */
            const int era_q = bn_head_port(self) == 0 ? (self->era ? *self->era > 0.5f : 1) : 1;
            if (era_q != self->era_now) {
                bn_psu_set_era(&self->psu, era_q);
                bn_psu_set_era(&self->psu_idle, era_q);
                bn_psu_set_era(&self->psu_urest, era_q);
                self->era_now = era_q;
#ifndef BN_ERA_SYNC_LATE   /* the gate's negative arm: the engine hears the era at the next chunk, as before */
                bn_ultra_sync_knobs(self);   /* the engine's reservoir moves in the same cell as the supply's (MK81#202) */
#endif
            }
        }
        self->psu_i_total = ipl + isc + 0.0025 + 0.0029 + 0.00091;
        /* The ripple implied by THAT current, kept per cell; the hum is added after the silence detector. With half
         * the reservoir the ripple doubles (x2.0 measured, idle and at full load), and the averaged supply model has
         * no ripple, so the factor is applied here: the old way hums 6 dB more. Only on the 100 W head: the 50 W
         * head keeps its reservoir in both eras (bn_psu.c). */
        self->hum_vpp = bn_hum_ripple_vpp(self->psu_i_total, self->head_now)
                      * ((self->era_now && !self->head_now) ? 2.0 : 1.0);
        const size_t cell_hi = (size_t)cell_len * (size_t)(self->os_factor > 0 ? self->os_factor : 1);
        {
            const double rsag = self->head_now ? 48.6 : 25.0;   /* ohm: the RSAG of rq's htraw point (the measurement builds' rail) */
            const double ich  = self->psu.vsrc > self->psu.vc23 ? (self->psu.vsrc - self->psu.vc23) / self->psu.rchg : 0.0;
            const double ichi = self->psu_idle.vsrc > self->psu_idle.vc23 ? (self->psu_idle.vsrc - self->psu_idle.vc23) / self->psu_idle.rchg : 0.0;
            const double rq_prev = self->rq_now;
            self->rq_now = (self->psu.vc23 + rsag * ich) / (self->psu_idle.vc23 + rsag * ichi);
            self->rq_cell0 = rq_prev;
            self->drq_hi = (self->rq_now - rq_prev) / (double)(cell_hi ? cell_hi : 1);
            /* The bias reservoir's target for this cell: the HT current above the engine's own rest current
             * (psu_urest), the circuit's g (per variant and era) and its lag coefficient at this fs_hi.
             * The reissue runs the 100 W UK's law (bv 0; `& 3` keeps the index inside the 4-row tables) at the
             * old way's supply (be 1). */
            const int bv = self->ri_now ? 0 : 2 * self->head_now + (bn_ultra_want(self) & 1);
            const int be = (self->era_now || self->ri_now) ? 1 : 0;
            const double ichu = (self->urest_ok && !self->oldrest && self->psu_urest.vsrc > self->psu_urest.vc23)
                                ? (self->psu_urest.vsrc - self->psu_urest.vc23) / self->psu_urest.rchg
                                : (self->urest_ok && !self->oldrest) ? 0.0 : ichi;
            self->bht_zss = 1.0 - BN_ULTRA_BIASHT_G[bv & 3][be] * (ich - ichu);
            self->bht_a = exp(-1.0 / (self->sample_rate * (double)(self->os_factor > 0 ? self->os_factor : 1)
                                      * BN_ULTRA_BIASHT_TAU[bv & 3][be]));
            {   /* the B+ the engine's netlist sees at its htraw (bn_ultra_u1_psu), ramped linearly over the cell */
                const double u1_prev = self->u1_now;
                self->u1_now = bn_ultra_u1_psu(self, bv);
                self->u1_cell0 = u1_prev > 0.0 ? u1_prev : self->u1_now;
                self->du1_hi = (self->u1_now - self->u1_cell0) / (double)(cell_hi ? cell_hi : 1);
            }
        }
        }   /* tick */

        /* The 100 W / 50 W head. The supply follows the head too (switching only the power stage would make
         * the 50 W sag like a 100 W — and the sag is what tells the two apart when playing). */
        {
            const int head_q = (bn_head_port(self) == 1);
            const int ri_q = (bn_head_port(self) == 2);   /* the reissue's supply (the old way's reservoir + its DC loads) */
            if (ri_q != self->ri_now) {
                bn_psu_set_ri(&self->psu, ri_q);
                bn_psu_set_ri(&self->psu_idle, ri_q);
                bn_psu_set_ri(&self->psu_urest, ri_q);
                self->ri_now = ri_q;
            }
            if (head_q != self->head_now) {
                bn_psu_set_head(&self->psu, head_q);
                bn_psu_set_head(&self->psu_idle, head_q);
                bn_psu_set_head(&self->psu_urest, head_q);
                self->head_now = head_q;
            }
        }
        /* PREAMP bypass — crossfade w2 <-> input, 10 ms ramp at fs_hi.
         * The preamp KEEPS running (its state stays warm: no click on return). */
        const double byp_pre_tgt = (self->bypass_preamp && *self->bypass_preamp > 0.5f) ? 1.0 : 0.0;
        const double byp_pre_step = 1.0 / (0.010 * self->sample_rate
                                    * (double)(self->os_factor > 0 ? self->os_factor : 1));
        const double seam_step = byp_pre_step;   /* same 10 ms ramp at fs_hi */
        const double cond_step = 1.0 / (BN_COND_RAMP_S * self->sample_rate
                                        * (double)(self->os_factor > 0 ? self->os_factor : 1));
        /* u[0]: the oversampled input at this fs_hi sample; u[1] is unused */
        double u[2] = { 0.0, 0.0 };
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
            /* the bias ramp (and its unread presence / Cabinet Feel twins) advance first: the engine reads the bias */
            bn_ramp_to(&self->bias_now, bias_tgt, cond_step);
            if (self->byp_pre_mix < byp_pre_tgt) {
                self->byp_pre_mix += byp_pre_step;
                if (self->byp_pre_mix > byp_pre_tgt) { self->byp_pre_mix = byp_pre_tgt; }
            } else if (self->byp_pre_mix > byp_pre_tgt) {
                self->byp_pre_mix -= byp_pre_step;
                if (self->byp_pre_mix < byp_pre_tgt) { self->byp_pre_mix = byp_pre_tgt; }
            }
            /* The preamp bypass fades w2 toward the input (u[0]) with the same level convention as the FX return
             * (x BN_STAGE_GAIN: 1.0 from the host = 60 V at the wiper): with the preamp AND the power stage bypassed
             * the output is the input, like the whole-plugin bypass, at the same declared latency. */
            /* the loop's ramps and its return at this sample, ahead of the engine, which takes the return */
            double ret_g = 0.0;
            if (fx_run) {
                bn_ramp_to(&self->fx_mix_now, fx_mix_tgt, seam_step);
                bn_ramp_to(&self->fx_lvl_now, fx_lvl_tgt, 2.0 * seam_step);
                /* send and return are at PREAMP LEVEL (w2 / BN_STAGE_GAIN): the
                 * return is put back into w2 volts on entry */
                /* the return plus what the host could not carry (bn_fx_comp.h): 0 until the estimator
                 * locks on a near-linear effect, exact at unity once it does. The complement takes the
                 * user's return level like the return itself: it is the part of the RETURN the host dropped,
                 * so a return level of 0 must leave the power stage with nothing of the preamp, complement
                 * included. */
                const double ret_c = self->fx_ret_hi[i] + bn_fxc_sample(&self->c->fxc, i);
                ret_g = ret_c * self->fx_lvl_now * BN_STAGE_GAIN;
            }
            double w2v_u = 0.0, x0_u = 0.0;
            {
                /* a head / edition switch runs through silence: fade out over 10 ms, re-start on the new
                 * variant's operating point (a DC solve), fade back in. If the ports return to the live
                 * variant before the fade-out ends, the gain simply ramps back up. */
                const int vw = bn_ultra_want(self);
                if (vw != self->ultra_v) {
                    bn_ramp_to(&self->ultra_gain, 0.0, seam_step);
                    if (self->ultra_gain <= 0.0 && !self->ultra_failed) {
                        /* no operating point: the amp stays muted (ultra_v < 0) and the start is retried next run() */
                        if (bn_ultra_start(self, vw, 0) == 0) {
                            self->ultra_hold = 0;
                            /* the effects card's system and its memory belong to the OLD variant (the variants differ in
                             * topology): processing it as the new one stalled the audio thread and burst (+48 dBFS). It
                             * is dropped; with the loop still on, the ON path below starts the new variant's card from the
                             * re-started engine, with its wait and crossfade, inside this muted hold */
                            bn_card_drop(self);
                        }
                    }
                } else if (self->ultra_hold > 0) {
                    --self->ultra_hold;
                } else if (self->ultra_gain < 1.0) {
                    bn_ramp_to(&self->ultra_gain, 1.0, seam_step);
                }
                if (self->ultra_v >= 0) {
                    const int v = self->ultra_v;
                    /* rq at this sample (measurement laws only); uu1 the B+ source, uu2 the bias source (the bias reservoir
                     * following the HT load) */
                    const double rail = self->rq_cell0 + self->drq_hi * (double)(phase_hi + i + 1);
                    BnUltraTap tap;
                    const double uu1 = BN_B8(self) ? self->psu.vsrc : BN_B7(self) ? bn_ultra_vht(v) : self->oldb ? bn_ultra_vht(v) * rail
                                     : self->u1_now > 0.0 ? self->u1_cell0 + self->du1_hi * (double)(phase_hi + i + 1)
                                                          : bn_ultra_u1_psu(self, v);
                    self->bht_z = self->bht_zss + (self->bht_z - self->bht_zss) * self->bht_a;   /* the bias reservoir's lag */
                    const double uu2 = (BN_B7(self) || BN_B9(self)) ? bn_ultra_vbias(v, self->bias_now) : self->oldlaw ? bn_ultra_vbias(v, self->bias_now) * (1.0 - ((v & 2) ? 1.0 : 11.35 / 32.43) * (1.0 - rail))
                                                    : bn_ultra_vbias(v, self->bias_now) * self->bht_z;
                    /* THE PREAMP BYPASS (the !card_done path below). While the bypass acts (or fades), the SEND system
                     * (src/prepig/: the preamp + phase inverter with the output valves' grids conducting, its inverter rail
                     * and NFB return read from the heard engine's nodes one engine sample late) plays the loop-free wiper,
                     * and the heard engine's PI, its C9 top detached, gets (1 - bypass) w2 + bypass x input x G, w2 the
                     * send system's; the PI's current still loads the heard engine's own preamp.
                     * With the bypass at 0 the attached engine alone runs, byte-identical to no bypass. The send system is
                     * born the sample the bypass leaves 0: its matrices at the present knobs (rebuilt only if they moved
                     * since), every node voltage and capacitor history it shares with the heard engine COPIED from it (a
                     * settled state, no DC solve), then the heard engine is detached. Idle again, the heard engine takes
                     * the send system's preamp state back and is re-attached. */
                    /* THE FX LOOP: the reissue's own effects card, series only (fx_mix is not applied). ON: the card
                     * variant starts at its DC point and its return half is reset; during the wait (card_wait, the card
                     * still inaudible) the heard engine plays and the send comes from the send twin; at the end of the
                     * wait the card takes every node and capacitor it shares with the heard engine (bn_card_seed_on) and
                     * the crossfade starts. OFF: the heard engine takes them back (bn_card_seed_off: exact, the round trip
                     * is bit-identical). A LEVEL move hands over through the heard engine. The host's return (ret_g: the
                     * jack voltage, the FX complement covering the host's latency) drives the return half; its output is
                     * the card system's fourth source. */
                    const double by = self->byp_pre_mix;
                    double t8;
                    double w2_dry = 0.0;
                    int card_done = 0;
                    {
                        /* unconnected or not a number: the default, Pedal (1) */
                        const int lvl = (self->fx_level && isfinite(*self->fx_level)) ? (*self->fx_level > 0.5f ? 1 : 0) : 1;
                        const int fi = self->os_factor >= 4 ? 1 : 0;
                        const double fs_e = self->sample_rate * (double)(self->os_factor > 0 ? self->os_factor : 1);
                        /* the crossfade's per-sample step */
                        const double xf_step = 1.0 / (BN_CARD_XF * fs_e);
                        if (fx_on && (!self->card_on || self->card_fading_out)) {
                            if (self->card_fading_out) {
                                /* ON again while the card fades out: it is still running, so it simply fades back in */
                                self->card_fading_out = 0; self->card_wait = 0;
                            } else if (bn_ultra_init(self->ultra_card, BN_ULTRA_FX(v, lvl), fs_e, &self->ultra_kn, uu1, uu2) == 0) {
#ifndef BN_CARD_LATE
                                bn_card_seed_on(self->ultra_card, self->ultra, v, lvl);
                                bn_ultra_set_knobs(self->ultra_card, BN_ULTRA_FX(v, lvl), &self->ultra_kn);
#endif
                                (void)cr_ret_reset(self->card_ret[lvl][fi]);
#ifdef BN_SEND_TWIN
                                (void)cs_send_reset(self->card_snd[lvl][fi]); self->card_w2p = self->card_w2pp = 0.0;
#endif
                                self->card_on = 1; self->card_lv = lvl;
                                self->card_g = 0.0;
                                /* the wait at ON, in engine samples: the HOST block the return comes back late (a host closes
                                 * the loop one of its blocks late; the chunk n can be much shorter, cut at a supply cell or
                                 * while a knob moves), the send's 10 ms ramp, and a margin of 64 base samples */
                                const double wait_blk = (double)n_samples;
                                self->card_wait = (long)(wait_blk * self->os_factor + 0.010 * fs_e + 64.0 * self->os_factor);
#ifdef BN_CARD_MIDFADE
                                {   /* the crossfade (two engines) is CENTRED on a chunk boundary, so its cost splits evenly over two
                                     * chunks instead of loading one: lh = half the crossfade in engine samples, modulo the chunk;
                                     * the extra wait moves the fade's centre onto the next boundary */
                                    const long nh = (long)n * (self->os_factor > 0 ? self->os_factor : 1);
                                    const long lh = (long)(0.5 * BN_CARD_XF * fs_e + 0.5) % (nh > 1 ? nh : 1);
                                    if (nh > 1) { self->card_wait += (((nh - lh) - ((long)i + self->card_wait) % nh) % nh + nh) % nh; }
                                }
#endif
                                self->card_wait0 = self->card_wait;
                            }
                        } else if (!fx_on && self->card_on && !self->card_fading_out) {
                            /* the heard engine takes the card's state and fades in; the card keeps running meanwhile. From a
                             * card fully in, the fade waits (the card alone, as while ON) until it is centred on a block
                             * boundary, like the fade at ON; from a fade still in progress it starts at once */
                            const long nh = (long)n * (self->os_factor > 0 ? self->os_factor : 1);
                            const long lh = (long)(0.5 * BN_CARD_XF * fs_e + 0.5) % (nh > 1 ? nh : 1);
                            const long wt = (self->card_g >= 1.0 && nh > 1) ? (((nh - lh) - (long)i % nh) % nh + nh) % nh : 0;
                            if (self->card_g <= 0.0) {
                                bn_card_seed_off(self->ultra, self->ultra_card, v, self->card_lv);
                                bn_ultra_set_knobs(self->ultra, v, &self->ultra_kn);
                                self->card_on = 0;
                            } else {
                                self->card_fading_out = 1; self->card_wait = wt;
                                if (wt == 0) { bn_card_seed_off(self->ultra, self->ultra_card, v, self->card_lv); bn_ultra_set_knobs(self->ultra, v, &self->ultra_kn); }
                            }
                        } else if (self->card_fading_out && self->card_wait > 0) {
                            /* the fade at OFF waiting for its place: the heard engine is seeded from the card when it starts */
                            if (--self->card_wait == 0) { bn_card_seed_off(self->ultra, self->ultra_card, v, self->card_lv); bn_ultra_set_knobs(self->ultra, v, &self->ultra_kn); }
                        } else if (self->card_on && !self->card_fading_out && lvl != self->card_lv) {
                            bn_card_seed_off(self->ultra, self->ultra_card, v, self->card_lv);
                            if (bn_ultra_init(self->ultra_card, BN_ULTRA_FX(v, lvl), fs_e, &self->ultra_kn, uu1, uu2) == 0) {
                                bn_card_seed_on(self->ultra_card, self->ultra, v, lvl);
                                bn_ultra_set_knobs(self->ultra_card, BN_ULTRA_FX(v, lvl), &self->ultra_kn);
                                (void)cr_ret_reset(self->card_ret[lvl][fi]);
#ifdef BN_SEND_TWIN
                                (void)cs_send_reset(self->card_snd[lvl][fi]); self->card_w2p = self->card_w2pp = 0.0;
#endif
                                self->card_lv = lvl;
                            } else {
                                bn_ultra_set_knobs(self->ultra, v, &self->ultra_kn);
                                self->card_on = 0; self->card_g = 0.0;
                            }
                        }
#if defined(BN_CARD_LATE) && defined(BN_SEND_TWIN)
                        /* while the card is still INAUDIBLE (the wait at ON, card_g 0) the card system does not run: the heard
                         * engine plays, the send comes from the twin on ITS w2, the return stage warms up on the host's return;
                         * at the end of the wait the card system is seeded from the heard engine and the crossfade starts (two
                         * engines for BN_CARD_XF only, not for the whole wait) */
                        if (self->card_on && !self->card_fading_out && self->card_g <= 0.0 && self->card_wait > 0) {
                            const double jack = fx_run ? ret_g : 0.0;
                            self->card_fxr = CARD_RET_TICK(self->card_ret[self->card_lv][fi], jack);
                            t8 = bn_ultra_process(self->ultra, v, u[0], uu1, uu2, 0.0, &tap);
                            w2_dry = tap.w2;
                            bn_card_send_twin(self, fi, tap.w2);
                            if (--self->card_wait == 0) {
                                bn_card_seed_on(self->ultra_card, self->ultra, v, self->card_lv);
                                bn_ultra_set_knobs(self->ultra_card, BN_ULTRA_FX(v, self->card_lv), &self->ultra_kn);
                            }
                            card_done = 1;
                        } else
#endif
                        if (self->card_on) {
                            const double jack = fx_run ? ret_g : 0.0;
                            const double fxr = CARD_RET_TICK(self->card_ret[self->card_lv][fi], jack);
                            self->card_fxr = fxr;
                            { CZ_T0(cz0); t8 = bn_ultra_process(self->ultra_card, BN_ULTRA_FX(v, self->card_lv), u[0], uu1, uu2, fxr, &tap);
                              CZ_ADD(cz_ultra, cz0); }
                            w2_dry = tap.w2;
#if defined(BN_SEND_TWIN)
                            /* the send from the TWIN, the card's send half alone at BN_SEND_K x the engine rate: the zeners'
                             * clamp pulses charge C8 in less than an engine sample, where the whole-amp system's own send is
                             * decided by its Newton's stopping rule. The system keeps its own send half for w2's load. w2
                             * inside the sample on the parabola through the last three samples; the last sub-step is the
                             * value at this sample */
                            bn_card_send_twin(self, fi, tap.w2);
#else
                            self->card_send = tap.fxc;
#endif
                            card_done = 1;
                            /* the transition: the heard (loop-free) engine runs too and the output crossfades */
                            if (self->card_fading_out) {
                                if (self->card_wait == 0) { self->card_g -= xf_step; }
                                if (self->card_g <= 0.0) { self->card_g = 0.0; self->card_on = 0; self->card_fading_out = 0; }
                            } else if (self->card_wait > 0) {
                                --self->card_wait;
                            } else if (self->card_g < 1.0) {
                                self->card_g += xf_step;
                                if (self->card_g > 1.0) { self->card_g = 1.0; }
                            }
                            if (self->card_g < 1.0) {
                                BnUltraTap th;
                                const double t8h = bn_ultra_process(self->ultra, v, u[0], uu1, uu2, 0.0, &th);
                                const double g = 0.5 - 0.5 * cos(3.14159265358979323846 * self->card_g);   /* raised cosine */
                                t8 = t8h + g * (t8 - t8h);
                                w2_dry = th.w2 + g * (w2_dry - th.w2);
                                /* the supply sees the amp that is heard: the plate and screen currents blend the same way */
                                tap.ipl = th.ipl + g * (tap.ipl - th.ipl); tap.isc = th.isc + g * (tap.isc - th.isc);
                            }
                        }
                    }
                    if (!card_done) {
                    /* with the card the loop is series and fx_mix is not applied (fx = 0): once the card has faded out,
                     * the mix's 10 ms ramp down must not start the send system; only the preamp bypass uses it */
                    const double fx = 0.0;
                    const int loop_acts = fx > 0.0 || by > 0.0 || byp_pre_tgt > 0.0;
                    /* the heard engine's nodes after its LAST sample (only read while the loop acts) */
                    const double* nv = (loop_acts || self->send_on) ? bn_ultra_nodes(self->ultra, v) : NULL;
                    if (loop_acts && !self->send_on) {
                        if (bn_prepig_knobs_differ(&self->ultra_kn, &self->send_kn)) {
                            const BnPrepigKnobs pk = bn_prepig_knobs(&self->ultra_kn);
                            bn_prepig_set_knobs(self->ultra_send, v, &pk);
                            self->send_kn = self->ultra_kn;
                        }
                        int nh = 0;
                        bn_prepig_seed(self->ultra_send, v, nv, bn_ultra_state(self->ultra, v, &nh));
                        bn_ultra_detach(self->ultra, v, 1);
                        self->send_on = 1;
                    } else if (!loop_acts && self->send_on) {
                        BnUltraArrays ua;
                        bn_ultra_arrays(self->ultra, v, &ua);
                        bn_prepig_restore(self->ultra_send, v, ua.V, ua.Vp1, ua.Vp2, ua.h);
                        bn_ultra_detach(self->ultra, v, 0);   /* C9 hangs from the wiper again: a rebuild, no stale LU */
                        self->send_on = 0;
                    }
                    if (self->send_on) {
                        { CZ_T0(cz0); w2_dry = bn_prepig_process(self->ultra_send, v, u[0], uu2, nv); CZ_ADD(cz_send, cz0); }
                        double top = (1.0 - fx) * (w2_dry + by * (u[0] * BN_STAGE_GAIN - w2_dry))
                                   + fx * (ret_g + self->c->w2_dc);
                        double xh = u[0];
                        { CZ_T0(cz0); t8 = bn_ultra_process(self->ultra, v, xh, uu1, uu2, top, &tap); CZ_ADD(cz_ultra, cz0); }
                    } else {
                        { CZ_T0(cz0); t8 = bn_ultra_process(self->ultra, v, u[0], uu1, uu2, 0.0, &tap); CZ_ADD(cz_ultra, cz0); }
                        w2_dry = tap.w2;
                    }
                    }
                    x0_u = self->ultra_gain * t8;
                    /* the wiper as the seam sees it (the loop-free amp's: the send never depends on the return), the
                     * preamp bypass fading it to the input */
                    w2v_u = w2_dry + by * (u[0] * BN_STAGE_GAIN - w2_dry);
                    self->ultra_ipl_acc += tap.ipl; self->ultra_isc_acc += tap.isc; ++self->ultra_ncur;
                }
            }
            const double w2v = w2v_u;
            /* --- THE w2 SEAM -----------------------------------------------
             * w2ac = DC-free w2 (the seam's own blocker): what GOES OUT through
             * the send (card off) and the power bypass's dry path. w2p is what
             * drives the power stage and the envelope: with the loop mixed in,
             * the return plus w2's CONSTANT rest value (w2_dc, ~0 V), never "w2
             * minus what the blocker removes" (that would leak the dry signal
             * into a series loop).
             * ⚠️ The mixes are written as `x + m*(a - b)`: with m = 0 exactly
             * they give `x + 0.0 = x` bit for bit, so the path is unchanged
             * with the ports at their defaults. */
            const double w2ac = w2v - self->c->seam_dc_x1 + self->c->seam_a * self->c->seam_dc_y1;
            self->c->seam_dc_x1 = w2v;
            self->c->seam_dc_y1 = w2ac;
            double w2p = w2v, w2p_ac = w2ac;
            if (fx_run) {
                w2p    = w2v  + self->fx_mix_now * ((ret_g + self->c->w2_dc) - w2v);
                w2p_ac = w2ac + self->fx_mix_now * (ret_g - w2ac);
                self->fx_send_hi[i] = w2ac * (1.0 / BN_STAGE_GAIN);
            }
            /* with the card in the loop the send is the card's send node and the power stage is driven by the card's
             * return half (series only): the envelope below follows that */
            if (self->card_on) {
                self->fx_send_hi[i] = self->card_send * (1.0 / BN_STAGE_GAIN);
                /* while the switch crossfades, the envelope follows the heard mix of the two drives, as the output does */
                const double g = self->card_g >= 1.0 ? 1.0 : 0.5 - 0.5 * cos(3.14159265358979323846 * self->card_g);
                w2p = w2p + g * (self->card_fxr - w2p); w2p_ac = w2p_ac + g * (self->card_fxr - w2p_ac);
            }
            /* the master envelope (PSU map and bias) follows what REALLY
             * drives the power stage: with the loop, the return */
            env_fast *= env_k;
            if (env_fast < 1e-9) { env_fast = 0.0; }          /* no denormal tail in silence */
            if (fabs(w2p) > env_fast) { env_fast = fabs(w2p); }
            if (env_fast > pk_fast) { pk_fast = env_fast; }
            const double y0 = x0_u;   /* ULTRA's secondary carries no DC: no output blocker */
            const double wet_hi = y0 * (1.0 / BN_POWER_GAIN);
            /* POWER bypass. The power stage KEEPS running (warm state, no
             * click on return); at the output t8 is faded toward the same
             * sample's DC-free seam signal at preamp level: w2, or the card's
             * return once the loop's card has taken over. */
            {
                const double dry_hi = w2p_ac * (1.0 / BN_STAGE_GAIN);
                bn_ramp_to(&self->byp_pow_mix, byp_pow_tgt, seam_step);
                self->buf_hi[i] = wet_hi + self->byp_pow_mix * (dry_hi - wet_hi);
            }
        }
        /* FX LOOP: the send goes down to the host rate and out with its fade */
        if (fx_run) {
            bn_fxp_down(&self->c->fxos, self->fx_send_hi, n, self->fx_ret_lo);
            bn_fxc_end(&self->c->fxc, self->fx_send_hi, self->fx_ret_lo, self->fx_rcap, n);
        }

        /* The master-wiper envelope the PSU's current map reads — the per-sample follower
         * above (release 5 ms: shorter than a low-E period). */
        self->env_fast = env_fast;
        /* the cell's MAXIMUM of the follower, not its value at the cell's end: on a
         * sustained tone that is the tone's peak whatever the block. The supply reads it
         * when the cell closes, with the engine's mean currents over the cell. */
        if (pk_fast > self->pk_cell) { self->pk_cell = pk_fast; }
        self->psu_phase += n;
        if (self->psu_phase >= cell_len) {
            self->w2_peak_fast = self->pk_cell;
            if (self->ultra_ncur > 0) {
                self->ultra_ipl = self->ultra_ipl_acc / (double)self->ultra_ncur;
                self->ultra_isc = self->ultra_isc_acc / (double)self->ultra_ncur;
                self->ultra_cur_ok = 1;
            }
            self->ultra_ipl_acc = 0.0; self->ultra_isc_acc = 0.0; self->ultra_ncur = 0;
            self->pk_cell = 0.0;
            self->psu_phase = 0;
        }

        { CZ_T0(cz0); bn_os_down(&self->c->os, self->buf_hi, n, self->buf_out); CZ_ADD(cz_down, cz0); }

        /* has the output gone still? (see the silence freeze above) */
        if (all_zero) {
            double dmax = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const double d = fabs(self->buf_out[i] - self->quiet_last);
                if (d > dmax) { dmax = d; }
                const double m = fabs(self->buf_out[i]);
                /* ⛔ Do NOT simplify to a bare `if (m > bn_quiet_level())`: that is NOT
                 * equivalent — with dmax already above the level (the `d` line above can
                 * leave it there) the two forms differ. */
                if (m > dmax && m > bn_quiet_level(self)) { dmax = 1e300; }
            }
            /* and the circuit state? (see bn_quiet_xeps) */
            /* the engine's own state (its capacitors' histories, all of them) must be still too, relative to its
             * magnitude; the freeze keeps the engine's last state untouched */
            const int ultra_q = self->ultra_v >= 0;
            if (ultra_q) {
                int nu = 0;
                const double* hu = bn_ultra_state(self->ultra, self->ultra_v, &nu);
                const int nx = nu < BN_QUIET_NX ? nu : BN_QUIET_NX;
                double dxmax = 0.0, xmax = 1.0;
#if BN_QUIET_NX >= 96
                if (nu > BN_QUIET_NX) { dxmax = 1e300; }   /* more states than the watch holds: never still */
#endif
                for (int q = 0; q < nx; ++q) {
                    const double d2 = fabs(hu[q] - self->x_snap[q]);
                    if (d2 > dxmax) { dxmax = d2; }
                    if (fabs(hu[q]) > xmax) { xmax = fabs(hu[q]); }
                    self->x_snap[q] = hu[q];
                }
                if (dxmax > bn_quiet_xeps(self) * xmax) { dmax = 1e300; }
            } else {
                dmax = 1e300;   /* no engine running (a start that found no operating point): nothing to freeze */
            }
            /* and the supply? The freeze stops the supply replica and the bias reservoir, which a real silence keeps
             * recovering for seconds: frozen early, the note after the silence starts on a supply that never
             * recovered. A per-block delta would pass a slow drift, so the supply is compared with ITSELF at the start
             * of the still run (q_psu, taken when the run starts): over the whole half second it must not have moved
             * more than BN_QUIET_PSU_K x BN_QUIET_TOL of each state (the bias reservoir likewise). */
            {
                const BnPsu* a = &self->psu;
                const double va[7] = { a->vc23, a->il, a->vc22, a->vpi, a->vx, a->vxp, self->bht_z };
                if (self->quiet_run == 0) {
                    for (int q = 0; q < 7; ++q) { self->q_psu[q] = va[q]; }
                } else {
                    for (int q = 0; q < 7; ++q) {
                        const double m = fabs(self->q_psu[q]) > 1.0 ? fabs(self->q_psu[q]) : 1.0;
                        const double dq = fabs(va[q] - self->q_psu[q]) / (BN_QUIET_PSU_K * BN_QUIET_TOL * m);
                        if (dq > 1.0) { dmax = 1e300; }
                    }
                }
            }
            if (dmax < bn_quiet_eps(self)) {
                self->quiet_run += (int)n;
                if (self->quiet_run >= (int)(0.5 * self->sample_rate)) {
                    self->frozen = 1;
                    { double acc = 0.0; const uint32_t m0 = n > 64 ? n - 64 : 0; uint32_t cnt = 0;
                      for (uint32_t q = m0; q < n; ++q) { acc += self->buf_out[q]; ++cnt; }
                      self->frozen_out = cnt ? acc / (double)cnt : self->buf_out[n - 1]; }
                }
            } else {
                self->quiet_run = 0;
            }
        }
        self->quiet_last = self->buf_out[n - 1];

        /* Mains hum. Out of the path when the control is at 0, the default. */
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
            /* the latency pad delays the WHOLE wet (amp + hum, after the output trim): a 2x path is delayed sample for
             * sample, trim ramps and hum included, to the largest configuration's latency */
            const double wet = bn_pad_push(self, y_amp * (double)self->trim_lin, fx_wet, eng_step);
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

#define BN_RUN_FN run
static const LV2_Descriptor descriptor = {
    BN_URI,
    instantiate,
    connect_port,
    activate,
    BN_RUN_FN,
    deactivate,
    cleanup,
    extension_data
};

LV2_SYMBOL_EXPORT const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : NULL;
}
