/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_mna.h @ d75dee0, sha256 677189cd27e0c46478967161d758912092fd443ba129f3a5c9196beb47b5dce0 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* mks_mna.h - the MK Silver preamp's sparse nodal engine (see mks_mna.c). */
#ifndef MKS_MNA_H
#define MKS_MNA_H

#include "nl_mna_device.h"
#include "mks_preamp_mna.h"
#include "nl_mna_pi_load.h"
#ifdef MKS_DEVICE_MK81
#if MKS_MK81_US
#define NL_MNA_6550 1   /* the export edition's 6550s */
#endif
#include "nl_mna_valves.h"   /* the valves as port blocks and their per-caller settings (NlMnaValveCfg) */
#if defined(MKS_N_FXZ) && MKS_N_FXZ
/* nlcore#25 (MK81#165): MKS_N_FXZ zener ports after the valves, each nl_dev_zener.h's law with its depletion charge;
 * their parameters come from the generated circuit (mks_fxz_params). No circuit without MKS_N_FXZ compiles any of it. */
#include "nl_dev_zener.h"
#endif
#endif

typedef struct {
    double fs;
    double V[MKS_MNA_N + MKS_MNA_NF];   /* node voltages: unknowns in elimination order, then the fixed nodes */
    double Vp1[MKS_MNA_N], Vp2[MKS_MNA_N];   /* the two previous samples' nonlinear nodes, for the start extrapolation */
    double gc[MKS_MNA_NC];              /* 2*C*fs per capacitor */
    double h[MKS_MNA_NC];               /* each capacitor's trapezoidal history: i = gc*v - h */
    double gam[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];   /* the windings' Gamma = Linv / (2 fs) */
    double hl[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];                  /* their histories: i = Gamma.v + hl */
#ifdef MKS_GAM_COLS
    double gamT[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];  /* Gamma transposed: the commit runs by columns (vectorises) */
#endif
#ifdef MKS_SPARSE_GAM
    /* MK81#104 (cost-pass row 27 on the windings): Gamma's nonzero entries per row, rebuilt with Gamma; the
     * per-sample commit skips the exact zeros (a skipped term adds +-0 to a sum that never is -0: bit-identical) */
    unsigned char gnz_n[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    unsigned char gnz_j[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];
#endif
    double m[MKS_MNA_NSLOT];            /* G + 2*fs*C with its LINEAR pivots factored (constant per rate) */
    double S[MKS_MNA_NSCH];             /* the Schur block over the nonlinear nodes, unfactored */
    double S0[MKS_MNA_NSCH];            /* the Schur block of the LINEAR network alone, factored: the DK's port residual */
    double w[MKS_MNA_NSCH];             /* S + the device Jacobian, factored */
    double y[MKS_MNA_N];
    double yp[MKS_MNA_N];               /* the last sample's y, for the tangent predictor */
#ifdef MKS_PREDERR
    double pe[MKS_MNA_N], vpred[MKS_MNA_N];   /* MK81#104: the last predictor error, added to the next tangent start */
#endif
    int    predict;                     /* tangent predictor: start from the last solution + the last LU applied to the change of y */
    double pred_max;                    /* the largest port move the predictor is trusted with, volts */
    int    pred_stop;                   /* the predictor's step counts as the previous one for the predicted stop */
    double prev0;
    double ls_gate;                     /* line search 2/3: skip the residual test after a step below this (volts) */
    int    chord_persist;               /* start each sample on the last factorisation (refresh still applies) */
    int    w_ok;                        /* w holds a factorisation from a converged sample */                /* the sample's right-hand side, forward-eliminated through the linear pivots */
    double fsm;                         /* the rate the matrices were built for (0 = DC) */
    double fc[MKS_MNA_NFIX];            /* the fixed couplings' coefficients at that rate */
    MksMnaKnobs knobs;                  /* the controls the matrices were built for */
    double kval[MKS_MNA_NKV];           /* the knob-driven element values (ohm / farad) */
    double J[MKS_MNA_NP * MKS_MNA_NP];  /* the device Jacobian (blocks) */
    double vp[MKS_MNA_NP], ip[MKS_MNA_NP];
    MksDevice dev;
    double tol;                         /* Newton step, volts */
    int    max_iter;
    int    extrapolate;                 /* the start's extrapolation order: 0, 1 (default) or 2 */
    double damp;                        /* a growing Newton step is applied scaled by this (1 = off) */
    int    port_norm;                   /* measure the Newton step on the port voltages (as the DK does) */
    int    resid_stop;                  /* with line_search 3: stop on the port residual (the DK's test), not the step */
    int    trace;                       /* diagnostic: print every Newton iteration of this sample (the runner sets it; only in a
                                         * build with MKS_NEWTON_TRACE: a product has no I/O in its Newton loop, nlcore#79) */
    int    line_search;                 /* 1: halve an update whose next Newton step does not shrink; 2: whose KCL residual does
                                         * not; 3: whose PORT residual (the DK's, volts) does not (up to 4 halvings) */
    double step_cap;                    /* the largest node update one Newton step may apply, volts */
    double refresh;                     /* re-take the Jacobian when a step does not shrink by this factor */
    int    predict_stop;                /* stop when the predicted next step is below tol */
    double pstop_rho;                   /* ... and only once the steps contract by at least this factor */
    int    junction_limit;              /* SPICE-style junction voltage limiting in the transient (on; dev.limit holds it) */
    int    last_iters, fallbacks, factorisations, bad_pivots;
    int    trust_rescues;   /* MK81#112: fallbacks the trust region converged (mks_mna.c) */
    int    unconverged;     /* MK81#112: samples that left Newton above tol (the last iterate was kept) */
    int    continuations;               /* samples rescued by the source continuation */
#ifdef MKS_SAMPLE_BUDGET
    /* POLICY MKS_SAMPLE_BUDGET=N (MK81 cold audit A1, nlcore#78): N Newton iterations per sample shared by the whole
     * fallback ladder, every rung and every rescue depth; once spent, no rung runs and the last finite iterate is kept */
    int    sb_left;                     /* iterations left in this sample */
    int    sb_used_max;                 /* the most one sample used since init (the margin's measurement) */
    long   budget_cuts;                 /* samples that spent the whole budget */
    long   budget_holds;                /* of them, the samples that held the last converged point (not converged) */
#endif
#if defined(MKS_RESCUE) && MKS_RESCUE
    int    rescues_ls, rescues_sub;     /* MK81#165: samples the last rescues converged (mks_mna.c, MKS_RESCUE) */
    double emf_prev;                    /* the EMF the last committed sample used (the sub-step's history conversion) */
#endif
#ifdef MKS_WALK_RECAP
    int    walk_armed;                  /* MK81#164: 1 while a PREDICTED first attempt runs, 2 once it took the walk settings */
    double walk_cap_keep;               /* the caller's step cap and line search, handed back when that attempt returns */
    int    walk_ls_keep;
#endif
    /* A BUFFERED SOURCE (MK81#3): source buf_src (an index into u[]; 0 = off, u[0] being the guitar: init leaves it 0) is a unity-impedance buffer
     * of node buf_node, u[buf_src] = buf_gain x V[buf_node] + buf_off, solved WITHIN the sample: Newton converges with
     * the caller's u[buf_src] as the guess, the node is read, and while the source misses it by more than tol the source
     * is corrected and Newton re-run warm (at most buf_max_pass passes). For a node the source reaches only through
     * stiff paths (MK 81's master wiper and the PI it feeds through the cut: the supply's capacitors) one pass fixes it. */
    int    buf_src, buf_node, buf_max_pass;
    /* buf_emf = 1: the buffered quantity is the EMF below instead of a source: emf = buf_gain x V[buf_node] + buf_off */
    int    buf_emf;
    double buf_dg;                      /* the last secant's slope dg/dx (0 = none yet): the next sample's first step */
    /* THE CAPACITOR EMF (MK81#3): with MKS_MNA_EMF_CAP generated (gen_preamp_mna.py MKS_EMF_CAP), that
     * capacitor's branch voltage is its nodes' difference + MKS_MNA_EMF_SIGN x emf (volts, set per sample by the caller
     * or by the buffered solve). 0 = the circuit as drawn, bit for bit. Ignored at DC (the capacitor is open there). */
    double emf;
    /* emf_detached = 1 (MK81#3, the twin scheme): the EMF capacitor's a terminal is DETACHED from its node: its branch
     * voltage is emf - V(b), emf being the a terminal's own voltage (a known source per sample), while its current still
     * flows out of node a. Set with mks_mna_set_emf_detached (a matrix rebuild). */
    int    emf_detached;
#ifdef MKS_CHORD_FIRST
    long   chord_accepts;   /* iterations ended by the last factorisation's step (MKS_CHORD_FIRST) */
#endif
#ifdef MKS_KSTOP
    /* the one-update stop (MKS_KSTOP, nl_mna.c): K of Newton's quadratic error, measured by the last sample that took a
     * second update (0 until then: the stop stays off), and whether this sample stopped after one update. The fields had
     * gone missing from the struct while the code stayed: never compiled since the copy (MK81#164) */
    double kq;
    int    one_step;
#endif
#ifdef MKS_THETA_RT
    /* POLICY MKS_THETA_RT (nlcore#13, MK81#164): the theta-method chosen at RUN time (mks_mna_set_theta) - on, the fixed
     * capacitors up to theta_cmax farad on theta_c and the windings on theta_l; off, the trapezoidal rule, the very code
     * of a build without the policy (bit for bit) */
    int    theta_on;
    double theta_c, theta_cmax, theta_l;
    /* precomputed by build() when on (the commit per sample multiplies: 140 divisions a sample at 100 W UK otherwise, the
     * first run-time build's process_d 18 % of the real-time instructions): h' = th_a v - th_b h, hl += th_l_inv acc */
    double th_a[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], th_b[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], th_l_inv;
#endif
    double buf_gain, buf_off;
    int    buf_passes;                  /* the extra solves this sample took (0 = the guess was within tol) */
    long   buf_extra;                   /* the extra solves since init */
#ifdef MKS_DEVICE_MK81
    NlMnaValveCfg mk81_cfg;              /* MK 81's device settings for THIS instance (valves per side, EL34 table) */
#if defined(MKS_N_FXZ) && MKS_N_FXZ
    NlZener fxz[MKS_N_FXZ];             /* the zener ports' laws (mks_fxz_params, nl_zener_init'ed) */
    double fxz_q[MKS_N_FXZ], fxz_ic[MKS_N_FXZ];   /* each one's charge and trapezoidal current at the last converged sample */
    int    fxz_started;                 /* 0 during the DC solve: no charge current */
#ifdef MKS_FXZ_TAB
#ifndef MKS_FXZ_TLO
#define MKS_FXZ_TLO (-20.0)
#endif
#ifndef MKS_FXZ_THI
#define MKS_FXZ_THI 1.5
#endif
/* params of MKS_FXZ_TAB (overridable together: MKS_FXZ_TN must be the integer (THI - TLO) / TH + 1, checked at init) */
#ifndef MKS_FXZ_TH
#define MKS_FXZ_TH 0.002
#endif
#ifndef MKS_FXZ_TN
#define MKS_FXZ_TN 10751   /* (MKS_FXZ_THI - MKS_FXZ_TLO) / MKS_FXZ_TH + 1: an integer constant (a static array's size) */
#endif
    const double *fxz_ti, *fxz_tg, *fxz_tq, *fxz_tc;   /* the shared tables (built once per compiled system) */
    int    fxz_tab_ok;
#endif
#endif
#endif
} MksMna;

/* 0 on success: builds the matrices for fs at the knobs k and solves the DC operating point with the sources at u
 * (u[0] = guitar, then the DC rails in mks_source_dc order) */
int  mks_mna_init(MksMna* s, double fs, const double* u, const MksMnaKnobs* k);   /* k NULL = the defaults */
#ifdef MKS_HANDOVER
void mks_mna_handover(MksMna* dst, const MksMna* src);   /* the state of src into dst (same circuit, another rate) */
#endif
/* move the controls: the knob-driven elements change and the matrices are rebuilt; the circuit's state (capacitor
 * histories, junction charge, node voltages) is kept, so a move is a re-tune, not a restart. Not for every sample: the
 * rebuild costs a linear factorisation (call it per block, with the knobs smoothed by the caller). */
void mks_mna_set_knobs(MksMna* s, const MksMnaKnobs* k);
#ifdef MKS_THETA_RT
/* theta on / off at run time (MKS_THETA_RT): rebuilds the matrices at the present rate and knobs. The histories are kept
 * as they are: call it before mks_mna_init, or at a silent switch (the engine re-rests) */
void mks_mna_set_fs(MksMna* s, double fs);   /* change the time step, the state kept */
/* walk a fresh start to the transient's own rest (nlcore#80): n backward-Euler steps at fs_coarse, sources held at u */
void mks_mna_settle(MksMna* s, const double* u, double fs_coarse, int n);
void mks_mna_set_theta(MksMna* s, int on, double theta_c, double theta_cmax, double theta_l);
#endif
/* the phase inverter's six load values at presence p (mks_pi_load.h, interpolated in the logarithm) */
void mks_pi_load_at(double p, double* out);
/* detach (1) / re-attach (0) the EMF capacitor's a terminal (see emf_detached); a rebuild, the state kept */
void mks_mna_set_emf_detached(MksMna* s, int on);
/* one sample: u as above; returns node B */
double mks_mna_process(MksMna* s, const double* u);

#endif
