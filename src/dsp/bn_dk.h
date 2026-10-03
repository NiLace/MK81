/* bn_dk.h — real-time circuit engine using the nodal DK method.
 *
 * Formulation: Holters & Zölzer, DAFx-11. Validated prototype in
 * dsp/prototype/dk.py; this is the C port, and it is checked against it
 * sample by sample.
 *
 *   x(n) = A·x(n-1) + B·u(n) + C·i(n)
 *   v(n) = D·x(n-1) + E·u(n) + F·i(n)     with i(n) = f(v(n))
 *   y(n) = L·x(n-1) + M·u(n) + N·i(n)
 *
 * Everything expensive (building the nodal analysis, inverting) happens in
 * bn_dk_init. bn_dk_process neither allocates memory nor makes system calls.
 */
#ifndef BN_DK_H
#define BN_DK_H

/* Caps on the STACK buffers of the generic path and of var_apply. Without an
 * explicit check, a netlist with more nonlinear ports or more controls would
 * silently corrupt the stack. bn_dk_init and bn_dkvar_init enforce them, where
 * failing is still legal. Actual maxima in the current circuits: 24 nonlinear
 * ports and 13 controls. */
#define BN_DK_NL_MAX 64
#define BN_DK_NV_MAX 32

#include "bn_circuit.h"

/* Nonlinear device model: given the port voltages v, returns the currents
 * that ENTER the device and their Jacobian di/dv.
 * (The engine inverts the sign.) */
typedef void (*BnDeviceFn)(const double* v, double* i, double* J, void* user);
/* "Currents only" variant for the chord Newton (no Jacobian). */
typedef void (*BnDeviceCurrFn)(const double* v, double* i, void* user);

typedef struct {
    int n_states, n_nl, n_out, n_in;

    double *A, *B, *C;          /* state update */
    double *D, *E, *F;          /* nonlinear port voltages */
    double *L, *M, *N;          /* outputs */

    double *x, *x_next, *v, *v_prev, *i_nl, *p;   /* working state */
    double *Jdev, *Jg, *rhs;    /* Jacobians and Newton system */
    int    *pivot;

    /* TRANSPOSED copies of the dense matrices, for the specialised kernels
     * (matvec in axpy form: same summation order per row → bit-exact, but
     * independent accumulators that do vectorise).
     * Filled in init and refreshed when controls move (bn_dkvar_set). */
    double *At, *Ct, *Dt, *Ft, *Lt, *Nt;

    BnDeviceFn     device;
    BnDeviceCurrFn device_currents;   /* optional: enables the chord Newton */
    void*          device_user;

    int    max_iter;
    double tol;
    int    jdev_blockdiag;      /* 1 if the device is 2x2 block-diagonal */
    /* 1 if the device GUARANTEES that the grid current does not depend on the
     * plate voltage, i.e. J[1] == 0.0 EXACTLY.
     * With 2x2 block-diagonal this makes the term `Ft[r0·NL+i] * d0` equal
     * `x * 0.0` in the odd columns for all 8 rows: 32 of the 128 madds of the
     * Jg assembly are null work.
     *   WHY SKIPPING IT IS BYTE-IDENTICAL: the accumulator starts at 0.0 or 1.0
     *   (never −0.0) and `base + 0.0 == base` exactly for those two values, so
     *   removing the term does not change a single bit. `Ft` is finite by
     *   construction (circuit matrix), so `Ft * 0.0` is 0.0 and not NaN.
     *   It is NOT an approximation: it is exact arithmetic.
     *   ⚠️ It holds for the OWN triode model (bn_tube_own.c sets `J[1] = 0.0`
     *   literally) and NOT for the third-party reference model of the oracle
     *   build (its `J[1] = dig_da` is not zero). That is why it is a flag of the DEVICE and not of the
     *   circuit. Default 0 = full path, which is the safe one. */
    int    jdev_ig_indep_va;
    const int* jdev_bstart;     /* GENERAL blocks: block start per port */
    const int* jdev_bsize;      /* ... and size (overrides jdev_blockdiag if != 0)  */
    int    extrapolate_v;       /* 1: Newton start by extrapolation */
    int    extrapolate_order;   /* 2 = quadratic 3v - 3v_prev + v_prev2 (the shipped setting); 1 = linear 2v - v_prev (BN_XV2=1) */
    double* v_prev2;            /* the port voltages two samples back, for the quadratic start */
    /* In-kernel, lever BN_PRED=1 read ONCE in bn_dk_init, default OFF.
     * Tangent predictor for the Newton's starting point: v0 = v[n-1] + Jg^-1 . (p[n] - p[n-1])
     * with the LU the last factorisation left in Jg (one solve, no device evaluation; step
     * capped at 100 V like the Newton's own). Replaces the linear extrapolation when it is
     * on. As an out-of-kernel wrapper (BN_DK_WARM=4) it cut the preamp's iterations 3.00 ->
     * 2.02 but cost +14 % of wall; here it costs one 8x8 solve. The converged answer is the
     * Newton's, never the guess's: the output moves only within `tol`. */
    int    predict_tangent;
    int    pred_ok;             /* the LU in Jg comes from a converged sample (else no guess) */
    /* Chord refresh ratio (lever BN_CHORD_REFRESH read once in bn_dk_init, default 0.3):
     * with a persistent chord, an iteration that does not shrink the residual below
     * `chord_refresh * prev_worst` triggers a fresh Jacobian + LU. Smaller = keeps the
     * stale LU longer (fewer factorisations, more chord steps). */
    double chord_refresh;
    /* Explicit inverse (lever BN_CHORD_INV=1 read once in bn_dk_init, default OFF): after each
     * factorisation the inverse of Jg is built by solving the nl unit vectors, and every Newton
     * step then applies it as a dense matvec instead of the two triangular sweeps (a serial
     * chain, 18 % of the preamp's time under perf). Same arithmetic order as the solve only in
     * exact arithmetic: the step differs at roundoff, the Newton still runs to `tol`. Pays only
     * with a persistent chord, where one inverse serves many samples. */
    int     solve_inverse;
    int     inv_ok;             /* Jinv matches the current LU */
    double* Jinv;               /* nl x nl */
    int    chord_persist;       /* 1: the Jg factorisation survives between
                                 * samples (at high fs the operating point moves
                                 * little; the adaptive refactor protects) */
    int    have_lu;             /* internal state of the persistent chord */
    int    last_iters;          /* diagnostic: iterations of the last sample */
    int    fallbacks;           /* times the Newton budget was exhausted */
    /* A "fallback" is NOT divergence — it is failing to get below `tol` in
     * `max_iter` iterations, and that can be 1e-7 (harmless) or 1e-1 (the
     * oracle lies). A bare count cannot tell which; these two fields turn it
     * into a NUMBER. Pure instrumentation: they enter no computation, the DSP
     * stays bit-exact. */
    double fb_resid_max;        /* worst residual among the non-converged samples */
    double fb_resid_sum;        /* sum, to give the mean */
    /* EXPERIMENT: warm start of the per-sample Newton behind the
     * environment lever BN_DK_WARM, read ONCE in bn_dk_init. NULL when the lever is
     * unset, and then bn_dk_process is the unchanged engine (one predictable branch).
     * The initial guess is ONLY a starting point: the Newton still runs to `tol`.
     * State and strategies live in bn_dk.c (struct BnDKWarm). */
    struct BnDKWarm* warm;

    /* CAPACITOR LOOPS. The netlist makes every capacitor a state, but a
     * loop made only of capacitors has one independent voltage fewer than it has
     * capacitors (KVL), and the trapezoidal companion turns each redundant one into a
     * mode at z = -1: unobservable, undamped, driven by roundoff (12 of 55 states in the
     * full amp, 5 of 24 in the preamp). The builder keeps a SPANNING FOREST of the capacitor graph as states and folds the
     * co-tree capacitors in algebraically: z = P.x_full with P = [I | fundamental-cycle
     * signs], the reduced capacitance matrix P.diag(2C/T).P' is dense but small, and the
     * nodal matrix is unchanged, so every port voltage and output is the same at roundoff.
     * State order: tree capacitors in netlist order, then inductors. For a loop-free
     * netlist P is the identity and the engine is byte-identical to the unfolded build. */
    int     cap_nc;             /* capacitors in the netlist */
    int     cap_nt;             /* of them, tree edges = capacitor states */
    int*    cap_tree;           /* cap_nt indices into circuit->capacitors, increasing */
    double* cap_P;              /* cap_nt x cap_nc: z = P . x_full */

    /* RANK-1 BLOCK SOLVE (optional, default OFF: r1_cut = 0). When the ports split at r1_cut
     * into two groups whose F off-diagonal blocks are rank 1 (F[a,b] = u.w', F[b,a] = s.t') and
     * no device block straddles the cut, Jg = I + F.Jdev has rank-1 off-diagonal blocks too,
     * and the Newton system is solved EXACTLY (in exact arithmetic) by an LU of each diagonal
     * block plus a Schur complement that is a rank-1 update of the second block. Only the
     * specialised kernels compiled with BN_K_R1_CUT use it; enabled by bn_dk_rank1_enable().
     * u, w, s, t are recomputed from F whenever F changes (bn_dk_refresh_transposes); if F
     * stops being rank 1 at the cut, r1_ok drops to 0 and the kernel uses the full LU. */
    int     r1_cut;             /* 0 = off */
    int     r1_ok;              /* the factors below describe the current F */
    double  r1_err;             /* worst |F_off - rank-1 reconstruction| / max|F_off|, both blocks */
    double *r1_u, *r1_w, *r1_s, *r1_t;   /* each n_nl long, indexed by port */

    /* Outputs the specialised kernel actually computes, at run time: 0 = all it was compiled
     * for (BN_K_NOU), 5 = only y[0..4] (the oracle writes y[0], y[1], y[4]). The strides do
     * not change, so the computed outputs are byte-identical; y[5..] are left untouched. */
    int     n_out_used;
} BnDK;

/* Enable the rank-1 block solve at port `cut`. Call after bn_dk_init and after setting
 * jdev_bstart/jdev_bsize. Returns 0 when the structure holds (F off-diagonal blocks rank 1
 * within 1e-9 of their largest entry, no device block across the cut, a kernel built for
 * that cut exists, no BN_DK_WARM), else -1 and the solver stays on the full LU. */
int  bn_dk_rank1_enable(BnDK* dk, int cut);

/* Returns 0 on success. Allocates memory: do NOT call from run(). */
int  bn_dk_init(BnDK* dk, const BnCircuit* c, double fs);
void bn_dk_free(BnDK* dk);
void bn_dk_reset(BnDK* dk);        /* FULL: leaves the circuit COLD */
void bn_dk_reset_estado(BnDK* dk); /* numerical state only; `x` is kept */

/* One sample: u are the sources, y the outputs. No allocations or syscalls. */
void bn_dk_process(BnDK* dk, const double* u, double* y);

/* Which specialised kernel bn_dk_process will dispatch to for this system, by the SAME
 * test the dispatch uses, or NULL for the generic path. Diagnostic: the gate
 * `make mk81-dk-dims` fails when a product circuit falls to the generic path, which can
 * happen silently when the codegen changes the state and output counts (as folding the
 * capacitor loops did to both full-amp kernels). */
const char* bn_dk_kernel_name(const BnDK* dk);

/* Initialises x and v at the DC operating point (solves the circuit without
 * capacitors and applies x=(2C/T)·vc). Call after bn_dk_init and after setting
 * device/device_currents/jdev_blockdiag. Returns 0 if the DC Newton converged. */
int bn_dk_dc_init(BnDK* dk, const BnCircuit* c, double fs, const double* u_dc);

/* ---- variable parts (Holters & Zölzer / Woodbury): move pots without -------
 * re-inverting S. Mirror of the prototype dsp/prototype/dk_variable.py. */
typedef struct {
    BnDK* dk;                   /* the system whose A..N are recomputed */
    int   nv, nx, nu, nl, no;
    int*    idx;                /* indices of the variable resistors */
    double* r0;                 /* reference values (the table's) */
    double* Q;                  /* Nv·S0inv·Nv'                (nv×nv) */
    double *PBx, *PBn, *PBo;    /* P·S0inv·Nv'  per block P */
    double *BRx, *BRu, *BRn;    /* Nv·S0inv·R   per block R */
    double *bxx,*bxu,*bxn, *bnx,*bnu,*bnn, *box,*bou,*bon;  /* bases P·S0inv·R */
    double* GXm;                /* unsigned reactive matrix (the Z sign is `zs`), nx x nx (dense on the
                                 * tree capacitors when the netlist has capacitor loops) */
    double* zs;                 /* +1 capacitor, -1 inductor, per state */
    double *Axs, *Bxs, *Cxs;    /* scratch: the unscaled Ax/Bx/Cx of set() */
    /* scratch for set(). */
    double *dG, *M, *W;
    int*    piv;
    int     n_singular;    /* moves refused because I + Q·dG was singular (the matrices kept the last good move) */
} BnDKVar;

int  bn_dkvar_init(BnDKVar* vk, BnDK* dk, const BnCircuit* c, double fs,
                   const int* var_idx, int nv);
/* 0 on success; -1 when I + Q·dG is singular: nothing is rewritten, n_singular counts it */
int  bn_dkvar_set(BnDKVar* vk, const double* vals);
void bn_dkvar_free(BnDKVar* vk);

/* INCLUDE GUARD — it must close here, after BnDKVar and its three functions.
 * Closed any earlier, they fall OUTSIDE the guard and are declared twice as
 * soon as this header is included twice, and the resulting error
 * («conflicting types for BnDKVar») does not point at a misplaced guard. */
#endif /* BN_DK_H */
