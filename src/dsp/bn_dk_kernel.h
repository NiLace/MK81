/* bn_dk_kernel.h — TEMPLATE for a kernel specialised per circuit.
 *
 * Profiling put the cost in the overhead of small loops (indices, run-time
 * bounds, calls), not in FLOPs. This file is the "codegen": bn_dk.c includes
 * it once per circuit with the dimensions as compile-time CONSTANTS, and the
 * compiler unrolls and keeps v/rhs in registers. It is NOT a new algorithm: it
 * is bn_dk_process with the same operations IN THE SAME ORDER (parity with the
 * Python prototype must remain exact; test_c_vs_python.py checks it).
 *
 * Macros the includer must define (and clean up afterwards):
 *   BN_K_NAME  — name of the emitted function
 *   BN_K_NL / BN_K_NX / BN_K_NU / BN_K_NO — circuit dimensions
 *   BN_K_JG_BLOCKDIAG2 — 1 if Jdev is 2x2 block-diagonal (standalone triodes)
 *   BN_K_JG_BLOCKS     — 1 if it uses jdev_bstart/jdev_bsize (triodes+pentodes)
 *   BN_K_NOU (optional) — how many outputs are actually COMPUTED. Default:
 *     all (BN_K_NO). The circuit emits taps for the nulls and the bench, but
 *     the plugin in real time reads ONE; computing the others throws away 306
 *     madds per oversampled sample (8.4% of the program's instructions,
 *     measured with callgrind). The STRIDE of Lt/Nt/M stays BN_K_NO: only the
 *     loops are shortened, so the outputs that are computed come out
 *     BYTE-IDENTICAL and the generic path keeps all 9.
 *   BN_K_R1_CUT (optional) — port index where the rank-1 block solve splits the Newton
 *     system (see bn_dk.h, r1_cut). Needs BN_K_JG_BLOCKS. The kernel takes that path at run
 *     time only when dk->r1_cut == BN_K_R1_CUT and dk->r1_ok; otherwise the full LU runs.
 * Exactly one of the two Jg variants must be 1: the bn_dk_process dispatch
 * only enters here when the device flags match.
 */

#ifndef BN_K_NOU
#define BN_K_NOU BN_K_NO
#endif

#define BN_K_CAT2(a, b) a##b
#define BN_K_CAT(a, b) BN_K_CAT2(a, b)
#define BN_K_LU_FACTOR BN_K_CAT(BN_K_NAME, _lu_factor)
#define BN_K_LU_SOLVE  BN_K_CAT(BN_K_NAME, _lu_solve)

/* LU with partial pivoting, fixed n = BN_K_NL. Same algorithm and same order
 * of operations as the generic lu_factor/lu_solve (LAPACK convention: all
 * RHS swaps first). */
static int BN_K_LU_FACTOR(double* restrict A, int* restrict pivot)
{
    for (int k = 0; k < BN_K_NL; ++k) {
        int    piv = k;
        double best = fabs(A[k * BN_K_NL + k]);
        for (int r = k + 1; r < BN_K_NL; ++r) {
            const double v = fabs(A[r * BN_K_NL + k]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) { return -1; }
        if (piv != k) {
            for (int c = 0; c < BN_K_NL; ++c) {
                const double t = A[k * BN_K_NL + c];
                A[k * BN_K_NL + c] = A[piv * BN_K_NL + c];
                A[piv * BN_K_NL + c] = t;
            }
        }
        pivot[k] = piv;
        /* RECIPROCAL outside the loop: a divsd INSIDE the r loop would mean
         * 28 divisions per factorisation, at one factorisation per sample. On
         * Sandy Bridge divsd is NOT pipelined (~20-22 cycles of throughput):
         * ~600 cycles/sample spent just dividing. Formally x/d != x*(1/d), but
         * the output port is float32: below ~6e-8 relative NOTHING can show up
         * in the output. Measured: null 0.000e+00 on a real DI recording. */
        /* The reciprocal is stored on the diagonal itself and BN_K_LU_SOLVE
         * multiplies instead of dividing — 8 fewer divsd per solve. The same
         * change, word for word, as the generic path (bn_dk.c): the two paths
         * must keep giving the SAME result byte for byte, and the bench tests
         * that. See there for the full reasoning on why writing to the
         * diagonal is safe. Against a plain division it is not byte-identical:
         * up to 1 ULP per avoided division. */
        const double dinv = 1.0 / A[k * BN_K_NL + k];
        A[k * BN_K_NL + k] = dinv;
        for (int r = k + 1; r < BN_K_NL; ++r) {
            const double f = A[r * BN_K_NL + k] * dinv;
            A[r * BN_K_NL + k] = f;
            if (f == 0.0) { continue; }
            for (int c = k + 1; c < BN_K_NL; ++c) {
                A[r * BN_K_NL + c] -= f * A[k * BN_K_NL + c];
            }
        }
    }
    return 0;
}

static void BN_K_LU_SOLVE(const double* restrict LU, const int* restrict pivot,
                          double* restrict b)
{
    for (int k = 0; k < BN_K_NL; ++k) {
        if (pivot[k] != k) { const double t = b[k]; b[k] = b[pivot[k]]; b[pivot[k]] = t; }
    }
    for (int k = 0; k < BN_K_NL; ++k) {
        for (int r = k + 1; r < BN_K_NL; ++r) { b[r] -= LU[r * BN_K_NL + k] * b[k]; }
    }
    for (int k = BN_K_NL - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < BN_K_NL; ++c) { s -= LU[k * BN_K_NL + c] * b[c]; }
        b[k] = s * LU[k * BN_K_NL + k];   /* the diagonal holds the RECIPROCAL */
    }
}

#ifdef BN_K_R1_CUT
#if !BN_K_JG_BLOCKS
#error "BN_K_R1_CUT needs BN_K_JG_BLOCKS"
#endif
#define BN_K_R1A BN_K_R1_CUT                  /* ports of the first block */
#define BN_K_R1B (BN_K_NL - BN_K_R1_CUT)      /* ports of the second block */
#define BN_K_R1_LUF BN_K_CAT(BN_K_NAME, _r1_luf)
#define BN_K_R1_LUS BN_K_CAT(BN_K_NAME, _r1_lus)
#define BN_K_R1_FACTOR BN_K_CAT(BN_K_NAME, _r1_factor)
#define BN_K_R1_SOLVE BN_K_CAT(BN_K_NAME, _r1_solve)
/* Layout of the rank-1 factorisation inside dk->Jg (nl*nl doubles, more than enough):
 *   [0, A*A)          LU of Ja = Jg[a,a]
 *   [A*A, +B*B)       LU of the Schur complement S = Jg[b,b] - alpha . s . c'
 *   then z (A)        z = Ja^-1 . u
 *   then c (B)        Jg[a,b] = u . c'
 *   then tp (A)       Jg[b,a] = s . tp'
 * pivot[0, A) are the first block's pivots, pivot[A, nl) the second's (local indices). */
#define BN_K_R1_OFF_S  (BN_K_R1A * BN_K_R1A)
#define BN_K_R1_OFF_Z  (BN_K_R1_OFF_S + BN_K_R1B * BN_K_R1B)
#define BN_K_R1_OFF_C  (BN_K_R1_OFF_Z + BN_K_R1A)
#define BN_K_R1_OFF_TP (BN_K_R1_OFF_C + BN_K_R1B)

/* The same LU as BN_K_LU_FACTOR / BN_K_LU_SOLVE (partial pivoting, reciprocal kept on the
 * diagonal) for an n x n block; always inlined with a constant n. */
static inline __attribute__((always_inline))
int BN_K_R1_LUF(double* restrict A, int* restrict pivot, const int n)
{
    for (int k = 0; k < n; ++k) {
        int    piv = k;
        double best = fabs(A[k * n + k]);
        for (int r = k + 1; r < n; ++r) {
            const double v = fabs(A[r * n + k]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) { return -1; }
        if (piv != k) {
            for (int c = 0; c < n; ++c) {
                const double t = A[k * n + c]; A[k * n + c] = A[piv * n + c]; A[piv * n + c] = t;
            }
        }
        pivot[k] = piv;
        const double dinv = 1.0 / A[k * n + k];
        A[k * n + k] = dinv;
        for (int r = k + 1; r < n; ++r) {
            const double f = A[r * n + k] * dinv;
            A[r * n + k] = f;
            if (f == 0.0) { continue; }
            for (int c = k + 1; c < n; ++c) { A[r * n + c] -= f * A[k * n + c]; }
        }
    }
    return 0;
}

static inline __attribute__((always_inline))
void BN_K_R1_LUS(const double* restrict LU, const int* restrict pivot, double* restrict b, const int n)
{
    for (int k = 0; k < n; ++k) {
        if (pivot[k] != k) { const double t = b[k]; b[k] = b[pivot[k]]; b[pivot[k]] = t; }
    }
    for (int k = 0; k < n; ++k) {
        for (int r = k + 1; r < n; ++r) { b[r] -= LU[r * n + k] * b[k]; }
    }
    for (int k = n - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < n; ++c) { s -= LU[k * n + c] * b[c]; }
        b[k] = s * LU[k * n + k];
    }
}

/* Assemble the two diagonal blocks of Jg = I + F.Jdev and the two rank-1 couplings, then
 * factor: Ja = LU, z = Ja^-1 u, alpha = tp.z, S = Jb - alpha s c', S = LU.
 *   Jg[a,b] = F[a,b].Jdev[b,b] = u (w'.Jdev[b,b]) = u c'   (no device block crosses the cut)
 *   Jg[b,a] = F[b,a].Jdev[a,a] = s (t'.Jdev[a,a]) = s tp'
 * Returns -1 when either block LU finds a zero pivot. */
static int BN_K_R1_FACTOR(BnDK* dk)
{
    const double* restrict Jdev = dk->Jdev;
    const double* restrict Ft = dk->Ft;
    double* restrict Jg = dk->Jg;
    double* restrict LA = Jg;
    double* restrict LS = Jg + BN_K_R1_OFF_S;
    double* restrict z  = Jg + BN_K_R1_OFF_Z;
    double* restrict cv = Jg + BN_K_R1_OFF_C;
    double* restrict tp = Jg + BN_K_R1_OFF_TP;
    const double* restrict ru = dk->r1_u;
    const double* restrict rw = dk->r1_w;
    const double* restrict rs = dk->r1_s;
    const double* restrict rt = dk->r1_t;
    {
        double JT[BN_K_R1A * BN_K_R1A];     /* column j of Ja, contiguous in i */
        for (int j = 0; j < BN_K_R1A; ++j) {
            double* restrict cj = JT + j * BN_K_R1A;
            for (int i = 0; i < BN_K_R1A; ++i) { cj[i] = 0.0; }
            cj[j] = 1.0;
            double tj = 0.0;
            const int b0 = dk->jdev_bstart[j], bn = dk->jdev_bsize[j];
            for (int r = b0; r < b0 + bn; ++r) {
                const double d = Jdev[r * BN_K_NL + j];
                const double* restrict fr = Ft + r * BN_K_NL;
                for (int i = 0; i < BN_K_R1A; ++i) { cj[i] += fr[i] * d; }
                tj += rt[r] * d;
            }
            tp[j] = tj;
        }
        for (int i = 0; i < BN_K_R1A; ++i) {
            for (int j = 0; j < BN_K_R1A; ++j) { LA[i * BN_K_R1A + j] = JT[j * BN_K_R1A + i]; }
        }
    }
    {
        double JT[BN_K_R1B * BN_K_R1B];     /* column j of Jb, contiguous in i */
        for (int j = 0; j < BN_K_R1B; ++j) {
            const int jg = BN_K_R1A + j;
            double* restrict cj = JT + j * BN_K_R1B;
            for (int i = 0; i < BN_K_R1B; ++i) { cj[i] = 0.0; }
            cj[j] = 1.0;
            double cjv = 0.0;
            const int b0 = dk->jdev_bstart[jg], bn = dk->jdev_bsize[jg];
            for (int r = b0; r < b0 + bn; ++r) {
                const double d = Jdev[r * BN_K_NL + jg];
                const double* restrict fr = Ft + r * BN_K_NL + BN_K_R1A;
                for (int i = 0; i < BN_K_R1B; ++i) { cj[i] += fr[i] * d; }
                cjv += rw[r] * d;
            }
            cv[j] = cjv;
        }
        for (int i = 0; i < BN_K_R1B; ++i) {
            for (int j = 0; j < BN_K_R1B; ++j) { LS[i * BN_K_R1B + j] = JT[j * BN_K_R1B + i]; }
        }
    }
    if (BN_K_R1_LUF(LA, dk->pivot, BN_K_R1A) != 0) { return -1; }
    for (int i = 0; i < BN_K_R1A; ++i) { z[i] = ru[i]; }
    BN_K_R1_LUS(LA, dk->pivot, z, BN_K_R1A);
    double alpha = 0.0;
    for (int i = 0; i < BN_K_R1A; ++i) { alpha += tp[i] * z[i]; }
    for (int i = 0; i < BN_K_R1B; ++i) {
        const double as = alpha * rs[BN_K_R1A + i];
        for (int j = 0; j < BN_K_R1B; ++j) { LS[i * BN_K_R1B + j] -= as * cv[j]; }
    }
    if (BN_K_R1_LUF(LS, dk->pivot + BN_K_R1A, BN_K_R1B) != 0) { return -1; }
    return 0;
}

/* b <- Jg^-1 b with the factorisation above:
 *   ya = Ja^-1 b_a ; g = tp.ya ; xb = S^-1 (b_b - g s) ; xa = ya - z (c.xb) */
static void BN_K_R1_SOLVE(const BnDK* dk, double* restrict b)
{
    const double* restrict Jg = dk->Jg;
    const double* restrict z  = Jg + BN_K_R1_OFF_Z;
    const double* restrict cv = Jg + BN_K_R1_OFF_C;
    const double* restrict tp = Jg + BN_K_R1_OFF_TP;
    const double* restrict rs = dk->r1_s + BN_K_R1A;
    double* restrict ba = b;
    double* restrict bb = b + BN_K_R1A;
    BN_K_R1_LUS(Jg, dk->pivot, ba, BN_K_R1A);
    double g = 0.0;
    for (int i = 0; i < BN_K_R1A; ++i) { g += tp[i] * ba[i]; }
    for (int i = 0; i < BN_K_R1B; ++i) { bb[i] -= g * rs[i]; }
    BN_K_R1_LUS(Jg + BN_K_R1_OFF_S, dk->pivot + BN_K_R1A, bb, BN_K_R1B);
    double cb = 0.0;
    for (int i = 0; i < BN_K_R1B; ++i) { cb += cv[i] * bb[i]; }
    for (int i = 0; i < BN_K_R1A; ++i) { ba[i] -= z[i] * cb; }
}
#define BN_K_SOLVE(dk_, b_) do { if (r1) { BN_K_R1_SOLVE((dk_), (b_)); } \
                                 else { BN_K_LU_SOLVE((dk_)->Jg, (dk_)->pivot, (b_)); } } while (0)
#else
#define BN_K_SOLVE(dk_, b_) BN_K_LU_SOLVE((dk_)->Jg, (dk_)->pivot, (b_))
#endif

/* The output taps y[i] for i < nou. Always inlined with a constant nou, so each call site
 * has fixed trip counts; the summation groups do not depend on nou, so every computed tap
 * is byte-identical whatever nou is. */
static inline __attribute__((always_inline))
void BN_K_CAT(BN_K_NAME, _outputs)(const BnDK* dk, const double* restrict x, const double* restrict i_nl,
                                   const double* restrict u, double* restrict y, const int nou)
{
    const double* restrict M  = dk->M;
    const double* restrict Lt = dk->Lt;
    const double* restrict Nt = dk->Nt;
    double s0[BN_K_NO], s1[BN_K_NO];
    for (int i = 0; i < nou; ++i) { s0[i] = 0.0; s1[i] = 0.0; }
    for (int j = 0; j < BN_K_NX; ++j) {
        const double xj = x[j];
        for (int i = 0; i < nou; ++i) { s0[i] += Lt[j * BN_K_NO + i] * xj; }
    }
    for (int j = 0; j < BN_K_NL; ++j) {
        const double ij = i_nl[j];
        for (int i = 0; i < nou; ++i) { s1[i] += Nt[j * BN_K_NO + i] * ij; }
    }
    for (int j = 0; j < BN_K_NU; ++j) {
        const double uj = u[j];
        for (int i = 0; i < nou; ++i) { s0[i] += M[i * BN_K_NU + j] * uj; }
    }
    for (int i = 0; i < nou; ++i) { y[i] = s0[i] + s1[i]; }
}

static void BN_K_NAME(BnDK* dk, const double* restrict u, double* restrict y)
{
#ifdef BN_K_R1_CUT
    const int r1 = (dk->r1_cut == BN_K_R1_CUT) && dk->r1_ok;
#endif
    const double* restrict E  = dk->E;
    const double* restrict Dt = dk->Dt;
    const double* restrict Ft = dk->Ft;
    double* restrict v    = dk->v;
    double* restrict p    = dk->p;
    double* restrict i_nl = dk->i_nl;
    double* restrict rhs  = dk->rhs;

    /* p = D·x + E·u — axpy over Dt: for each row i the sum walks j in the
     * SAME order as the generic dot product → bit-exact, but the BN_K_NL
     * accumulators are independent and the compiler vectorises. */
    /* The accumulation goes into a LOCAL array and is copied to dk->p once at
     * the end. Accumulating directly on the heap (dk->p) keeps the compiler
     * from promoting those 8 accumulators to registers, because it cannot
     * know nobody else looks at them: it reloads and rewrites them on each of
     * the 24+2 passes of the outer loop.
     *   Byte-identical: same values, same summation groups, same order.
     *   ⚠️ COMPILER LOTTERY, measured: gcc-14 -4.2 %, clang -4.8 %, but
     *   **gcc-12 +5.8 % SLOWER**. This is built with gcc-14. When porting to
     *   another toolchain (e.g. a Raspberry Pi 5) this must be RE-MEASURED
     *   specifically, not inherited. */
    const int use_pred = dk->predict_tangent && dk->pred_ok;   /* tangent predictor, see bn_dk.h */
    double dpp[BN_K_NL];
    {
        const double* restrict x = dk->x;
        double pl[BN_K_NL];
        for (int i = 0; i < BN_K_NL; ++i) { pl[i] = 0.0; }
        for (int j = 0; j < BN_K_NX; ++j) {
            const double xj = x[j];
            for (int i = 0; i < BN_K_NL; ++i) { pl[i] += Dt[j * BN_K_NL + i] * xj; }
        }
        for (int j = 0; j < BN_K_NU; ++j) {
            const double uj = u[j];
            for (int i = 0; i < BN_K_NL; ++i) { pl[i] += E[i * BN_K_NU + j] * uj; }
        }
        /* tangent predictor: dp = p[n] − p[n−1], taken before p is overwritten */
        if (use_pred) { for (int i = 0; i < BN_K_NL; ++i) { dpp[i] = pl[i] - p[i]; } }
        for (int i = 0; i < BN_K_NL; ++i) { p[i] = pl[i]; }
    }

    /* Damped Newton on v = p + F·i(v) (same flow as the generic path) */
    int it = 0;
    {
        if (!dk->chord_persist) { dk->have_lu = 0; }
        if (use_pred) {
            /* Tangent predictor: one solve with the LU the last factorisation left in
             * Jg, step capped at 100 V; replaces the extrapolation, keeps v_prev's meaning */
            BN_K_SOLVE(dk, dpp);
            double step = 0.0;
            for (int i = 0; i < BN_K_NL; ++i) { if (fabs(dpp[i]) > step) { step = fabs(dpp[i]); } }
            const double scale = (step > 100.0) ? (100.0 / step) : 1.0;
            for (int i = 0; i < BN_K_NL; ++i) {
                const double vp = v[i];
                v[i] = vp + scale * dpp[i];
                dk->v_prev[i] = vp;
            }
        } else if (dk->extrapolate_v) {
            if (dk->extrapolate_order == 2) {
                for (int i = 0; i < BN_K_NL; ++i) {
                    const double vp = v[i], vq = dk->v_prev[i];
                    v[i] = 3.0 * vp - 3.0 * vq + dk->v_prev2[i];
                    dk->v_prev2[i] = vq; dk->v_prev[i] = vp;
                }
            } else {
            for (int i = 0; i < BN_K_NL; ++i) {
                const double vp = v[i];
                v[i] = 2.0 * vp - dk->v_prev[i];
                dk->v_prev[i] = vp;
            }
            }
        }
        double prev_worst = 1e300;
        double last_worst = 0.0;   /* the residual the loop exits with */
        /* CARRIED residual: the backtracking already computes r(v_accepted) —
         * bit-identical to the one the next iteration would assemble again,
         * so it is reused instead of recomputed (CSE, not a new algorithm:
         * same values and same decisions as the generic engine). */
        double cr[BN_K_NL];
        double cworst = 0.0;
        int    have_res = 0;
        /* Sample start: if the CHORD is still alive (have_lu, only possible
         * with chord_persist) the Jacobian is not needed — the currents are
         * enough, and they are MUCH cheaper (the triode's Jacobian is four more
         * derivatives per port). Without this, chord_persist would save the Jg
         * assembly and the LU but still pay the full Jacobian on every sample,
         * i.e. the expensive part. If the residual then does not drop enough,
         * the adaptive refresh below asks for a fresh Jacobian.
         * The product ships chord_persist=1, so have_lu IS set here on most
         * samples and this path runs; kernel/generic bit-identity holds by
         * construction (same operations, same order), not by have_lu=0. */
        if (dk->have_lu && dk->device_currents) {
            dk->device_currents(v, i_nl, dk->device_user);
        } else {
            dk->device(v, i_nl, dk->Jdev, dk->device_user);
        }
        for (it = 0; it < dk->max_iter; ++it) {
            double worst = 0.0;
            if (have_res) {
                for (int i = 0; i < BN_K_NL; ++i) { rhs[i] = -cr[i]; }
                worst = cworst;
            } else {
                double r_[BN_K_NL];
                for (int i = 0; i < BN_K_NL; ++i) { r_[i] = v[i] - p[i]; }
                for (int j = 0; j < BN_K_NL; ++j) {
                    const double ij = i_nl[j];
                    for (int i = 0; i < BN_K_NL; ++i) { r_[i] += Ft[j * BN_K_NL + i] * ij; }
                }
                for (int i = 0; i < BN_K_NL; ++i) {
                    rhs[i] = -r_[i];
                    /* `!(x <= worst)` and not `x > worst`: a NaN residual must POISON worst so the
                     * `worst < tol` test below fails and the sample goes to the fallback. With
                     * `>` a NaN never beats 0.0 and the Newton "converges" at iteration 1
                     * (observed with a bad warm start driving a port to 1e4 V). */
                    if (!(fabs(r_[i]) <= worst)) { worst = fabs(r_[i]); }
                }
            }
            last_worst = worst;
            if (worst < dk->tol) { break; }
            if (dk->have_lu && worst > dk->chord_refresh * prev_worst) {
                dk->device(v, i_nl, dk->Jdev, dk->device_user);
                dk->have_lu = 0;
            }
            prev_worst = worst;
            if (!dk->have_lu) {
                /* Jg = I + F·Jdev — the "columns" of F the block needs are
                 * contiguous ROWS of Ft.
                 * How often this runs depends on the chord mode: with
                 * chord_persist=0 it is ONCE PER SAMPLE (gprof 1.00 Jacobians/sample,
                 * the LU 5.5 % of the total, block + LU ~26 % of the plugin); with
                 * chord_persist=1 (shipped) and refresh 0.3 only when the
                 * chord's residual stops halving -- the factorisation was 18.5 % of
                 * the plugin without the persistent chord. Re-measure before quoting
                 * either number. */
                int lu_fail;
#ifdef BN_K_R1_CUT
                if (r1) {
                    lu_fail = BN_K_R1_FACTOR(dk);
                } else
#endif
                {
                const double* restrict Jdev = dk->Jdev;
                double* restrict Jg = dk->Jg;
#if BN_K_JG_BLOCKS
                {
                    /* Jg = I + F·Jdev as axpys over a local transposed accumulator: row j of JgT is column j of
                     * Jg, contiguous in i. Same products, same summation order per entry as the dot-product form,
                     * so the result is bit-identical (md5 of rendered outputs, verified on 18 renders incl. abuse
                     * and all circuit variants); -9 % of the exact amp's time at 16x on an Intel N150, because the
                     * dot form's 2-3-trip inner loop and strided stores were the solver's hottest code (28 % of its
                     * instructions). */
                    double JgT[BN_K_NL * BN_K_NL];
                    for (int j = 0; j < BN_K_NL; ++j) {
                        double* restrict cj = JgT + j * BN_K_NL;
                        for (int i = 0; i < BN_K_NL; ++i) { cj[i] = 0.0; }
                        cj[j] = 1.0;
                        const int b0 = dk->jdev_bstart[j];
                        const int bn = dk->jdev_bsize[j];
                        for (int r = b0; r < b0 + bn; ++r) {
                            const double d = Jdev[r * BN_K_NL + j];
                            const double* restrict fr = Ft + r * BN_K_NL;
                            for (int i = 0; i < BN_K_NL; ++i) { cj[i] += fr[i] * d; }
                        }
                    }
                    for (int i = 0; i < BN_K_NL; ++i) {
                        for (int j = 0; j < BN_K_NL; ++j) { Jg[i * BN_K_NL + j] = JgT[j * BN_K_NL + i]; }
                    }
                }
#elif BN_K_JG_BLOCKDIAG2
                /* With `jdev_ig_indep_va` the device guarantees J[1] == 0.0
                 * EXACTLY (the grid current does not depend on the plate), so
                 * in the ODD columns the d0 term is `x * 0.0` in all 8 rows: 32
                 * of the 128 madds per sample are null work. Skipping it is
                 * byte-identical, not approximate — see the long comment in
                 * bn_dk.h.
                 * The loop is split into even/odd so the decision is NOT a
                 * branch per column: the compiler sees both forms whole. With
                 * the flag at 0 (the oracle's reference model, and any device that does not
                 * guarantee it) the full path runs. */
                if (dk->jdev_ig_indep_va) {
                    for (int j = 0; j < BN_K_NL; j += 2) {
                        const int r0 = j, r1 = j + 1;
                        /* EVEN column (grid port): full */
                        {
                            const double d0 = Jdev[r0 * BN_K_NL + j];
                            const double d1 = Jdev[r1 * BN_K_NL + j];
                            for (int i = 0; i < BN_K_NL; ++i) {
                                Jg[i * BN_K_NL + j] = ((i == j) ? 1.0 : 0.0)
                                    + Ft[r0 * BN_K_NL + i] * d0
                                    + Ft[r1 * BN_K_NL + i] * d1;
                            }
                        }
                        /* ODD column (plate port): d0 is 0.0 */
                        {
                            const int jj = j + 1;
                            const double d1 = Jdev[r1 * BN_K_NL + jj];
                            for (int i = 0; i < BN_K_NL; ++i) {
                                Jg[i * BN_K_NL + jj] = ((i == jj) ? 1.0 : 0.0)
                                    + Ft[r1 * BN_K_NL + i] * d1;
                            }
                        }
                    }
                } else {
                for (int j = 0; j < BN_K_NL; ++j) {
                    const int r0 = j & ~1, r1 = r0 + 1;
                    const double d0 = Jdev[r0 * BN_K_NL + j];
                    const double d1 = Jdev[r1 * BN_K_NL + j];
                    for (int i = 0; i < BN_K_NL; ++i) {
                        Jg[i * BN_K_NL + j] = ((i == j) ? 1.0 : 0.0)
                            + Ft[r0 * BN_K_NL + i] * d0
                            + Ft[r1 * BN_K_NL + i] * d1;
                    }
                }
                }
#else
#error "define BN_K_JG_BLOCKDIAG2 or BN_K_JG_BLOCKS"
#endif
                lu_fail = BN_K_LU_FACTOR(dk->Jg, dk->pivot);
                }
                if (lu_fail != 0) {
                    /* This exit must be COUNTED, like its twin in the generic
                     * path (bn_dk.c): this kernel is the path the product runs
                     * (dispatched from bn_dk_process), so without it the
                     * contract «fallbacks = 0 => the Newton always converged»
                     * would hold only on the generic path, which almost nothing
                     * executes.
                     * POSITIVE CONTROL: with the LU forced to fail and no count
                     * here, bn_bench reported «fallbacks 0» with the output
                     * moved by +3.7 dB — a textbook false clean.
                     * Note: the max_iter exit further below is counted too, but
                     * it is not reached from here, because this break leaves
                     * with it < max_iter. */
                    dk->fallbacks++;
                    if (last_worst > dk->fb_resid_max) { dk->fb_resid_max = last_worst; }
                    dk->fb_resid_sum += last_worst;
                    /* SELF-HEALING, the same as the max_iter exit further
                     * below. This break leaves with it < max_iter, so it does
                     * NOT reach that one: without resetting here, a
                     * non-converged sample would poison the extrapolation of
                     * the next one and the plugin would LATCH UP (the failure
                     * described at the max_iter exit). */
                    for (int i = 0; i < BN_K_NL; ++i) { dk->v_prev[i] = v[i]; dk->v_prev2[i] = v[i]; }
                    dk->pred_ok = 0;      /* predictor: a singular LU predicts nothing */
                    break;
                }
                dk->have_lu = 1;
                dk->pred_ok = 1;          /* predictor: Jg now holds a usable factorisation */
                dk->inv_ok = 0;
            }
            if (dk->solve_inverse) {
                /* The explicit inverse, rebuilt after each factorisation, applied as a matvec */
                double* restrict Jinv = dk->Jinv;
                if (!dk->inv_ok) {
                    for (int c = 0; c < BN_K_NL; ++c) {
                        double e[BN_K_NL];
                        for (int i = 0; i < BN_K_NL; ++i) { e[i] = (i == c) ? 1.0 : 0.0; }
                        BN_K_SOLVE(dk, e);
                        for (int i = 0; i < BN_K_NL; ++i) { Jinv[i * BN_K_NL + c] = e[i]; }
                    }
                    dk->inv_ok = 1;
                }
                double t[BN_K_NL];
                for (int i = 0; i < BN_K_NL; ++i) { t[i] = 0.0; }
                for (int j = 0; j < BN_K_NL; ++j) {
                    const double rj = rhs[j];
                    for (int i = 0; i < BN_K_NL; ++i) { t[i] += Jinv[i * BN_K_NL + j] * rj; }
                }
                for (int i = 0; i < BN_K_NL; ++i) { rhs[i] = t[i]; }
            } else {
                BN_K_SOLVE(dk, rhs);
            }
            double step = 0.0;
            for (int i = 0; i < BN_K_NL; ++i) {
                if (fabs(rhs[i]) > step) { step = fabs(rhs[i]); }
            }
            /* NEWTON STEP CAP — a 5.0 V cap was the ROOT CAUSE of
             * non-convergence.
             *
             * The mechanism, MEASURED: this cap limits how far the Newton can
             * ADVANCE per iteration. With musical material the port voltages
             * move a few volts between oversampled samples and the cap NEVER
             * BITES (0 failures with a real DI recording and with the amp
             * saturating at peak 0.95). But with transients of extreme slope
             * (a square wave, full-band noise) the solution moves HUNDREDS of
             * volts in one sample: at 5 V per iteration and 48 iterations the
             * solver only reaches 240 V, so it CANNOT get there however long
             * it iterates. It stopped halfway and returned garbage, which
             * poisoned the state and fed back.
             *
             * Sweep measured on noise (failures out of 192512 samples):
             *   cap 1 -> 62508 · cap 5 -> 18549 · cap 20 -> 222 · cap 50 -> 0
             *   cap 100 -> 0 · no cap -> 0
             * And on a square wave at 0 dBFS: cap 5 -> 1159 · cap 100 -> 0.
             *
             * 100 V is chosen: ten times above where the problem disappears,
             * and still a safety net against an absurd step (the backtracking
             * below is what really damps). Since the cap does not bite on
             * musical material, raising it is BIT-EXACT there — it only
             * changes things where the solver was already failing. */
            const double scale = (step > 100.0) ? (100.0 / step) : 1.0;
            {
                double alpha = 1.0;
                double vt[BN_K_NL], it_[BN_K_NL];
                int accepted = 0;
                for (int bt = 0; bt < 4 && !accepted; ++bt) {
                    for (int i = 0; i < BN_K_NL; ++i) {
                        vt[i] = v[i] + alpha * scale * rhs[i];
                    }
                    if (dk->device_currents) {
                        dk->device_currents(vt, it_, dk->device_user);
                    } else {
                        dk->device(vt, it_, dk->Jdev, dk->device_user);
                        dk->have_lu = 0;
                    }
                    double wt = 0.0;
                    for (int i = 0; i < BN_K_NL; ++i) { cr[i] = vt[i] - p[i]; }
                    for (int j = 0; j < BN_K_NL; ++j) {
                        const double ij = it_[j];
                        for (int i = 0; i < BN_K_NL; ++i) { cr[i] += Ft[j * BN_K_NL + i] * ij; }
                    }
                    for (int i = 0; i < BN_K_NL; ++i) {
                        /* Same negated form as the residual above, and for the same
                         * reason: `>` against NaN is false, so wt would stay 0.0, this step
                         * would be ACCEPTED as the best one, and the next iteration would read
                         * worst = 0 < tol and return "converged" with v = NaN — without
                         * incrementing `fallbacks`, breaking the fallback-count contract. Both
                         * sides of the if feed the same variable, so both need the guard. */
                        if (!(fabs(cr[i]) <= wt)) { wt = fabs(cr[i]); }
                    }
                    if (wt < worst || bt == 3) { accepted = 1; cworst = wt; have_res = 1; }
                    else {
                        alpha *= 0.5;
                    }
                }
                for (int i = 0; i < BN_K_NL; ++i) {
                    v[i] = vt[i];
                    i_nl[i] = it_[i];
                }
            }
        }
        if (it >= dk->max_iter) {
            dk->fallbacks++;
            if (last_worst > dk->fb_resid_max) { dk->fb_resid_max = last_worst; }
            dk->fb_resid_sum += last_worst;
            dk->pred_ok = 0;              /* predictor: a sample that did not converge predicts nothing */
            /* SELF-HEALING. Without this the plugin LATCHES UP: if v_prev
             * were written regardless, a non-converged sample would poison
             * the extrapolation of the next one, that one would not converge
             * either, and from then on ALL samples would exhaust the budget.
             * MEASURED at 4x (the default oversampling) with a square wave at
             * 0 dBFS: output rms 0.067 and, when the playing drops to soft,
             * 0.0003 — the amp stays mute until the instance is reloaded.
             * At 1x, 2x and 8x it did not happen: 4x is right on the edge.
             * The fix: set v_prev = v so the next sample does NOT extrapolate
             * (it starts from the last v, which is bounded even if it did not
             * converge). It is BIT-EXACT on every sample that DOES converge —
             * it only touches the path that was already broken. */
            for (int i = 0; i < BN_K_NL; ++i) { dk->v_prev[i] = v[i]; dk->v_prev2[i] = v[i]; }
        }
        for (int i = 0; i < BN_K_NL; ++i) { i_nl[i] = -i_nl[i]; }
    }
    dk->last_iters = it + 1;

    /* outputs and state update — axpy; the generic path's two accumulators
     * s0/s1 are kept (identical summation groups → bit-exact). */
    {
        /* `x` is not const: the new state is written DIRECTLY here, without
         * going through dk->x_next and without a final copy. That saves 24
         * stores + 24 loads + 24 stores per sample.
         *   SAFE: all reads of x[] (the At and Lt blocks) have finished before
         *   the loop that writes. The order of the sums is untouched, so it is
         *   byte-identical.
         *   ⚠️ ONE SINGLE restrict pointer to dk->x. A second restrict pointer
         *   to the SAME object would be undefined behaviour and a latent
         *   miscompile waiting for a more aggressive compiler. `x` is reused.
         *   ⚠️ And this holds ONLY for the kernel: in the generic path the
         *   equivalent block (in bn_dk.c) is i-outer/j-inner, not axpy, so
         *   reads and writes overlap — applying it there is MEASURED as
         *   destructive (+3.7 dB of residual). Leave it alone. */
        double* restrict x  = dk->x;
        const double* restrict B  = dk->B;
        const double* restrict At = dk->At;
        const double* restrict Ct = dk->Ct;
        if (BN_K_NOU >= 5 && dk->n_out_used == 5) {
            BN_K_CAT(BN_K_NAME, _outputs)(dk, x, i_nl, u, y, 5);
        } else {
            BN_K_CAT(BN_K_NAME, _outputs)(dk, x, i_nl, u, y, BN_K_NOU);
        }
        {
            double s0[BN_K_NX], s1[BN_K_NX];
            for (int i = 0; i < BN_K_NX; ++i) { s0[i] = 0.0; s1[i] = 0.0; }
            for (int j = 0; j < BN_K_NX; ++j) {
                const double xj = x[j];
                for (int i = 0; i < BN_K_NX; ++i) { s0[i] += At[j * BN_K_NX + i] * xj; }
            }
            for (int j = 0; j < BN_K_NL; ++j) {
                const double ij = i_nl[j];
                for (int i = 0; i < BN_K_NX; ++i) { s1[i] += Ct[j * BN_K_NX + i] * ij; }
            }
            for (int j = 0; j < BN_K_NU; ++j) {
                const double uj = u[j];
                for (int i = 0; i < BN_K_NX; ++i) { s0[i] += B[i * BN_K_NU + j] * uj; }
            }
            for (int i = 0; i < BN_K_NX; ++i) { x[i] = s0[i] + s1[i]; }
        }
    }
}

#undef BN_K_NOU        /* MANDATORY: if left defined, the next
                        * instantiation would inherit this one's NOU */
#undef BN_K_LU_FACTOR
#undef BN_K_LU_SOLVE
#undef BN_K_SOLVE
#ifdef BN_K_R1_CUT
#undef BN_K_R1A
#undef BN_K_R1B
#undef BN_K_R1_LUF
#undef BN_K_R1_LUS
#undef BN_K_R1_FACTOR
#undef BN_K_R1_SOLVE
#undef BN_K_R1_OFF_S
#undef BN_K_R1_OFF_Z
#undef BN_K_R1_OFF_C
#undef BN_K_R1_OFF_TP
#undef BN_K_R1_CUT     /* same reason as BN_K_NOU */
#endif
#undef BN_K_CAT
#undef BN_K_CAT2
