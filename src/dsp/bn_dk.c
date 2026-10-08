/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_dk.c — nodal DK method in C. Port of the prototype dsp/prototype/dk.py.
 *
 * It is checked against the prototype sample by sample (dsp/prototype/
 * test_c_vs_python.py): if the two do not match, the port is wrong.
 */
#include "bn_dk.h"

#include <math.h>
#include <stdio.h>    /* exit report of the BN_DK_WARM experiment lever only */
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ util -- */
/* warm start of the Newton (experiment lever), defined below the kernels */
struct BnDKWarm;
static struct BnDKWarm* bn_dk_warm_alloc(int nl, int nx);
static void bn_dk_warm_free(struct BnDKWarm* w);
static void bn_dk_warm_reset(struct BnDKWarm* w);

#define IDX(M, cols, r, c) ((M)[(size_t)(r) * (size_t)(cols) + (size_t)(c)])

static double* mat_alloc(int rows, int cols)
{
    /* aligned to 64 B: the axpy kernels load whole rows vectorised */
    size_t n = (rows <= 0 || cols <= 0) ? 1 : (size_t)rows * (size_t)cols;
    size_t bytes = (n * sizeof(double) + 63u) & ~(size_t)63u;
    double* m = (double*)aligned_alloc(64, bytes);
    if (m) { memset(m, 0, bytes); }
    return m;
}

/* In-place LU factorisation with partial pivoting + solve with the stored
 * factorisation. The chord Newton reuses the factorisation across
 * iterations (new Jacobian only if convergence stalls). */
static int lu_factor(double* A, int n, int* pivot)
{
    for (int k = 0; k < n; ++k) {
        int    piv = k;
        double best = fabs(IDX(A, n, k, k));
        for (int r = k + 1; r < n; ++r) {
            const double v = fabs(IDX(A, n, r, k));
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) { return -1; }
        if (piv != k) {
            for (int c = 0; c < n; ++c) {
                const double t = IDX(A, n, k, c);
                IDX(A, n, k, c) = IDX(A, n, piv, c);
                IDX(A, n, piv, c) = t;
            }
        }
        pivot[k] = piv;
        /* same change as in the kernel: kernel and generic paths must keep
         * giving the SAME result byte for byte. */
        /* The reciprocal is STORED on the diagonal itself — it was already
         * computed for the elimination — and lu_solve multiplies instead of
         * dividing. That saves 8 divsd per solve (~19-24 per sample), and on
         * Sandy Bridge divsd is NOT pipelined (~20-22 cycles of throughput)
         * against 5 for mulsd.
         *   STORING IT THERE IS SAFE: the factorisation never reads A[k][k]
         *   again after this point (rows r>k read COLUMN k and ROW k FROM
         *   COLUMN k+1 on, never the diagonal already written), and later pivot
         *   swaps only move rows > k, so no reciprocal already written is
         *   displaced. Outside lu_solve nobody reads the diagonal of a
         *   factorised matrix: the only consumers are the generic Newton and
         *   var_apply, both through the factor/solve pair.
         *   ⚠️ CAREFUL, and that is why it is NOT applied there: the solver
         *   further below (solve_dense) runs its inner loop from `c = k`, so it
         *   DOES re-read the diagonal. It is also an init path, not hot: it
         *   gains nothing and would break.
         *   ⚠️ It is NOT byte-identical to a division: x/d and x*(1/d) differ
         *   by up to 1 ULP. It goes into the kernel AND the generic path AT THE
         *   SAME TIME, or the byte-for-byte parity between the two paths, which
         *   is tested, breaks. */
        const double dinv = 1.0 / IDX(A, n, k, k);
        IDX(A, n, k, k) = dinv;
        for (int r = k + 1; r < n; ++r) {
            const double f = IDX(A, n, r, k) * dinv;
            IDX(A, n, r, k) = f;              /* store the multiplier */
            if (f == 0.0) { continue; }
            for (int c = k + 1; c < n; ++c) {
                IDX(A, n, r, c) -= f * IDX(A, n, k, c);
            }
        }
    }
    return 0;
}

static void lu_solve(const double* LU, const int* pivot, double* b, int n)
{
    /* LAPACK convention: the stored multipliers were swapped TOGETHER with
     * the rows at later pivots, so ALL the RHS swaps go first and the
     * elimination afterwards (getrs = laswp + trsv). With interleaved swaps
     * the result is wrong — measured: error 8.9e3 in the isolated test. */
    for (int k = 0; k < n; ++k) {
        if (pivot[k] != k) { const double t = b[k]; b[k] = b[pivot[k]]; b[pivot[k]] = t; }
    }
    for (int k = 0; k < n; ++k) {
        for (int r = k + 1; r < n; ++r) { b[r] -= IDX(LU, n, r, k) * b[k]; }
    }
    for (int k = n - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < n; ++c) { s -= IDX(LU, n, k, c) * b[c]; }
        b[k] = s * IDX(LU, n, k, k);   /* the diagonal holds the RECIPROCAL */
    }
}

/* Solves A·x = b by Gaussian elimination with partial pivoting.
 * Serves both for inverting in init and for the Newton step in run;
 * the systems are tiny (2..8 unknowns), so neither LAPACK nor an external
 * dependency pays off. */
static int solve_dense(double* A, double* b, int n, int* pivot)
{
    for (int k = 0; k < n; ++k) {
        int    piv = k;
        double best = fabs(IDX(A, n, k, k));
        for (int r = k + 1; r < n; ++r) {
            const double v = fabs(IDX(A, n, r, k));
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-300) {
            return -1;                      /* singular */
        }
        if (piv != k) {
            for (int c = 0; c < n; ++c) {
                const double t = IDX(A, n, k, c);
                IDX(A, n, k, c) = IDX(A, n, piv, c);
                IDX(A, n, piv, c) = t;
            }
            const double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        pivot[k] = piv;
        /* same change as in the kernel: kernel and generic paths must keep
         * giving the SAME result byte for byte. */
        const double dinv = 1.0 / IDX(A, n, k, k);
        for (int r = k + 1; r < n; ++r) {
            const double f = IDX(A, n, r, k) * dinv;
            if (f == 0.0) { continue; }
            for (int c = k; c < n; ++c) {
                IDX(A, n, r, c) -= f * IDX(A, n, k, c);
            }
            b[r] -= f * b[k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < n; ++c) {
            s -= IDX(A, n, k, c) * b[c];
        }
        b[k] = s / IDX(A, n, k, k);
    }
    return 0;
}

static int invert_dense(const double* A, double* Ainv, int n)
{
    double* work = mat_alloc(n, n);
    double* col  = (double*)calloc((size_t)n, sizeof(double));
    int*    piv  = (int*)calloc((size_t)n, sizeof(int));
    int     rc   = 0;
    /* All three allocations are checked: on OOM the memcpy below would write
     * to NULL -> SIGSEGV in instantiate(), instead of the host receiving a
     * clean NULL and disabling the plugin. */
    if (!work || !col || !piv) {
        free(work); free(col); free(piv);
        return -1;
    }
    for (int j = 0; j < n && rc == 0; ++j) {
        memcpy(work, A, (size_t)n * (size_t)n * sizeof(double));
        for (int r = 0; r < n; ++r) { col[r] = (r == j) ? 1.0 : 0.0; }
        rc = solve_dense(work, col, n, piv);
        for (int r = 0; r < n; ++r) { IDX(Ainv, n, r, j) = col[r]; }
    }
    free(work); free(col); free(piv);
    return rc;
}

static void mat_mul(const double* A, int ar, int ac,
                    const double* B, int bc, double* out)
{
    for (int i = 0; i < ar; ++i) {
        for (int j = 0; j < bc; ++j) {
            double s = 0.0;
            for (int k = 0; k < ac; ++k) {
                s += IDX(A, ac, i, k) * IDX(B, bc, k, j);
            }
            IDX(out, bc, i, j) = s;
        }
    }
}

/* Rank-1 factors of the two off-diagonal blocks of F at dk->r1_cut, pivoting on the largest
 * entry of each block: F[a,b] ~ u.w' with u = F[:,q], w = F[p,:]/F[p,q]; F[b,a] ~ s.t' alike.
 * r1_ok = 1 only when both reconstructions are within 1e-9 of the block's largest entry. The
 * factors only shape the Newton STEP: the residual is always evaluated with the full F, so
 * a reconstruction error changes the convergence rate, never the converged answer.
 * Any change of F invalidates the factorisation held in Jg (its layout depends on r1_ok). */
static void bn_dk_rank1_factor(BnDK* dk)
{
    const int nl = dk->n_nl, k = dk->r1_cut;
    dk->r1_ok = 0;
    dk->have_lu = 0; dk->pred_ok = 0; dk->inv_ok = 0;
    if (k <= 0 || k >= nl || !dk->r1_u) { return; }
    double worst = 0.0;
    for (int blk = 0; blk < 2; ++blk) {
        const int r0 = blk ? k : 0, r1 = blk ? nl : k;      /* rows */
        const int c0 = blk ? 0 : k, c1 = blk ? k : nl;      /* columns */
        double* cu = blk ? dk->r1_s : dk->r1_u;             /* indexed by row port */
        double* rw = blk ? dk->r1_t : dk->r1_w;             /* indexed by column port */
        int pr = r0, pc = c0; double big = 0.0;
        for (int r = r0; r < r1; ++r) {
            for (int c = c0; c < c1; ++c) {
                const double a = fabs(IDX(dk->F, nl, r, c));
                if (a > big) { big = a; pr = r; pc = c; }
            }
        }
        if (!(big > 0.0)) { return; }                       /* decoupled blocks: not this solver */
        const double piv = IDX(dk->F, nl, pr, pc);
        for (int r = r0; r < r1; ++r) { cu[r] = IDX(dk->F, nl, r, pc); }
        for (int c = c0; c < c1; ++c) { rw[c] = IDX(dk->F, nl, pr, c) / piv; }
        double e = 0.0;
        for (int r = r0; r < r1; ++r) {
            for (int c = c0; c < c1; ++c) {
                const double d = fabs(IDX(dk->F, nl, r, c) - cu[r] * rw[c]);
                if (d > e) { e = d; }
            }
        }
        if (e / big > worst) { worst = e / big; }
    }
    dk->r1_err = worst;
    dk->r1_ok = (worst <= 1e-9);
}

/* Transposed copies for the axpy kernels (see bn_dk.h). Called after any
 * rewrite of A..N: init and bn_dkvar_set. */
static void bn_dk_refresh_transposes(BnDK* dk)
{
    if (dk->r1_cut) { bn_dk_rank1_factor(dk); }
    const int nx = dk->n_states, nl = dk->n_nl, no = dk->n_out;
    if (!dk->At) { return; }
    for (int r = 0; r < nx; ++r) {
        for (int c = 0; c < nx; ++c) { IDX(dk->At, nx, c, r) = IDX(dk->A, nx, r, c); }
        for (int c = 0; c < nl; ++c) { IDX(dk->Ct, nx, c, r) = IDX(dk->C, nl, r, c); }
    }
    for (int r = 0; r < nl; ++r) {
        for (int c = 0; c < nx; ++c) { IDX(dk->Dt, nl, c, r) = IDX(dk->D, nx, r, c); }
        for (int c = 0; c < nl; ++c) { IDX(dk->Ft, nl, c, r) = IDX(dk->F, nl, r, c); }
    }
    for (int r = 0; r < no; ++r) {
        for (int c = 0; c < nx; ++c) { IDX(dk->Lt, no, c, r) = IDX(dk->L, nx, r, c); }
        for (int c = 0; c < nl; ++c) { IDX(dk->Nt, no, c, r) = IDX(dk->N, nl, r, c); }
    }
}

/* ------------------------------------------------------------------ init -- */
static void fill_incidence(double* N, const BnPort* ports, int n_ports, int n_nodes)
{
    for (int r = 0; r < n_ports; ++r) {
        if (ports[r].a != BN_GND) { IDX(N, n_nodes, r, ports[r].a - 1) =  1.0; }
        if (ports[r].b != BN_GND) { IDX(N, n_nodes, r, ports[r].b - 1) = -1.0; }
    }
}

static void fill_incidence_elem(double* N, const BnElement* el, int n_el, int n_nodes)
{
    for (int r = 0; r < n_el; ++r) {
        if (el[r].a != BN_GND) { IDX(N, n_nodes, r, el[r].a - 1) =  1.0; }
        if (el[r].b != BN_GND) { IDX(N, n_nodes, r, el[r].b - 1) = -1.0; }
    }
}

/* -------------------------------------------------- capacitor forest ---- */
static int uf_find(int* par, int i)
{
    while (par[i] != i) { par[i] = par[par[i]]; i = par[i]; }
    return i;
}

/* Spanning forest of the capacitor graph (ground is a vertex too: two capacitors from
 * one node to ground are a loop through ground). Tree capacitors become states; each
 * co-tree capacitor e gets the column P[:, e] = the signs of the tree path from e.a to
 * e.b, so that its incidence row is the signed sum of the tree rows (KVL). Returns 0,
 * with *tree (nt entries) and *P (nt x nc) allocated; the caller frees them. */
static int cap_forest_build(const BnCircuit* c, int* nt_out, int** tree_out, double** P_out)
{
    const int nc = c->n_capacitors, nv = c->n_nodes + 1;
    int* par   = (int*)malloc(sizeof(int) * (size_t)nv);
    int* tpos  = (int*)malloc(sizeof(int) * (size_t)(nc > 0 ? nc : 1));
    int* tree  = (int*)malloc(sizeof(int) * (size_t)(nc > 0 ? nc : 1));
    if (!par || !tpos || !tree) { free(par); free(tpos); free(tree); return -1; }
    for (int i = 0; i < nv; ++i) { par[i] = i; }
    int nt = 0;
    for (int e = 0; e < nc; ++e) {
        const int ra = uf_find(par, c->capacitors[e].a), rb = uf_find(par, c->capacitors[e].b);
        if (ra != rb) { par[ra] = rb; tpos[e] = nt; tree[nt++] = e; }
        else          { tpos[e] = -1; }
    }
    double* P = (double*)calloc((size_t)(nt > 0 ? nt : 1) * (size_t)(nc > 0 ? nc : 1), sizeof(double));
    /* BFS scratch over the tree: parent vertex and parent edge per vertex */
    int* pv = (int*)malloc(sizeof(int) * (size_t)nv);
    int* pe = (int*)malloc(sizeof(int) * (size_t)nv);
    int* q  = (int*)malloc(sizeof(int) * (size_t)nv);
    if (!P || !pv || !pe || !q) { free(P); free(pv); free(pe); free(q); free(par); free(tpos); free(tree); return -1; }
    for (int t = 0; t < nt; ++t) { IDX(P, nc, t, tree[t]) = 1.0; }
    for (int e = 0; e < nc; ++e) {
        if (tpos[e] >= 0) { continue; }
        const int a = c->capacitors[e].a, b = c->capacitors[e].b;
        if (a == b) { continue; }                 /* degenerate: no voltage, no state */
        for (int i = 0; i < nv; ++i) { pv[i] = -1; pe[i] = -1; }
        int qh = 0, qt = 0;
        q[qt++] = a; pv[a] = a;
        while (qh < qt && pv[b] < 0) {
            const int v = q[qh++];
            for (int t = 0; t < nt; ++t) {
                const int ta = c->capacitors[tree[t]].a, tb = c->capacitors[tree[t]].b;
                const int w = (ta == v) ? tb : (tb == v) ? ta : -1;
                if (w < 0 || pv[w] >= 0) { continue; }
                pv[w] = v; pe[w] = t; q[qt++] = w;
            }
        }
        if (pv[b] < 0) { free(P); free(pv); free(pe); free(q); free(par); free(tpos); free(tree); return -1; }
        /* walk b -> a: the step parent(v) -> v uses tree edge pe[v]; its row counts +1
         * when it is oriented parent -> v and -1 otherwise (the sum telescopes to a - b) */
        for (int v = b; v != a; v = pv[v]) {
            const int t = pe[v];
            const int ta = c->capacitors[tree[t]].a;
            IDX(P, nc, t, e) = (ta == pv[v]) ? 1.0 : -1.0;
        }
    }
    free(pv); free(pe); free(q); free(par); free(tpos);
    *nt_out = nt; *tree_out = tree; *P_out = P;
    return 0;
}

/* The reactive part shared by bn_dk_init and bn_dkvar_init (the parity test in
 * tools/bn_test_var.c catches any drift between the two callers):
 * NX (nx x nn) = tree-capacitor rows then inductor rows; GXm (nx x nx) =
 * P.diag(2C/T).P' on the capacitor block, diag(T/2L) and the coupled (T/2).L^-1
 * blocks on the inductors; zs = +1 / -1 per state. All outputs allocated here. */
static int reactive_build(const BnCircuit* c, double T, int nt, const int* tree, const double* P,
                          double** NX_out, double** GXm_out, double** zs_out)
{
    const int nn = c->n_nodes, nc = c->n_capacitors, nL = c->n_inductors, nx = nt + nL;
    double* NX  = mat_alloc(nx, nn);
    double* GXm = mat_alloc(nx, nx);
    double* zs  = (double*)calloc((size_t)(nx > 0 ? nx : 1), sizeof(double));
    if (!NX || !GXm || !zs) { free(NX); free(GXm); free(zs); return -1; }
    for (int t = 0; t < nt; ++t) {
        const BnElement* el = &c->capacitors[tree[t]];
        if (el->a != BN_GND) { IDX(NX, nn, t, el->a - 1) =  1.0; }
        if (el->b != BN_GND) { IDX(NX, nn, t, el->b - 1) = -1.0; }
        zs[t] = 1.0;
    }
    if (nL) { fill_incidence_elem(NX + (size_t)nt * nn, c->inductors, nL, nn); }
    /* capacitor block: skipping the zero entries of P keeps a loop-free netlist
     * byte-identical (1.0 * g * 1.0 with nothing added) */
    for (int i = 0; i < nt; ++i) {
        for (int j = 0; j < nt; ++j) {
            double s = 0.0;
            for (int e = 0; e < nc; ++e) {
                const double pi = IDX(P, nc, i, e);
                if (pi == 0.0) { continue; }
                const double pj = IDX(P, nc, j, e);
                if (pj == 0.0) { continue; }
                s += pi * (2.0 * c->capacitors[e].value / T) * pj;
            }
            IDX(GXm, nx, i, j) = s;
        }
    }
    for (int e = 0; e < nL; ++e) {
        IDX(GXm, nx, nt + e, nt + e) = T / (2.0 * c->inductors[e].value);
        zs[nt + e] = -1.0;
    }
    for (int m = 0; m < c->n_mutuals; ++m) {
        const BnMutual* mu = &c->mutuals[m];
        double Lm[BN_MUTUAL_MAX * BN_MUTUAL_MAX];
        double Gi[BN_MUTUAL_MAX * BN_MUTUAL_MAX];
        memcpy(Lm, mu->L, sizeof(double) * (size_t)mu->n * (size_t)mu->n);
        /* The failure is propagated, not ignored: ignoring it would leave Gi
         * UNINITIALISED (stack) and put garbage into GXm. A singular mutual
         * L matrix is a malformed netlist, not a recoverable condition. */
        if (invert_dense(Lm, Gi, mu->n) != 0) { free(NX); free(GXm); free(zs); return -1; }
        for (int a = 0; a < mu->n; ++a) {
            for (int b = 0; b < mu->n; ++b) {
                IDX(GXm, nx, nt + mu->idx[a], nt + mu->idx[b]) = 0.5 * T * IDX(Gi, mu->n, a, b);
            }
        }
    }
    *NX_out = NX; *GXm_out = GXm; *zs_out = zs;
    return 0;
}

int bn_dk_init(BnDK* dk, const BnCircuit* c, double fs)
{
    memset(dk, 0, sizeof(*dk));
    const int nn = c->n_nodes;
    const int nr = c->n_resistors;
    const int nc_ = c->n_capacitors;
    const int nL = c->n_inductors;
    const double T = 1.0 / fs;
    if (cap_forest_build(c, &dk->cap_nt, &dk->cap_tree, &dk->cap_P) != 0) { return -1; }
    dk->cap_nc = nc_;
    const int nx = dk->cap_nt + nL;
    const int nu = c->n_sources;
    const int nl = c->n_nonlinear;
    const int no = c->n_outputs;

    /* The generic path of bn_dk_step uses STACK buffers
     * `double vt[BN_DK_NL_MAX], it_[BN_DK_NL_MAX]` for the backtracking.
     * Without this check, a netlist with more nonlinear ports would silently
     * corrupt the stack. Failing is still legal here (the host gets NULL and
     * disables the plugin); inside step it is not.
     * The actual maximum is 24 (the whole amp), i.e. 2.7x headroom. */
    if (nl > BN_DK_NL_MAX) { free(dk->cap_tree); free(dk->cap_P); memset(dk, 0, sizeof(*dk)); return -1; }

    dk->n_states = nx; dk->n_nl = nl; dk->n_out = no; dk->n_in = nu;
    dk->max_iter = 100;         /* same as the Python prototype: with the preamp
                                 * clipping hard, 50 run out (189 fallbacks in
                                 * 2 s of signal) and the result diverges from
                                 * the prototype. Measured. */
    dk->tol      = 1e-10;

    double* NR = mat_alloc(nr, nn);
    double* NU = mat_alloc(nu, nn);
    double* NN = mat_alloc(nl, nn);
    double* NO = mat_alloc(no, nn);
    fill_incidence_elem(NR, c->resistors, nr, nn);
    fill_incidence(NU, c->sources, nu, nn);
    fill_incidence(NN, c->nonlinear, nl, nn);
    fill_incidence(NO, c->outputs, no, nn);

    /* General GX: P.diag(2C/T).P' for the forest capacitors, diag(T/2L) for
     * standalone inductors, and (T/2)·L^-1 blocks for the COUPLED groups. zs = sign
     * Z (+1 capacitor, -1 inductor). */
    double *NX, *GXm, *zs;
    if (reactive_build(c, T, dk->cap_nt, dk->cap_tree, dk->cap_P, &NX, &GXm, &zs) != 0) {
        free(NR); free(NU); free(NN); free(NO);
        free(dk->cap_tree); free(dk->cap_P); memset(dk, 0, sizeof(*dk));
        return -1;
    }

    /* S = [[NR' GR NR + NX' GX NX, NU'], [NU, 0]] */
    const int ns = nn + nu;
    double* S = mat_alloc(ns, ns);
    for (int e = 0; e < nr; ++e) {
        const double g = 1.0 / c->resistors[e].value;
        for (int i = 0; i < nn; ++i) {
            const double ni = IDX(NR, nn, e, i);
            if (ni == 0.0) { continue; }
            for (int j = 0; j < nn; ++j) {
                IDX(S, ns, i, j) += ni * g * IDX(NR, nn, e, j);
            }
        }
    }
    /* NX' * GXm * NX (GXm is not diagonal when there are coupled inductors) */
    for (int e = 0; e < nx; ++e) {
        for (int f2 = 0; f2 < nx; ++f2) {
            const double g = IDX(GXm, nx, e, f2);
            if (g == 0.0) { continue; }
            for (int i = 0; i < nn; ++i) {
                const double ni = IDX(NX, nn, e, i);
                if (ni == 0.0) { continue; }
                for (int j = 0; j < nn; ++j) {
                    IDX(S, ns, i, j) += ni * g * IDX(NX, nn, f2, j);
                }
            }
        }
    }
    for (int e = 0; e < nu; ++e) {
        for (int i = 0; i < nn; ++i) {
            IDX(S, ns, i, nn + e) = IDX(NU, nn, e, i);
            IDX(S, ns, nn + e, i) = IDX(NU, nn, e, i);
        }
    }

    double* Sinv = mat_alloc(ns, ns);
    if (invert_dense(S, Sinv, ns) != 0) {
        free(NR); free(NX); free(NU); free(NN); free(NO); free(S); free(Sinv);
        free(GXm); free(zs);
        free(dk->cap_tree); free(dk->cap_P); memset(dk, 0, sizeof(*dk));
        return -1;
    }

    /* Selectors [M 0] and injectors [M'; 0] */
    double* Px = mat_alloc(nx, ns);
    double* Pn = mat_alloc(nl, ns);
    double* Po = mat_alloc(no, ns);
    for (int r = 0; r < nx; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Px, ns, r, cc) = IDX(NX, nn, r, cc); } }
    for (int r = 0; r < nl; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Pn, ns, r, cc) = IDX(NN, nn, r, cc); } }
    for (int r = 0; r < no; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Po, ns, r, cc) = IDX(NO, nn, r, cc); } }

    double* Rx = mat_alloc(ns, nx);
    double* Ru = mat_alloc(ns, nu);
    double* Rn = mat_alloc(ns, nl);
    for (int r = 0; r < nx; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Rx, nx, cc, r) = IDX(NX, nn, r, cc); } }
    for (int r = 0; r < nl; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Rn, nl, cc, r) = IDX(NN, nn, r, cc); } }
    for (int e = 0; e < nu; ++e) { IDX(Ru, nu, nn + e, e) = 1.0; }

    /* vx = Ax·x + Bx·u + Cx·i, etc. */
    double* tmp_x = mat_alloc(nx, ns);
    double* tmp_n = mat_alloc(nl, ns);
    double* tmp_o = mat_alloc(no, ns);
    mat_mul(Px, nx, ns, Sinv, ns, tmp_x);
    mat_mul(Pn, nl, ns, Sinv, ns, tmp_n);
    mat_mul(Po, no, ns, Sinv, ns, tmp_o);

    double* Ax = mat_alloc(nx, nx); mat_mul(tmp_x, nx, ns, Rx, nx, Ax);
    double* Bx = mat_alloc(nx, nu); mat_mul(tmp_x, nx, ns, Ru, nu, Bx);
    double* Cx = mat_alloc(nx, nl); mat_mul(tmp_x, nx, ns, Rn, nl, Cx);

    dk->A = mat_alloc(nx, nx);
    dk->B = mat_alloc(nx, nu);
    dk->C = mat_alloc(nx, nl);
    /* A = Z(2·GXm·Ax − I), B = 2·Z·GXm·Bx, C = 2·Z·GXm·Cx — with GXm a full matrix */
    for (int r = 0; r < nx; ++r) {
        for (int cc = 0; cc < nx; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(GXm, nx, r, k) * IDX(Ax, nx, k, cc); }
            IDX(dk->A, nx, r, cc) = zs[r] * (2.0 * s2 - ((r == cc) ? 1.0 : 0.0));
        }
        for (int cc = 0; cc < nu; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(GXm, nx, r, k) * IDX(Bx, nu, k, cc); }
            IDX(dk->B, nu, r, cc) = zs[r] * 2.0 * s2;
        }
        for (int cc = 0; cc < nl; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(GXm, nx, r, k) * IDX(Cx, nl, k, cc); }
            IDX(dk->C, nl, r, cc) = zs[r] * 2.0 * s2;
        }
    }
    free(GXm); free(zs);

    dk->D = mat_alloc(nl, nx); mat_mul(tmp_n, nl, ns, Rx, nx, dk->D);
    dk->E = mat_alloc(nl, nu); mat_mul(tmp_n, nl, ns, Ru, nu, dk->E);
    dk->F = mat_alloc(nl, nl); mat_mul(tmp_n, nl, ns, Rn, nl, dk->F);
    dk->L = mat_alloc(no, nx); mat_mul(tmp_o, no, ns, Rx, nx, dk->L);
    dk->M = mat_alloc(no, nu); mat_mul(tmp_o, no, ns, Ru, nu, dk->M);
    dk->N = mat_alloc(no, nl); mat_mul(tmp_o, no, ns, Rn, nl, dk->N);

    dk->At = mat_alloc(nx, nx);
    dk->Ct = mat_alloc(nl, nx);
    dk->Dt = mat_alloc(nx, nl);
    dk->Ft = mat_alloc(nl, nl);
    dk->Lt = mat_alloc(nx, no);
    dk->Nt = mat_alloc(nl, no);
    bn_dk_refresh_transposes(dk);

    dk->x      = (double*)calloc((size_t)(nx > 0 ? nx : 1), sizeof(double));
    dk->x_next = (double*)calloc((size_t)(nx > 0 ? nx : 1), sizeof(double));
    dk->v      = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    dk->v_prev = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    dk->v_prev2 = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    {   /* start-extrapolation order, read once (experiment lever) */
        static int xv2 = -1;
        if (xv2 < 0) { const char* e = getenv("BN_XV2"); xv2 = (e && e[0] == '1') ? 1 : 2; }   /* DEFAULT 2 = quadratic start (-3.6 % of iterations, output -86.6 dB from the linear start's); BN_XV2=1 = the linear start */
        dk->extrapolate_order = xv2;
    }
    dk->i_nl = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    dk->p    = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    dk->Jdev = mat_alloc(nl, nl);
    dk->Jg   = mat_alloc(nl, nl);
    dk->rhs  = (double*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(double));
    dk->pivot = (int*)calloc((size_t)(nl > 0 ? nl : 1), sizeof(int));
    dk->warm  = bn_dk_warm_alloc(nl, nx);   /* NULL unless BN_DK_WARM is set */
    {
        /* Explicit inverse: the caller decides (the plugin reads BN_CHORD / BN_CHORD_INV once,
         * the bench takes `cpinv`). The engine must NOT read BN_CHORD_INV too: with opposite
         * default semantics ('1' = on here, '0' = off in the plugin) a bench with the variable
         * set would rebuild the inverse every sample without a persistent chord -- a
         * combination that never pays. One reader per artefact. */
        dk->solve_inverse = 0; dk->inv_ok = 0;
        dk->Jinv = mat_alloc(nl, nl);
    }
    {
        /* chord refresh ratio, read once */
        static double refresh_lever = -1.0;
        if (refresh_lever < 0.0) { const char* e = getenv("BN_CHORD_REFRESH"); double v = e ? atof(e) : 0.0; refresh_lever = (v > 0.0 && v < 1.0) ? v : 0.3; }   /* DEFAULT 0.3 (-1 % of iterations); BN_CHORD_REFRESH=0.5 = the previous setting. An earlier measurement saw 0.3 + the quadratic start move the hum gate's 1/f ratio past its floor: re-check that gate when touching this */
        dk->chord_refresh = refresh_lever;
    }
    {
        /* tangent-predictor lever, read once (a getenv per sample is a syscall in the audio thread) */
        static int pred_lever = -1;
        if (pred_lever < 0) { const char* e = getenv("BN_PRED"); pred_lever = (e && e[0] == '1') ? 1 : 0; }
        dk->predict_tangent = pred_lever;
        dk->pred_ok = 0;
    }

    free(NR); free(NX); free(NU); free(NN); free(NO); free(S); free(Sinv);
    free(Px); free(Pn); free(Po); free(Rx); free(Ru); free(Rn);
    free(tmp_x); free(tmp_n); free(tmp_o); free(Ax); free(Bx); free(Cx);
    return 0;
}

void bn_dk_free(BnDK* dk)
{
    free(dk->A); free(dk->B); free(dk->C);
    free(dk->D); free(dk->E); free(dk->F);
    free(dk->L); free(dk->M); free(dk->N);
    free(dk->x); free(dk->x_next); free(dk->v); free(dk->v_prev); free(dk->v_prev2); free(dk->i_nl); free(dk->p);
    free(dk->Jdev); free(dk->Jg); free(dk->rhs); free(dk->pivot);
    free(dk->At); free(dk->Ct); free(dk->Dt); free(dk->Ft); free(dk->Lt); free(dk->Nt);
    free(dk->cap_tree); free(dk->cap_P); free(dk->Jinv);
    free(dk->r1_u); free(dk->r1_w); free(dk->r1_s); free(dk->r1_t);
    bn_dk_warm_free(dk->warm);
    memset(dk, 0, sizeof(*dk));
}

/* Clears the solver's WORKING state without touching `x` (the circuit's
 * PHYSICAL state: capacitor voltages and inductor currents).
 * Every piece of numerical state (v_prev, x_next, have_lu, ...) must be
 * cleared here, or a "reset" is not one. The two things are kept apart
 * because they are not the same: the physical state is RESTORED to its
 * operating point (the caller holds the snapshot), and the numerical state
 * is CLEARED. */
void bn_dk_reset_estado(BnDK* dk)
{
    for (int i = 0; i < dk->n_nl; ++i)     { dk->v[i] = 0.0; }
    if (dk->v_prev) { for (int i = 0; i < dk->n_nl; ++i)     { dk->v_prev[i] = 0.0; } }
    if (dk->v_prev2) { for (int i = 0; i < dk->n_nl; ++i)    { dk->v_prev2[i] = 0.0; } }
    if (dk->x_next) { for (int i = 0; i < dk->n_states; ++i) { dk->x_next[i] = 0.0; } }
    dk->have_lu = 0;
    dk->pred_ok = 0;
    dk->inv_ok = 0;
    bn_dk_warm_reset(dk->warm);
}

/* FULL reset, physical state included: leaves the circuit COLD (all
 * capacitors at zero). It is not what an activate() wants — there the
 * operating point is restored — but it is kept complete and correct rather
 * than half done. */
void bn_dk_reset(BnDK* dk)
{
    for (int i = 0; i < dk->n_states; ++i) { dk->x[i] = 0.0; }
    bn_dk_reset_estado(dk);
}

/* ------------------------------------------- specialised kernels -------- */
/* The same operations as the generic bn_dk_process below, with the circuit
 * dimensions as compile-time constants (bn_dk_kernel.h). The bn_dk_process
 * dispatch chooses by dimensions + device flags; any other circuit (e.g. the
 * static system of bn_dk_dc_init) falls to the generic path. */

#define BN_K_NAME bn_dk_kern_preamp2203
#define BN_K_NL 8
#define BN_K_NX 18      /* 5 capacitor loops folded; the inert bplus-0 capacitor pruned at codegen, 24 C -> 23 */
#define BN_K_NU 2
#define BN_K_NO 9
/* The circuit emits 9 taps (pa1,k1,pa2,k2,pa3,k3,cf,out_ts + the output) for
 * the nulls and the bench, but the PLUGIN reads y[0] and nothing else.
 * Computing the other eight cost 306 madds per oversampled sample = 8.4% of
 * the program's instructions (callgrind). With NOU=1 the output is
 * BYTE-IDENTICAL and whoever needs all 9 asks for them via the generic path.
 * CAREFUL: that "generic path" must be REQUESTED at compile time,
 * -DBN_DISABLE_KERNELS; the build/bn_bench_generic target does so. Without it,
 * `bn_bench op preamp` would print taps that nothing computes. */
#define BN_K_NOU 1
#define BN_K_JG_BLOCKDIAG2 1
#define BN_K_JG_BLOCKS 0
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS

/* Complete full amp. LS_3 (the open 16 Ω winding) is removed at codegen.
 * With the capacitor loops folded: 58 capacitors with 16 loops + 5 inductors = 47 states.
 * ⚠️ If this constant disagrees with the netlist's state count, the kernel never
 * dispatches and the full amp silently runs the generic path; `make mk81-dk-dims`
 * checks the dimension. */
#define BN_K_NAME bn_dk_kern_fullamp
#define BN_K_NL 24
#define BN_K_NX 48     /* 4 interwinding capacitors (2 to t8, 2 to ground) = 62 C, 19 loops, +1 state */
#define BN_K_NU 3
#define BN_K_NO 15     /* 15 taps since the codegen added one; 14 was a dead kernel */
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS

/* COLLAPSED full amp (EL34 pairs → equivalent valve, exact to −194 dB) */
#define BN_K_NAME bn_dk_kern_fullamp_col
#define BN_K_NL 18
#define BN_K_NX 44      /* 54 C, 15 loops, 5 L = 44 (without the interwinding capacitors: 50 capacitors with
                         * 12 loops + 5 inductors). A mismatch with the netlist is a dead kernel (see above) */
#define BN_K_NU 3
#define BN_K_NO 15     /* same */
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#define BN_K_R1_CUT 8   /* preamp ports 0-7 | PI + power 8-17: F couples them through rank-1 blocks */
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS

/* The collapsed amp with the LCR speaker load (nx 47). Without the rank-1 solve these circuits
 * keep the generic path; this kernel is dispatched only when bn_dk_rank1_enable() succeeded. */
#define BN_K_NAME bn_dk_kern_fullamp_col_lcr
#define BN_K_NL 18
#define BN_K_NX 47
#define BN_K_NU 3
#define BN_K_NO 15
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#define BN_K_R1_CUT 8
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS


/* The SAME three amps with the SPLIT output transformer of the 100 W heads (dsp/prototype/
 * ot_split.py: each primary half in six layer sections, 14 coupled inductors, the derived self and interwinding
 * capacitances at the internal nodes). The 50 W circuits keep the lumped transformer and the kernels above. Counts read
 * off `make mk81-dk-dims`: full amp 77 C, 21 loops, 15 L = 71; collapsed 69 C, 17 loops, 15 L = 67; collapsed LCR 70. */
#define BN_K_NAME bn_dk_kern_fullamp_ots
#define BN_K_NL 24
#define BN_K_NX 71
#define BN_K_NU 3
#define BN_K_NO 15
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS
#define BN_K_NAME bn_dk_kern_fullamp_col_ots
#define BN_K_NL 18
#define BN_K_NX 67
#define BN_K_NU 3
#define BN_K_NO 15
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#define BN_K_R1_CUT 8
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS
#define BN_K_NAME bn_dk_kern_fullamp_col_lcr_ots
#define BN_K_NL 18
#define BN_K_NX 70
#define BN_K_NU 3
#define BN_K_NO 15
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#define BN_K_R1_CUT 8
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS

/* The 50 W amp with ITS own output transformer (dsp/prototype/
 * ot_split.py: 17 primary sections + 3 secondary ones coupled, the derived copper and capacitances). Counts read
 * off `make mk81-dk-dims`: collapsed 81 C, 23 loops, 21 L = 79. The LCR twin (82) stays on the generic path. */
#define BN_K_NAME bn_dk_kern_fullamp_col_otd
#define BN_K_NL 18
#define BN_K_NX 79
#define BN_K_NU 3
#define BN_K_NO 15
#define BN_K_JG_BLOCKDIAG2 0
#define BN_K_JG_BLOCKS 1
#define BN_K_R1_CUT 8
#include "bn_dk_kernel.h"
#undef BN_K_NAME
#undef BN_K_NL
#undef BN_K_NX
#undef BN_K_NU
#undef BN_K_NO
#undef BN_K_JG_BLOCKDIAG2
#undef BN_K_JG_BLOCKS

/* ------------------------------------------------------- warm start ---- */
/* EXPERIMENT LEVER: a warm start for the per-sample Newton. The
 * converged answer is the Newton's, never the guess's (Holters & Zölzer,
 * EUSIPCO 2016 use a solution cache; here it is ONLY an initial guess).
 *
 *   BN_DK_WARM=<mode>[,<K>[,<delta>[,<lambda>]]]   read ONCE, in bn_dk_init
 *     0  lever set but no change of guess: statistics only (baseline count)
 *     1  linear extrapolation      v0 = 2·v[n-1] − v[n-2]
 *     2  solution cache            v0 = v of the stored (p, v) pair nearest to
 *                                  this sample's p (Euclidean, volts), if it is
 *                                  nearer than the previous sample's p; else v[n-1]
 *     3  mode 1 + one damped fixed-point step  v0 = v_e − lambda·r(v_e)
 *     4  tangent predictor         v0 = v[n-1] + Jg⁻¹·(p[n] − p[n-1]) with the LU
 *                                  the last sample left in dk->Jg (stale chord)
 *     5  cache + tangent           v0 = v_c + Jg⁻¹·(p[n] − p_c)
 *   K      cache entries (modes 2, 5), default 64
 *   delta  minimum distance (volts) from the nearest entry for a converged
 *          (p, v) to be inserted, default 0 (= every sample, ring buffer)
 *   lambda damping of mode 3, default 1.0
 *
 * Unset: dk->warm == NULL and bn_dk_process is the engine as before. Modes 2..5
 * recompute p = D·x + E·u here (the kernels compute their own copy; that
 * duplicate is counted in the wall time, ~nl·nx madds). Nothing here allocates
 * or calls the system after init; the exit report is a destructor. */
typedef struct BnDKWarm {
    int     mode, K, n, next, have_hist, have_jg, last_fallbacks;
    double  delta2, lambda;
    double *v1, *v2, *p_prev, *v0, *p_now;   /* nl each */
    double *cp, *cv;                          /* K·nl each: cached (p, v) */
    double  d_near2;                          /* squared distance to the nearest entry */
    int     guess_from;                       /* 0 baseline, 1 extrap, 2 cache, 3 tangent */
    /* RETRY: a guess is only a guess. If the Newton does not converge from it, the
     * sample is redone from the baseline guess (the previous solution) with the state
     * restored — so a warm start can never converge less often than the baseline. */
    double *x_save;                           /* nx: the state before the sample */
    int     fb_save;                          /* engine counters before the sample */
    double  fbmax_save, fbsum_save;
} BnDKWarm;

static int    bn_dk_warm_env_read = 0;
static int    bn_dk_warm_env_set  = 0;
static int    bn_dk_warm_mode     = 0;
static int    bn_dk_warm_K        = 64;
static double bn_dk_warm_delta    = 0.0;
static double bn_dk_warm_lambda   = 1.0;
/* statistics over every instance (the exit report) */
static long   bn_dk_warm_samples = 0, bn_dk_warm_iters = 0, bn_dk_warm_from[4] = {0, 0, 0, 0};
static long   bn_dk_warm_inserts = 0, bn_dk_warm_retries = 0, bn_dk_warm_retry_failed = 0;
static double bn_dk_warm_err_sum = 0.0, bn_dk_warm_err_max = 0.0;

static void bn_dk_warm_read_env(void)
{
    if (bn_dk_warm_env_read) { return; }
    bn_dk_warm_env_read = 1;
    const char* e = getenv("BN_DK_WARM");
    if (!e || !*e) { return; }
    bn_dk_warm_env_set = 1;
    char* end = NULL;
    bn_dk_warm_mode = (int)strtol(e, &end, 10);
    if (end && *end == ',') { bn_dk_warm_K = (int)strtol(end + 1, &end, 10); }
    if (end && *end == ',') { bn_dk_warm_delta = strtod(end + 1, &end); }
    if (end && *end == ',') { bn_dk_warm_lambda = strtod(end + 1, &end); }
    if (bn_dk_warm_K < 1) { bn_dk_warm_K = 1; }
}

__attribute__((destructor)) static void bn_dk_warm_report(void)
{
    if (!bn_dk_warm_env_set || bn_dk_warm_samples == 0) { return; }
    fprintf(stderr, "[BN_DK_WARM mode %d K %d delta %g lambda %g] samples %ld | iters/sample %.4f | "
            "guess from: baseline %ld extrap %ld cache %ld tangent %ld | inserts %ld | "
            "retries %ld (failed again %ld) | max|v0-v|: mean %.3e max %.3e\n",
            bn_dk_warm_mode, bn_dk_warm_K, bn_dk_warm_delta, bn_dk_warm_lambda,
            bn_dk_warm_samples, (double)bn_dk_warm_iters / (double)bn_dk_warm_samples,
            bn_dk_warm_from[0], bn_dk_warm_from[1], bn_dk_warm_from[2], bn_dk_warm_from[3],
            bn_dk_warm_inserts, bn_dk_warm_retries, bn_dk_warm_retry_failed,
            bn_dk_warm_err_sum / (double)bn_dk_warm_samples, bn_dk_warm_err_max);
}

static BnDKWarm* bn_dk_warm_alloc(int nl, int nx)
{
    bn_dk_warm_read_env();
    if (!bn_dk_warm_env_set || nl <= 0) { return NULL; }
    BnDKWarm* w = (BnDKWarm*)calloc(1, sizeof(BnDKWarm));
    if (!w) { return NULL; }
    w->mode = bn_dk_warm_mode; w->K = bn_dk_warm_K;
    w->delta2 = bn_dk_warm_delta * bn_dk_warm_delta; w->lambda = bn_dk_warm_lambda;
    w->v1 = (double*)calloc((size_t)nl, sizeof(double));
    w->v2 = (double*)calloc((size_t)nl, sizeof(double));
    w->p_prev = (double*)calloc((size_t)nl, sizeof(double));
    w->v0 = (double*)calloc((size_t)nl, sizeof(double));
    w->p_now = (double*)calloc((size_t)nl, sizeof(double));
    w->cp = (double*)calloc((size_t)w->K * (size_t)nl, sizeof(double));
    w->cv = (double*)calloc((size_t)w->K * (size_t)nl, sizeof(double));
    w->x_save = (double*)calloc((size_t)(nx > 0 ? nx : 1), sizeof(double));
    if (!w->v1 || !w->v2 || !w->p_prev || !w->v0 || !w->p_now || !w->cp || !w->cv || !w->x_save) {
        free(w->v1); free(w->v2); free(w->p_prev); free(w->v0); free(w->p_now);
        free(w->cp); free(w->cv); free(w->x_save); free(w);
        return NULL;
    }
    return w;
}

static void bn_dk_warm_free(BnDKWarm* w)
{
    if (!w) { return; }
    free(w->v1); free(w->v2); free(w->p_prev); free(w->v0); free(w->p_now);
    free(w->cp); free(w->cv); free(w->x_save); free(w);
}

static void bn_dk_warm_reset(BnDKWarm* w)
{
    if (!w) { return; }
    w->have_hist = 0; w->have_jg = 0; w->n = 0; w->next = 0;
}

/* p = D·x + E·u, the generic path's loop (modes 2..5 need it BEFORE the solve) */
static void bn_dk_warm_p(const BnDK* dk, const double* u, double* p)
{
    const int nx = dk->n_states, nl = dk->n_nl, nu = dk->n_in;
    for (int i = 0; i < nl; ++i) {
        double s = 0.0;
        for (int j = 0; j < nx; ++j) { s += IDX(dk->D, nx, i, j) * dk->x[j]; }
        for (int j = 0; j < nu; ++j) { s += IDX(dk->E, nu, i, j) * u[j]; }
        p[i] = s;
    }
}

/* one solve with the LU the last sample left behind, step capped at 100 V as
 * the Newton's own speed limit; dst += Jg⁻¹·rhs */
static void bn_dk_warm_tangent(const BnDK* dk, double* rhs, double* dst)
{
    const int nl = dk->n_nl;
    lu_solve(dk->Jg, dk->pivot, rhs, nl);
    double step = 0.0;
    for (int i = 0; i < nl; ++i) { if (fabs(rhs[i]) > step) { step = fabs(rhs[i]); } }
    const double scale = (step > 100.0) ? (100.0 / step) : 1.0;
    for (int i = 0; i < nl; ++i) { dst[i] += scale * rhs[i]; }
}

static void bn_dk_warm_before(BnDK* dk, const double* u)
{
    BnDKWarm* w = dk->warm;
    const int nl = dk->n_nl;
    w->guess_from = 0;
    w->d_near2 = -1.0;
    if (w->mode >= 2) { bn_dk_warm_p(dk, u, w->p_now); }
    if (!w->have_hist) {
        /* seed the history with the current v (the DC point, or the last v after a
         * sample that did not converge), exactly as the engine seeds v_prev: the
         * first extrapolation is then the no-op 2·v − v = v */
        for (int i = 0; i < nl; ++i) { w->v1[i] = dk->v[i]; w->v2[i] = dk->v[i]; }
        w->have_hist = 1;
        w->have_jg = 0;
    }
    {
        switch (w->mode) {
        case 1:
            for (int i = 0; i < nl; ++i) { dk->v[i] = 2.0 * w->v1[i] - w->v2[i]; }
            w->guess_from = 1;
            break;
        case 6:
            /* the PLAIN previous solution: dk->v is left as the last sample left it and the
             * core's own extrapolation is held off below — the control the product's preamp
             * needs, because there the engine ships with extrapolate_v = 1 */
            w->guess_from = 6;
            break;
        case 3: {
            double vt[BN_DK_NL_MAX], it_[BN_DK_NL_MAX];
            for (int i = 0; i < nl; ++i) { vt[i] = 2.0 * w->v1[i] - w->v2[i]; }
            if (dk->device_currents) { dk->device_currents(vt, it_, dk->device_user); }
            else                     { dk->device(vt, it_, dk->Jdev, dk->device_user); }
            for (int i = 0; i < nl; ++i) {
                double r = vt[i] - w->p_now[i];
                for (int j = 0; j < nl; ++j) { r += IDX(dk->F, nl, i, j) * it_[j]; }
                dk->v[i] = vt[i] - w->lambda * r;
            }
            w->guess_from = 1;
            break;
        }
        case 4:
            if (w->have_jg) {
                double d[BN_DK_NL_MAX];
                for (int i = 0; i < nl; ++i) { d[i] = w->p_now[i] - w->p_prev[i]; dk->v[i] = w->v1[i]; }
                bn_dk_warm_tangent(dk, d, dk->v);
                w->guess_from = 3;
            }
            break;
        default:
            break;
        }
    }
    if (w->mode == 2 || w->mode == 5) {
        /* nearest cached p, brute force; the previous sample's p competes */
        int best = -1;
        double bd = 1e300;
        for (int k = 0; k < w->n; ++k) {
            const double* c = w->cp + (size_t)k * (size_t)nl;
            double d = 0.0;
            for (int i = 0; i < nl; ++i) { const double e = w->p_now[i] - c[i]; d += e * e; }
            if (d < bd) { bd = d; best = k; }
        }
        w->d_near2 = bd;
        double dprev = 1e300;
        if (w->have_jg || w->n > 0) {   /* p_prev is valid once a sample converged */
            dprev = 0.0;
            for (int i = 0; i < nl; ++i) { const double e = w->p_now[i] - w->p_prev[i]; dprev += e * e; }
        }
        if (best >= 0 && bd < dprev) {
            const double* c = w->cp + (size_t)best * (size_t)nl;
            const double* cv = w->cv + (size_t)best * (size_t)nl;
            for (int i = 0; i < nl; ++i) { dk->v[i] = cv[i]; }
            w->guess_from = 2;
            if (w->mode == 5 && w->have_jg) {
                double d[BN_DK_NL_MAX];
                for (int i = 0; i < nl; ++i) { d[i] = w->p_now[i] - c[i]; }
                bn_dk_warm_tangent(dk, d, dk->v);
                w->guess_from = 3;
            }
        } else if (w->mode == 5 && w->have_jg) {
            double d[BN_DK_NL_MAX];
            for (int i = 0; i < nl; ++i) { d[i] = w->p_now[i] - w->p_prev[i]; dk->v[i] = w->v1[i]; }
            bn_dk_warm_tangent(dk, d, dk->v);
            w->guess_from = 3;
        }
    }
    for (int i = 0; i < nl; ++i) { w->v0[i] = dk->v[i]; }
    for (int i = 0; i < dk->n_states; ++i) { w->x_save[i] = dk->x[i]; }
    w->fb_save = dk->fallbacks; w->fbmax_save = dk->fb_resid_max; w->fbsum_save = dk->fb_resid_sum;
}

static void bn_dk_warm_after(BnDK* dk)
{
    BnDKWarm* w = dk->warm;
    const int nl = dk->n_nl;
    const int failed = (dk->fallbacks != w->last_fallbacks);
    w->last_fallbacks = dk->fallbacks;
    bn_dk_warm_samples++;
    bn_dk_warm_iters += dk->last_iters;
    bn_dk_warm_from[w->guess_from]++;
    {
        double e = 0.0;
        for (int i = 0; i < nl; ++i) { const double a = fabs(w->v0[i] - dk->v[i]); if (a > e) { e = a; } }
        bn_dk_warm_err_sum += e;
        if (e > bn_dk_warm_err_max) { bn_dk_warm_err_max = e; }
    }
    if (failed) {
        /* the engine's own self-healing: a sample that did not converge must not
         * feed an extrapolation, nor the cache, nor the tangent */
        w->have_hist = 0; w->have_jg = 0;
        return;
    }
    for (int i = 0; i < nl; ++i) { w->v2[i] = w->v1[i]; w->v1[i] = dk->v[i]; }
    if (w->mode >= 2) {
        /* the kernel wrote dk->p for this sample: the same vector as p_now */
        for (int i = 0; i < nl; ++i) { w->p_prev[i] = dk->p[i]; }
    }
    /* a factorisation happened in this sample iff the first residual test failed */
    if (dk->last_iters >= 2) { w->have_jg = 1; }
    if ((w->mode == 2 || w->mode == 5) && (w->n == 0 || w->d_near2 > w->delta2)) {
        double* c = w->cp + (size_t)w->next * (size_t)nl;
        double* cv = w->cv + (size_t)w->next * (size_t)nl;
        for (int i = 0; i < nl; ++i) { c[i] = dk->p[i]; cv[i] = dk->v[i]; }
        w->next = (w->next + 1) % w->K;
        if (w->n < w->K) { w->n++; }
        bn_dk_warm_inserts++;
    }
}

/* --------------------------------------------------------------- process -- */
/* one place decides the dispatch, so that bn_dk_kernel_name() cannot drift from it */
static int kernel_id(const BnDK* dk)
{
    const int nx = dk->n_states, nl = dk->n_nl, no = dk->n_out, nu = dk->n_in;
    if (nl == 8 && nx == 18 && nu == 2 && no == 9
        && dk->jdev_blockdiag && !dk->jdev_bstart && dk->At) { return 1; }
    if (nl == 24 && nx == 48 && nu == 3 && no == 15
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 2; }
    if (nl == 18 && nx == 44 && nu == 3 && no == 15
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 3; }
    if (nl == 18 && nx == 47 && nu == 3 && no == 15 && dk->r1_cut == 8
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 4; }
    if (nl == 24 && nx == 71 && nu == 3 && no == 15
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 5; }
    if (nl == 18 && nx == 67 && nu == 3 && no == 15
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 6; }
    if (nl == 18 && nx == 70 && nu == 3 && no == 15 && dk->r1_cut == 8
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 7; }
    if (nl == 18 && nx == 79 && nu == 3 && no == 15
        && dk->jdev_bstart && dk->jdev_bsize && dk->At) { return 8; }
    return 0;
}

int bn_dk_rank1_enable(BnDK* dk, int cut)
{
    const int nl = dk->n_nl;
    if (cut != 8 || nl != 18 || !dk->jdev_bstart || !dk->jdev_bsize || dk->warm) { return -1; }
    for (int j = 0; j < nl; ++j) {          /* no device block may straddle the cut */
        const int b0 = dk->jdev_bstart[j], b1 = b0 + dk->jdev_bsize[j];
        if ((j < cut) != (b0 < cut) || (j < cut) != (b1 - 1 < cut)) { return -1; }
    }
    if (!dk->r1_u) {
        dk->r1_u = (double*)calloc((size_t)nl, sizeof(double));
        dk->r1_w = (double*)calloc((size_t)nl, sizeof(double));
        dk->r1_s = (double*)calloc((size_t)nl, sizeof(double));
        dk->r1_t = (double*)calloc((size_t)nl, sizeof(double));
        if (!dk->r1_u || !dk->r1_w || !dk->r1_s || !dk->r1_t) { return -1; }
    }
    dk->r1_cut = cut;
    bn_dk_rank1_factor(dk);
    const int kid = kernel_id(dk);
    if (!dk->r1_ok || (kid != 3 && kid != 4 && kid != 6 && kid != 7 && kid != 8)) {
        dk->r1_cut = 0; dk->r1_ok = 0;
        dk->have_lu = 0; dk->pred_ok = 0; dk->inv_ok = 0;
        return -1;
    }
    return 0;
}

const char* bn_dk_kernel_name(const BnDK* dk)
{
    static const char* const names[9] = { 0, "preamp2203", "fullamp", "fullamp_col", "fullamp_col_lcr",
                                          "fullamp_ots", "fullamp_col_ots", "fullamp_col_lcr_ots", "fullamp_col_otd" };
    return names[kernel_id(dk)];
}

static void bn_dk_process_core(BnDK* dk, const double* u, double* y)
{
    const int nx = dk->n_states, nl = dk->n_nl, no = dk->n_out, nu = dk->n_in;

    switch (kernel_id(dk)) {
        case 1: bn_dk_kern_preamp2203(dk, u, y); return;
        case 2: bn_dk_kern_fullamp(dk, u, y); return;
        case 3: bn_dk_kern_fullamp_col(dk, u, y); return;
        case 4: bn_dk_kern_fullamp_col_lcr(dk, u, y); return;
        case 5: bn_dk_kern_fullamp_ots(dk, u, y); return;
        case 6: bn_dk_kern_fullamp_col_ots(dk, u, y); return;
        case 7: bn_dk_kern_fullamp_col_lcr_ots(dk, u, y); return;
        case 8: bn_dk_kern_fullamp_col_otd(dk, u, y); return;
        default: break;
    }

    /* p = D·x + E·u — dp = p[n] − p[n−1] is taken BEFORE dk->p is overwritten (tangent predictor) */
    const int use_pred = dk->predict_tangent && dk->pred_ok;
    double dpp[BN_DK_NL_MAX];
    for (int i = 0; i < nl; ++i) {
        double s = 0.0;
        for (int j = 0; j < nx; ++j) { s += IDX(dk->D, nx, i, j) * dk->x[j]; }
        for (int j = 0; j < nu; ++j) { s += IDX(dk->E, nu, i, j) * u[j]; }
        if (use_pred) { dpp[i] = s - dk->p[i]; }
        dk->p[i] = s;
    }

    /* Damped Newton on v = p + F·i(v).
     * The sign is inverted here: the model returns the current that enters
     * the device and the DK injects it into the nodes. */
    int it = 0;
    if (nl > 0) {
        if (!dk->chord_persist) { dk->have_lu = 0; }
        if (use_pred) {
            /* tangent predictor with the last LU; v_prev keeps the extrapolation's meaning */
            lu_solve(dk->Jg, dk->pivot, dpp, nl);
            double step = 0.0;
            for (int i = 0; i < nl; ++i) { if (fabs(dpp[i]) > step) { step = fabs(dpp[i]); } }
            const double scale = (step > 100.0) ? (100.0 / step) : 1.0;
            for (int i = 0; i < nl; ++i) {
                const double vp = dk->v[i];
                dk->v[i] = vp + scale * dpp[i];
                dk->v_prev[i] = vp;
            }
        } else if (dk->extrapolate_v) {
            /* start by extrapolation: linear v_guess = 2v - v_prev (or the
             * quadratic form when extrapolate_order == 2). At high fs the
             * samples change little and this saves iterations; if the
             * prediction is bad, the Newton's damping corrects it. */
            if (dk->extrapolate_order == 2) {
                for (int i = 0; i < nl; ++i) {
                    const double vp = dk->v[i], vq = dk->v_prev[i];
                    dk->v[i] = 3.0 * vp - 3.0 * vq + dk->v_prev2[i];
                    dk->v_prev2[i] = vq; dk->v_prev[i] = vp;
                }
            } else {
            for (int i = 0; i < nl; ++i) {
                const double vp = dk->v[i];
                dk->v[i] = 2.0 * vp - dk->v_prev[i];
                dk->v_prev[i] = vp;
            }
            }
        }
        double prev_worst = 1e300;
        double last_worst = 0.0;   /* the residual the loop exits with */
        /* same idea as in the kernel: with the chord alive the currents are
         * enough, the Jacobian is only needed when refactorising.
         * Bit-exact with chord_persist=0. */
        if (dk->have_lu && dk->device_currents) {
            dk->device_currents(dk->v, dk->i_nl, dk->device_user);
        } else {
            dk->device(dk->v, dk->i_nl, dk->Jdev, dk->device_user);
        }
        for (it = 0; it < dk->max_iter; ++it) {
            double worst = 0.0;
            {
                /* profiling (gprof) put 58% of the time in these inline
                 * loops: restrict so the compiler can pipeline them */
                const double* restrict vv = dk->v;
                const double* restrict pp = dk->p;
                const double* restrict ii = dk->i_nl;
                double* restrict rr = dk->rhs;
                for (int i = 0; i < nl; ++i) {
                    const double* restrict Fr = dk->F + (size_t)i * nl;
                    double s = vv[i] - pp[i];
                    for (int j = 0; j < nl; ++j) { s += Fr[j] * ii[j]; }
                    rr[i] = -s;
                    if (!(fabs(s) <= worst)) { worst = fabs(s); }   /* NaN-proof, see bn_dk_kernel.h */
                }
            }
            last_worst = worst;
            if (worst < dk->tol) { break; }
            /* CHORD: if the residual does not drop enough with the old
             * factorisation, a fresh Jacobian is due. The ratio is `chord_refresh`
             * (BN_CHORD_REFRESH; in a sweep the plugin's optimum at tol 1e-4 is 0.3-0.5,
             * the bench at 1e-8 wants 0.1). */
            if (dk->have_lu && worst > dk->chord_refresh * prev_worst) {
                /* Whether device_currents exists does not matter here: when
                 * the Jacobian must be refreshed only dk->device computes it,
                 * and it brings `i` up to date as well. */
                dk->device(dk->v, dk->i_nl, dk->Jdev, dk->device_user);
                dk->have_lu = 0;
            }
            prev_worst = worst;
            /* Jdev is 2x2 block-diagonal (a triode only couples with
             * itself): column j has only 2 non-zero rows. 128 FMA instead
             * of 512 with nl=8. For devices that are NOT block-diagonal this
             * shortcut would be wrong: it is enabled only if the device
             * declares it (dk->jdev_blockdiag). */
            if (!dk->have_lu) {
                if (dk->jdev_bstart && dk->jdev_bsize) {
                    /* variable-size blocks (2x2 triodes + 3x3 pentodes) */
                    for (int j = 0; j < nl; ++j) {
                        const int b0 = dk->jdev_bstart[j];
                        const int bn = dk->jdev_bsize[j];
                        for (int i = 0; i < nl; ++i) {
                            double s = (i == j) ? 1.0 : 0.0;
                            for (int r = b0; r < b0 + bn; ++r) {
                                s += IDX(dk->F, nl, i, r) * IDX(dk->Jdev, nl, r, j);
                            }
                            IDX(dk->Jg, nl, i, j) = s;
                        }
                    }
                } else if (dk->jdev_blockdiag) {
                    for (int j = 0; j < nl; ++j) {
                        const int r0 = j & ~1, r1 = r0 + 1;
                        const double d0 = IDX(dk->Jdev, nl, r0, j);
                        const double d1 = IDX(dk->Jdev, nl, r1, j);
                        for (int i = 0; i < nl; ++i) {
                            IDX(dk->Jg, nl, i, j) = ((i == j) ? 1.0 : 0.0)
                                + IDX(dk->F, nl, i, r0) * d0
                                + IDX(dk->F, nl, i, r1) * d1;
                        }
                    }
                } else {
                    for (int i = 0; i < nl; ++i) {
                        for (int j = 0; j < nl; ++j) {
                            double s = (i == j) ? 1.0 : 0.0;
                            for (int k = 0; k < nl; ++k) {
                                s += IDX(dk->F, nl, i, k) * IDX(dk->Jdev, nl, k, j);
                            }
                            IDX(dk->Jg, nl, i, j) = s;
                        }
                    }
                }
                if (lu_factor(dk->Jg, nl, dk->pivot) != 0) {
                    /* The contract is «fallbacks = 0 => the Newton always
                     * converged». A singular Jacobian that broke the loop
                     * without touching the counter would look EXACTLY like a
                     * clean convergence, so it counts as what it is: a
                     * fallback, with its residual. */
                    dk->fallbacks++;
                    if (last_worst > dk->fb_resid_max) { dk->fb_resid_max = last_worst; }
                    dk->fb_resid_sum += last_worst;
                    dk->pred_ok = 0;      /* a singular LU predicts nothing */
                    break;
                }
                dk->have_lu = 1;
                dk->pred_ok = 1;          /* predictor: Jg now holds a usable factorisation */
                dk->inv_ok = 0;
            }
            if (dk->solve_inverse) {
                if (!dk->inv_ok) {
                    /* explicit inverse: Jinv column by column from the unit vectors, with the LU just made */
                    for (int c = 0; c < nl; ++c) {
                        double e[BN_DK_NL_MAX];
                        for (int i = 0; i < nl; ++i) { e[i] = (i == c) ? 1.0 : 0.0; }
                        lu_solve(dk->Jg, dk->pivot, e, nl);
                        for (int i = 0; i < nl; ++i) { IDX(dk->Jinv, nl, i, c) = e[i]; }
                    }
                    dk->inv_ok = 1;
                }
                double t[BN_DK_NL_MAX];
                for (int i = 0; i < nl; ++i) {
                    double s = 0.0;
                    for (int j = 0; j < nl; ++j) { s += IDX(dk->Jinv, nl, i, j) * dk->rhs[j]; }
                    t[i] = s;
                }
                for (int i = 0; i < nl; ++i) { dk->rhs[i] = t[i]; }
            } else {
                lu_solve(dk->Jg, dk->pivot, dk->rhs, nl);
            }
            double step = 0.0;
            for (int i = 0; i < nl; ++i) {
                if (fabs(dk->rhs[i]) > step) { step = fabs(dk->rhs[i]); }
            }
            /* SAME step cap as in the kernel: the cap is a SPEED limit, not
             * a damper (a 5 V cap is too low — see bn_dk_kernel.h). Circuits
             * whose dimensions match no kernel (e.g. the oracle with the LCR
             * load) land here, so this path needs it too. Kernel and generic
             * path MUST carry the same value: a gate compares their outputs
             * byte for byte. */
            const double scale = (step > 100.0) ? (100.0 / step) : 1.0;
            /* BACKTRACKING on the residual norm: with many ports at corners
             * simultaneously the fixed step BOUNCES (measured: the whole amp
             * at 500 mV hit the full budget even at 8x). EXACT mirror of the
             * prototype (dk.py): 4 tries, halving the step each time, and if
             * none improves, the shortest is accepted. */
            {
                double alpha = 1.0;
                /* bound guaranteed by the guard in bn_dk_init */
                double vt[BN_DK_NL_MAX], it_[BN_DK_NL_MAX];
                int accepted = 0;
                for (int bt = 0; bt < 4 && !accepted; ++bt) {
                    for (int i = 0; i < nl; ++i) {
                        vt[i] = dk->v[i] + alpha * scale * dk->rhs[i];
                    }
                    if (dk->device_currents) {
                        dk->device_currents(vt, it_, dk->device_user);
                    } else {
                        dk->device(vt, it_, dk->Jdev, dk->device_user);
                        dk->have_lu = 0;
                    }
                    double wt = 0.0;
                    {
                        const double* restrict pp = dk->p;
                        for (int i = 0; i < nl; ++i) {
                            const double* restrict Fr = dk->F + (size_t)i * nl;
                            double s2 = vt[i] - pp[i];
                            for (int j = 0; j < nl; ++j) { s2 += Fr[j] * it_[j]; }
                            /* `!(x <= wt)` and not `x > wt`: with `>` a NaN residual never
                             * beats 0.0, so wt stays 0, this step reads as the BEST one and
                             * is accepted at the first try instead of backtracking. Same
                             * reason as the twin in bn_dk_kernel.h and the residual test
                             * above; every copy needs it. */
                            if (!(fabs(s2) <= wt)) { wt = fabs(s2); }
                        }
                    }
                    if (wt < worst || bt == 3) { accepted = 1; }
                    else { alpha *= 0.5; }
                }
                for (int i = 0; i < nl; ++i) {
                    dk->v[i] = vt[i];
                    dk->i_nl[i] = it_[i];
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
            for (int i = 0; i < nl; ++i) { dk->v_prev[i] = dk->v[i]; dk->v_prev2[i] = dk->v[i]; }
        }
        for (int i = 0; i < nl; ++i) { dk->i_nl[i] = -dk->i_nl[i]; }
    }
    dk->last_iters = it + 1;

    /* outputs and state update — restrict pointers and dual accumulators
     * so the compiler vectorises (measured: the generic IDX form did not
     * pipeline; -O3 alone did nothing) */
    {
        const double* restrict xv2 = dk->x;
        const double* restrict iv2 = dk->i_nl;
        for (int i = 0; i < no; ++i) {
            const double* restrict Lr = dk->L + (size_t)i * nx;
            const double* restrict Nr = dk->N + (size_t)i * nl;
            double s0 = 0.0, s1 = 0.0;
            for (int j = 0; j < nx; ++j) { s0 += Lr[j] * xv2[j]; }
            for (int j = 0; j < nl; ++j) { s1 += Nr[j] * iv2[j]; }
            for (int j = 0; j < nu; ++j) { s0 += IDX(dk->M, nu, i, j) * u[j]; }
            y[i] = s0 + s1;
        }
        for (int i = 0; i < nx; ++i) {
            const double* restrict Ar = dk->A + (size_t)i * nx;
            const double* restrict Cr = dk->C + (size_t)i * nl;
            double s0 = 0.0, s1 = 0.0;
            for (int j = 0; j < nx; ++j) { s0 += Ar[j] * xv2[j]; }
            for (int j = 0; j < nl; ++j) { s1 += Cr[j] * iv2[j]; }
            for (int j = 0; j < nu; ++j) { s0 += IDX(dk->B, nu, i, j) * u[j]; }
            dk->x_next[i] = s0 + s1;
        }
    }
    for (int i = 0; i < nx; ++i) { dk->x[i] = dk->x_next[i]; }
}

void bn_dk_process(BnDK* dk, const double* u, double* y)
{
    /* dk->warm is NULL unless BN_DK_WARM was set at init: one predictable branch */
    if (dk->warm) { bn_dk_warm_before(dk, u); }
    /* ⛔ While a warm guess is active the core must NOT extrapolate on top of it: it would
     * do v = 2·guess − v_prev and store the GUESS as v_prev, so a mode measured inside the
     * plugin (extrapolate_v = 1) would be 'guess + a corrupted extrapolation', not the guess
     * (which made the tangent mode look worse than it is on the preamp). The flag is held off for
     * the core call and restored after; the bench (extrapolate_v = 0) is unaffected. */
    const int extr_saved = dk->extrapolate_v;
    if (dk->warm && dk->warm->guess_from != 0) { dk->extrapolate_v = 0; }
    bn_dk_process_core(dk, u, y);
    if (dk->warm) {
        BnDKWarm* w = dk->warm;
        if (dk->fallbacks != w->fb_save && w->guess_from != 0) {
            /* the warm guess did not converge: redo the sample from the previous
             * solution with the state restored — the baseline's own path */
            const int nl = dk->n_nl, nx = dk->n_states;
            const int it1 = dk->last_iters;
            for (int i = 0; i < nx; ++i) { dk->x[i] = w->x_save[i]; }
            for (int i = 0; i < nl; ++i) { dk->v[i] = w->v1[i]; w->v0[i] = w->v1[i]; }
            dk->fallbacks = w->fb_save; dk->fb_resid_max = w->fbmax_save; dk->fb_resid_sum = w->fbsum_save;
            dk->have_lu = 0;
            w->guess_from = 0;
            bn_dk_warm_retries++;
            bn_dk_process_core(dk, u, y);
            if (dk->fallbacks != w->fb_save) { bn_dk_warm_retry_failed++; }
            dk->last_iters += it1;   /* the sample cost both attempts */
        }
        bn_dk_warm_after(dk);
    }
    dk->extrapolate_v = extr_saved;
}

/* Start at the operating point WITHOUT simulating the warm-up (the
 * prototype's trick, test_preamp_dk.py::warm_system): the STATIC DC is
 * solved (same circuit without capacitors, outputs at every node, one Newton)
 * and x = (2C/T)·vc is initialised per capacitor — the exact fixed point of
 * the state recursion. With C21 (τ=0.5 s) a brute-force warm-up would take
 * hundreds of thousands of samples in instantiate. */
int bn_dk_dc_init(BnDK* dk, const BnCircuit* c, double fs, const double* u_dc)
{
    const int nn = c->n_nodes;
    const int nL = c->n_inductors;
    BnPort* all_nodes = (BnPort*)calloc((size_t)nn, sizeof(BnPort));
    if (!all_nodes) { return -1; }
    for (int k = 0; k < nn; ++k) { all_nodes[k].a = k + 1; all_nodes[k].b = BN_GND; }

    /* In the static DC the inductors are SHORTS (1 mΩ), not companions; their
     * state is initialised afterwards with x_L = −I_L (fixed point with Z=−1). */
    const double RSHORT = 1e-3;
    BnElement* res2 = (BnElement*)calloc((size_t)(c->n_resistors + (nL > 0 ? nL : 1)),
                                         sizeof(BnElement));
    if (!res2) { free(all_nodes); return -1; }
    memcpy(res2, c->resistors, sizeof(BnElement) * (size_t)c->n_resistors);
    for (int e = 0; e < nL; ++e) {
        res2[c->n_resistors + e].a = c->inductors[e].a;
        res2[c->n_resistors + e].b = c->inductors[e].b;
        res2[c->n_resistors + e].value = RSHORT;
    }

    BnCircuit sc = *c;
    sc.resistors = res2; sc.n_resistors = c->n_resistors + nL;
    sc.capacitors = 0; sc.n_capacitors = 0;
    sc.inductors = 0; sc.n_inductors = 0;
    sc.mutuals = 0; sc.n_mutuals = 0;
    sc.outputs = all_nodes; sc.n_outputs = nn;

    BnDK sdk;
    if (bn_dk_init(&sdk, &sc, fs) != 0) { free(all_nodes); free(res2); return -1; }
    sdk.device          = dk->device;
    sdk.device_currents = dk->device_currents;
    sdk.device_user     = dk->device_user;
    sdk.jdev_blockdiag  = dk->jdev_blockdiag;
    sdk.jdev_ig_indep_va = dk->jdev_ig_indep_va;   /* propagated, or the static DC
                                                    * solve would assemble Jg by a
                                                    * different path */
    /* ⚠️ bstart/bsize MUST be propagated. Without them, with jdev_blockdiag=1
     * the Jg assembly takes the 2x2-block shortcut (r0 = j&~1) on devices whose
     * blocks are 3x3 — the full amp's pentodes: column 14 would read rows 14,15
     * (and 15 is ANOTHER pentode) and miss rows 12,13. The Newton residual is
     * still exact, so if it converges the DC point is correct and the defect
     * only costs iterations; the risk is failing to converge unnoticed. */
    sdk.jdev_bstart     = dk->jdev_bstart;
    sdk.jdev_bsize      = dk->jdev_bsize;
    sdk.tol             = 1e-12;      /* the DC is solved once, and solved tight */
    sdk.max_iter        = 200;

    double* vn = (double*)calloc((size_t)nn, sizeof(double));
    if (!vn) { bn_dk_free(&sdk); free(all_nodes); free(res2); return -1; }
    bn_dk_process(&sdk, u_dc, vn);
    const int fell = sdk.fallbacks;

    /* the companion source of every capacitor, then folded onto the forest states
     * (z = P.x_full; the DC node voltages satisfy KVL, so nothing is lost) */
    const int nc = c->n_capacitors, nt = dk->cap_nt;
    double* xf = (double*)calloc((size_t)(nc > 0 ? nc : 1), sizeof(double));
    if (!xf) { free(vn); bn_dk_free(&sdk); free(all_nodes); free(res2); return -1; }
    for (int e = 0; e < nc; ++e) {
        const int a = c->capacitors[e].a, b = c->capacitors[e].b;
        const double va = (a == BN_GND) ? 0.0 : vn[a - 1];
        const double vb = (b == BN_GND) ? 0.0 : vn[b - 1];
        xf[e] = 2.0 * c->capacitors[e].value * fs * (va - vb);
    }
    for (int t = 0; t < nt; ++t) {
        double s = 0.0;
        for (int e = 0; e < nc; ++e) {
            const double p = IDX(dk->cap_P, nc, t, e);
            if (p != 0.0) { s += p * xf[e]; }
        }
        dk->x[t] = s;
    }
    free(xf);
    for (int e = 0; e < nL; ++e) {
        const int a = c->inductors[e].a, b = c->inductors[e].b;
        const double va = (a == BN_GND) ? 0.0 : vn[a - 1];
        const double vb = (b == BN_GND) ? 0.0 : vn[b - 1];
        dk->x[nt + e] = -(va - vb) / RSHORT;   /* x_L = −I_L */
    }
    for (int p = 0; p < c->n_nonlinear; ++p) {
        const int a = c->nonlinear[p].a, b = c->nonlinear[p].b;
        const double va = (a == BN_GND) ? 0.0 : vn[a - 1];
        const double vb = (b == BN_GND) ? 0.0 : vn[b - 1];
        dk->v[p] = va - vb;
        if (dk->v_prev) { dk->v_prev[p] = dk->v[p]; }
    }
    free(vn); bn_dk_free(&sdk); free(all_nodes); free(res2);
    return fell ? -2 : 0;
}

/* ------------------------------------------------------------ variables -- */
/* Variable parts via Woodbury: S(g) = S0 + Nv'·ΔG·Nv, and every state-space
 * matrix is P·S⁻¹·R, so
 *   P·S⁻¹·R = base − PB·ΔG·(I + Q·ΔG)⁻¹·BR
 * with base/PB/BR/Q precomputed on S0. The (I+QΔG)⁻¹ form tolerates
 * sections that do not move (ΔG=0). Mirror of dk_variable.py; the assembly
 * of S below DUPLICATES the one in bn_dk_init on purpose — the parity test in
 * tools/bn_test_var.c catches any drift between the two. */

int bn_dkvar_init(BnDKVar* vk, BnDK* dk, const BnCircuit* c, double fs,
                  const int* var_idx, int nv)
{
    memset(vk, 0, sizeof(*vk));
    const int nn = c->n_nodes, nr = c->n_resistors;
    const int nL = c->n_inductors;
    const int nx = dk->cap_nt + nL;          /* the capacitor forest's states, as in bn_dk_init */
    const int nu = c->n_sources, nl = c->n_nonlinear, no = c->n_outputs;
    const int ns = nn + nu;
    const double T = 1.0 / fs;
    if (dk->cap_nc != c->n_capacitors || nx != dk->n_states) { return -1; }
    /* Same case as nl in bn_dk_init — var_apply uses
     * `double col[BN_DK_NV_MAX]` on the stack, so the bound is checked here.
     * The actual maximum is 13 (the whole amp's controls): 2.4x headroom. */
    if (nv > BN_DK_NV_MAX) { return -1; }

    vk->dk = dk; vk->nv = nv; vk->nx = nx; vk->nu = nu; vk->nl = nl; vk->no = no;
    /* TODO (knobs of the full amp): the A/B/C transformation already uses the whole GXm
     * (bn_dkvar_set); what is left for coupled inductors is the mutual-inductance
     * block that reactive_build adds, which bn_dkvar_set does not rebuild yet. The dkvar
     * covers C + uncoupled L (the preamp has no L). */
    if (c->n_mutuals > 0) { return -1; }

    vk->idx = (int*)calloc((size_t)nv, sizeof(int));
    vk->r0  = (double*)calloc((size_t)nv, sizeof(double));
    for (int k = 0; k < nv; ++k) {
        vk->idx[k] = var_idx[k];
        vk->r0[k]  = c->resistors[var_idx[k]].value;
    }

    double* NR = mat_alloc(nr, nn);
    double* NU = mat_alloc(nu, nn);
    double* NN = mat_alloc(nl, nn);
    double* NO = mat_alloc(no, nn);
    fill_incidence_elem(NR, c->resistors, nr, nn);
    fill_incidence(NU, c->sources, nu, nn);
    fill_incidence(NN, c->nonlinear, nl, nn);
    fill_incidence(NO, c->outputs, no, nn);

    /* NX / GXm / zs from the SAME builder as bn_dk_init. The shared builder makes a
     * second, drifting copy of the mutuals loop here impossible. GXm and zs are KEPT for set(), where the state transform is a
     * matrix product (the capacitor block is dense when the netlist has loops). */
    double *NX, *GXm, *zs;
    if (reactive_build(c, T, dk->cap_nt, dk->cap_tree, dk->cap_P, &NX, &GXm, &zs) != 0) {
        free(NR); free(NU); free(NN); free(NO); free(vk->idx); free(vk->r0);
        memset(vk, 0, sizeof(*vk)); return -1;
    }

    double* S = mat_alloc(ns, ns);
    for (int e = 0; e < nr; ++e) {
        const double g = 1.0 / c->resistors[e].value;
        for (int i = 0; i < nn; ++i) {
            const double ni = IDX(NR, nn, e, i);
            if (ni == 0.0) { continue; }
            for (int j = 0; j < nn; ++j) {
                IDX(S, ns, i, j) += ni * g * IDX(NR, nn, e, j);
            }
        }
    }
    /* NX' * GXm * NX (GXm is not diagonal when there are coupled inductors) */
    for (int e = 0; e < nx; ++e) {
        for (int f2 = 0; f2 < nx; ++f2) {
            const double g = IDX(GXm, nx, e, f2);
            if (g == 0.0) { continue; }
            for (int i = 0; i < nn; ++i) {
                const double ni = IDX(NX, nn, e, i);
                if (ni == 0.0) { continue; }
                for (int j = 0; j < nn; ++j) {
                    IDX(S, ns, i, j) += ni * g * IDX(NX, nn, f2, j);
                }
            }
        }
    }
    for (int e = 0; e < nu; ++e) {
        for (int i = 0; i < nn; ++i) {
            IDX(S, ns, i, nn + e) = IDX(NU, nn, e, i);
            IDX(S, ns, nn + e, i) = IDX(NU, nn, e, i);
        }
    }
    double* Sinv = mat_alloc(ns, ns);
    if (invert_dense(S, Sinv, ns) != 0) {
        free(S); free(Sinv); free(NR); free(NX); free(NU); free(NN); free(NO);
        free(GXm); free(zs); free(vk->idx); free(vk->r0); memset(vk, 0, sizeof(*vk));
        return -1;
    }

    /* incidence of the variable edges, in the [phi; is] space */
    double* Nv = mat_alloc(nv, ns);
    for (int r = 0; r < nv; ++r) {
        const BnElement* el = &c->resistors[var_idx[r]];
        if (el->a != BN_GND) { IDX(Nv, ns, r, el->a - 1) =  1.0; }
        if (el->b != BN_GND) { IDX(Nv, ns, r, el->b - 1) = -1.0; }
    }

    /* selectors and injectors as in bn_dk_init */
    double* Px = mat_alloc(nx, ns);
    double* Pn = mat_alloc(nl, ns);
    double* Po = mat_alloc(no, ns);
    for (int r = 0; r < nx; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Px, ns, r, cc) = IDX(NX, nn, r, cc); } }
    for (int r = 0; r < nl; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Pn, ns, r, cc) = IDX(NN, nn, r, cc); } }
    for (int r = 0; r < no; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Po, ns, r, cc) = IDX(NO, nn, r, cc); } }
    double* Rx = mat_alloc(ns, nx);
    double* Ru = mat_alloc(ns, nu);
    double* Rn = mat_alloc(ns, nl);
    for (int r = 0; r < nx; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Rx, nx, cc, r) = IDX(NX, nn, r, cc); } }
    for (int r = 0; r < nl; ++r) { for (int cc = 0; cc < nn; ++cc) { IDX(Rn, nl, cc, r) = IDX(NN, nn, r, cc); } }
    for (int e = 0; e < nu; ++e) { IDX(Ru, nu, nn + e, e) = 1.0; }

    /* arms and bases */
    double* SiNvT = mat_alloc(ns, nv);          /* S0inv · Nv' */
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nv; ++j) {
            double s = 0.0;
            for (int k = 0; k < ns; ++k) { s += IDX(Sinv, ns, i, k) * IDX(Nv, ns, j, k); }
            IDX(SiNvT, nv, i, j) = s;
        }
    }
    vk->Q = mat_alloc(nv, nv);
    mat_mul(Nv, nv, ns, SiNvT, nv, vk->Q);

    double* tx = mat_alloc(nx, ns);
    double* tn = mat_alloc(nl, ns);
    double* to = mat_alloc(no, ns);
    mat_mul(Px, nx, ns, Sinv, ns, tx);
    mat_mul(Pn, nl, ns, Sinv, ns, tn);
    mat_mul(Po, no, ns, Sinv, ns, to);

    vk->PBx = mat_alloc(nx, nv); mat_mul(Px, nx, ns, SiNvT, nv, vk->PBx);
    vk->PBn = mat_alloc(nl, nv); mat_mul(Pn, nl, ns, SiNvT, nv, vk->PBn);
    vk->PBo = mat_alloc(no, nv); mat_mul(Po, no, ns, SiNvT, nv, vk->PBo);
    vk->BRx = mat_alloc(nv, nx); mat_mul(Nv, nv, ns, Sinv, ns, S);  /* S = Nv·S0inv (reused) */
    mat_mul(S, nv, ns, Rx, nx, vk->BRx);
    vk->BRu = mat_alloc(nv, nu); mat_mul(S, nv, ns, Ru, nu, vk->BRu);
    vk->BRn = mat_alloc(nv, nl); mat_mul(S, nv, ns, Rn, nl, vk->BRn);

    vk->bxx = mat_alloc(nx, nx); mat_mul(tx, nx, ns, Rx, nx, vk->bxx);
    vk->bxu = mat_alloc(nx, nu); mat_mul(tx, nx, ns, Ru, nu, vk->bxu);
    vk->bxn = mat_alloc(nx, nl); mat_mul(tx, nx, ns, Rn, nl, vk->bxn);
    vk->bnx = mat_alloc(nl, nx); mat_mul(tn, nl, ns, Rx, nx, vk->bnx);
    vk->bnu = mat_alloc(nl, nu); mat_mul(tn, nl, ns, Ru, nu, vk->bnu);
    vk->bnn = mat_alloc(nl, nl); mat_mul(tn, nl, ns, Rn, nl, vk->bnn);
    vk->box = mat_alloc(no, nx); mat_mul(to, no, ns, Rx, nx, vk->box);
    vk->bou = mat_alloc(no, nu); mat_mul(to, no, ns, Ru, nu, vk->bou);
    vk->bon = mat_alloc(no, nl); mat_mul(to, no, ns, Rn, nl, vk->bon);

    vk->GXm = GXm; vk->zs = zs;                 /* owned by vk from here */
    vk->Axs = mat_alloc(nx, nx);
    vk->Bxs = mat_alloc(nx, nu);
    vk->Cxs = mat_alloc(nx, nl);

    vk->dG = (double*)calloc((size_t)nv, sizeof(double));
    vk->M  = mat_alloc(nv, nv);
    vk->W  = mat_alloc(nv, (nx > nu ? (nx > nl ? nx : nl) : (nu > nl ? nu : nl)));
    vk->piv = (int*)calloc((size_t)nv, sizeof(int));

    free(NR); free(NX); free(NU); free(NN); free(NO);
    free(S); free(Sinv); free(Nv); free(SiNvT);
    free(Px); free(Pn); free(Po); free(Rx); free(Ru); free(Rn);
    free(tx); free(tn); free(to);
    return 0;
}

/* out(rows×cols) = base − PB(rows×nv) · [ΔG·(I+QΔG)⁻¹·BR](nv×cols) */
static void var_apply(BnDKVar* vk, const double* base, const double* PB,
                      const double* BR, int rows, int cols, double* out)
{
    const int nv = vk->nv;
    /* W = M⁻¹·BR (column by column), then scaled by rows with dG */
    for (int j = 0; j < cols; ++j) {
        double col[BN_DK_NV_MAX];   /* bound guaranteed by bn_dkvar_init */
        for (int i = 0; i < nv; ++i) { col[i] = IDX(BR, cols, i, j); }
        lu_solve(vk->M, vk->piv, col, nv);
        for (int i = 0; i < nv; ++i) { IDX(vk->W, cols, i, j) = vk->dG[i] * col[i]; }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double s = IDX(base, cols, i, j);
            for (int k = 0; k < nv; ++k) {
                s -= IDX(PB, nv, i, k) * IDX(vk->W, cols, k, j);
            }
            IDX(out, cols, i, j) = s;
        }
    }
}

int bn_dkvar_set(BnDKVar* vk, const double* vals)
{
    const int nv = vk->nv, nx = vk->nx, nu = vk->nu, nl = vk->nl, no = vk->no;
    BnDK* dk = vk->dk;
    for (int k = 0; k < nv; ++k) {
        vk->dG[k] = 1.0 / vals[k] - 1.0 / vk->r0[k];
    }
    /* M = I + Q·diag(dG), factorised once per move */
    for (int i = 0; i < nv; ++i) {
        for (int j = 0; j < nv; ++j) {
            IDX(vk->M, nv, i, j) = ((i == j) ? 1.0 : 0.0)
                + IDX(vk->Q, nv, i, j) * vk->dG[j];
        }
    }
    if (lu_factor(vk->M, nv, vk->piv) != 0) { ++vk->n_singular; return -1; }

    /* Ax/Bx/Cx first (scratch), then the SAME transform as bn_dk_init, term by term:
     * A = Z(2·GXm·Ax − I), B = 2·Z·GXm·Bx, C = 2·Z·GXm·Cx — GXm as a matrix */
    var_apply(vk, vk->bxx, vk->PBx, vk->BRx, nx, nx, vk->Axs);
    var_apply(vk, vk->bxu, vk->PBx, vk->BRu, nx, nu, vk->Bxs);
    var_apply(vk, vk->bxn, vk->PBx, vk->BRn, nx, nl, vk->Cxs);
    for (int r = 0; r < nx; ++r) {
        for (int cc = 0; cc < nx; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(vk->GXm, nx, r, k) * IDX(vk->Axs, nx, k, cc); }
            IDX(dk->A, nx, r, cc) = vk->zs[r] * (2.0 * s2 - ((r == cc) ? 1.0 : 0.0));
        }
        for (int cc = 0; cc < nu; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(vk->GXm, nx, r, k) * IDX(vk->Bxs, nu, k, cc); }
            IDX(dk->B, nu, r, cc) = vk->zs[r] * 2.0 * s2;
        }
        for (int cc = 0; cc < nl; ++cc) {
            double s2 = 0.0;
            for (int k = 0; k < nx; ++k) { s2 += IDX(vk->GXm, nx, r, k) * IDX(vk->Cxs, nl, k, cc); }
            IDX(dk->C, nl, r, cc) = vk->zs[r] * 2.0 * s2;
        }
    }
    var_apply(vk, vk->bnx, vk->PBn, vk->BRx, nl, nx, dk->D);
    var_apply(vk, vk->bnu, vk->PBn, vk->BRu, nl, nu, dk->E);
    var_apply(vk, vk->bnn, vk->PBn, vk->BRn, nl, nl, dk->F);
    var_apply(vk, vk->box, vk->PBo, vk->BRx, no, nx, dk->L);
    var_apply(vk, vk->bou, vk->PBo, vk->BRu, no, nu, dk->M);
    var_apply(vk, vk->bon, vk->PBo, vk->BRn, no, nl, dk->N);
    bn_dk_refresh_transposes(dk);
    /* The chord's factorisation, its explicit inverse and the tangent predictor's stored
     * state were built with the OLD F/D/E: numeric state, not physical (re-tuning keeps x
     * and v, not a stale LU). Without this the first Newton step of the first sample after
     * every knob application would use another circuit's Jacobian (the refresh cannot fire
     * on iteration 0). Costs one factorisation + nl solves per application, i.e. once per
     * host block while a knob moves. */
    dk->have_lu = 0; dk->inv_ok = 0; dk->pred_ok = 0;
    return 0;
}

void bn_dkvar_free(BnDKVar* vk)
{
    free(vk->idx); free(vk->r0); free(vk->Q);
    free(vk->PBx); free(vk->PBn); free(vk->PBo);
    free(vk->BRx); free(vk->BRu); free(vk->BRn);
    free(vk->bxx); free(vk->bxu); free(vk->bxn);
    free(vk->bnx); free(vk->bnu); free(vk->bnn);
    free(vk->box); free(vk->bou); free(vk->bon);
    free(vk->GXm); free(vk->zs); free(vk->Axs); free(vk->Bxs); free(vk->Cxs);
    free(vk->dG); free(vk->M); free(vk->W);
    free(vk->piv);
    memset(vk, 0, sizeof(*vk));
}
