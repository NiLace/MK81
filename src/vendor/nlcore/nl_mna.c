/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_mna.c @ d75dee0, sha256 a79918be29e59ba7536476f9949f1a49aa239ea854e53a0c7c442132b835c344 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* mks_mna.c - the MK Silver preamp as a sparse nodal system, reduced to its nonlinear nodes.
 *
 * The same circuit, ports and device as the DK engine (bn_dk.c over mks_preamp_circuit.h), solved on node voltages.
 * Why, measured: the circuit is a chain of blocks joined at single nodes, and the nodal matrix keeps that structure
 * where the DK turns it into dense matrices (35 states, 14 ports in lead, about 4,200 multiply-adds per sample).
 *
 * The nodes are ordered linear first (mks_preamp_mna.h, generated): their pivots are factored ONCE per rate, which
 * leaves the Schur complement over the nodes a port touches (22 in lead, 88 nonzeros, no fill). Newton runs on those
 * only; the linear nodes come back by one back substitution per sample. In exact arithmetic these are the iterates
 * of Newton on the full nodal system, which are the DK's: the linear equations are eliminated, not approximated.
 *
 * Discretisation: the trapezoidal rule, as the DK's. A capacitor is its companion i = gc*v - h, gc = 2C*fs, and after
 * each converged sample h <- 2*gc*v - h. The junction charge keeps its own companion inside the device.
 *
 * Newton: a first-order extrapolation of the last two samples as the start (MK 81 ships second order; here first is
 * cheaper, measured); the
 * Jacobian is retaken whenever a step does not shrink by `refresh` (0 = every iteration); the step is capped at 100 V;
 * converged when the largest nonlinear-node update is below tol.
 */
#include "nl_mna.h"

#include <stdio.h>
#include <string.h>
/* MKS_FORCE_FAIL (measurement builds only, never a product): NL_FORCE_FAIL=N in the environment makes every Nth sample
 * fail every rung of the ladder (tol unreachable, the relative tolerance off), to time the whole ladder and the
 * sample budget's bound on it (MK81 cold audit A1, nlcore#78) */
#define MKS_RTOL_EFF MKS_RTOL

#ifdef MKS_DEVICE_MK81
/* an amp's valves: nl_mna_valves.h over nl_dev_valve.h (MK 81's 100 W head first), no junctions */
/* the valve settings of THIS ENGINE INSTANCE (s->mk81_cfg), handed to the device per call: no global settings
 * exist (MK 81's bn_fullamp_npar / bn_fullamp_pentode_tab are left behind) (MK81#5). ⚠️ The macros read `s`, the engine every call
 * site has in scope. Once a FILE-scope settings struct (one per compiled variant) was still shared by every
 * INSTANCE of that variant: the first instance solved its operating point on the closed-form EL34 and the second found
 * the table already switched on, so two ULTRA instances of MK 81 differed from sample 0 (and one could switch the
 * other's valve model mid-block from another thread). MKS_MK81_CFG_SHARED_NEGATIVE rebuilds that shared struct, for
 * the gate's negative arm only. */
#define MKS_MK81_CFG (&s->mk81_cfg)
#ifdef NL_MNA_DEVICE
/* a caller's own device in place of the circuit's valves (MK 81's FX-loop send systems: the PI with its EL34 grids only),
 * with the valves' signature and settings: void f(const double* v, double* i, double* J, void* cfg) and its currents */
void NL_MNA_DEVICE(const double* v, double* i, double* J, void* user);
void NL_MNA_DEVICE_CURRENTS(const double* v, double* i, void* user);
#define mks_device(v, i, J, u)          NL_MNA_DEVICE((v), (i), (J), MKS_MK81_CFG)
#define mks_device_currents(v, i, u)    NL_MNA_DEVICE_CURRENTS((v), (i), MKS_MK81_CFG)
#else
/* the circuit's valves (the export edition's 6550s with MKS_MK81_US: nl_mna.h set NL_MNA_6550 before nl_mna_valves.h) */
#define mks_device(v, i, J, u)          nl_mna_valves((v), (i), (J), MKS_MK81_CFG)
#define mks_device_currents(v, i, u)    nl_mna_valves_currents((v), (i), MKS_MK81_CFG)
#endif
#if defined(MKS_N_FXZ) && MKS_N_FXZ
#ifdef NL_MNA_DEVICE
#error "MKS_N_FXZ (zener ports after the valves) with a caller's own NL_MNA_DEVICE: not supported"
#endif
/* nlcore#25 (MK81#165): the valves, then MKS_N_FXZ zener ports (the last ports, 1x1 blocks), each with DFOR's depletion
 * charge integrated trapezoidally at the engine's rate: i = I(v) + 2 fs (q(v) - q_prev) - i_prev once started */
#undef mks_device
#undef mks_device_currents
#define NL_FXZ_P0 (MKS_N_PORTS - MKS_N_FXZ)
#if defined(MKS_FXZ_ITOL) && (!defined(MKS_RTOL) || defined(MKS_RTOL_NODIV))
#error "MKS_FXZ_ITOL rides on MKS_RTOL's weighted port stop (and not MKS_RTOL_NODIV's): without it the switch would be inert"
#endif
#ifdef MKS_FXZ_TAB
/* POLICY MKS_FXZ_TAB (MK81#183): the zener ports' law (current + slope, charge + capacitance) read from a cubic
 * Hermite table on [MKS_FXZ_TLO, MKS_FXZ_THI] every MKS_FXZ_TH volts, the exact nl_zener_iv / nl_zener_qc outside it and
 * whenever the ports' parameters differ (one table per compiled system, built at the first init - a client warms it at
 * instantiate, never in the audio thread). The exact law's exp / log / pow per evaluation was ~13 % of MK81's loop-ON
 * cost at 4x (perf); the table took it out at the same answer (MK81 VH take: ANMR unmoved; test/fxz arm TAB) */
static inline int fxz_herm(const double* tv, const double* td, double x, double* y, double* dy)
{
    if (!(x > MKS_FXZ_TLO && x < MKS_FXZ_THI)) return 0;
    const double f = (x - MKS_FXZ_TLO) * (1.0 / MKS_FXZ_TH); const int n = (int)f; const double t = f - n, h = MKS_FXZ_TH, t2 = t * t, t3 = t2 * t;
    *y = (2 * t3 - 3 * t2 + 1) * tv[n] + (t3 - 2 * t2 + t) * h * td[n] + (-2 * t3 + 3 * t2) * tv[n + 1] + (t3 - t2) * h * td[n + 1];
    if (dy) *dy = (6 * t2 - 6 * t) * (tv[n] - tv[n + 1]) * (1.0 / h) + (3 * t2 - 4 * t + 1) * td[n] + (3 * t2 - 2 * t) * td[n + 1];
    return 1;
}
#define FXZ_IV(s, z, v, i, g) do { if (!((s)->fxz_tab_ok && fxz_herm((s)->fxz_ti, (s)->fxz_tg, (v), (i), (g)))) nl_zener_iv(&(s)->fxz[z], (v), (i), (g)); } while (0)
#define FXZ_QC(s, z, v, q, c) do { if (!((s)->fxz_tab_ok && fxz_herm((s)->fxz_tq, (s)->fxz_tc, (v), (q), (c)))) nl_zener_qc(&(s)->fxz[z], (v), (q), (c)); } while (0)
#else
#define FXZ_IV(s, z, v, i, g) nl_zener_iv(&(s)->fxz[z], (v), (i), (g))
#define FXZ_QC(s, z, v, q, c) nl_zener_qc(&(s)->fxz[z], (v), (q), (c))
#endif
static void fxz_ports(MksMna* s, const double* v, double* i, double* J)
{
    for (int k = 0; k < MKS_N_FXZ; ++k) {
        const int p = NL_FXZ_P0 + k;
        double id, gd;
        FXZ_IV(s, k, v[p], &id, &gd);
        if (s->fxz_started) {
            double q, c;
            FXZ_QC(s, k, v[p], &q, &c);
            id += 2.0 * s->fs * (q - s->fxz_q[k]) - s->fxz_ic[k];
            gd += 2.0 * s->fs * c;
        }
        i[p] = id;
        if (J) { J[p * MKS_N_PORTS + p] = gd; }
    }
}
static void fxz_commit(MksMna* s, const double* v)
{
    if (!s->fxz_started) { return; }
    for (int k = 0; k < MKS_N_FXZ; ++k) {
        double q, c;
        FXZ_QC(s, k, v[NL_FXZ_P0 + k], &q, &c);
        s->fxz_ic[k] = 2.0 * s->fs * (q - s->fxz_q[k]) - s->fxz_ic[k];
        s->fxz_q[k] = q;
    }
}
static void fxz_start(MksMna* s, const double* v)
{
    for (int k = 0; k < MKS_N_FXZ; ++k) {
        double c;
        nl_zener_qc(&s->fxz[k], v[NL_FXZ_P0 + k], &s->fxz_q[k], &c);
        s->fxz_ic[k] = 0.0;
    }
    s->fxz_started = 1;
}
#define mks_device(v, i, J, u)          (nl_mna_valves((v), (i), (J), MKS_MK81_CFG), fxz_ports(s, (v), (i), (J)))
#define mks_device_currents(v, i, u)    (nl_mna_valves_currents((v), (i), MKS_MK81_CFG), fxz_ports(s, (v), (i), NULL))
#define mks_device_commit(v, u)         fxz_commit(s, (v))
#define mks_device_start(d, v, fs)      fxz_start(s, (v))
#else
#define mks_device_commit(v, u)         ((void)0)
#define mks_device_start(d, v, fs)      ((void)0)
#endif
#define mks_device_init(d)              ((void)0)
#endif

#define ST(x) ((void)0)
/* MK81#104: the COST CENSUS build only (-DBN_CENSUS, never shipped): reference cycles inside the engine -
 * the device evaluations, the factorisations, the Newton solves, and the whole per-sample process - printed at unload by
 * every compiled copy that ran (MKS_CZ_NAME tells the copies apart) */
#define CZ_T0(v) ((void)0)
#define CZ_ADD(acc, v) ((void)0)
#define CZ(x) ((void)0)
#ifdef MKS_THETA_C
/* MK81#164: the theta of capacitor k - MKS_THETA_C for the fixed capacitors up to MKS_THETA_CMAX farad (the valves'
 * interelectrode and the transformer's winding capacitances: the stiff modes), the trapezoidal 1/2 for the rest */
#ifndef MKS_THETA_CMAX
#define MKS_THETA_CMAX 1.0
#endif
#define MKS_THC(k) ((mks_mna_cap[(k)].knob < 0 && mks_mna_cap[(k)].v <= MKS_THETA_CMAX) ? MKS_THETA_C : 0.5)
#endif
#ifdef MKS_THETA_RT
#ifdef MKS_THETA_C
#error "MKS_THETA_RT chooses theta at run time: do not also set the compile-time MKS_THETA_C / MKS_THETA_L"
#endif
#define THC_RT(s, k) ((mks_mna_cap[(k)].knob < 0 && mks_mna_cap[(k)].v <= (s)->theta_cmax) ? (s)->theta_c : 0.5)
#endif
/* an inductor at DC: a short of this conductance (1 microohm: below any winding resistance by 1e7) */
#define MKS_MNA_GSHORT 1e6

void mks_pi_load_at(double p, double* out)
{
    const double x = (p < 0.0 ? 0.0 : p > 1.0 ? 1.0 : p) * (MKS_PI_LOAD_N - 1);
    int i = (int)x;
    if (i > MKS_PI_LOAD_N - 2) { i = MKS_PI_LOAD_N - 2; }
    const double t = x - i;
    for (int k = 0; k < 6; ++k) {
        out[k] = exp((1.0 - t) * log(mks_pi_load[i][k]) + t * log(mks_pi_load[i + 1][k]));
    }
}

static void stamp_elem(MksMna* s, const MksMnaElem* e, double g)
{
    if (e->saa >= 0) { s->m[e->saa] += g; }
    if (e->sbb >= 0) { s->m[e->sbb] += g; }
    if (e->sab >= 0) { s->m[e->sab] -= g; }
    if (e->sba >= 0) { s->m[e->sba] -= g; }
    if (e->fa >= 0) { s->fc[e->fa] -= g; }
    if (e->fb >= 0) { s->fc[e->fb] -= g; }
}

/* the matrices for rate fs (0 = DC: capacitors open) at the current knobs */
static void build(MksMna* s, double fs)
{
    mks_pi_load_at(s->knobs.PRES, s->kval);
    mks_mna_pot_values(&s->knobs, s->kval);
    memset(s->m, 0, sizeof s->m);
    memset(s->fc, 0, sizeof s->fc);
    for (int k = 0; k < MKS_MNA_NR; ++k) {
        const MksMnaElem* e = &mks_mna_res[k];
        stamp_elem(s, e, 1.0 / (e->knob >= 0 ? s->kval[e->knob] : e->v));
    }
    for (int k = 0; k < MKS_MNA_NC; ++k) {
        const MksMnaElem* e = &mks_mna_cap[k];
#ifdef MKS_THETA_C
        s->gc[k] = (e->knob >= 0 ? s->kval[e->knob] : e->v) * s->fs * (1.0 / MKS_THC(k));   /* MK81#164: theta-method, below */
#elif defined(MKS_THETA_RT)
        if (s->theta_on) {
            const double th = THC_RT(s, k);
            s->gc[k] = (e->knob >= 0 ? s->kval[e->knob] : e->v) * s->fs * (1.0 / th);
            s->th_a[k] = s->gc[k] * (1.0 / th); s->th_b[k] = (1.0 - th) / th;   /* the commit's, in its operation order */
        }
        else { s->gc[k] = 2.0 * (e->knob >= 0 ? s->kval[e->knob] : e->v) * s->fs; }
#else
        s->gc[k] = 2.0 * (e->knob >= 0 ? s->kval[e->knob] : e->v) * s->fs;
#endif
        if (fs > 0.0) { stamp_elem(s, e, s->gc[k]); }
    }
#ifdef MKS_MNA_EMF_CAP
    if (s->emf_detached && fs > 0.0) {
        /* the EMF capacitor's a terminal detached: its current gc (emf - V(b)) - h no longer depends on V(a), so the
         * a column's two entries go (the a row keeps -gc V(b): the current still leaves node a) */
        const MksMnaElem* e = &mks_mna_cap[MKS_MNA_EMF_CAP];
        if (e->saa >= 0) { s->m[e->saa] -= s->gc[MKS_MNA_EMF_CAP]; }
        if (e->sba >= 0) { s->m[e->sba] += s->gc[MKS_MNA_EMF_CAP]; }
    }
#endif
    /* the windings: their Gamma block at this rate, or (DC) each one a short of MKS_MNA_GSHORT */
#ifdef MKS_THETA_L
    /* MK81#164: the windings on the theta-method instead of the trapezoidal rule (theta = 1/2):
     * i' = i + (Linv / fs) (theta v' + (1 - theta) v), so Gamma = theta Linv / fs and the commit hl += Gamma v / theta. A
     * stiff mode (the plates' parasitic capacitance against the windings, far above Nyquist) maps to z = -(1 - theta) / theta
     * instead of z = -1: the trapezoidal rule leaves it ringing at Nyquist, sample to sample, and Newton chases it */
    for (int k = 0; k < MKS_MNA_NI * MKS_MNA_NI; ++k) { s->gam[k] = MKS_THETA_L * mks_mna_linv[k] / s->fs; }
#elif defined(MKS_THETA_RT)
    if (s->theta_on) { for (int k = 0; k < MKS_MNA_NI * MKS_MNA_NI; ++k) { s->gam[k] = s->theta_l * mks_mna_linv[k] / s->fs; }
                       s->th_l_inv = 1.0 / s->theta_l; }
    else { for (int k = 0; k < MKS_MNA_NI * MKS_MNA_NI; ++k) { s->gam[k] = mks_mna_linv[k] / (2.0 * s->fs); } }
#else
    for (int k = 0; k < MKS_MNA_NI * MKS_MNA_NI; ++k) { s->gam[k] = mks_mna_linv[k] / (2.0 * s->fs); }
#endif
#if defined(MKS_MNA_IND_KNOBS) && MKS_MNA_IND_KNOBS
    /* knob-driven windings (MK81#4, the 4x12's LVC and LP): uncoupled (checked at generation), so their Gamma is the
     * diagonal alone, 1 / (2 fs L(knob)) */
    for (int k = 0; k < MKS_MNA_NI; ++k) {
#ifdef MKS_THETA_L
        if (mks_mna_ind_knob[k] >= 0) { s->gam[k * MKS_MNA_NI + k] = MKS_THETA_L / (s->fs * s->kval[mks_mna_ind_knob[k]]); }
#elif defined(MKS_THETA_RT)
        if (mks_mna_ind_knob[k] >= 0) {
            s->gam[k * MKS_MNA_NI + k] = s->theta_on ? s->theta_l / (s->fs * s->kval[mks_mna_ind_knob[k]])
                                                     : 1.0 / (2.0 * s->fs * s->kval[mks_mna_ind_knob[k]]);
        }
#else
        if (mks_mna_ind_knob[k] >= 0) { s->gam[k * MKS_MNA_NI + k] = 1.0 / (2.0 * s->fs * s->kval[mks_mna_ind_knob[k]]); }
#endif
    }
#endif
#ifdef MKS_GAM_COLS
    for (int k = 0; k < MKS_MNA_NI; ++k) { for (int j = 0; j < MKS_MNA_NI; ++j) { s->gamT[j * MKS_MNA_NI + k] = s->gam[k * MKS_MNA_NI + j]; } }
#endif
#ifdef MKS_SPARSE_GAM
    for (int k = 0; k < MKS_MNA_NI; ++k) {
        int n = 0;
        for (int j = 0; j < MKS_MNA_NI; ++j) { if (s->gam[k * MKS_MNA_NI + j] != 0.0) { s->gnz_j[k * MKS_MNA_NI + n++] = (unsigned char)j; } }
        s->gnz_n[k] = (unsigned char)n;
    }
#endif
    for (int k = 0; k < (MKS_MNA_NI > 0 ? MKS_MNA_NIST : 0); ++k) {
        const MksMnaIStamp* t = &mks_mna_istamp[k];
        const double g = fs > 0.0 ? s->gam[t->k * MKS_MNA_NI + t->j] : (t->k == t->j ? MKS_MNA_GSHORT : 0.0);
        s->m[t->slot] += t->sign * g;
    }
    mks_mna_factor_lin(s->m);
    mks_mna_load_schur(s->m, s->S);
    memcpy(s->S0, s->S, sizeof s->S0);
    mks_mna_factor_schur(s->S0);
    s->fsm = fs;
}

void mks_mna_set_emf_detached(MksMna* s, int on)
{
#ifdef MKS_MNA_EMF_CAP
    _Static_assert(MKS_MNA_EMF_SIGN > 0.0, "the detached EMF needs the capacitor's a terminal on the EMF's node");
    if (s->emf_detached == on) { return; }
    s->emf_detached = on;
    s->w_ok = 0;   /* the last Newton factorisation belongs to the other matrix */
    build(s, s->fsm);
#else
    (void)s; (void)on;
#endif
}

/* each capacitor's branch voltage (the EMF capacitor's with its EMF, as the commit takes it) and each winding's, from V */
static void branch_v(const MksMna* s, const double* V, double emf, double* vc, double* vw)
{
    for (int k = 0; k < MKS_MNA_NC; ++k) {
        const int a = mks_mna_cap[k].a, b = mks_mna_cap[k].b;
        vc[k] = (a < MKS_MNA_N + MKS_MNA_NF ? V[a] : 0.0) - (b < MKS_MNA_N + MKS_MNA_NF ? V[b] : 0.0);
    }
#ifdef MKS_MNA_EMF_CAP
    {
        const int a = mks_mna_cap[MKS_MNA_EMF_CAP].a;
        vc[MKS_MNA_EMF_CAP] += s->emf_detached ? emf - (a < MKS_MNA_N + MKS_MNA_NF ? V[a] : 0.0) : MKS_MNA_EMF_SIGN * emf;
    }
#else
    (void)s; (void)emf;
#endif
    for (int k = 0; k < MKS_MNA_NI; ++k) { vw[k] = V[mks_mna_ind[k].a] - V[mks_mna_ind[k].b]; }
}

/* the trapezoidal state is (v, i) per element: i = gc v - h  =>  h = gc v_n + i_n after the commit; i = Gamma v + hl  =>
 * hl = i_n + Gamma v_n. Changing the step keeps v_n and i_n: h += (gc' - gc) v_n, hl += (Gamma' - Gamma) v_n. */
static void convert_hist(MksMna* s, const double* gc0, const double* gam0, const double* vc, const double* vw)
{
    for (int k = 0; k < MKS_MNA_NC; ++k) { s->h[k] += (s->gc[k] - gc0[k]) * vc[k]; }
    for (int k = 0; k < MKS_MNA_NI; ++k) {
        double acc = 0.0;
        for (int j = 0; j < MKS_MNA_NI; ++j) { acc += (s->gam[k * MKS_MNA_NI + j] - gam0[k * MKS_MNA_NI + j]) * vw[j]; }
        s->hl[k] += acc;
    }
}

/* A knob that drives a capacitor or a winding (gc / Gamma depend on it) changes that element's conductance under its
 * committed history: kept as it was, the element's current would jump by (gc' - gc) v_n at the move, a click (a
 * capacitor switched by a front-panel switch, a cabinet load swept by a knob). The rebuild converts the histories as a
 * change of step does (convert_hist: each element's voltage and current kept). A knob that drives only resistors leaves
 * every gc and Gamma exactly equal, so the conversion adds exactly zero. NL_MNA_KNOB_NO_CONVERT: the old rebuild, for a
 * gate's negative arm only. */
void mks_mna_set_knobs(MksMna* s, const MksMnaKnobs* k)
{
    double gc0[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], gam0[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];
    double vc[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    memcpy(gc0, s->gc, sizeof(double) * MKS_MNA_NC); memcpy(gam0, s->gam, sizeof(double) * MKS_MNA_NI * MKS_MNA_NI);
    branch_v(s, s->V, s->emf, vc, vw);
    s->knobs = *k;
    build(s, s->fsm);
    convert_hist(s, gc0, gam0, vc, vw);
}

/* the sample's right-hand side y = -(constant part of the residual), forward-eliminated */
static void rhs(MksMna* s)
{
#ifdef MKS_MNA_EMF_CAP
    /* the capacitor EMF: i = gc (v + sign e) - h = gc v - (h - sign gc e), so the history it contributes is shifted */
    const double h_keep = s->h[MKS_MNA_EMF_CAP];
    s->h[MKS_MNA_EMF_CAP] -= MKS_MNA_EMF_SIGN * s->gc[MKS_MNA_EMF_CAP] * s->emf;
#endif
    mks_mna_rhs(s->fc, s->V, s->h, s->fsm > 0.0, s->y);
#ifdef MKS_MNA_EMF_CAP
    s->h[MKS_MNA_EMF_CAP] = h_keep;
#endif
    for (int k = 0; k < MKS_MNA_NI; ++k) {      /* the windings' history currents (zero at DC) */
        const int a = mks_mna_ind[k].a, b = mks_mna_ind[k].b;
        if (a < MKS_MNA_N) { s->y[a] -= s->hl[k]; }
        if (b < MKS_MNA_N) { s->y[b] += s->hl[k]; }
    }
    mks_mna_forward_lin(s->m, s->y);
}

static void port_voltages(MksMna* s)
{
    mks_mna_port_v(s->V, s->vp);
}

#define ND_ITER() ((void)0)
#define ND_BT() ((void)0)
#define ND_STEP(st_, f_) ((void)0)

/* Newton on the nonlinear nodes; returns the iterations, or -1 if it did not converge */
#ifdef MKS_CHORD_FIRST
/* POLICY MKS_CHORD_FIRST (nlcore#13, MK81#164; a cold reader's proposal): the convergence test of a solve with
 * the LAST factorisation, measured as newton() measures its step (on the ports, MKS_RTOL's weighting) */
static inline double cf_stepw(const MksMna* s, const double* g)
{
    double sw = 0.0;
    if (s->port_norm) {
        for (int p = 0; p < MKS_MNA_NP; ++p) {
            const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
            const double d = (a < MKS_MNA_N ? g[a] : 0.0) - (b < MKS_MNA_N ? g[b] : 0.0);
#ifdef MKS_RTOL
            const double dw = fabs(d) * (s->tol / (s->tol + MKS_RTOL_EFF * fabs(s->vp[p])));
#else
            const double dw = fabs(d);
#endif
            if (!(dw <= sw)) { sw = dw; }
        }
    } else {
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { if (!(fabs(g[i]) <= sw)) { sw = fabs(g[i]); } }
    }
    return sw;
}
#endif

#ifdef MKS_SAMPLE_BUDGET
#if !(defined(MKS_RESCUE) && MKS_RESCUE)
#error "MKS_SAMPLE_BUDGET resets its counter in the rescue's entry point: build it with MKS_RESCUE"
#endif
#define SB_LEFT(s) ((s)->sb_left > 0)
#else
#define SB_LEFT(s) 1
#endif
static int newton(MksMna* s, int fresh_every_iteration)
{
    double f[MKS_MNA_N];
    double prev = s->prev0 > 0.0 ? s->prev0 : 1e300;   /* a predicted start hands over its own step as the previous one */
    s->prev0 = 0.0;
    double rprev = 1e300;
    double dlast[MKS_MNA_N];
#ifdef MKS_LAZYRES
    double f0[MKS_MNA_N];
#ifdef MKS_XFORM
    double Vs0[MKS_MNA_N + MKS_MNA_NF], ips0[MKS_MNA_NP];
#endif
    int rprev_pending = 0;   /* iteration 0's port residual, computed only if iteration 1 compares against it */
#endif
    int bt = 0;
    int prev_lim = prev < 1e300 ? 0 : 1;
#ifdef MKS_CYCLE_ABORT
#ifndef MKS_CYCLE_MIN
#define MKS_CYCLE_MIN 1.0   /* volts: a smaller repeated step is not a cycle worth cutting */
#if defined(MKS_WALK_RECAP) && !defined(MKS_WALK_CAP)
#define MKS_WALK_CAP 800.0   /* volts: the walk's port step cap (MK81: 400 / 800 / none a plateau) */
#endif
#endif
    double cyc[8]; int cyc_n = 0;
#endif
#ifdef MKS_CAP_GROW
#ifndef MKS_CAP_MAX
#define MKS_CAP_MAX 1600.0
#endif
    double cg_cap = s->step_cap, cg_req = 0.0;   /* the grown cap and the step the last capped update was asked for */
#endif
#ifdef MKS_PTOL
    /* per-port tolerance (lever census 4-Oct): the convergence tests read each port's step scaled by tol / tol_port;
     * the cap and the line-search gate keep the volts */
    double prevw = prev;
    static double wt[MKS_MNA_NP]; static double wt_tol = -1.0; static int wt_started = -1;
    if (wt_tol != s->tol || wt_started != s->dev.started) for (int p = 0; p < MKS_MNA_NP; ++p) {
        const int p0 = 2 * MKS_N_TRIODES, j0 = p0 + 3 * MKS_N_PENTODES;
        double tp = s->tol;
        if (p < p0 && (p & 1)) { tp = MKS_TTOL; }                        /* triode plate-cathode */
        else if (p >= p0 && p < j0 && (p - p0) % 3 == 2) { tp = MKS_PTOL; }   /* pentode plate-cathode */
        else if (p >= p0 && p < j0 && (p - p0) % 3 == 1) { tp = MKS_STOL; }   /* pentode screen-cathode */
        wt[p] = s->dev.started ? s->tol / (tp > s->tol ? tp : s->tol) : 1.0;   /* the DC point keeps tol everywhere */
        wt_tol = s->tol; wt_started = s->dev.started;
    }
#define STEPW stepw
#define PREVW prevw
#elif defined(MKS_RTOL)
    /* MK81#104 (LowDude's lever d, docs/POWER_ENGINE.md s.11.2 3b): SPICE's relative stop, |dv| < tol + rtol |v|
     * on each port, i.e. each port's step weighted by tol / (tol + rtol |v_port|) against tol (MKS_RTOL = rtol) */
    double prevw = prev;
#define STEPW stepw
#define PREVW prevw
#else
#define STEPW step
#define PREVW prev
#endif
    /* persistent chord (MK 81's DK `cp`): the sample starts on the last sample's factorisation; the refresh rule
     * below retakes it when a step does not contract enough */
    int have_lu = s->chord_persist && s->w_ok && !fresh_every_iteration;
#ifdef MKS_CHORD_FIRST
    int cf_lu = 0;   /* this call has factorised at least once */
#endif
    const int limit_on = s->dev.limit;
#ifdef MKS_LAZYRES
    int ls_skip0 = 0;
#endif
    for (int it = 0; it < s->max_iter; ++it) {
#ifdef MKS_SAMPLE_BUDGET
        if (s->sb_left <= 0) { break; }   /* the sample's budget is spent: this attempt fails where it stands */
        --s->sb_left;
#endif
        /* the junction limiting tames the first steps from an overshooting start; a junction that stays limited would
         * never let the iteration count as converged, so it is dropped after 8 iterations (restored on return) */
        s->dev.limit = limit_on && it < 8;
        port_voltages(s);
        int need_lu = 0;
#ifdef MKS_CHORD_FIRST
        /* an iteration that would retake the Jacobian first tries the last factorisation: currents only now, the
         * Jacobian (and the factorisation) only if that chord step does not already land within tol */
        const int chord_try = !have_lu && it > 0 && !fresh_every_iteration && cf_lu;   /* w: THIS call's last LU */
        if (chord_try) {
            mks_device_currents(s->vp, s->ip, &s->dev);
        } else
#endif
        if (!have_lu) {
            CZ_T0(cz0);
            mks_device(s->vp, s->ip, s->J, &s->dev);
            CZ_ADD(cz_dev, cz0); CZ(cz_dev_n++);
            ST(st_dev++);
            need_lu = 1;   /* stamped and factored below, once the residual test has had its say */
        } else {
            CZ_T0(cz0);
            mks_device_currents(s->vp, s->ip, &s->dev);
            CZ_ADD(cz_cur, cz0);
        }
        ND_ITER();
#ifdef MKS_XFORM
        /* the iterate form (lever census 4-Oct): solve for the new voltages, (S + PJP') V' = y - P(i - J vp), instead of
         * the update from the residual: the block product J.vp on the ports replaces S.V over the Schur block. The true
         * residual (fr) is formed only where the line search reads it. */
        double fr[MKS_MNA_N];
        int fr_ok = 0;
        {
            double r[MKS_MNA_NP];
            const int nl = MKS_MNA_NP;
            for (int p = 0; p < nl; ++p) {
                const int b0 = s->dev.bstart[p], bn = s->dev.bsize[p];
                double jv = 0.0;
                for (int c = b0; c < b0 + bn; ++c) { jv += s->J[p * nl + c] * s->vp[c]; }
                r[p] = s->ip[p] - jv;
            }
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { f[i] = s->y[i]; }
            mks_mna_scatter(r, f);
        }
#define XRES() do { if (!fr_ok) { mks_mna_schur_matvec(s->S, s->V, fr); \
            for (int i_ = MKS_MNA_NLIN; i_ < MKS_MNA_N; ++i_) { fr[i_] = s->y[i_] - fr[i_]; } \
            mks_mna_scatter(s->ip, fr); fr_ok = 1; } } while (0)
#define FRES fr
#else
        mks_mna_schur_matvec(s->S, s->V, f);
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { f[i] = s->y[i] - f[i]; }   /* -(S.V - y + port currents) */
        mks_mna_scatter(s->ip, f);
#define XRES() ((void)0)
#define FRES f
#endif
        /* line_search 2: MK 81's DK backtracking, on the RESIDUAL (the nonlinear nodes' KCL error before the solve): an
         * update whose residual is not smaller than the last one is halved back (at most 4 halvings; the 4th is
         * accepted, as the DK accepts its shortest). The Jacobian is retaken at the halved point. */
        /* the residual test is only paid while Newton takes large steps: once the last step is below ls_gate volts the
         * iteration is in its quadratic regime and the residual falls on its own (rprev is then forgotten) */
        const int ls_now = s->line_search >= 2 && (bt > 0 || !(prev <= s->ls_gate));
        if (s->line_search >= 2 && !ls_now) { rprev = 1e300;
#ifdef MKS_LAZYRES
            rprev_pending = 0;
#endif
        }
#ifdef MKS_LAZYRES
        if (ls_now && it == 0 && s->line_search == 3 && !s->resid_stop) {
#ifdef MKS_XFORM
            memcpy(Vs0, s->V, sizeof Vs0); memcpy(ips0, s->ip, sizeof ips0);   /* the residual there, only if asked */
#else
            memcpy(f0, f, sizeof f0);
#endif
            rprev_pending = 1; ls_skip0 = 1;
        }
        if (ls_now && rprev_pending && it > 0) {
            double g[MKS_MNA_N]; double rn = 0.0;
#ifdef MKS_XFORM
            mks_mna_schur_matvec(s->S, Vs0, g);
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { g[i] = s->y[i] - g[i]; }
            mks_mna_scatter(ips0, g);
#else
            memcpy(g, f0, sizeof g);
#endif
            ST(st_solve++);
            mks_mna_solve_schur(s->S0, g);
            for (int p = 0; p < MKS_MNA_NP; ++p) {
                const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                const double d = (a < MKS_MNA_N ? g[a] : 0.0) - (b < MKS_MNA_N ? g[b] : 0.0);
                if (!(fabs(d) <= rn)) { rn = fabs(d); }
            }
            rprev = rn; rprev_pending = 0;
        }
        if (ls_skip0) { ls_skip0 = 0; } else
#endif
        if (ls_now) {
            double rn = 0.0;
            if (s->line_search == 3) {
                /* the DK's residual: how far each PORT would move if the valves held their present currents (the linear
                 * network alone, S0, solved with this KCL error), in volts: v - p - F.i in its terms */
                double g[MKS_MNA_N];
                XRES(); memcpy(g, FRES, sizeof g);
                ST(st_solve++);
                mks_mna_solve_schur(s->S0, g);
                for (int p = 0; p < MKS_MNA_NP; ++p) {
                    const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                    const double d = (a < MKS_MNA_N ? g[a] : 0.0) - (b < MKS_MNA_N ? g[b] : 0.0);
                    if (!(fabs(d) <= rn)) { rn = fabs(d); }
                }
            } else {
                for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { if (!(fabs(f[i]) <= rn)) { rn = fabs(f[i]); } }
            }
            /* the DK's stop: the port residual below tol ends the sample BEFORE another factorisation and solve (the
             * step test pays one whole iteration only to confirm) */
            if (s->resid_stop && s->line_search == 3 && rn < s->tol && !s->dev.limited) { s->dev.limit = limit_on; return it + 1; }
            if (it > 0 && !(rn < rprev) && bt < 4) {
                for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { dlast[i] *= 0.5; s->V[i] -= dlast[i]; }
                ++bt; ND_BT();
                have_lu = 0;
                continue;
            }
            rprev = rn;
        }
#ifdef MKS_CHORD_FIRST
        if (chord_try) {
            double g[MKS_MNA_N];
            memcpy(g, f, sizeof g);
            mks_mna_solve_schur(s->w, g);
            const double sw = cf_stepw(s, g);
            if (sw < s->tol && !s->dev.limited) {           /* (a NaN fails the test: the Newton step below sees it) */
                for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] += g[i]; }
                s->chord_accepts++;
                s->dev.limit = limit_on;
                return it + 1;
            }
            mks_device(s->vp, s->ip, s->J, &s->dev);           /* the Newton step from this same point */
            need_lu = 1;
        }
#endif
        if (need_lu) {
            CZ_T0(cz0);
#if defined(MKS_LOAD_STAMP) && defined(MKS_MNA_HAS_LOAD_STAMP)
            /* POLICY MKS_LOAD_STAMP (nlcore#13, MK81#164): the block loaded and stamped in one generated pass (the same
             * additions in the same order: bit for bit the copy and the stamp below) */
            mks_mna_load_stamp(s->S, s->w, s->J);
#else
            memcpy(s->w, s->S, sizeof s->w);
            mks_mna_stamp(s->w, s->J);
#endif
            mks_mna_factor_schur(s->w);
            CZ_ADD(cz_fact, cz0); CZ(cz_fact_n++);
            ST(st_fact++);
            s->factorisations++;
            have_lu = 1;
#ifdef MKS_CHORD_FIRST
            cf_lu = 1;
#endif
        }
        { CZ_T0(cz0); mks_mna_solve_schur(s->w, f); CZ_ADD(cz_solve, cz0); CZ(cz_it++); }
#ifdef MKS_XFORM
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { f[i] -= s->V[i]; }   /* the update */
#endif
        ST(st_solve++); ST(st_it++);
        double step = 0.0;
#if defined(MKS_PTOL) || defined(MKS_RTOL)
        double stepw = 0.0;
#endif
        if (s->port_norm) {
            /* the step measured where the DK measures it: on the PORT voltages (the valves' and junctions' own
             * terminals), not on every node of the block (a rail moving by its millivolts held the test) */
            int pmax = 0, pwmax = 0; (void)pwmax;
#if defined(MKS_RTOL) && defined(MKS_RTOL_NODIV)
            double rt_num = 0.0, rt_den = 1.0;   /* the weighted maximum as a fraction (MKS_RTOL_NODIV) */
#endif
            for (int p = 0; p < MKS_MNA_NP; ++p) {
                const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                const double d = (a < MKS_MNA_N ? f[a] : 0.0) - (b < MKS_MNA_N ? f[b] : 0.0);
                if (!(fabs(d) <= step)) { step = fabs(d); pmax = p; }
#ifdef MKS_PTOL
                if (!(fabs(d) * wt[p] <= stepw)) { stepw = fabs(d) * wt[p]; pwmax = p; }
#endif
#ifdef MKS_RTOL
#ifdef MKS_RTOL_NODIV
                /* the weighted maximum without a division per port: |d| / (tol + rtol |vp|) compared by cross-multiplying
                 * against the best so far; the winner's weighted step is formed once below, in the same expression */
                { const double den = s->tol + MKS_RTOL_EFF * fabs(s->vp[p]);
                  if (!(fabs(d) * rt_den <= rt_num * den)) { rt_num = fabs(d); rt_den = den; pwmax = p; } }
#else
                { double dw = fabs(d) * (s->tol / (s->tol + MKS_RTOL_EFF * fabs(s->vp[p])));
#if defined(MKS_FXZ_ITOL) && defined(MKS_N_FXZ) && MKS_N_FXZ
                  /* POLICY MKS_FXZ_ITOL (MK81#165 / #183): a send zener port also stops on its CURRENT step, |dv| x g <=
                   * MKS_FXZ_ITOL amps (g its Jacobian diagonal) - a zener port 3 mV off is a large current error, i.e. a wrong
                   * charge into the coupling capacitor at every clamp, and MK81's closed FX loop heard it: the output decided by
                   * Newton's stop (+28 ANMR between two stops) at 1e-9 A back to the loop off's +1.7; a voltage stop instead cost
                   * +44 % iterations (it chases the near-floating ports while the zeners are cut off) */
                  if (p >= NL_FXZ_P0) { const double wi = fabs(d) * fabs(s->J[p * MKS_MNA_NP + p]) * (s->tol / MKS_FXZ_ITOL); if (wi > dw) { dw = wi; } }
#endif
                  if (!(dw <= stepw)) { stepw = dw; pwmax = p; } }
#endif
#endif
            }
#if defined(MKS_RTOL) && defined(MKS_RTOL_NODIV)
            stepw = rt_num * (s->tol / (s->tol + MKS_RTOL_EFF * fabs(s->vp[pwmax])));
#endif
            (void)pmax;
        } else {
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) {
                if (!(fabs(f[i]) <= step)) { step = fabs(f[i]); }   /* a NaN poisons the step */
            }
        }
        ND_STEP(step, f);
        if (!isfinite(step)) { s->bad_pivots++; s->dev.limit = limit_on; return -1; }
#ifdef MKS_WALK_RECAP
        /* POLICY MKS_WALK_RECAP (MK81#164): a PREDICTED first attempt whose FIRST Newton step (the largest port step,
         * before the cap) exceeds MKS_WALK_RECAP volts runs its remaining iterations with the port step cap at MKS_WALK_CAP
         * (800 V) and no residual line search; the caller's settings come back when the attempt returns. MK81's failed first
         * attempts were WALKERS: a pentode plate asked for ~4 kV (its node near-linear where the valve is cut off), the
         * 100 V cap and the line search's halvings moved it ~10 V per update to the 48-iteration limit. The first step is
         * the signature (MK81 2x trace: > 3000 V in 60 % of the failed attempts, 1.3 % of the converged >= 4-iteration
         * ones); the remedy is a LARGER step from the SAME start - abandoning the start for the extrapolated one, or a
         * smaller cap, cost more (MK81 docs/NEWTON_TAIL.md). Every sample whose first step stays under the threshold, and
         * every non-predicted attempt, is untouched. */
        if (it == 0 && s->walk_armed == 1 && step > MKS_WALK_RECAP) {
            s->walk_cap_keep = s->step_cap; s->walk_ls_keep = s->line_search;
            s->step_cap = MKS_WALK_CAP; s->line_search = 0; s->walk_armed = 2;
        }
#endif
#ifdef MKS_CYCLE_ABORT
        /* POLICY MKS_CYCLE_ABORT (MK81#164 / #176): a Newton that has come back to a point it already
         * stepped from never converges - MK81's failed first attempts were EXACT cycles from their first iteration (an
         * output plate asks for 250-440 V, the 100 V cap takes it, the next step is exactly 100 V back; period 2, or 5
         * with backtracks) to the iteration limit. From iteration 8 (the junction limiting is off: the map no longer
         * changes) a step above MKS_CYCLE_MIN (1 V) equal to one of the last 8 steps (to 1e-12 relative) ends the attempt as failed, so
         * the caller's next rung starts at once from the same point it would have started from at the limit */
        if (it >= 8 && step > MKS_CYCLE_MIN) {
            for (int q = 0; q < cyc_n && q < 8; ++q) {
                if (fabs(step - cyc[q]) <= 1e-12 * step) { s->dev.limit = limit_on; return -1; }
            }
        }
        if (it >= 8) { cyc[cyc_n++ & 7] = step; }
#endif
#ifdef MKS_NEWTON_TRACE
        if (s->trace) {
            double rn = 0.0; int imax = -1;
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { if (fabs(f[i]) > rn) { rn = fabs(f[i]); imax = i; } }
            fprintf(stderr, "  it %3d fresh %d step %.3e at node %d limited %d bt %d\n", it, fresh_every_iteration, step, imax, s->dev.limited, bt);
        }
#endif
        /* line search (Deuflhard's natural monotonicity test, the role MK 81's DK backtracking plays): if the Newton step
         * from here is not smaller than the last one, the last update overshot: take back half of it, retake the
         * Jacobian there, and try again (at most 4 halvings per update). The convergence test is untouched. */
        if (s->line_search == 1 && it > 0 && !(step < prev) && bt < 4) {
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { dlast[i] *= 0.5; s->V[i] -= dlast[i]; }
            ++bt;
            have_lu = 0;
            continue;
        }
        bt = 0;
#ifdef MKS_CAP_GROW
        /* MK81#164: the step cap GROWS while the Newton target stays put. Measured (VH take, 2x, worst knob
         * point): the samples of 12+ iterations walk a plate 100 V per iteration towards a target 1-4 kV away, the step
         * shrinking by exactly the cap each time (a cut-off pentode leaves the plate node linear: the full step would
         * land), while removing the cap DOUBLED the fallbacks (the full step overshoots where the target is not fixed).
         * So: after a capped step, if the new step is what the last one left (|step - (req - cap)| <= 10 % of req: the
         * target did not move) the cap is multiplied by MKS_CAP_GROW (up to MKS_CAP_MAX); otherwise it returns to
         * step_cap. Not on the fallback's fresh-Jacobian passes (their trust region sets the cap) */
        if (cg_req > 0.0) {
            cg_cap = (!fresh_every_iteration && fabs(step - (cg_req - cg_cap)) <= 0.1 * cg_req)
                   ? (cg_cap * MKS_CAP_GROW < MKS_CAP_MAX ? cg_cap * MKS_CAP_GROW : MKS_CAP_MAX) : s->step_cap;
        }
        if (fresh_every_iteration) { cg_cap = s->step_cap; }
        cg_req = step > cg_cap ? step : 0.0;
        double sc = step > cg_cap ? cg_cap / step : 1.0;
#else
        double sc = step > s->step_cap ? s->step_cap / step : 1.0;
#endif
#ifdef MKS_STOP_UNCAPPED
        const int capped = sc < 1.0;   /* this update is the step cap's, not Newton's (POLICY MKS_STOP_UNCAPPED below) */
#endif
#ifdef MKS_PLATE_FTB
        /* POLICY MKS_PLATE_FTB (MK81#164): fraction to the boundary on the conducting PENTODE ports (plate and screen,
         * not the control grid) - an update may not take such a port that conducts (|i| >= MKS_FTB_ION) and sits above MKS_FTB_VMIN volts below MKS_PLATE_FTB of its
         * present voltage; the whole step is scaled so that the worst such port lands there. It breaks the knee
         * cycle: a conducting output plate linearised in its pentode region asks for hundreds of volts down, the cap gives
         * 100, the valve is cut off there and asks for +100 back (period 2 around a root inside the knee). Pentodes only:
         * on the triode plates (MK81's 2-30 V, 4 mA stages) it turned period-3 cycles the ladder resolves into period-5
         * ones it walks to the iteration limit (one sample 78 -> 755 iterations, the worst block +2.5 % at 4x).
         * REFUTED with it (1-Nov): the step cap read only on the conducting ports (MKS_CAP_OFFPORTS): the same cost, but
         * at the shipped stop the run lands -58.0 dB rms from the 3e-9 build where without it -89.0 (test/mna, 100uk).
         * MK81 100 W UK, VH take 20 s, worst knob point, quiet machine (docs/REFERENCE.md), 5 paired rounds, % of a core: 4x mean
         * 26.07 -> 24.96, worst 256-sample block 34.83 -> 31.82; 2x 16.04 -> 14.96, 25.03 -> 21.74; vs the circuit unchanged */
#ifndef MKS_FTB_VMIN
#define MKS_FTB_VMIN 5.0
#endif
#ifndef MKS_FTB_ION
#define MKS_FTB_ION 1e-6
#endif
        {
            const int p0 = 2 * MKS_N_TRIODES, j0 = p0 + 3 * MKS_N_PENTODES;
            for (int p = p0; p < j0; ++p) {
                if ((p - p0) % 3 == 0 || fabs(s->ip[p]) < MKS_FTB_ION) { continue; }   /* the control grid; a cut-off port */
                const double v0 = s->vp[p];
                if (!(v0 > MKS_FTB_VMIN)) { continue; }
                const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                const double dv = sc * ((a < MKS_MNA_N ? f[a] : 0.0) - (b < MKS_MNA_N ? f[b] : 0.0));
                if (dv < 0.0 && v0 + dv < MKS_PLATE_FTB * v0) { sc *= (MKS_PLATE_FTB - 1.0) * v0 / dv; }
            }
        }
#endif
        /* damping: a step that GROWS against the previous one is the oscillation that ends in a fallback; it is
         * applied by `damp` (1 = off) */
        if (it > 0 && step > prev && s->damp < 1.0) { sc *= s->damp; }
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { dlast[i] = sc * f[i]; s->V[i] += dlast[i]; }
#ifdef MKS_STOP_UNCAPPED
        /* POLICY MKS_STOP_UNCAPPED (MK81#207 / #201, 11-Nov-2026): an update the port step cap TRUNCATED is not a Newton
         * step, so it ends nothing - neither the step stop nor the predicted stop, whose remaining-error estimate
         * step^2 / (prev - step) holds only for the step Newton asked for - and, like a limited one, it is no previous
         * step for the next iteration's predicted stop. Without it, MK81 at 2x on hot input (two fuzzes at full volume,
         * Roadcase's +-0.5 V noise): a tangent that moved a port 52.7 kV, a first Newton step of 52.8 kV capped to 800 V,
         * accepted at iteration 0 by the predicted stop (the RTOL-weighted step 12.5 against the predictor's raw 52.7 kV);
         * that state ran the amp away (output 116x full scale) and every following sample down the whole ladder (up to
         * 8,028 iterations, a 125 ms period on an N150). */
        const int lim = s->dev.limited || capped;
#else
        const int lim = s->dev.limited;   /* a limited evaluation is not a converged point */
#endif
        if (!s->resid_stop && STEPW < s->tol && !lim) { s->dev.limit = limit_on; return it + 1; }
        /* Termination by the predicted remaining error: with the steps contracting at rate r = step / prev, what is left
         * after this update is at most the geometric tail step * r / (1 - r) = step^2 / (prev - step) (exact for a
         * linear rate, the chord's; an overestimate for Newton's quadratic one). Below tol, the iteration that would
         * only confirm it is skipped. (Measured 3-Oct: the plain next-step estimate step^2 / prev under-read the chord
         * and left a -52 dB worst point at Gain 1 with a hot input.) */
        /* (the previous step must be a Newton step too: a LIMITED step is not, and its size predicts nothing) */
        if (s->predict_stop && !s->resid_stop && !lim && !prev_lim && (it > 0 || prev < 1e300) && STEPW < s->pstop_rho * PREVW
            && STEPW * STEPW < s->tol * (PREVW - STEPW)) {
            s->dev.limit = limit_on;
            return it + 1;
        }
        if (fresh_every_iteration || step > s->refresh * prev) { have_lu = 0; }
#ifdef MKS_KSTOP
        /* one-update stop: Newton's error after this update is about K * step^2 (quadratic), K measured on this circuit
         * by the last sample that took a second update; stop when that is below tol by the safety factor */
        if (it == 0 && prev >= 1e300 && !lim && !fresh_every_iteration && s->kq > 0.0
            && MKS_KSTOP * s->kq * STEPW * STEPW < s->tol) {
            s->one_step = 1; s->dev.limit = limit_on; return 1;
        }
        if (it == 1 && !lim && !prev_lim && PREVW > 0.0) {
            const double kn = STEPW / (PREVW * PREVW);
            s->kq = kn > 0.5 * s->kq ? kn : 0.5 * s->kq;
        }
#endif
        prev = step;
#if defined(MKS_PTOL) || defined(MKS_RTOL)
        prevw = stepw;
#endif
        prev_lim = lim;
    }
    s->dev.limit = limit_on;
    return -1;
}

int mks_mna_init(MksMna* s, double fs, const double* u, const MksMnaKnobs* k)
{
    memset(s, 0, sizeof *s);
#ifdef MKS_DEVICE_MK81
    s->mk81_cfg.npar = MKS_MK81_NPAR;   /* valves per side of this variant; the closed-form EL34 until a caller sets the table */
    s->mk81_cfg.pentode_tab = 0;
#if defined(MKS_N_FXZ) && MKS_N_FXZ
    for (int z = 0; z < MKS_N_FXZ; ++z) { s->fxz[z] = mks_fxz_params[z]; nl_zener_init(&s->fxz[z]); }
#ifdef MKS_FXZ_TAB
    {   /* built ONCE per compiled system (the first init: the plugin warms it at instantiate, BN_FXZ_WARM), then shared */
        static double ti[MKS_FXZ_TN], tg[MKS_FXZ_TN], tq[MKS_FXZ_TN], tc[MKS_FXZ_TN]; static int built;
        s->fxz_tab_ok = fabs(MKS_FXZ_TLO + (MKS_FXZ_TN - 1) * MKS_FXZ_TH - MKS_FXZ_THI) < 1e-9;   /* else the exact law */
        for (int z = 1; z < MKS_N_FXZ; ++z) if (memcmp(&mks_fxz_params[z], &mks_fxz_params[0], sizeof mks_fxz_params[0])) s->fxz_tab_ok = 0;
        if (s->fxz_tab_ok && !built) {
            for (int n = 0; n < MKS_FXZ_TN; ++n) { const double v = MKS_FXZ_TLO + n * MKS_FXZ_TH; nl_zener_iv(&s->fxz[0], v, &ti[n], &tg[n]); nl_zener_qc(&s->fxz[0], v, &tq[n], &tc[n]); }
            built = 1;
        }
        s->fxz_ti = ti; s->fxz_tg = tg; s->fxz_tq = tq; s->fxz_tc = tc;
    }
#endif
#endif
#endif
#ifdef MKS_SAMPLE_BUDGET
    s->sb_left = 1 << 30;   /* the start's DC solve is not a sample: no budget (each process call resets it) */
#endif
    s->fs = fs;
    s->tol = 3e-5;
    s->max_iter = 48;
    s->extrapolate = 1;   /* first order: 2.42 Newton iterations against 2.68 at second order (MK 81's), 3-Oct */
    s->refresh = 0.1;
    s->predict_stop = 1;
    s->pstop_rho = 1.0;
    s->junction_limit = 1;
    s->step_cap = 100.0;
    s->line_search = 0;
    s->port_norm = 0;
    s->damp = 1.0;   /* OFF: at 0.7 it cut the masters-full fallbacks 19 -> 2 but RAISED the alias at 2 kHz, 0.0513 V, from -45.0 to
                         -24.7 dB ASR (3-Oct night): a damped step leaves a limit cycle that the stop accepts */
    mks_device_init(&s->dev);
    s->knobs = k ? *k : mks_mna_knobs_default;
    /* DC: capacitors open, no junction charge (the device is not started yet), from the prototype's DC solution */
    for (int i = 0; i < MKS_MNA_N; ++i) { s->V[i] = mks_mna_vdc[i]; }
    s->V[MKS_MNA_N] = 0.0;
    for (int k = 0; k < MKS_MNA_NF - 1; ++k) { s->V[MKS_MNA_N + 1 + k] = u[k]; }
    build(s, 0.0);
#if MKS_MNA_DC_RAMP
    /* no operating point known: the sources ramp from zero, a Newton solve per step from the last (source stepping,
     * what circuit simulators fall back to); the step is halved where a solve fails */
    {
        const int max_it = s->max_iter;
        s->max_iter = 200;
        double lam = 0.0, dl = 0.02;
        while (lam < 1.0) {
            const double next = lam + dl > 1.0 ? 1.0 : lam + dl;
            double Vsave[MKS_MNA_N];
            memcpy(Vsave, s->V, sizeof Vsave);
            for (int k = 0; k < MKS_MNA_NF - 1; ++k) { s->V[MKS_MNA_N + 1 + k] = next * u[k]; }
            rhs(s);
            if (newton(s, 1) < 0) {
                memcpy(s->V, Vsave, sizeof Vsave);
                dl *= 0.5;
                if (dl < 1e-6) { s->max_iter = max_it; return -1; }
                continue;
            }
            lam = next;
            if (dl < 0.05) { dl *= 1.5; }
        }
        s->max_iter = max_it;
    }
#else
    rhs(s);
    if (newton(s, 1) < 0) { return -1; }
#endif
    mks_mna_back_lin(s->m, s->y, s->V);
    /* the windings' DC currents (each a short of MKS_MNA_GSHORT at DC), kept for their transient histories */
    double idc[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1], vdcw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    for (int k = 0; k < MKS_MNA_NI; ++k) {
        vdcw[k] = s->V[mks_mna_ind[k].a] - s->V[mks_mna_ind[k].b];
        idc[k] = MKS_MNA_GSHORT * vdcw[k];
    }
    /* transient: the capacitors' companions, each history set for zero current at the operating point */
    build(s, fs);
    for (int k = 0; k < MKS_MNA_NI; ++k) {       /* i = Gamma.v + hl must give the DC current */
        double acc = 0.0;
        for (int j = 0; j < MKS_MNA_NI; ++j) { acc += s->gam[k * MKS_MNA_NI + j] * vdcw[j]; }
        s->hl[k] = idc[k] - acc;
    }
    for (int k = 0; k < MKS_MNA_NC; ++k) {
        s->h[k] = s->gc[k] * (s->V[mks_mna_cap[k].a] - s->V[mks_mna_cap[k].b]);
    }
    port_voltages(s);
    mks_device_start(&s->dev, s->vp, fs);
    s->dev.limit = s->junction_limit;
    for (int i = 0; i < MKS_MNA_N; ++i) { s->Vp1[i] = s->V[i]; s->Vp2[i] = s->V[i]; }
    s->factorisations = 0;
    return 0;
}

/* MK81#165: the rescue of a sample that leaves the whole ladder below (fresh Newton, trust region, source
 * continuation) unconverged. Measured on 100 W UK 2x, Cabinet Feel 1, the FX loop's return 256 samples late (73 such
 * samples in 20 s): the ladder's Newton runs with the DK's PORT-residual line search (line_search 3), and there the
 * residual stagnates (~71 V) while the Newton direction asks for kV: the search halves every step back. Bit 1: the same
 * fresh Newton with the plain step-monotonicity test instead (line_search 1, Deuflhard's), more iterations. Bit 2: the
 * sample taken as two trapezoidal half-steps from the same state (the matrices rebuilt at 2 fs, the capacitor and winding
 * histories converted exactly, the sources and the EMF at the midpoint linearly), each with the whole ladder, recursively
 * down to MKS_RESCUE_DEPTH halvings. Only a sample that failed everything pays: every other sample is bit-identical. */
#if defined(MKS_RESCUE) && MKS_RESCUE
#ifndef MKS_RESCUE_DEPTH
#define MKS_RESCUE_DEPTH 2
#endif
#ifndef MKS_RESCUE_LS_IT
#define MKS_RESCUE_LS_IT 300
#endif
static double process_d(MksMna* s, const double* u, int depth);
#endif


/* MK81#172: set_theta sat INSIDE the rescue's #if above, so a build with MKS_THETA_RT and without MKS_RESCUE's
 * bit 2 (MK81's no-rescue twins, its forced-4x reference) left mks_mna_set_theta undefined: dlopen failed. The two helpers
 * above serve both; the theta switch and the half-step rescue each under their own policy. */
#ifdef MKS_HANDOVER
/* POLICY MKS_HANDOVER (MK81#169): the state of src handed to dst, the SAME circuit built at another rate (or
 * theta): node voltages, the EMF, and each capacitor's and winding's history converted so that its voltage and current
 * are kept (convert_hist with dst's coefficients: h' = h + (gc' - gc) v, hl' = hl + (Gamma' - Gamma) v). The predictor
 * restarts from the handed-over point (no extrapolation history at the new step) and dst's last factorisation is dropped.
 * For local refinement (edges): a coarse engine hands a hard sample to a fine twin and takes the state back. */
void mks_mna_handover(MksMna* dst, const MksMna* src)
{
    double vc[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    branch_v(src, src->V, src->emf, vc, vw);
    memcpy(dst->V, src->V, sizeof dst->V);
    memcpy(dst->h, src->h, sizeof dst->h); memcpy(dst->hl, src->hl, sizeof dst->hl);
    convert_hist(dst, src->gc, src->gam, vc, vw);
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { dst->Vp1[i] = src->V[i]; dst->Vp2[i] = src->V[i]; }
    dst->emf = src->emf;
#if defined(MKS_RESCUE) && MKS_RESCUE
    dst->emf_prev = src->emf;
#endif
    dst->buf_dg = src->buf_dg;
    dst->w_ok = 0;
}
#endif

#ifdef MKS_THETA_RT
/* MKS_THETA_RT: theta on / off at run time, after mks_mna_init (which clears the engine). The matrices are rebuilt at the
 * present rate and knobs and the histories converted as a change of step converts them (convert_hist: each element's
 * voltage and current kept), so the sample after the switch starts from the same physical state */
/* the time step changed, the state (v, i) of every element kept: the matrices rebuilt at fs, the histories converted,
 * the extrapolation restarted from the present point */
void mks_mna_set_fs(MksMna* s, double fs)
{
    double gc0[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], gam0[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];
    double vc[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    memcpy(gc0, s->gc, sizeof(double) * MKS_MNA_NC); memcpy(gam0, s->gam, sizeof(double) * MKS_MNA_NI * MKS_MNA_NI);
    branch_v(s, s->V, s->emf, vc, vw);
    s->fs = fs; build(s, fs); convert_hist(s, gc0, gam0, vc, vw);
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->Vp1[i] = s->V[i]; s->Vp2[i] = s->V[i]; }
    s->w_ok = 0;
}

/* nlcore#80: the operating point mks_mna_init solves is the circuit's rest (a settled engine sits within millivolts of
 * it), but a transient started there is NOT at rest: MK81's push-pull plates swung apart by 9 V on the first sample and
 * 54 V by the third, a -17.7 dBFS thump, and a restart mid-stream needed 5,500-8,000 Newton iterations in its first
 * samples. The settle walks the state to the transient's own rest in PSEUDO-TIME: backward Euler (theta 1 on every
 * capacitor and winding: it damps the stiff modes a coarse trapezoid would ring) at fs_coarse for n steps with the sources
 * held at u, then back to the caller's step and method with the histories converted. MK81 100 W UK, 96 kHz: 300 steps at
 * 1 kHz leave the first real step moving the plates by 3 uV (9 V without), for ~0.3 ms (an init alone ~35 us). */
void mks_mna_settle(MksMna* s, const double* u, double fs_coarse, int n)
{
    const double fs0 = s->fs;
    const int th_on = s->theta_on; const double thc = s->theta_c, thm = s->theta_cmax, thl = s->theta_l;
    mks_mna_set_theta(s, 1, 1.0, 1e300, 1.0);
    mks_mna_set_fs(s, fs_coarse);
    for (int i = 0; i < n; ++i) {
#if defined(MKS_RESCUE) && MKS_RESCUE
#ifdef MKS_SAMPLE_BUDGET
        s->sb_left = 1 << 30;   /* the settle is not a sample: no budget, and it does not count in sb_used_max */
#endif
        (void)process_d(s, u, 0);
#else
        (void)mks_mna_process(s, u);
#endif
    }
    mks_mna_set_fs(s, fs0);
    mks_mna_set_theta(s, th_on, thc, thm, thl);
}

void mks_mna_set_theta(MksMna* s, int on, double theta_c, double theta_cmax, double theta_l)
{
    double gc0[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], gam0[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];
    double vc[MKS_MNA_NC > 0 ? MKS_MNA_NC : 1], vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
    memcpy(gc0, s->gc, sizeof(double) * MKS_MNA_NC); memcpy(gam0, s->gam, sizeof(double) * MKS_MNA_NI * MKS_MNA_NI);
    branch_v(s, s->V, s->emf, vc, vw);
    s->theta_on = on; s->theta_c = theta_c; s->theta_cmax = theta_cmax; s->theta_l = theta_l;
    if (s->fs > 0.0) { build(s, s->fs); convert_hist(s, gc0, gam0, vc, vw); }
}
#endif

#if defined(MKS_RESCUE) && (MKS_RESCUE & 2)
/* the failed sample again as two half-steps; V0 / uprev: the last committed sample's nonlinear nodes and sources (the
 * linear nodes and the histories are still the committed ones: nothing below the ladder touched them) */
static double substep(MksMna* s, const double* u, const double* uprev, const double* V0, int depth)
{
    const double fs0 = s->fs, emf1 = s->emf, emf0 = s->emf_prev;
    double gc0[MKS_MNA_NC], gam0[MKS_MNA_NI > 0 ? MKS_MNA_NI * MKS_MNA_NI : 1];
    double vc[MKS_MNA_NC], vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1], Vp1o[MKS_MNA_N];
    double umid[MKS_MNA_NF > 1 ? MKS_MNA_NF - 1 : 1];
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; Vp1o[i] = s->Vp1[i]; }
    for (int k = 0; k < MKS_MNA_NF - 1; ++k) { s->V[MKS_MNA_N + 1 + k] = uprev[k]; umid[k] = 0.5 * (uprev[k] + u[k]); }
    branch_v(s, s->V, emf0, vc, vw);
    memcpy(gc0, s->gc, sizeof gc0); memcpy(gam0, s->gam, sizeof gam0);
    s->fs = 2.0 * fs0; build(s, s->fs);
    convert_hist(s, gc0, gam0, vc, vw);
    /* the extrapolated start at the half step: the sample before, halfway */
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->Vp1[i] = 0.5 * (V0[i] + Vp1o[i]); }
    s->w_ok = 0;                                       /* the last factorisation belongs to the full step's matrices */
    s->emf = 0.5 * (emf0 + emf1);
    (void)process_d(s, umid, depth + 1);
    s->emf = emf1;
    const double y = process_d(s, u, depth + 1);
    /* back to the full step: the histories at the end point, the matrices at fs */
    memcpy(gc0, s->gc, sizeof gc0); memcpy(gam0, s->gam, sizeof gam0);
    branch_v(s, s->V, emf1, vc, vw);
    s->fs = fs0; build(s, fs0);
    convert_hist(s, gc0, gam0, vc, vw);
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->Vp2[i] = Vp1o[i]; s->Vp1[i] = V0[i]; }
    s->w_ok = 0;
    return y;
}
#endif

#if defined(MKS_RESCUE) && MKS_RESCUE
#ifdef MKS_SAMPLE_BUDGET
double mks_mna_process(MksMna* s, const double* u)
{
    s->sb_left = MKS_SAMPLE_BUDGET;
    const double y = process_d(s, u, 0);
    const int used = MKS_SAMPLE_BUDGET - s->sb_left;
    if (used > s->sb_used_max) { s->sb_used_max = used; }
    if (s->sb_left <= 0) { s->budget_cuts++; }
    return y;
}
#else
double mks_mna_process(MksMna* s, const double* u)
{
    return process_d(s, u, 0);
}
#endif

static double process_d(MksMna* s, const double* u, int depth)
{
    (void)depth;
#else
double mks_mna_process(MksMna* s, const double* u)
{
#endif
    CZ_T0(czp0);
    double uprev[MKS_MNA_NF > 1 ? MKS_MNA_NF - 1 : 1];
    for (int k = 0; k < MKS_MNA_NF - 1; ++k) { uprev[k] = s->V[MKS_MNA_N + 1 + k]; s->V[MKS_MNA_N + 1 + k] = u[k]; }
    rhs(s);
    double V0[MKS_MNA_N];
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { V0[i] = s->V[i]; }
    int predicted = 0;
    if (s->predict && s->w_ok) {
        /* the tangent predictor (MK 81's DK P2): the last sample's solution moved by the last factorisation applied to the
         * change of the right-hand side, i.e. the first Newton step from there without a device evaluation nor a new LU;
         * capped at step_cap on the ports like a Newton step */
        double d[MKS_MNA_N];
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { d[i] = s->y[i] - s->yp[i]; }
        mks_mna_solve_schur(s->w, d);
        double st = 0.0;
        for (int p = 0; p < MKS_MNA_NP; ++p) {
            const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
            const double dd = (a < MKS_MNA_N ? d[a] : 0.0) - (b < MKS_MNA_N ? d[b] : 0.0);
            if (!(fabs(dd) <= st)) { st = fabs(dd); }
        }
        /* a tangent that wants a port to move more than pred_max is extrapolating across a knee: not trusted, the
         * ordinary start is taken instead (measured 3-Oct: a 100 V predicted plate move sent Newton to 4 kV steps) */
        if (isfinite(st) && st <= s->pred_max) {
#ifdef MKS_MIXPRED
            /* the nodes a junction or a valve grid touches keep the extrapolated start: their law is exponential and the
             * tangent overshoots it (MK Silver's x2.5 with the plain predictor, docs/MK81_TWIN.md section 6) */
            static int soft[MKS_MNA_N]; static int soft_ready = 0;
            if (!soft_ready) {
                for (int p = 0; p < MKS_MNA_NP; ++p) {
                    const int p0 = 2 * MKS_N_TRIODES, j0 = p0 + 3 * MKS_N_PENTODES;
                    const int hard = (p < p0 && !(p & 1)) || (p >= p0 && p < j0 && (p - p0) % 3 == 0) || p >= j0;
                    if (hard) { const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                                if (a < MKS_MNA_N) { soft[a] = 1; } if (b < MKS_MNA_N) { soft[b] = 1; } }
                }
                soft_ready = 1;
            }
            st = 0.0;
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) {
                const double xv = 2.0 * V0[i] - s->Vp1[i];
                s->V[i] = soft[i] ? xv : V0[i] + d[i];
            }
            {   /* the step handed to the predicted stop: the predicted move on the ports */
                for (int p = 0; p < MKS_MNA_NP; ++p) {
                    const int a = mks_mna_port[p].a, b = mks_mna_port[p].b;
                    const double dd = (a < MKS_MNA_N ? s->V[a] - V0[a] : 0.0) - (b < MKS_MNA_N ? s->V[b] - V0[b] : 0.0);
                    if (!(fabs(dd) <= st)) { st = fabs(dd); }
                }
            }
#else
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i] + d[i]; }
#endif
#ifdef MKS_PREDERR
            /* MK81#104 (LowDude's s.10.2 "predictor-error extrapolation"): the tangent start moved by the error the last
             * predicted sample left (its converged point minus its prediction) */
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->vpred[i] = s->V[i]; s->V[i] += s->pe[i]; }
#endif
            predicted = 1;
            if (s->pred_stop) { s->prev0 = st; }
        }
    }
    if (!predicted && s->extrapolate) {
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = (s->extrapolate == 2) ? 3.0 * V0[i] - 3.0 * s->Vp1[i] + s->Vp2[i] : 2.0 * V0[i] - s->Vp1[i]; }
    }
#ifdef MKS_FIRST_BUDGET
    /* MK81#104 round c (cost-pass row 25): the first attempts (predicted start, then the extrapolated one) get a Newton budget
     * of MKS_FIRST_BUDGET iterations instead of max_iter; a sample that does not converge inside it takes the fallback
     * ladder at once (fresh Jacobian from the last converged point, max_iter, the trust region, the continuation) */
    const int mi_first = s->max_iter;
    s->max_iter = MKS_FIRST_BUDGET;
#endif
#ifdef MKS_RETRY_CAP
    double rc_cap = 0.0; int rc_ls = 0, rc_on = 0;
#endif
#ifdef MKS_WALK_RECAP
    s->walk_armed = predicted;
#endif
    int it = newton(s, 0);
#ifdef MKS_WALK_RECAP
    if (s->walk_armed == 2) { s->step_cap = s->walk_cap_keep; s->line_search = s->walk_ls_keep; }
    s->walk_armed = 0;
#endif
    s->w_ok = it >= 0;
#ifdef MKS_PREDERR
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->pe[i] = (predicted && it >= 0) ? s->V[i] - s->vpred[i] : 0.0; }
#endif
    if (it < 0 && predicted && s->extrapolate) {
        /* a predicted start that did not converge: the ordinary extrapolated start before the fallback proper */
#ifdef MKS_RETRY_CAP
        /* POLICY MKS_RETRY_CAP (MK81#164 / #176): the RETRIES of a sample whose predicted start failed (the
         * extrapolated start, the fresh-Jacobian fallback; the trust region keeps its own caps) run with the port step
         * capped at MKS_RETRY_CAP volts and no residual line search; the caller's settings come back at the sample's end.
         * The first attempt is untouched, so every sample it converges is bit-identical to the build without it.
         * MK81 100 W UK, VH take 20 s, worst knob point (docs/REFERENCE.md): fallbacks 6905 -> 1474 (4x), 4457 -> 1035 (2x); the
         * worst 256-sample block 45.7 -> 39.6 % (4x) and 34.0 -> 28.8 % of a core (2x), the mean -5 %; against the
         * circuit at 768 kHz the ANMR worst / mean unchanged to 0.01 dB at both rates */
        rc_cap = s->step_cap; rc_ls = s->line_search;
        s->step_cap = MKS_RETRY_CAP; s->line_search = 0; rc_on = 1;
#endif
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = 2.0 * V0[i] - s->Vp1[i]; }
        it = newton(s, 0);
        s->w_ok = it >= 0;
    }
#ifdef MKS_FIRST_BUDGET
    s->max_iter = mi_first;
#endif
    if (it < 0) {
        /* the fallback: from the last converged voltages, a fresh Jacobian every iteration */
        s->fallbacks++; ST(st_fb++); CZ(cz_fb++);
        for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; }
        it = newton(s, 1);
        /* MK81#112 (carried from MK 81's copy): a TRUST REGION before the source continuation. Measured on hot all-1
         * material (50 W US, OC all-1, 20 s at 96 kHz): of 117 fallbacks, 26 ended unconverged and the last iterate was
         * output - a 30-55 V one-sample spike each, the worst ANMR frames. Traced: from the last converged point, full
         * Newton steps capped at 100 V on the ports swing the output plates (pa_top / pa_bot) and the PI plate by
         * hundreds of volts across the pentodes' knee and cut-off and cycle there; the residual line search (4 halvings)
         * does not stop it, and neither does walking the sources (K up to 64). The same fresh-Jacobian Newton with the
         * port step capped at 20 V, then 5 V, with room for the extra iterations, converges all of them (0 of 118 left).
         * Only a sample whose fresh Newton failed pays it: every other sample is bit-identical. */
        if (it < 0) {
            const double cap_keep = s->step_cap; const int mi_keep = s->max_iter;
            static const double caps[2] = { 20.0, 5.0 };
            static const int    its[2]  = { 200, 400 };
            for (int c = 0; c < 2 && it < 0; ++c) {
                for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; }
                s->step_cap = caps[c]; s->max_iter = its[c];
                it = newton(s, 1);
            }
            s->step_cap = cap_keep; s->max_iter = mi_keep;
            if (it >= 0) { s->trust_rescues++; }
        }
        /* source continuation (SPICE's source stepping, inside the sample): the sources walk from the last sample's
         * values to this one's in K sub-steps, each solved from the one before; the end point is the same equations'
         * solution, only the path to it changes. Only a sample whose Newton failed pays it. */
        for (int K = 4; it < 0 && K <= 64 && SB_LEFT(s); K *= 4) {
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; }
            int ok = 1;
            for (int j = 1; j <= K && ok; ++j) {
                const double lam = (double)j / K;
                for (int k = 0; k < MKS_MNA_NF - 1; ++k) { s->V[MKS_MNA_N + 1 + k] = uprev[k] + lam * (u[k] - uprev[k]); }
                rhs(s);
                ok = newton(s, 0) >= 0;
            }
            if (ok) { s->continuations++; it = s->max_iter; }
            for (int k = 0; k < MKS_MNA_NF - 1; ++k) { s->V[MKS_MNA_N + 1 + k] = u[k]; }
            if (!ok) { rhs(s); }
        }
#if defined(MKS_RESCUE) && (MKS_RESCUE & 1)
        if (it < 0 && SB_LEFT(s)) {
            const int lk = s->line_search, mk = s->max_iter;
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; }
            s->line_search = 1; s->max_iter = MKS_RESCUE_LS_IT;
            it = newton(s, 1);
            s->line_search = lk; s->max_iter = mk;
            if (it >= 0) { s->rescues_ls++; }
        }
#endif
#if defined(MKS_RESCUE) && (MKS_RESCUE & 2)
        if (it < 0 && depth < MKS_RESCUE_DEPTH && SB_LEFT(s)) {
            s->rescues_sub++;
#ifdef MKS_RETRY_CAP
            if (rc_on) { s->step_cap = rc_cap; s->line_search = rc_ls; }   /* the half-steps run the caller's settings */
#endif
            return substep(s, u, uprev, V0, depth);
        }
#endif
        if (it < 0) {
            /* still not below tol: keep the last iterate if it is finite (it is the best estimate the solver has; the
             * previous sample's voltages would be a step back in time), else the previous sample */
            s->unconverged++;
            int finite = 1;
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { finite &= isfinite(s->V[i]); }
            if (!finite) { for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; } }
#ifdef MKS_SAMPLE_BUDGET
            /* a sample the BUDGET cut holds the last converged point: the iterate it stopped at can be far from the
             * solution, and committed as state it ran the amp away (MK81, 9-Nov-2026: +31..+33 dBFS for seconds on hot
             * noise and head switches) */
            if (!SB_LEFT(s)) { for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->V[i] = V0[i]; } s->budget_holds++; }
#endif
            it = s->max_iter;
        }
    }
#ifdef MKS_RETRY_CAP
    if (rc_on) { s->step_cap = rc_cap; s->line_search = rc_ls; }
#endif
    s->last_iters = it; ST(st_samp++); ST(st_hist[it < 7 ? it : 7]++);
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->yp[i] = s->y[i]; }
    mks_mna_back_lin(s->m, s->y, s->V);
    s->buf_passes = 0;
    if (s->buf_emf || (s->buf_src > 0 && s->buf_src < MKS_MNA_NF - 1)) {
        /* the buffered source or EMF (see mks_mna.h): correct it to the node it follows and re-solve warm, before the
         * commit */
        double* us = s->buf_emf ? &s->emf : &s->V[MKS_MNA_N + 1 + s->buf_src];
        /* g(x) = x - (gain V[node](x) + off) = 0, by SECANT after the first fixed-point step: through a coupling the
         * node follows x strongly (MK 81's C9 into the PI's grid), the plain fixed point crawled at 11 extra solves a
         * sample on a series loop at tol 1e-9 (6-Oct); the node's response to x is near-linear within a sample */
        double xp = 0.0, gp = 0.0;
        for (int pass = 0; pass < s->buf_max_pass; ++pass) {
            const double want = s->buf_gain * s->V[s->buf_node] + s->buf_off;
            const double g = *us - want;
            if (!(fabs(g) > s->tol)) { break; }
            double xn = want;                                    /* the fixed-point step */
            if (pass > 0 && g != gp) {                           /* the secant step, its slope kept for the next sample */
                s->buf_dg = (g - gp) / (*us - xp);
                xn = *us - g / s->buf_dg;
            } else if (pass == 0 && s->buf_dg > 0.0) {           /* first step: Newton on the last sample's slope */
                xn = *us - g / s->buf_dg;
            }
            xp = *us; gp = g;
            *us = xn;
            rhs(s);
            for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->yp[i] = s->y[i]; }   /* the next predictor starts from the FINAL rhs */
            const int it2 = newton(s, 0);
            if (it2 >= 0) { s->last_iters += it2; } else { s->last_iters += s->max_iter; }
            mks_mna_back_lin(s->m, s->y, s->V);
            s->buf_passes++; s->buf_extra++;
        }
    }
    /* commit: the histories, the junction charge, the extrapolation's past */
#ifdef MKS_THETA_C
    /* MK81#164: the capacitors on the theta-method, C (v' - v) fs = theta i' + (1 - theta) i: gc = C fs / theta
     * and h' = (gc / theta) v - ((1 - theta) / theta) h (theta = 1/2 is the trapezoidal rule, h' = 2 gc v - h) */
    for (int k = 0; k < MKS_MNA_NC; ++k) {
        const double v = s->V[mks_mna_cap[k].a] - s->V[mks_mna_cap[k].b];
        const double th = MKS_THC(k);
        s->h[k] = s->gc[k] * (1.0 / th) * v - ((1.0 - th) / th) * s->h[k];
    }
#elif defined(MKS_THETA_RT)
    if (s->theta_on) {
        for (int k = 0; k < MKS_MNA_NC; ++k) {
            const double v = s->V[mks_mna_cap[k].a] - s->V[mks_mna_cap[k].b];
            s->h[k] = s->th_a[k] * v - s->th_b[k] * s->h[k];
        }
    } else { mks_mna_commit_caps(s->gc, s->V, s->h); }
#else
    mks_mna_commit_caps(s->gc, s->V, s->h);
#endif
#ifdef MKS_MNA_EMF_CAP
    /* the EMF capacitor's history from its BRANCH voltage: h' = 2 gc (v + sign e) - h */
    {   /* detached: the branch is emf - V(b), so the correction from the generated (V(a) - V(b)) is emf - V(a) */
        const int a = mks_mna_cap[MKS_MNA_EMF_CAP].a;
        const double corr = s->emf_detached ? s->emf - (a < MKS_MNA_N + MKS_MNA_NF ? s->V[a] : 0.0) : MKS_MNA_EMF_SIGN * s->emf;
#ifdef MKS_THETA_C
        s->h[MKS_MNA_EMF_CAP] += s->gc[MKS_MNA_EMF_CAP] * (1.0 / MKS_THC(MKS_MNA_EMF_CAP)) * corr;
#elif defined(MKS_THETA_RT)
        if (s->theta_on) { s->h[MKS_MNA_EMF_CAP] += s->gc[MKS_MNA_EMF_CAP] * (1.0 / THC_RT(s, MKS_MNA_EMF_CAP)) * corr; }
        else { s->h[MKS_MNA_EMF_CAP] += 2.0 * s->gc[MKS_MNA_EMF_CAP] * corr; }
#else
        s->h[MKS_MNA_EMF_CAP] += 2.0 * s->gc[MKS_MNA_EMF_CAP] * corr;
#endif
    }
#endif
    if (MKS_MNA_NI > 0) {                        /* the windings: hl += 2 Gamma v */
        double vw[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
        for (int k = 0; k < MKS_MNA_NI; ++k) { vw[k] = s->V[mks_mna_ind[k].a] - s->V[mks_mna_ind[k].b]; }
#ifdef MKS_GAM_COLS
        /* the same sums in the same order per row (acc_k = 0 + g_k0 v_0 + ... + g_k16 v_16), run by columns so the rows
         * advance together (two-wide SSE2) instead of 17 serial chains: bit-identical */
        double accv[MKS_MNA_NI > 0 ? MKS_MNA_NI : 1];
        for (int k = 0; k < MKS_MNA_NI; ++k) { accv[k] = 0.0; }
        for (int j = 0; j < MKS_MNA_NI; ++j) {
            const double vj = vw[j]; const double* g = &s->gamT[j * MKS_MNA_NI];
            for (int k = 0; k < MKS_MNA_NI; ++k) { accv[k] += g[k] * vj; }
        }
#endif
        for (int k = 0; k < MKS_MNA_NI; ++k) {
            double acc = 0.0;
#ifdef MKS_GAM_COLS
            acc = accv[k];
#elif defined(MKS_SPARSE_GAM)
            const double* g = &s->gam[k * MKS_MNA_NI];
            const unsigned char* jj = &s->gnz_j[k * MKS_MNA_NI];
            for (int q = 0; q < s->gnz_n[k]; ++q) { acc += g[jj[q]] * vw[jj[q]]; }
#else
            for (int j = 0; j < MKS_MNA_NI; ++j) { acc += s->gam[k * MKS_MNA_NI + j] * vw[j]; }
#endif
#ifdef MKS_THETA_L
            s->hl[k] += acc * (1.0 / MKS_THETA_L);
#elif defined(MKS_THETA_RT)
            if (s->theta_on) { s->hl[k] += acc * s->th_l_inv; } else { s->hl[k] += 2.0 * acc; }
#else
            s->hl[k] += 2.0 * acc;
#endif
        }
    }
    port_voltages(s);
#ifdef MKS_KSTOP
    if (s->one_step) { mks_device_currents(s->vp, s->ip, &s->dev); s->one_step = 0; ST(st_cur++); }
#endif
    mks_device_commit(s->vp, &s->dev);
#if defined(MKS_RESCUE) && MKS_RESCUE
    s->emf_prev = s->emf;
#endif
    for (int i = MKS_MNA_NLIN; i < MKS_MNA_N; ++i) { s->Vp2[i] = s->Vp1[i]; s->Vp1[i] = V0[i]; }
    CZ_ADD(cz_proc, czp0); CZ(cz_samp++);
    return s->V[MKS_MNA_OUT];
}
