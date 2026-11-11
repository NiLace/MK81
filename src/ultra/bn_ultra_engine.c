/* bn_ultra_engine.c - ONE of ULTRA's 15 systems behind plain functions. Compiled once per system with
 * -DBN_UV=<100uk|...|fx0_100uk|...|fxI_100ri>, -I src/ultra/gen<...>_<variant> first, and the solver's global symbols
 * renamed per system (tools/build_ultra_objs.sh), so they all link into one .so. The device's settings (valves per
 * side, the pentode table) live in each engine instance (MksMna.mk81_cfg), never in a global. */
#include <math.h>
#include <string.h>
#include "nl_mna.h"   /* nlcore's nodal amp solver (src/vendor/nlcore) */
#include "bn_fullamp_own.h"
#include "bn_ultra.h"

#define BN_CAT2(a, b, c) a##b##c
#define BN_CAT(a, b, c) BN_CAT2(a, b, c)
#define BN_UF(name) BN_CAT(bn_ultra_, BN_UV, _##name)

double BN_UF(vht)(void) { return mks_source_dc[1]; }

static MksMnaKnobs to_kn(const BnUltraKnobs* k)
{
    MksMnaKnobs kn;
    memset(&kn, 0, sizeof kn);
    kn.VOL = k->vol; kn.TREB = k->treb; kn.BASS = k->bass; kn.MID = k->mid; kn.MASTER = k->master;
    kn.PRES = k->pres; kn.LOWIN = k->lowin; kn.CAB = k->cab;
#if MKS_MNA_NKNOB > 8
    kn.ERA = k->era;   /* the rectifier's era (systems generated with MKS_MK81_RECT=1) */
#endif
    return kn;
}

#ifndef BN_START_SETTLE_FS
#define BN_START_SETTLE_FS 1000.0   /* the pseudo-time step of the start's settle (backward Euler), Hz */
#endif
#ifndef BN_START_SETTLE_N
#define BN_START_SETTLE_N 300       /* its steps: 0.3 s of pseudo-time */
#endif
static int engine_init(MksMna* s, double fs, const BnUltraKnobs* k, double u1, double u2)
{
    const double u[MKS_N_SOURCES] = { 0.0, u1, u2 };
    const MksMnaKnobs kn = to_kn(k);
    /* the DC operating point on the closed-form pentode law, then the tabulated law for THIS instance (the EL34's; the
     * 6550's when built with BN_6550_TAB, as the product is) */
    const int r = mks_mna_init(s, fs, u, &kn);
    s->mk81_cfg.pentode_tab = 1;
#ifdef MKS_THETA_RT
    /* the theta-method by the engine rate. Below BN_ULTRA_THETA_FS (2x of a 44.1 / 48 kHz host) theta > 0.5 on the
     * small capacitors (up to 2e-10 F; the larger ones stay trapezoidal) and on the windings damps the stiff ultrasonic
     * modes the trapezoid leaves ringing at Nyquist, and cuts Newton's iterations. At 4x, and at 2x of a 96 kHz host,
     * the plain trapezoid (there theta damps the audio band too and costs fidelity against the circuit). The product
     * builds with BN_ULTRA_THETA=0.52 (plugin.mk). */
#ifndef BN_ULTRA_THETA_FS
#define BN_ULTRA_THETA_FS 150000.0
#endif
#ifndef BN_ULTRA_THETA
#define BN_ULTRA_THETA 0.55
#endif
    if (r == 0 && fs < BN_ULTRA_THETA_FS) { mks_mna_set_theta(s, 1, BN_ULTRA_THETA, 2e-10, BN_ULTRA_THETA); }
#endif
    /* the product's Newton settings. tol 3e-3: Newton's stop on the ports (1e-3 costs ~4 % more for the same
     * fidelity against the circuit; 1e-2 does not hold the fidelity over the whole knob travel) */
    s->tol = 3e-3;
#if defined(MKS_THETA_RT)
    /* nlcore#80 / MK81#198: the operating point is the circuit's rest, but a transient started there is not at rest (the
     * push-pull plates swing apart: a -17.7 dBFS thump, and a restart mid-stream sent Newton down its whole ladder for
     * 5,500-8,000 iterations). Every start - activate(), a head / factor / card restart mid-stream - settles in
     * pseudo-time first (nlcore's mks_mna_settle: ~0.3 ms). */
    if (r == 0) { mks_mna_settle(s, u, BN_START_SETTLE_FS, BN_START_SETTLE_N); }
#endif
    s->ls_gate = 100.0;
    s->pred_stop = 1;
    s->predict = 1;
    s->pred_max = 1e9;
    s->line_search = 3;
    s->port_norm = 1;
    s->refresh = 0.05;
#if defined(MKS_N_FXZ) && MKS_N_FXZ
/* card systems that carry the send zeners (MKS_N_FXZ; the shipped card systems do not, so this block is inert in the
 * product, whose card systems take BN_CARD_LS0 below): their Newton setting, n = 7 by default (no residual line search
 * and a 20 V port step cap, grown x2 while the target holds: MKS_CAP_GROW). BN_CARD_NEWTON=0 = the product's settings */
#ifndef BN_CARD_NEWTON
#define BN_CARD_NEWTON 7
#endif
    /* the other values, measurement builds only: 1 no predicted start, 2 no residual line search, 3 the monotonicity
     * line search, 4 no predicted stop, 5 no theta at 2x, 6 = 1 + 2 + 4 + 5, 7 = 2 + a 20 V step cap, 8 = 2 + 5 V */
#if BN_CARD_NEWTON == 1 || BN_CARD_NEWTON == 6
    s->predict = 0;
#endif
#if BN_CARD_NEWTON == 2 || BN_CARD_NEWTON == 6 || BN_CARD_NEWTON == 7 || BN_CARD_NEWTON == 8
    s->line_search = 0;
#endif
#if BN_CARD_NEWTON == 7
    s->step_cap = 20.0;
#endif
#if BN_CARD_NEWTON == 8
    s->step_cap = 5.0;
#endif
#if BN_CARD_NEWTON == 3
    s->line_search = 1;
#endif
#if BN_CARD_NEWTON == 4 || BN_CARD_NEWTON == 6
    s->pred_stop = 0;
#endif
#if (BN_CARD_NEWTON == 5 || BN_CARD_NEWTON == 6) && defined(MKS_THETA_RT)
    if (r == 0) { mks_mna_set_theta(s, 0, 0.5, 0.0, 0.5); }
#endif
#endif
#if defined(BN_CARD_LS0) && defined(MKS_MK81_CARD) && MKS_MK81_CARD
    /* the card systems (the loop ON): no residual line search (the product's search sends many of their samples down
     * the fallback ladder). Set by the product build (-DBN_CARD_LS0, plugin.mk) */
    s->line_search = 0;
#endif
    return r;
}

size_t BN_UF(size)(void) { return sizeof(MksMna); }
int BN_UF(init)(void* mem, double fs, const BnUltraKnobs* k, double u1, double u2)
{
    return engine_init((MksMna*)mem, fs, k, u1, u2);
}

void BN_UF(set_knobs)(void* mem, const BnUltraKnobs* k)
{
    const MksMnaKnobs kn = to_kn(k);
    mks_mna_set_knobs((MksMna*)mem, &kn);
}

double BN_UF(process)(void* mem, double x, double u1, double u2, double top, BnUltraTap* tap)
{
    MksMna* s = (MksMna*)mem;
#if (defined(MKS_N_FXZ) && MKS_N_FXZ) || defined(MKS_MK81_CARD)
    /* a card system (the loop ON), with or without the send zeners in the system: `top` is the card's return, the
     * fourth source fx_ret; no EMF (C9 hangs from fx_w3) */
    const double u[MKS_N_SOURCES] = { x, u1, u2, top };
#else
    const double u[MKS_N_SOURCES] = { x, u1, u2 };
    /* detached (the FX loop's main engine): C9's top is the caller's; attached: the amp as drawn, no EMF at all */
    s->emf = s->emf_detached ? top : 0.0;
#endif
    const double y = mks_mna_process(s, u);
    if (tap) {
        /* the port currents, amperes: 2 per triode (grid-cathode, plate-cathode), then 3 per pentode (g1-k, g2-k, p-k):
         * 2 MKS_N_TRIODES + 3 q + 1 is pentode q's screen current, + 2 its plate current */
        double ipl = 0.0, isc = 0.0;
        for (int q = 0; q < MKS_N_PENTODES; ++q) {
            ipl += s->ip[2 * MKS_N_TRIODES + 3 * q + 2];
            isc += s->ip[2 * MKS_N_TRIODES + 3 * q + 1];
        }
        tap->w2 = s->V[MKS_MNA_V_W2];
        tap->ipl = ipl;
        tap->isc = isc;
#ifdef MKS_MNA_V_FX_C
        tap->fxc = s->V[MKS_MNA_V_FX_C];
#else
        tap->fxc = 0.0;
#endif
    }
    return y;
}

int BN_UF(fallbacks)(const void* mem) { return ((const MksMna*)mem)->fallbacks; }
int BN_UF(unconverged)(const void* mem) { return ((const MksMna*)mem)->unconverged; }
void BN_UF(currents)(const void* mem, double* ipl, double* isc)
{
    const MksMna* s = (const MksMna*)mem;
    double a = 0.0, b = 0.0;
    for (int q = 0; q < MKS_N_PENTODES; ++q) { a += s->ip[2 * MKS_N_TRIODES + 3 * q + 2]; b += s->ip[2 * MKS_N_TRIODES + 3 * q + 1]; }
    *ipl = a; *isc = b;
}
void BN_UF(detach)(void* mem, int on)
{
    mks_mna_set_emf_detached((MksMna*)mem, on);
}
const double* BN_UF(state)(const void* mem, int* n) { *n = MKS_MNA_NC; return ((const MksMna*)mem)->h; }
/* the node voltages, the generator's V layout (src/ultra/gen_<v>/mks_preamp_mna.h): the FX loop's send system reads its
 * sources from them and is seeded from them (src/prepig/) */
const double* BN_UF(nodes)(const void* mem) { return ((const MksMna*)mem)->V; }
void BN_UF(arrays)(void* mem, BnUltraArrays* a)
{
    MksMna* s = (MksMna*)mem;
    a->V = s->V; a->Vp1 = s->Vp1; a->Vp2 = s->Vp2; a->h = s->h;
}
