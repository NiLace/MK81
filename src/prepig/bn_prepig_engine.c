/* bn_prepig_engine.c - ONE of the five FX-loop send systems (bn_prepig.h) behind plain functions. Compiled once per
 * variant with -DBN_GV=<100uk|100us|50uk|50us|100ri>, -I src/prepig/gen_<variant> first, the solver's global symbols
 * renamed per variant and its device set to the grid-conducting one (bn_prepig_device.c, NL_MNA_DEVICE), so they link
 * into ULTRA's .so beside its own systems (tools/build_ultra_objs.sh). */
#include <string.h>

#include "nl_mna.h"   /* nlcore's nodal amp solver (src/vendor/nlcore) */
#include "bn_pentode_own.h"
#include "bn_prepig.h"
#include "bn_prepig_map.h"

#define BN_CAT2(a, b, c) a##b##c
#define BN_CAT(a, b, c) BN_CAT2(a, b, c)
#define BN_GF(name) BN_CAT(bn_prepig_, BN_GV, _##name)

/* the generated map covers exactly this system */
_Static_assert(sizeof bn_pg_vmap / sizeof bn_pg_vmap[0] == MKS_MNA_N + MKS_MNA_NF, "bn_prepig_map.h: V slots");
_Static_assert(sizeof bn_pg_cmap / sizeof bn_pg_cmap[0] == MKS_MNA_NC, "bn_prepig_map.h: capacitors");
_Static_assert(MKS_N_SOURCES == 4, "the send system's sources are [in, xpi, nbias0, t4]");

size_t BN_GF(size)(void) { return sizeof(MksMna); }

static MksMnaKnobs to_kn(const BnPrepigKnobs* k)
{
    MksMnaKnobs kn = { k->vol, k->treb, k->bass, k->mid, k->master, k->pres, k->lowin };
    return kn;
}

int BN_GF(init)(void* mem, double fs, const BnPrepigKnobs* k, double vbias, const double* uv)
{
    MksMna* s = (MksMna*)mem;
    const double u[MKS_N_SOURCES] = { 0.0, uv[BN_PG_U_XPI], vbias, uv[BN_PG_U_T4] };
    const MksMnaKnobs kn = to_kn(k);
    const int r = mks_mna_init(s, fs, u, &kn);
    /* the grid law from its table after the DC solve, as ULTRA's instances: the EL34's, and the 6550's when it has one */
#if defined(BN_GRID_TAB) && defined(BN_6550_TAB)
    s->mk81_cfg.pentode_tab = 1;
#else
    if (!MKS_MK81_US) { s->mk81_cfg.pentode_tab = 1; }
#endif
#ifdef MKS_THETA_RT
    /* the theta-method by the engine rate, as ULTRA's systems (src/ultra/bn_ultra_engine.c; the product builds with
     * BN_ULTRA_THETA=0.52) */
#ifndef BN_ULTRA_THETA_FS
#define BN_ULTRA_THETA_FS 150000.0
#endif
#ifndef BN_ULTRA_THETA
#define BN_ULTRA_THETA 0.55
#endif
    if (r == 0 && fs < BN_ULTRA_THETA_FS) { mks_mna_set_theta(s, 1, BN_ULTRA_THETA, 2e-10, BN_ULTRA_THETA); }
#endif
    /* ULTRA's product Newton settings (src/ultra/bn_ultra_engine.c) */
    s->tol = 3e-3;
    s->ls_gate = 100.0;
    s->pred_stop = 1;
    s->predict = 1;
    s->pred_max = 1e9;
    s->line_search = 3;
    s->port_norm = 1;
    s->refresh = 0.05;
    return r;
}

void BN_GF(set_knobs)(void* mem, const BnPrepigKnobs* k)
{
    const MksMnaKnobs kn = to_kn(k);
    mks_mna_set_knobs((MksMna*)mem, &kn);
}

void BN_GF(seed)(void* mem, const double* uv, const double* uh)
{
    MksMna* s = (MksMna*)mem;
    for (int i = 0; i < MKS_MNA_N + MKS_MNA_NF; ++i) { s->V[i] = bn_pg_vmap[i] < 0 ? 0.0 : uv[bn_pg_vmap[i]]; }
    for (int k = 0; k < MKS_MNA_NC; ++k) {
        s->h[k] = bn_pg_cmap[k] >= 0 ? uh[bn_pg_cmap[k]]
                                     : s->gc[k] * (s->V[mks_mna_cap[k].a] - s->V[mks_mna_cap[k].b]);   /* zero current */
    }
    /* the start as after an init: no extrapolation from a past the system did not have, no factorisation kept */
    memcpy(s->Vp1, s->V, sizeof s->Vp1);
    memcpy(s->Vp2, s->V, sizeof s->Vp2);
    s->w_ok = 0;
    s->prev0 = 0.0;
}

void BN_GF(restore)(const void* mem, double* uv, double* uvp1, double* uvp2, double* uh)
{
    const MksMna* s = (const MksMna*)mem;
    for (int i = 0; i < MKS_MNA_N; ++i) {   /* the send system's unknowns (the generator asserts they are ULTRA's unknowns) */
        const int j = bn_pg_vmap[i];
        uv[j] = s->V[i]; uvp1[j] = s->V[i]; uvp2[j] = s->V[i];
    }
    for (int k = 0; k < MKS_MNA_NC; ++k) { if (bn_pg_cmap[k] >= 0) { uh[bn_pg_cmap[k]] = s->h[k]; } }
}

double BN_GF(process)(void* mem, double x, double vbias, const double* uv)
{
    const double u[MKS_N_SOURCES] = { x, uv[BN_PG_U_XPI], vbias, uv[BN_PG_U_T4] };
    return mks_mna_process((MksMna*)mem, u);
}

int BN_GF(fallbacks)(const void* mem) { return ((const MksMna*)mem)->fallbacks; }
int BN_GF(unconverged)(const void* mem) { return ((const MksMna*)mem)->unconverged; }
