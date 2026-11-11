/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_card_map.c - the loop switch's state hand-over between ULTRA's product system (variant v, the loop OFF) and the
 * reissue card's system (BN_ULTRA_FX(v, level), the loop ON), through the generated maps
 * (src/ultra/genfx<L>_<v>/bn_card_map.h, tools/gen_card_map.py). Exact both ways: product -> card -> product at one
 * sample is bit-identical to never switching (the laboratory's switch probe checks the round trip). The caller follows a seed with bn_ultra_set_knobs on the seeded engine
 * (a rebuild: no stale factorisation). */
#include "bn_card.h"

#include "bn_ultra.h"
#include "genfx0_100uk/bn_card_map.h"
#include "genfx0_100us/bn_card_map.h"
#include "genfx0_50uk/bn_card_map.h"
#include "genfx0_50us/bn_card_map.h"
#include "genfx0_100ri/bn_card_map.h"
#include "genfxI_100uk/bn_card_map.h"
#include "genfxI_100us/bn_card_map.h"
#include "genfxI_50uk/bn_card_map.h"
#include "genfxI_50us/bn_card_map.h"
#include "genfxI_100ri/bn_card_map.h"

/* cnu / cnc: the card system's unknowns and capacitors; pnu / pnc: the product's. on_v[i] / on_c[k]: the product's
 * index for the card's unknown i / capacitor k, -1 = the card's own (kept). off_v[i] / off_c[k]: the card's index for
 * the product's unknown i / capacitor k, -1 = left as it is (off_c is never -1: every product capacitor exists in the
 * card) */
typedef struct { int cnu, cnc, pnu, pnc; const short *on_v, *on_c, *off_v, *off_c; } BnCardMap;
#define BN_CM(T, t) { BN_CM_##T##_CNU, BN_CM_##T##_CNC, BN_CM_##T##_PNU, BN_CM_##T##_PNC, \
                      bn_cm_##t##_on_v, bn_cm_##t##_on_c, bn_cm_##t##_off_v, bn_cm_##t##_off_c }
static const BnCardMap MAPS[2][BN_ULTRA_NV] = {
    { BN_CM(FX0_100UK, fx0_100uk), BN_CM(FX0_100US, fx0_100us), BN_CM(FX0_50UK, fx0_50uk), BN_CM(FX0_50US, fx0_50us), BN_CM(FX0_100RI, fx0_100ri) },
    { BN_CM(FXI_100UK, fxi_100uk), BN_CM(FXI_100US, fxi_100us), BN_CM(FXI_50UK, fxi_50uk), BN_CM(FXI_50US, fxi_50us), BN_CM(FXI_100RI, fxi_100ri) },
};

/* the product variant clamped into the table (as bn_ultra.c clamps its dispatch): a stray v reads a real map */
static int cm_v(int v) { return v < 0 ? 0 : v >= BN_ULTRA_NV ? BN_ULTRA_NV - 1 : v; }

void bn_card_seed_on(void* card, void* prod, int v, int level)
{
    v = cm_v(v);
    const BnCardMap* m = &MAPS[level ? 1 : 0][v];
    BnUltraArrays c, p;
    bn_ultra_arrays(card, BN_ULTRA_FX(v, level ? 1 : 0), &c);
    bn_ultra_arrays(prod, v, &p);
    for (int i = 0; i < m->cnu; ++i) {
        const int j = m->on_v[i];
        /* the start extrapolation's past samples travel too: the next sample starts from the product's history */
        if (j >= 0) { c.V[i] = p.V[j]; c.Vp1[i] = p.Vp1[j]; c.Vp2[i] = p.Vp2[j]; }
    }
    for (int k = 0; k < m->cnc; ++k) {
        const int j = m->on_c[k];
        if (j >= 0) { c.h[k] = p.h[j]; }
    }
}

void bn_card_seed_off(void* prod, void* card, int v, int level)
{
    v = cm_v(v);
    const BnCardMap* m = &MAPS[level ? 1 : 0][v];
    BnUltraArrays c, p;
    bn_ultra_arrays(card, BN_ULTRA_FX(v, level ? 1 : 0), &c);
    bn_ultra_arrays(prod, v, &p);
    for (int i = 0; i < m->pnu; ++i) {
        const int j = m->off_v[i];
        if (j >= 0) { p.V[i] = c.V[j]; p.Vp1[i] = c.Vp1[j]; p.Vp2[i] = c.Vp2[j]; }
    }
    for (int k = 0; k < m->pnc; ++k) { p.h[k] = c.h[m->off_c[k]]; }
}
