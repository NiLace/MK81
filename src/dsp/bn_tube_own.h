/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_tube_own.h — the project's OWN triode (12AX7, Dempwolf & Zölzer DAFx-11
 * form, parameters fitted by us to the RCA datasheet).
 *
 * THIS is the shipping model: no third-party material. The derivation and
 * the provenance of every constant are in dsp/prototype/tube_own.py (the
 * docstring's facts block); this file is its C port, verified against the
 * prototype by tools/bn_test_tube_own.c.
 *
 * Ports as in bn_tube.h: v[0]=vgk, v[1]=vak → i[0]=ig, i[1]=ia (flowing in).
 */
#ifndef BN_TUBE_OWN_H
#define BN_TUBE_OWN_H

/* for BnDeviceFn / BnDeviceCurrFn (the device selectors). bn_dk.h does not
 * include this file, so there is no cycle. */
#include "bn_dk.h"

void bn_tube_own(const double* v, double* i, double* J, void* user);
void bn_tube_own_currents(const double* v, double* i);

/* Four triodes over 8 ports (2x2 block-diagonal Jacobian). */
void bn_tube_own_x4(const double* v, double* i, double* J, void* user);
void bn_tube_own_currents_x4(const double* v, double* i, void* user);

/* TABULATED variant (two 1-D LUTs: Ik(x) and Ig(vgk); closed form out of
 * range). Call bn_tube_own_tables_init() first (or _tab_x4 does it itself). */
void bn_tube_own_tables_init(void);
void bn_tube_own_tab(const double* v, double* i, double* J, void* user);
void bn_tube_own_tab_currents(const double* v, double* i);
void bn_tube_own_tab_x4(const double* v, double* i, double* J, void* user);
void bn_tube_own_tab_currents_x4(const double* v, double* i, void* user);
/* Variants without the trims check + the selectors that pick one or the other
 * ONCE. Use the selectors, not the bare pointers. */
void bn_tube_own_tab_x4_notrim(const double* v, double* i, double* J, void* user);
void bn_tube_own_tab_currents_x4_notrim(const double* v, double* i, void* user);
BnDeviceFn     bn_tube_own_pick_tab_x4(void);
BnDeviceCurrFn bn_tube_own_pick_tab_currents_x4(void);

/* Is any trim other than 1.0 set by BN_TRIMS? Queried by the places that
 * pick a device that does NOT consume trims (the oracle's reference model), so they warn
 * instead of silently measuring that "the trims do nothing". */
int bn_own_trims_puestos(void);

#endif /* BN_TUBE_OWN_H */
