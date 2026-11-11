/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* ret_nodal_api.h - the effects card's return half behind four calls, compiled in its own translation unit
 * (ret_nodal_api.c) so its headers never meet the whole-amp engine's. level 0 = "0", 1 = "I". Input: the return
 * jack, volts; output: IC1b's output against VREF, volts (the amp's fx_ret source). */
#ifndef RET_NODAL_API_H
#define RET_NODAL_API_H
void*  cr_ret_new(double fs, int level);              /* allocates (a bench's call; never in an audio thread) */
int    cr_ret_reset(void* r);                          /* back to the DC point, no allocation (an audio thread may call
                                                        * it); 0 = ok */
double cr_ret_tick(void* r, double jack);              /* one sample at the engine rate given to cr_ret_new */
long   cr_ret_nonconv(void* r);                        /* samples that did not converge (always 0 here: explicit) */
#endif
