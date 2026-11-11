/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* send_nodal_api.h - the effects card's SEND half behind four calls, compiled in its own translation unit
 * (send_nodal_api.c) so its headers never meet the whole-amp engine's. level 0 = "0", 1 = "I". Input: w2 against
 * VREF, volts; output: the send node c against VREF, volts. */
#ifndef SEND_NODAL_API_H
#define SEND_NODAL_API_H
void*  cs_send_new(double fs, int level);    /* allocates (never in an audio thread) */
int    cs_send_reset(void* s);               /* the DC point; 0 = ok */
double cs_send_tick(void* s, double w2);     /* one sample at the engine rate given to cs_send_new */
long   cs_send_nonconv(void* s);             /* samples whose Newton hit LS_NEWTON_MAX */
#endif
