/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_card.h - the reissue effects card in ULTRA (the loop ON): the state hand-over between the product's system
 * (variant v) and the card's (BN_ULTRA_FX(v, level)), bn_card_map.c. Both engines bn_ultra_size() bytes. */
#ifndef BN_CARD_H
#define BN_CARD_H
void bn_card_seed_on(void* card, void* prod, int v, int level);    /* product -> card (the card's own nodes kept) */
void bn_card_seed_off(void* prod, void* card, int v, int level);   /* card -> product (every product slot written) */
#endif
