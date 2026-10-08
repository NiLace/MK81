/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* ui_fonts.h — loading of the bundled fonts (FreeType → cairo-ft) for the MK 81 panel.
 * The cairo faces are exposed in bn_face[]; if one failed to load (NULL), ui_photo.h falls back
 * to Cairo's "toy API" (system font). Role → file mapping in ui_fonts.c. */
#ifndef BN_UI_FONTS_H
#define BN_UI_FONTS_H

#include <cairo/cairo.h>

/* The photoreal panel (ui_photo.h) carries all its lettering in the renders; the only text drawn live is the LED's
 * (Doto), a pot's value while it turns and the help popup (Oswald, the typeface of the printed names). */
typedef enum {
  BNF_LED = 0,      /* Doto Round Bold (OFL) — the LED's variant display and its drop-down */
  BNF_LABEL,        /* Oswald (OFL) — the pots' name typeface: the live value and the popup */
  BNF_COUNT
} BnFontId;

extern cairo_font_face_t *bn_face[BNF_COUNT];

/* Loads the TTFs from <dir> and takes a reference. Returns 1 if a reference was acquired
 * (the caller MUST pair it with exactly one bn_fonts_free), 0 only if FreeType could not
 * initialise. Per-font failures leave that bn_face[] NULL (fallback).
 * Reference-counted across instances. */
int  bn_fonts_load(const char *dir);
void bn_fonts_free(void);

#endif /* BN_UI_FONTS_H */
