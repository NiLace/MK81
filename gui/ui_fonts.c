/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* ui_fonts.c — implementation of ui_fonts.h. FreeType loads each TTF; cairo-ft wraps the
 * FT_Face in a cairo_font_face_t. The FT_Face must outlive the cairo face (cairo references
 * it, it does not copy it), and cairo may keep the face in its process-wide caches long after this
 * UI is closed and its module unloaded by the host (mk81_ui.so has no STB_GNU_UNIQUE symbol, so a
 * dlclose really unmaps it).
 *
 * So each face owns a PRIVATE FT_Library holding its one FT_Face, hung on the cairo face as user
 * data whose destructor is FreeType's own FT_Done_FreeType: when cairo finally drops the face,
 * FreeType frees the library and the face with it, and nothing calls back into this module. One
 * FT_Library per process with each FT_Face released by a function in THIS module crashes a host
 * that draws text after the dlclose.
 * The destructor is FT_Done_FreeType (FT_Error(*)(FT_Library)) called as cairo's void(*)(void*): a
 * call through another function type, which C leaves undefined and which is exact on the ABI this
 * plugin targets (x86-64 System V: the pointer argument travels in the same register, the int
 * result is discarded). A wrapper of the right type would be code in THIS module, which is what
 * crashed.
 *
 * MK81_UI_FONTS_FT_CALLBACK=1 builds that crashing way, for a test's negative arm only. */
#include "ui_fonts.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include <cairo/cairo-ft.h>
#include <stdio.h>


cairo_font_face_t *bn_face[BNF_COUNT] = {0};

static int        s_loaded = 0;
static int        s_refs   = 0;   /* number of live UIs: the faces are released only when it reaches 0 */
static const cairo_user_data_key_t s_key;

static const char *const FILES[BNF_COUNT] = {
  "Doto-Round-Bold.ttf",
  "Oswald.ttf",
};

/* one face from one bundled file, owning its own FreeType library (see the top of this file) */
static cairo_font_face_t *load_face(const char *path) {
  FT_Library lib;
  if (FT_Init_FreeType(&lib)) return NULL;
  FT_Face ff;
  if (FT_New_Face(lib, path, 0, &ff)) { FT_Done_FreeType(lib); return NULL; }
  cairo_font_face_t *f = cairo_ft_font_face_create_for_ft_face(ff, 0);
  if (cairo_font_face_status(f) != CAIRO_STATUS_SUCCESS
      || cairo_font_face_set_user_data(f, &s_key, lib, (cairo_destroy_func_t)(void (*)(void))FT_Done_FreeType)
         != CAIRO_STATUS_SUCCESS) {
    cairo_font_face_destroy(f); FT_Done_FreeType(lib); return NULL;
  }
  return f;
}

int bn_fonts_load(const char *dir) {
  if (s_loaded) { s_refs++; return 1; }
  if (!dir) return 0;
  s_loaded = 1; s_refs = 1;
  for (int i = 0; i < BNF_COUNT; i++) {
    char path[2048];
    snprintf(path, sizeof path, "%s/%s", dir, FILES[i]);
    bn_face[i] = load_face(path);
  }
  return 1;
}

void bn_fonts_free(void) {
  if (!s_loaded) return;
  if (--s_refs > 0) return;
  for (int i = 0; i < BNF_COUNT; i++)
    if (bn_face[i]) { cairo_font_face_destroy(bn_face[i]); bn_face[i] = NULL; }   /* its FreeType library goes when cairo releases it */
  s_loaded = 0;
}
