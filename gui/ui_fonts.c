/* ui_fonts.c — implementation of ui_fonts.h. FreeType loads each TTF; cairo-ft wraps the
 * FT_Face in a cairo_font_face_t. The FT_Face must outlive the cairo face (cairo references
 * it, it does not copy it): it is tied to it with a destructor in user_data. */
#include "ui_fonts.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include <cairo/cairo-ft.h>
#include <stdio.h>

cairo_font_face_t *bn_face[BNF_COUNT] = {0};

/* The FreeType library lives for the WHOLE process and each FT_Face is released when CAIRO drops its last
 * reference (destructor in user_data, the pattern the cairo-ft documentation requires). Calling FT_Done_Face right
 * after cairo_font_face_destroy is NOT safe: cairo keeps faces and scaled fonts in its cache beyond the destroy, so
 * closing and reopening the editor could leave it holding a freed FT_Face. */
static FT_Library s_ft;
static int        s_ft_ok  = 0;
static int        s_loaded = 0;
static int        s_refs   = 0;   /* number of live UIs: the faces are released only when it reaches 0 */
static const cairo_user_data_key_t s_key;

static const char *const FILES[BNF_COUNT] = {
  "Doto-Round-Bold.ttf",
  "Oswald.ttf",
};

static void done_face(void *p) { FT_Done_Face((FT_Face)p); }

int bn_fonts_load(const char *dir) {
  if (s_loaded) { s_refs++; return 1; }
  if (!dir) return 0;
  if (!s_ft_ok) { if (FT_Init_FreeType(&s_ft)) return 0; s_ft_ok = 1; }
  s_loaded = 1; s_refs = 1;
  for (int i = 0; i < BNF_COUNT; i++) {
    char path[2048];
    FT_Face ff;
    snprintf(path, sizeof path, "%s/%s", dir, FILES[i]);
    bn_face[i] = NULL;
    if (FT_New_Face(s_ft, path, 0, &ff)) continue;
    cairo_font_face_t *f = cairo_ft_font_face_create_for_ft_face(ff, 0);
    if (cairo_font_face_status(f) != CAIRO_STATUS_SUCCESS
        || cairo_font_face_set_user_data(f, &s_key, ff, done_face) != CAIRO_STATUS_SUCCESS) {
      cairo_font_face_destroy(f); FT_Done_Face(ff); continue;
    }
    bn_face[i] = f;
  }
  return 1;
}

void bn_fonts_free(void) {
  if (!s_loaded) return;
  if (--s_refs > 0) return;
  for (int i = 0; i < BNF_COUNT; i++)
    if (bn_face[i]) { cairo_font_face_destroy(bn_face[i]); bn_face[i] = NULL; }   /* the FT_Face goes when cairo releases it */
  s_loaded = 0;
}
