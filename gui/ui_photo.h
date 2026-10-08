/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* ui_photo.h — MK 81 : the photoreal panel (front and back), a PURE drawing module: no window, no host.
 * ui.c drives it from Pugl; ui_preview.c renders it to a PNG without a host. ONE source for what is drawn and for
 * where a click lands (the click zones are the Blender scene's own projections, ui_photo_layout.h).
 *
 * Assets (gui/photo/, packed by gui/scene/pack_assets.py from Blender renders at 2x the logical size):
 *   plate_<state>.png  the whole frame per state — front: on/byp/off x cable in HIGH/LOW; back: on/off
 *   knob_<k>_{lit,dark}.png, toggle_<t>.png, rocker_power.png  vertical strips, blend mask baked into alpha
 * Plates are loaded LAZILY and kept only resampled to the window, and only those one click away from the current
 * state (a full-res plate is 8 MB decoded). Strip frames are small and
 * drawn straight from the full-res strip, scaled on the fly.
 *
 * Behaviour:
 *   - every knob cap, front AND back, glows with the badge: power ON and not bypassed; dark otherwise
 *   - the power rocker animates between its states; the cable sits in the HIGH or LOW jack (`low_input`)
 *   - the LED is DARK with the amp off and lights while hovered (or while its list is open)
 *   - the LED is a drop-down of the four variants (writes `head` + `us_spec`)
 *   - a click on the grille cloth (front) or on the vent (back) turns the head round; anywhere else does nothing
 */
#ifndef MK_UI_PHOTO_H
#define MK_UI_PHOTO_H

#include <cairo/cairo.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ui_fonts.h"
#include "ui_photo_layout.h"

#define MK_LOGICAL_W (MK_ASSET_W / 2)
#define MK_LOGICAL_H (MK_ASSET_H / 2)

enum { MK_FRONT = 0, MK_BACK = 1 };
enum { MK_PL_FRONT_ON_HIGH = 0, MK_PL_FRONT_ON_LOW, MK_PL_FRONT_BYP_HIGH, MK_PL_FRONT_BYP_LOW,
       MK_PL_FRONT_OFF_HIGH, MK_PL_FRONT_OFF_LOW, MK_PL_BACK_ON, MK_PL_BACK_OFF, MK_N_PLATES };
static const char *const MK_PLATE_FILE[MK_N_PLATES] = {
  "plate_front_on_high.png", "plate_front_on_low.png", "plate_front_byp_high.png", "plate_front_byp_low.png",
  "plate_front_off_high.png", "plate_front_off_low.png", "plate_back_on.png", "plate_back_off.png" };

/* the four variants of the LED drop-down, in list order = 2*head + us_spec (head 1 = 50 W, us_spec 1 = 6550; mk81.ttl) */
static const char *const MK_VARIANT_NAME[4] = { "100W EL34", "100W 6550", "50W EL34", "50W 6550" };
static inline int mk_variant_head(int v)   { return (v == 2 || v == 3) ? 1 : 0; }
static inline int mk_variant_us(int v)     { return (v == 1 || v == 3) ? 1 : 0; }
static inline int mk_variant_of(int head, int us) { return (head ? 2 : 0) + (us ? 1 : 0); }

/* what a pixel is: the click zones' codes (also MkState.show_val's knob codes) */
enum { MK_HIT_NONE = -1, MK_HIT_FRONT_KNOB = 0, MK_HIT_BACK_KNOB = 100, MK_HIT_TOGGLE = 200,
       MK_HIT_ROCKER = 300, MK_HIT_JACK_HIGH, MK_HIT_JACK_LOW, MK_HIT_LED, MK_HIT_FLIP, MK_HIT_MENU = 400 };

typedef struct {
  double front[MK_N_FRONT_KNOB];   /* 0..1, frame 0 = fully anticlockwise = the port's minimum */
  double back[MK_N_BACK_KNOB];
  int    toggle[MK_N_TOGGLE];      /* 1 = bat UP */
  int    power, bypass, low, variant, side;
  double rocker;                   /* 0 = OFF .. 1 = ON; animated towards `power` */
  int    led_hover, menu_open, menu_hover;   /* menu_hover: row under the pointer, -1 none */
  int    show_val;                 /* the knob (an MK_HIT_* knob code) whose VALUE replaces its name, -1 none */
  int    tip;                      /* the help card is up (after the pointer rests on a control) */
  double tip_x, tip_y;             /* where the pointer rested, asset px */
  char   tip_title[48], tip_line[160], tip_value[48];
} MkState;

typedef struct {
  char             dir[2048];
  cairo_surface_t *scaled[MK_N_PLATES];      /* plates resampled to the window; NULL = not yet */
  int              sw, sh;                    /* the window size they were made for */
  cairo_surface_t *fk_lit[MK_N_FRONT_KNOB], *fk_dark[MK_N_FRONT_KNOB];
  cairo_surface_t *bk_lit[MK_N_BACK_KNOB], *bk_dark[MK_N_BACK_KNOB];
  cairo_surface_t *tog[MK_N_TOGGLE], *rocker;
  cairo_surface_t *lab_f[MK_N_FRONT_KNOB], *lab_b[MK_N_BACK_KNOB];   /* what lies under each knob's printed name */
  cairo_surface_t *led;                       /* the lit LED, cached per variant and window size (its glow is costly) */
  int              led_variant, led_w, led_h;
} MkAssets;

/* the window -> asset transform: letterboxed, aspect kept */
typedef struct { double s, ox, oy; } MkFit;
static inline MkFit mk_fit(int w, int h) {
  MkFit f; double sx = (double)w / MK_ASSET_W, sy = (double)h / MK_ASSET_H;
  f.s = sx < sy ? sx : sy;
  f.ox = floor((w - MK_ASSET_W * f.s) / 2.0); f.oy = floor((h - MK_ASSET_H * f.s) / 2.0);
  return f;
}

static cairo_surface_t *mk_load_png(const char *dir, const char *file) {
  char p[2304]; snprintf(p, sizeof p, "%s/%s", dir, file);
  cairo_surface_t *s = cairo_image_surface_create_from_png(p);
  if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
    fprintf(stderr, "mk81 ui: missing asset %s\n", p);
    cairo_surface_destroy(s); return NULL;
  }
  return s;
}

static void mk_assets_init(MkAssets *A, const char *photo_dir) {
  memset(A, 0, sizeof *A);
  snprintf(A->dir, sizeof A->dir, "%s", photo_dir);
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) {
    A->fk_lit[k] = mk_load_png(A->dir, MK_FRONT_KNOB_LIT[k].file);
    A->fk_dark[k] = mk_load_png(A->dir, MK_FRONT_KNOB_DARK[k].file);
  }
  for (int k = 0; k < MK_N_BACK_KNOB; k++) {
    A->bk_lit[k] = mk_load_png(A->dir, MK_BACK_KNOB_LIT[k].file);
    A->bk_dark[k] = mk_load_png(A->dir, MK_BACK_KNOB_DARK[k].file);
  }
  for (int t = 0; t < MK_N_TOGGLE; t++) A->tog[t] = mk_load_png(A->dir, MK_TOGGLE[t].file);
  A->rocker = mk_load_png(A->dir, MK_ROCKER.file);
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) A->lab_f[k] = mk_load_png(A->dir, MK_FRONT_LABEL[k].file);
  for (int k = 0; k < MK_N_BACK_KNOB; k++)  A->lab_b[k] = mk_load_png(A->dir, MK_BACK_LABEL[k].file);
}

static void mk_drop_scaled(MkAssets *A) {
  for (int i = 0; i < MK_N_PLATES; i++)
    if (A->scaled[i]) { cairo_surface_destroy(A->scaled[i]); A->scaled[i] = NULL; }
  if (A->led) { cairo_surface_destroy(A->led); A->led = NULL; }
}

static void mk_assets_free(MkAssets *A) {
  mk_drop_scaled(A);
  cairo_surface_t **all[] = { A->fk_lit, A->fk_dark };
  for (int j = 0; j < 2; j++) for (int k = 0; k < MK_N_FRONT_KNOB; k++) if (all[j][k]) cairo_surface_destroy(all[j][k]);
  for (int k = 0; k < MK_N_BACK_KNOB; k++) {
    if (A->bk_lit[k]) cairo_surface_destroy(A->bk_lit[k]);
    if (A->bk_dark[k]) cairo_surface_destroy(A->bk_dark[k]);
  }
  for (int t = 0; t < MK_N_TOGGLE; t++) if (A->tog[t]) cairo_surface_destroy(A->tog[t]);
  if (A->rocker) cairo_surface_destroy(A->rocker);
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) if (A->lab_f[k]) cairo_surface_destroy(A->lab_f[k]);
  for (int k = 0; k < MK_N_BACK_KNOB; k++)  if (A->lab_b[k]) cairo_surface_destroy(A->lab_b[k]);
  memset(A, 0, sizeof *A);
}

static inline int mk_lit(const MkState *st) { return st->power && !st->bypass; }

static int mk_plate_id(const MkState *st) {
  if (st->side == MK_BACK) return st->power ? MK_PL_BACK_ON : MK_PL_BACK_OFF;
  int base = !st->power ? MK_PL_FRONT_OFF_HIGH : st->bypass ? MK_PL_FRONT_BYP_HIGH : MK_PL_FRONT_ON_HIGH;
  return base + (st->low ? 1 : 0);
}

/* the plate at window size: load, resample ONCE (CAIRO_FILTER_GOOD), free the full-res copy */
static cairo_surface_t *mk_plate(MkAssets *A, int id, int win_w, int win_h) {
  if (A->sw != win_w || A->sh != win_h) { mk_drop_scaled(A); A->sw = win_w; A->sh = win_h; }
  if (A->scaled[id]) return A->scaled[id];
  cairo_surface_t *full = mk_load_png(A->dir, MK_PLATE_FILE[id]);
  MkFit f = mk_fit(win_w, win_h);
  int w = (int)ceil(MK_ASSET_W * f.s), h = (int)ceil(MK_ASSET_H * f.s);
  if (w < 1) w = 1;
  if (h < 1) h = 1;
  cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
  cairo_t *c = cairo_create(s);
  if (full) {
    cairo_scale(c, f.s, f.s);
    cairo_set_source_surface(c, full, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(c), f.s == 1.0 ? CAIRO_FILTER_NEAREST : CAIRO_FILTER_GOOD);
    cairo_pattern_set_extend(cairo_get_source(c), CAIRO_EXTEND_PAD);
    cairo_paint(c);
    cairo_surface_destroy(full);
  } else {                                    /* a missing plate: a flat placeholder, never a crash */
    cairo_set_source_rgb(c, 0.09, 0.09, 0.1); cairo_paint(c);
  }
  cairo_destroy(c);
  A->scaled[id] = s;
  return s;
}

/* the plates ONE click away from `st` (power toggle, the flip, the other jack) plus the current one; the rest are
 * dropped. Keeps 3-4 plates resident instead of 8 (8 MB each at ui:scaleFactor 2) */
static int mk_reachable(const MkState *st, int ids[4]) {
  int n = 0; MkState t = *st;
  ids[n++] = mk_plate_id(st);
  t.power = !st->power; ids[n++] = mk_plate_id(&t); t = *st;
  t.side = !st->side;   ids[n++] = mk_plate_id(&t); t = *st;
  if (st->side == MK_FRONT) { t.low = !st->low; ids[n++] = mk_plate_id(&t); }
  return n;
}

/* resample ONE missing reachable plate for this window size (and drop the unreachable ones); returns 1 if it did
 * work. Called from the UI's idle so a power toggle or a flip never waits for a load (~35 ms per plate on the development machine) */
static inline int mk_prewarm(MkAssets *A, const MkState *st, int win_w, int win_h) {
  if (A->sw != win_w || A->sh != win_h) return 0;          /* only once the current size has painted */
  int ids[4], n = mk_reachable(st, ids), keep[MK_N_PLATES] = { 0 };
  for (int i = 0; i < n; i++) keep[ids[i]] = 1;
  for (int i = 0; i < MK_N_PLATES; i++)
    if (!keep[i] && A->scaled[i]) { cairo_surface_destroy(A->scaled[i]); A->scaled[i] = NULL; }
  for (int i = 0; i < n; i++)
    if (!A->scaled[ids[i]]) { mk_plate(A, ids[i], win_w, win_h); return 1; }
  return 0;
}

static inline int mk_frame(double v, int n) {
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  return (int)lround(v * (n - 1));
}

/* one frame of a strip, in ASSET coordinates (the caller has scaled the context) */
static void mk_draw_frame(cairo_t *cr, cairo_surface_t *strip, const MkStrip *g, int i) {
  if (!strip) return;
  if (i < 0) i = 0;
  if (i >= g->n) i = g->n - 1;
  cairo_save(cr);
  cairo_rectangle(cr, g->x, g->y, g->w, g->h);
  cairo_clip(cr);
  cairo_set_source_surface(cr, strip, g->x, g->y - (double)i * g->h);
  cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_GOOD);
  cairo_paint(cr);
  cairo_restore(cr);
}

/* ---------------------------------------------------------------- the LED and its drop-down */
#define MK_LED_R 1.0
#define MK_LED_G 0.30
#define MK_LED_B 0.22

static void mk_led_font(cairo_t *cr, double size) {
  if (bn_face[BNF_LED]) cairo_set_font_face(cr, bn_face[BNF_LED]);
  else cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, size);
}

/* glowing dot-matrix text centred at (cx, cy): a soft wide stroke (the glow), then a thin stroke + fill so the
 * sub-pixel dots never read dark */
static void mk_led_text(cairo_t *cr, const char *txt, double cx, double cy, double size, double alpha) {
  mk_led_font(cr, size);
  cairo_text_extents_t te; cairo_text_extents(cr, txt, &te);
  cairo_move_to(cr, cx - te.width / 2 - te.x_bearing, cy - te.height / 2 - te.y_bearing);
  cairo_text_path(cr, txt);
  cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G * 0.7, MK_LED_B * 0.6, 0.10 * alpha);
  cairo_set_line_width(cr, size * 0.22); cairo_stroke_preserve(cr);
  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G, MK_LED_B, 0.55 * alpha);
  cairo_set_line_width(cr, size * 0.035); cairo_stroke_preserve(cr);
  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G + 0.08, MK_LED_B + 0.06, alpha);
  cairo_fill(cr);
}

/* the LED window: 100 x 18 mm in the scene, text 11.5 mm high, the ▼ 6 mm from the right end (amp_mk81.json) */
static void mk_draw_led(cairo_t *cr, const MkState *st) {
  const MkBox *b = &MK_LED;
  double w = b->x1 - b->x0, h = b->y1 - b->y0, mm = w / 100.0;
  double cy = (b->y0 + b->y1) / 2;
  mk_led_text(cr, MK_VARIANT_NAME[st->variant & 3], (b->x0 + b->x1) / 2, cy, h * 11.5 / 18.0, 1.0);
  double ax = b->x1 - 6.0 * mm, aw = 2.6 * mm;                 /* the ▼: the display is a drop-down */
  cairo_move_to(cr, ax - aw, cy - aw * 0.6); cairo_line_to(cr, ax + aw, cy - aw * 0.6); cairo_line_to(cr, ax, cy + aw * 0.7);
  cairo_close_path(cr);
  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G + 0.08, MK_LED_B + 0.06, 1.0); cairo_fill(cr);
}

/* the list opens under the LED's bezel, as wide as it, one row per variant */
/* the rows shrink if four of them would run past the frame's bottom (at the bezel's height they would run 6 px off) */
static inline double mk_menu_row_h(void) {
  double rh = (MK_LED_BEZEL.y1 - MK_LED_BEZEL.y0) * 1.05, room = (MK_ASSET_H - 8.0 - (MK_LED_BEZEL.y1 + 4.0)) / 4.0;
  return rh < room ? rh : room;
}
static inline double mk_menu_y0(void)    { return MK_LED_BEZEL.y1 + 4.0; }

static void mk_draw_menu(cairo_t *cr, const MkState *st) {
  double x0 = MK_LED_BEZEL.x0, x1 = MK_LED_BEZEL.x1, rh = mk_menu_row_h(), y0 = mk_menu_y0();
  double h = rh * 4;
  cairo_rectangle(cr, x0 - 3, y0 - 3, (x1 - x0) + 6, h + 6);
  cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.55); cairo_fill(cr);                  /* a soft drop shadow */
  cairo_rectangle(cr, x0, y0, x1 - x0, h);
  cairo_set_source_rgb(cr, 0.045, 0.043, 0.042); cairo_fill_preserve(cr);
  cairo_set_source_rgb(cr, 0.22, 0.22, 0.23); cairo_set_line_width(cr, 2.0); cairo_stroke(cr);
  for (int r = 0; r < 4; r++) {
    double ry = y0 + r * rh;
    if (r == (st->variant & 3)) { cairo_rectangle(cr, x0 + 2, ry + 1, x1 - x0 - 4, rh - 2);
                                  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G, MK_LED_B, 0.22); cairo_fill(cr); }
    if (r == st->menu_hover)    { cairo_rectangle(cr, x0 + 2, ry + 1, x1 - x0 - 4, rh - 2);
                                  cairo_set_source_rgba(cr, 1, 1, 1, 0.10); cairo_fill(cr); }
    mk_led_text(cr, MK_VARIANT_NAME[r], (x0 + x1) / 2, ry + rh / 2, (MK_LED.y1 - MK_LED.y0) * 11.5 / 18.0 * 0.85, 1.0);
  }
}

/* ---------------------------------------------------------------- a knob's value, and the help card */
/* the value in the knob's own unit (the name shows the value while it turns).
 * side/k as in MK_FRONT_KNOB_* / MK_BACK_KNOB_*; `variant` for the bias, whose current depends on the valves */
static void mk_knob_value(int side, int k, double v, int variant, char *out, int n) {
  if (v < 0) v = 0;
  if (v > 1) v = 1;
  if (side == MK_FRONT) {
    if (k == 6) snprintf(out, n, "%+.1f dB", -24.0 + 36.0 * v);           /* Output: trim, -24..+12 dB */
    else        snprintf(out, n, "%.1f", 1.0 + 10.0 * v);                 /* the dial's own 1..11 */
    return;
  }
  switch (k) {
    case 1:                  /* Bias: mA per valve, 25 / 40 / 55 for EVERY variant (the axis the nets were trained on,
                              * harness/nn_datos.py BIAS_AXIS); the 6550's own 42..92 axis of July was never shipped */
      (void)variant;
      snprintf(out, n, "%.0f mA", 25.0 + 30.0 * v); break;
    case 3: {                                                              /* Return Level: 0..2 linear */
      double g = 2.0 * v;
      if (g < 1e-4) snprintf(out, n, "-inf dB"); else snprintf(out, n, "%+.1f dB", 20.0 * log10(g));
      break; }
    default: snprintf(out, n, "%.0f %%", 100.0 * v); break;                /* Analog Noise, Cabinet Feel, FX Mix */
  }
}

static void mk_label_font(cairo_t *cr, double size) {
  if (bn_face[BNF_LABEL]) cairo_set_font_face(cr, bn_face[BNF_LABEL]);
  else cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(cr, size);
}

/* the patch over the printed name, then the value printed like it: white with a dark keyline */
static void mk_draw_value(cairo_t *cr, MkAssets *A, const MkState *st) {
  int fr = st->side == MK_FRONT;
  int base = fr ? MK_HIT_FRONT_KNOB : MK_HIT_BACK_KNOB, n = fr ? MK_N_FRONT_KNOB : MK_N_BACK_KNOB;
  int k = st->show_val - base;
  if (k < 0 || k >= n) return;
  const MkPatch *p = fr ? &MK_FRONT_LABEL[k] : &MK_BACK_LABEL[k];
  cairo_surface_t *ps = fr ? A->lab_f[k] : A->lab_b[k];
  if (ps) {
    cairo_save(cr); cairo_rectangle(cr, p->x, p->y, p->w, p->h); cairo_clip(cr);
    cairo_set_source_surface(cr, ps, p->x, p->y); cairo_paint(cr); cairo_restore(cr);
  }
  char buf[32]; mk_knob_value(st->side, k, fr ? st->front[k] : st->back[k], st->variant, buf, sizeof buf);
  double size = fr ? MK_FRONT_LABEL_PX : MK_BACK_LABEL_PX;
  mk_label_font(cr, size);
  cairo_text_extents_t te; cairo_text_extents(cr, buf, &te);
  double cx = (p->lx0 + p->lx1) / 2, cy = fr ? (p->ly0 + p->ly1) / 2 : p->ly0 + size * 0.36;   /* back: the first line */
  cairo_move_to(cr, cx - te.width / 2 - te.x_bearing, cy - te.height / 2 - te.y_bearing);
  cairo_text_path(cr, buf);
  cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);
  cairo_set_source_rgba(cr, 0.06, 0.02, 0.015, 0.9); cairo_set_line_width(cr, size * 0.09); cairo_stroke_preserve(cr);
  cairo_set_source_rgb(cr, 1, 1, 1); cairo_fill(cr);
}

/* the help card: title, the port's one line (wrapped), the current value; kept inside the frame */
static void mk_draw_tip(cairo_t *cr, const MkState *st) {
  const double pad = 18, tsz = 30, lsz = 25, wmax = 560;
  char lines[4][96]; int nl = 0;
  mk_label_font(cr, lsz);
  {                                                         /* word-wrap the line to wmax */
    const char *s = st->tip_line; char cur[96] = "";
    while (*s && nl < 4) {
      const char *e = s; while (*e && *e != ' ') e++;
      char trial[200]; snprintf(trial, sizeof trial, "%s%s%.*s", cur, cur[0] ? " " : "", (int)(e - s), s);
      cairo_text_extents_t te; cairo_text_extents(cr, trial, &te);
      if (te.x_advance > wmax && cur[0]) { snprintf(lines[nl++], 96, "%s", cur); snprintf(cur, sizeof cur, "%.*s", (int)(e - s), s); }
      else snprintf(cur, sizeof cur, "%.95s", trial);    /* a line never exceeds 95 chars (the .ttl gate caps 120 per text) */
      s = *e ? e + 1 : e;
    }
    if (cur[0] && nl < 4) snprintf(lines[nl++], 96, "%s", cur);
  }
  double w = 0;
  cairo_text_extents_t te;
  for (int i = 0; i < nl; i++) { cairo_text_extents(cr, lines[i], &te); if (te.x_advance > w) w = te.x_advance; }
  mk_label_font(cr, tsz); cairo_text_extents(cr, st->tip_title, &te); if (te.x_advance > w) w = te.x_advance;
  if (st->tip_value[0]) { mk_label_font(cr, lsz); cairo_text_extents(cr, st->tip_value, &te); if (te.x_advance > w) w = te.x_advance; }
  double W = w + 2 * pad, H = pad * 2 + tsz * 1.25 + nl * lsz * 1.3 + (st->tip_value[0] ? lsz * 1.5 : 0);
  double x = st->tip_x + 22, y = st->tip_y + 28;
  if (x + W > MK_ASSET_W - 8) x = st->tip_x - 22 - W;
  if (y + H > MK_ASSET_H - 8) y = st->tip_y - 28 - H;
  if (x < 8) x = 8;
  if (y < 8) y = 8;
  cairo_new_path(cr);
  double r = 10;
  cairo_arc(cr, x + W - r, y + r, r, -M_PI / 2, 0); cairo_arc(cr, x + W - r, y + H - r, r, 0, M_PI / 2);
  cairo_arc(cr, x + r, y + H - r, r, M_PI / 2, M_PI); cairo_arc(cr, x + r, y + r, r, M_PI, 3 * M_PI / 2); cairo_close_path(cr);
  cairo_set_source_rgba(cr, 0.05, 0.045, 0.04, 0.94); cairo_fill_preserve(cr);
  cairo_set_source_rgba(cr, 0.85, 0.42, 0.25, 0.6); cairo_set_line_width(cr, 2.0); cairo_stroke(cr);
  double ty = y + pad + tsz * 0.8;
  mk_label_font(cr, tsz); cairo_set_source_rgb(cr, 1, 1, 1);
  cairo_move_to(cr, x + pad, ty); cairo_show_text(cr, st->tip_title);
  mk_label_font(cr, lsz); cairo_set_source_rgb(cr, 0.80, 0.78, 0.74);
  for (int i = 0; i < nl; i++) { ty += lsz * 1.3 + (i == 0 ? tsz * 0.35 : 0); cairo_move_to(cr, x + pad, ty); cairo_show_text(cr, lines[i]); }
  if (st->tip_value[0]) {
    ty += lsz * 1.5; cairo_set_source_rgb(cr, 1.0, 0.62, 0.45);
    cairo_move_to(cr, x + pad, ty); cairo_show_text(cr, st->tip_value);
  }
}

/* ---------------------------------------------------------------- the whole panel */
static void mk_draw(cairo_t *cr, MkAssets *A, const MkState *st, int win_w, int win_h) {
  MkFit f = mk_fit(win_w, win_h);
  cairo_set_source_rgb(cr, 0x16 / 255.0, 0x16 / 255.0, 0x18 / 255.0);   /* the host panel colour round the frame */
  cairo_paint(cr);
  cairo_surface_t *pl = mk_plate(A, mk_plate_id(st), win_w, win_h);
  cairo_set_source_surface(cr, pl, f.ox, f.oy);
  cairo_paint(cr);
  cairo_save(cr);
  cairo_translate(cr, f.ox, f.oy);
  cairo_scale(cr, f.s, f.s);
  int lit = mk_lit(st);
  if (st->side == MK_FRONT) {
    for (int k = 0; k < MK_N_FRONT_KNOB; k++) {
      const MkStrip *g = lit ? &MK_FRONT_KNOB_LIT[k] : &MK_FRONT_KNOB_DARK[k];
      mk_draw_frame(cr, lit ? A->fk_lit[k] : A->fk_dark[k], g, mk_frame(st->front[k], g->n));
    }
    mk_draw_frame(cr, A->rocker, &MK_ROCKER, mk_frame(st->rocker, MK_ROCKER.n));
    if (st->power || st->led_hover || st->menu_open) {
      /* the LED from its cache: stroking the glyphs' glow every frame costs 1.5 ms of the front's 1.64 ms per frame
       * (measured on the development machine); it only changes with the variant or the window size */
      /* the cache sits on a whole-pixel origin and is drawn with the window's EXACT transform shifted to it, so it
       * lands on the very pixels a direct draw would (a cache drawn at the box's own origin would come out half a px off) */
      double bx = floor(f.ox + MK_LED.x0 * f.s) - 1, by = floor(f.oy + MK_LED.y0 * f.s) - 1;
      int lw = (int)ceil((MK_LED.x1 - MK_LED.x0) * f.s) + 3, lh = (int)ceil((MK_LED.y1 - MK_LED.y0) * f.s) + 3;
      if (!A->led || A->led_variant != st->variant || A->led_w != lw || A->led_h != lh) {
        if (A->led) cairo_surface_destroy(A->led);
        A->led = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, lw, lh);
        cairo_t *lc = cairo_create(A->led);
        cairo_translate(lc, f.ox - bx, f.oy - by);
        cairo_scale(lc, f.s, f.s);
        mk_draw_led(lc, st);
        cairo_destroy(lc);
        A->led_variant = st->variant; A->led_w = lw; A->led_h = lh;
      }
      cairo_save(cr);
      cairo_identity_matrix(cr);
      cairo_set_source_surface(cr, A->led, bx, by);
      cairo_paint(cr);
      cairo_restore(cr);
    }
    if (st->menu_open) mk_draw_menu(cr, st);
    if (st->show_val >= 0) mk_draw_value(cr, A, st);
  } else {
    for (int k = 0; k < MK_N_BACK_KNOB; k++) {
      const MkStrip *g = lit ? &MK_BACK_KNOB_LIT[k] : &MK_BACK_KNOB_DARK[k];
      mk_draw_frame(cr, lit ? A->bk_lit[k] : A->bk_dark[k], g, mk_frame(st->back[k], g->n));
    }
    for (int t = 0; t < MK_N_TOGGLE; t++) mk_draw_frame(cr, A->tog[t], &MK_TOGGLE[t], st->toggle[t] ? 1 : 0);
    if (st->show_val >= 0) mk_draw_value(cr, A, st);
  }
  if (st->tip) mk_draw_tip(cr, st);
  cairo_restore(cr);
}

/* ---------------------------------------------------------------- hit-testing, in ASSET pixels */

static inline int mk_in_circle(double x, double y, double cx, double cy, double r) {
  return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;
}
static inline int mk_in_box(double x, double y, const MkBox *b) {
  return x >= b->x0 && x <= b->x1 && y >= b->y0 && y <= b->y1;
}

/* click radii relative to the part: on the back, ERA and BIAS sit 23 mm apart, so a knob's zone is its skirt + 15 %
 * and a toggle's 80 % of its strip's circle (the bat's travel); mk81-ui-photo checks that no two zones overlap */
#define MK_KNOB_HIT 1.15
#define MK_TOGGLE_HIT 0.8

static inline int mk_hit(const MkState *st, double x, double y) {
  if (st->side == MK_FRONT) {
    if (st->menu_open) {
      double rh = mk_menu_row_h(), y0 = mk_menu_y0();
      if (x >= MK_LED_BEZEL.x0 && x <= MK_LED_BEZEL.x1 && y >= y0 && y < y0 + 4 * rh)
        return MK_HIT_MENU + (int)((y - y0) / rh);
    }
    for (int k = 0; k < MK_N_FRONT_KNOB; k++) {
      const MkStrip *g = &MK_FRONT_KNOB_LIT[k];
      if (mk_in_circle(x, y, g->cx, g->cy, g->r * MK_KNOB_HIT)) return MK_HIT_FRONT_KNOB + k;
    }
    if (mk_in_box(x, y, &MK_ROCKER_BOX)) return MK_HIT_ROCKER;
    {                                                       /* the two jacks' zones overlap: the NEARER one wins */
      double dh = hypot(x - MK_JACK_HIGH.x, y - MK_JACK_HIGH.y), dl = hypot(x - MK_JACK_LOW.x, y - MK_JACK_LOW.y);
      if (dh <= MK_JACK_HIGH.r * 1.5 && dh <= dl) return MK_HIT_JACK_HIGH;
      if (dl <= MK_JACK_LOW.r * 1.5) return MK_HIT_JACK_LOW;
    }
    if (mk_in_box(x, y, &MK_LED_BEZEL)) return MK_HIT_LED;
    if (mk_in_box(x, y, &MK_FLIP_FRONT)) return MK_HIT_FLIP;   /* the grille cloth turns the head round */
  } else {
    for (int k = 0; k < MK_N_BACK_KNOB; k++) {
      const MkStrip *g = &MK_BACK_KNOB_LIT[k];
      if (mk_in_circle(x, y, g->cx, g->cy, g->r * MK_KNOB_HIT)) return MK_HIT_BACK_KNOB + k;
    }
    for (int t = 0; t < MK_N_TOGGLE; t++) {
      const MkStrip *g = &MK_TOGGLE[t];
      if (mk_in_circle(x, y, g->cx, g->cy, g->r * MK_TOGGLE_HIT)) return MK_HIT_TOGGLE + t;
    }
    if (mk_in_box(x, y, &MK_FLIP_BACK)) return MK_HIT_FLIP;    /* the vent turns it back */
  }
  return MK_HIT_NONE;
}

#endif /* MK_UI_PHOTO_H */
