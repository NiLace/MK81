/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* ui_photo.h — MK 81 : the photoreal panel (front and back), a PURE drawing module: no window, no host.
 * ui.c drives it from Pugl; ui_preview.c renders it to a PNG without a host. ONE source for what is drawn and for
 * where a click lands (the click zones are the Blender scene's own projections, ui_photo_layout.h); the ports' ranges
 * live in ui.c, and the values shown here (mk_knob_value) must follow them.
 *
 * Assets (gui/photo/, packed by gui/scene/pack_assets.py from Blender renders at 2x the logical size):
 *   plate_<state>.png  the whole frame per state — front: on/byp/off x cable in HIGH/LOW; back: on/off
 *   knob_<k>_{lit,dark}.png, toggle_<t>.png, rocker_power.png, hq_*.png, back_level.png, toggle_era_c.png
 *                      vertical strips, blend mask baked into alpha
 *   label_<k>.png      what lies under each knob's printed name (the value is drawn over it)
 * Plates are resampled to the window LAZILY, and only those one click away from the current state are kept so
 * (a full-res plate is 8 MB decoded). Strip frames are small and drawn straight from the full-res strip, scaled on
 * the fly.
 *
 * The decoded images are ONE copy per process, shared by every open editor, counted by its users under a mutex; an editor that opens while another decodes WAITS for it. The strips are decoded at open
 * on up to MK_MAX_THREADS threads (one image per job, the largest first: each job is the serial path's own code on
 * its own surface, so the result does not depend on the threads); a plate's full-res original is decoded the first
 * time it is needed and KEPT while an editor is open, so a new window size only resamples. The set is freed with
 * its last user (a closed editor holds nothing: ~150 MB decoded), and once more by a destructor at unload.
 * Negative arms, for the gates only: MK81_UI_NO_SHARED_CACHE=1 decodes per editor; MK81_UI_DROP_ORIGINALS=1 lets a
 * plate's original go once resampled (a new size decodes it again); MK81_UI_KEEP_CACHE=1 keeps
 * the set after the last editor and at unload (a leak at dlclose).
 *
 * Behaviour:
 *   - every knob cap, front AND back, glows with the badge: power ON and not bypassed; dark otherwise
 *   - the power rocker animates between its states; the cable sits in the HIGH or LOW jack (`low_input`)
 *   - the LED is DARK with the amp off and lights while hovered (or while its list is open)
 *   - the LED is a drop-down of the five variants (writes `head` + `us_spec`)
 *   - a click on the grille cloth (front) or on the vent (back) turns the head round; anywhere else does nothing
 *   - the HQ button is the LETTERS «HQ» under the badge (port `hq`): released they are smoked (an unlit tube), pressed
 *     they sink into the cloth and light exactly as the badge does (power ON, not bypassed)
 *   - the back has no FX MIX knob (its port stays, undrawn); the card's LEVEL is a toggle, LINE up / PEDAL down
 *   - the Era bat sits at its CENTRE where the switch does nothing (the 50 W head, the reissue)
 */
#ifndef MK_UI_PHOTO_H
#define MK_UI_PHOTO_H

#include <cairo/cairo.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ui_fonts.h"
#include "ui_photo_layout.h"
#include "ui_texts.h"   /* MK_LO_ / MK_HI_ / MK_DEF_<symbol>: the ports' ranges, generated from the TTL */

#define MK_LOGICAL_W (MK_ASSET_W / 2)
#define MK_LOGICAL_H (MK_ASSET_H / 2)

enum { MK_FRONT = 0, MK_BACK = 1 };
enum { MK_PL_FRONT_ON_HIGH = 0, MK_PL_FRONT_ON_LOW, MK_PL_FRONT_BYP_HIGH, MK_PL_FRONT_BYP_LOW,
       MK_PL_FRONT_OFF_HIGH, MK_PL_FRONT_OFF_LOW, MK_PL_BACK_ON, MK_PL_BACK_OFF, MK_N_PLATES };
static const char *const MK_PLATE_FILE[MK_N_PLATES] = {
  "plate_front_on_high.png", "plate_front_on_low.png", "plate_front_byp_high.png", "plate_front_byp_low.png",
  "plate_front_off_high.png", "plate_front_off_low.png", "plate_back_on.png", "plate_back_off.png" };

/* the five variants of the LED drop-down, in list order: 0..3 = 2*head + us_spec for the 100 W / 50 W heads (head 1 =
 * 50 W, us_spec 1 = 6550; mk81.ttl), 4 = the reissue (head 2, EL34 only) */
#define MK_N_VARIANT 5
static const char *const MK_VARIANT_NAME[MK_N_VARIANT] = { "100W EL34", "100W 6550", "50W EL34", "50W 6550", "100W EL34 RI" };
static inline int mk_variant_head(int v)   { return v == 4 ? 2 : (v == 2 || v == 3) ? 1 : 0; }   /* the head port's value */
static inline int mk_variant_us(int v)     { return (v == 1 || v == 3) ? 1 : 0; }
static inline int mk_variant_of(int head, int us) { return head == 2 ? 4 : (head ? 2 : 0) + (us ? 1 : 0); }
/* the Era toggle (MK_TOGGLE[3]) does nothing on the 50 W head, whose supply has one reservoir in both eras, nor on the
 * reissue: a click on it is ignored and its bat is drawn at its CENTRE (MK_TOGGLE_ERA_C). The port itself is unchanged,
 * so saved sessions load as before. */
#define MK_TOGGLE_ERA 3
#define MK_TOGGLE_FX_LEVEL 5   /* the card's LEVEL: bat UP = LINE = port 0, DOWN = PEDAL = port 1 (inverted, ui.c) */

/* what a pixel is: the click zones' codes (also MkState.show_val's knob codes) */
enum { MK_HIT_NONE = -1, MK_HIT_FRONT_KNOB = 0, MK_HIT_BACK_KNOB = 100, MK_HIT_TOGGLE = 200,
       MK_HIT_ROCKER = 300, MK_HIT_JACK_HIGH, MK_HIT_JACK_LOW, MK_HIT_LED, MK_HIT_FLIP, MK_HIT_HQ, MK_HIT_MENU = 400 };

typedef struct {
  double front[MK_N_FRONT_KNOB];   /* 0..1, frame 0 = fully anticlockwise = the port's minimum */
  double back[MK_N_BACK_KNOB];
  int    toggle[MK_N_TOGGLE];      /* 1 = bat UP */
  int    power, bypass, low, variant, side;
  int    hq;                       /* the HQ letters pressed (port `hq` 1 = 4x) */
  double rocker;                   /* 0 = OFF .. 1 = ON; animated towards `power` */
  int    led_hover, menu_open, menu_hover;   /* menu_hover: row under the pointer, -1 none */
  int    show_val;                 /* the knob (an MK_HIT_* knob code) whose VALUE replaces its name, -1 none */
  int    tip;                      /* the help card is up (after the pointer rests on a control) */
  double tip_x, tip_y;             /* where the pointer rested, asset px */
  char   tip_title[48], tip_line[160], tip_value[48];
} MkState;
static inline int mk_toggle_inert(const MkState *st, int t) { return t == MK_TOGGLE_ERA && mk_variant_head(st->variant) != 0; }   /* 50 W and the reissue */

#ifndef MK81_UI_NO_SHARED_CACHE
#define MK81_UI_NO_SHARED_CACHE 0
#endif
#ifndef MK81_UI_DROP_ORIGINALS
#define MK81_UI_DROP_ORIGINALS 0
#endif
#ifndef MK81_UI_KEEP_CACHE
#define MK81_UI_KEEP_CACHE 0
#endif

/* the decoded images: the strips and labels (decoded at open), the plates' full-res originals (on first need) */
typedef struct {
  cairo_surface_t *fk_lit[MK_N_FRONT_KNOB], *fk_dark[MK_N_FRONT_KNOB];
  cairo_surface_t *bk_lit[MK_N_BACK_KNOB], *bk_dark[MK_N_BACK_KNOB];
  cairo_surface_t *tog[MK_N_TOGGLE], *rocker;
  cairo_surface_t *lab_f[MK_N_FRONT_KNOB], *lab_b[MK_N_BACK_KNOB];
  cairo_surface_t *hq_lit, *hq_dark, *back_level, *tog_era_c;   /* MK_N_EXTRA_IMAGES of them */
  cairo_surface_t *plate[MK_N_PLATES];
} MkImages;   /* ONLY cairo_surface_t pointers: mk_images_free walks it as an array */
_Static_assert(sizeof(MkImages) == sizeof(cairo_surface_t *) * (3 * MK_N_FRONT_KNOB + 3 * MK_N_BACK_KNOB + MK_N_TOGGLE + 1 + 4
               + MK_N_PLATES), "MkImages must hold cairo_surface_t pointers only: mk_images_free walks it as an array");
#define MK_N_EXTRA_IMAGES 4

typedef struct {
  char             dir[2048];
  cairo_surface_t *scaled[MK_N_PLATES];      /* plates resampled to the window; NULL = not yet */
  int              sw, sh;                    /* the window size they were made for */
  double           ps;                        /* the scale they were resampled at: the window's, capped at 1 */
  /* the strips this editor draws from: the shared set's (`shared`) or its own (`own`) */
  cairo_surface_t *fk_lit[MK_N_FRONT_KNOB], *fk_dark[MK_N_FRONT_KNOB];
  cairo_surface_t *bk_lit[MK_N_BACK_KNOB], *bk_dark[MK_N_BACK_KNOB];
  cairo_surface_t *tog[MK_N_TOGGLE], *rocker;
  cairo_surface_t *lab_f[MK_N_FRONT_KNOB], *lab_b[MK_N_BACK_KNOB];   /* what lies under each knob's printed name */
  cairo_surface_t *hq_lit, *hq_dark, *back_level, *tog_era_c;
  cairo_surface_t *led;                       /* the lit LED, cached per variant and window size (its glow is costly) */
  int              led_variant, led_w, led_h;
  int              shared, inited;
  MkImages         own;
} MkAssets;

/* the window -> asset transform: letterboxed, aspect kept */
typedef struct { double s, ox, oy; } MkFit;
static inline MkFit mk_fit(int w, int h) {
  MkFit f; double sx = (double)w / MK_ASSET_W, sy = (double)h / MK_ASSET_H;
  f.s = sx < sy ? sx : sy;
  f.ox = floor((w - MK_ASSET_W * f.s) / 2.0); f.oy = floor((h - MK_ASSET_H * f.s) / 2.0);
  return f;
}

/* PNG decodes since the process started: the gates read it to tell a decode from a cache hit */
static atomic_uint mk_decodes;

static cairo_surface_t *mk_load_png(const char *dir, const char *file) {
  char p[2304]; snprintf(p, sizeof p, "%s/%s", dir, file);
  atomic_fetch_add(&mk_decodes, 1u);
  cairo_surface_t *s = cairo_image_surface_create_from_png(p);
  if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
    fprintf(stderr, "mk81 ui: missing asset %s\n", p);
    cairo_surface_destroy(s); return NULL;
  }
  return s;
}

/* --- a few threads over a list of independent jobs --- */
#define MK_MAX_THREADS 6
/* one job slot per image mk_images_decode queues: the knob strips lit and dark, the rocker, the toggles, the labels and
 * the four extra images */
#define MK_LOAD_N (2 * (MK_N_FRONT_KNOB + MK_N_BACK_KNOB) + 1 + MK_N_TOGGLE + MK_N_FRONT_KNOB + MK_N_BACK_KNOB + 4)
typedef struct { const char *dir; const char *file[MK_LOAD_N]; cairo_surface_t **dst[MK_LOAD_N]; int n; atomic_int next; } MkLoadJobs;
static void *mk_load_worker(void *p) {
  MkLoadJobs *j = (MkLoadJobs *)p;
  for (int i; (i = atomic_fetch_add(&j->next, 1)) < j->n;) *j->dst[i] = mk_load_png(j->dir, j->file[i]);
  return NULL;
}
static void mk_add_job(MkLoadJobs *j, cairo_surface_t **dst, const char *file) {
  if (j->n >= MK_LOAD_N) return;                        /* MK_LOAD_N counts what mk_images_decode queues: never reached */
  j->dst[j->n] = dst; j->file[j->n++] = file;
}

/* every strip and label into `im`, in parallel; the knob strips (most of the bytes) first */
static void mk_images_decode(MkImages *im, const char *dir) {
  MkLoadJobs j; memset(&j, 0, sizeof j); j.dir = dir; atomic_init(&j.next, 0);
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) {
    mk_add_job(&j, &im->fk_lit[k], MK_FRONT_KNOB_LIT[k].file); mk_add_job(&j, &im->fk_dark[k], MK_FRONT_KNOB_DARK[k].file);
  }
  for (int k = 0; k < MK_N_BACK_KNOB; k++) {
    mk_add_job(&j, &im->bk_lit[k], MK_BACK_KNOB_LIT[k].file); mk_add_job(&j, &im->bk_dark[k], MK_BACK_KNOB_DARK[k].file);
  }
  mk_add_job(&j, &im->rocker, MK_ROCKER.file);
  for (int t = 0; t < MK_N_TOGGLE; t++) mk_add_job(&j, &im->tog[t], MK_TOGGLE[t].file);
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) mk_add_job(&j, &im->lab_f[k], MK_FRONT_LABEL[k].file);
  for (int k = 0; k < MK_N_BACK_KNOB; k++) mk_add_job(&j, &im->lab_b[k], MK_BACK_LABEL[k].file);
  mk_add_job(&j, &im->hq_lit, MK_HQ_LIT.file); mk_add_job(&j, &im->hq_dark, MK_HQ_DARK.file);
  mk_add_job(&j, &im->back_level, MK_BACK_LEVEL.file); mk_add_job(&j, &im->tog_era_c, MK_TOGGLE_ERA_C.file);
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  int nt = (int)(cpus < 1 ? 1 : cpus > MK_MAX_THREADS ? MK_MAX_THREADS : cpus), started = 0;
  pthread_t th[MK_MAX_THREADS];
  while (started < nt - 1 && pthread_create(&th[started], NULL, mk_load_worker, &j) == 0) started++;
  mk_load_worker(&j);                                   /* the caller's thread works too; a thread not made leaves its share */
  for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
}

static void mk_images_free(MkImages *im) {
  cairo_surface_t **all = (cairo_surface_t **)im;
  for (size_t i = 0; i < sizeof *im / sizeof *all; i++) if (all[i]) cairo_surface_destroy(all[i]);
  memset(im, 0, sizeof *im);
}

/* the process-wide set: one bundle's images, counted by the editors that use it */
static pthread_mutex_t mk_cache_mx = PTHREAD_MUTEX_INITIALIZER;
static MkImages        mk_cache;
static char            mk_cache_dir[2048];
static int             mk_cache_users = 0, mk_cache_loaded = 0;

__attribute__((destructor)) static void mk_cache_free_at_unload(void) {
  if (!MK81_UI_KEEP_CACHE) { mk_images_free(&mk_cache); mk_cache_loaded = 0; }
}

/* this editor's images: the shared set (decoded here if nobody holds it), or its own copy when the set holds another
 * bundle's images or the cache is switched off */
static void mk_assets_init(MkAssets *A, const char *photo_dir) {
  memset(A, 0, sizeof *A);
  snprintf(A->dir, sizeof A->dir, "%s", photo_dir);
  const MkImages *im = &A->own;
  if (!MK81_UI_NO_SHARED_CACHE) {
    pthread_mutex_lock(&mk_cache_mx);
    if (!mk_cache_loaded || (mk_cache_users == 0 && strcmp(mk_cache_dir, A->dir))) {
      mk_images_free(&mk_cache);
      mk_images_decode(&mk_cache, A->dir);
      snprintf(mk_cache_dir, sizeof mk_cache_dir, "%s", A->dir);
      mk_cache_loaded = 1;
    }
    if (!strcmp(mk_cache_dir, A->dir)) { mk_cache_users++; A->shared = 1; im = &mk_cache; }
    pthread_mutex_unlock(&mk_cache_mx);
  }
  if (!A->shared) mk_images_decode(&A->own, A->dir);
  memcpy(A->fk_lit, im->fk_lit, sizeof A->fk_lit); memcpy(A->fk_dark, im->fk_dark, sizeof A->fk_dark);
  memcpy(A->bk_lit, im->bk_lit, sizeof A->bk_lit); memcpy(A->bk_dark, im->bk_dark, sizeof A->bk_dark);
  memcpy(A->tog, im->tog, sizeof A->tog); A->rocker = im->rocker;
  memcpy(A->lab_f, im->lab_f, sizeof A->lab_f); memcpy(A->lab_b, im->lab_b, sizeof A->lab_b);
  A->hq_lit = im->hq_lit; A->hq_dark = im->hq_dark; A->back_level = im->back_level; A->tog_era_c = im->tog_era_c;
  A->inited = 1;
}

static void mk_drop_scaled(MkAssets *A) {
  for (int i = 0; i < MK_N_PLATES; i++)
    if (A->scaled[i]) { cairo_surface_destroy(A->scaled[i]); A->scaled[i] = NULL; }
  if (A->led) { cairo_surface_destroy(A->led); A->led = NULL; }
}

static void mk_assets_free(MkAssets *A) {
  mk_drop_scaled(A);
  if (A->shared) {
    pthread_mutex_lock(&mk_cache_mx);
    if (--mk_cache_users == 0 && !MK81_UI_KEEP_CACHE) { mk_images_free(&mk_cache); mk_cache_loaded = 0; }
    pthread_mutex_unlock(&mk_cache_mx);
  } else if (A->inited) {
    mk_images_free(&A->own);
  }
  memset(A, 0, sizeof *A);
}

/* a plate resampled from its full-res original into `dst` (CAIRO_FILTER_GOOD; NEAREST at scale 1) */
static void mk_resample_plate(cairo_t *c, cairo_surface_t *full, MkFit f) {
  cairo_scale(c, f.s, f.s);
  cairo_set_source_surface(c, full, 0, 0);
  cairo_pattern_set_filter(cairo_get_source(c), f.s == 1.0 ? CAIRO_FILTER_NEAREST : CAIRO_FILTER_GOOD);
  cairo_pattern_set_extend(cairo_get_source(c), CAIRO_EXTEND_PAD);
  cairo_paint(c);
}
static inline int mk_lit(const MkState *st) { return st->power && !st->bypass; }

static int mk_plate_id(const MkState *st) {
  if (st->side == MK_BACK) return st->power ? MK_PL_BACK_ON : MK_PL_BACK_OFF;
  int base = !st->power ? MK_PL_FRONT_OFF_HIGH : st->bypass ? MK_PL_FRONT_BYP_HIGH : MK_PL_FRONT_ON_HIGH;
  return base + (st->low ? 1 : 0);
}

/* the plate at window size: its full-res original (decoded once, kept while an editor is open), resampled ONCE. The
 * originals are MK_ASSET_W x MK_ASSET_H pixels: a window larger than that (ui:scaleFactor above 2) would only upsample
 * them into bigger surfaces with no more detail, so the resample is capped at the original's own size (A->ps <= 1) and
 * mk_paint_plate scales the rest on screen. Up to scale 1 nothing changes. */
#ifndef MK_PLATE_S_MAX
#define MK_PLATE_S_MAX 1.0
#endif
static cairo_surface_t *mk_plate(MkAssets *A, int id, int win_w, int win_h) {
  if (A->sw != win_w || A->sh != win_h) { mk_drop_scaled(A); A->sw = win_w; A->sh = win_h; }
  if (A->scaled[id]) return A->scaled[id];
  MkFit f = mk_fit(win_w, win_h);
  if (f.s > MK_PLATE_S_MAX) f.s = MK_PLATE_S_MAX;
  A->ps = f.s;
  int w = (int)ceil(MK_ASSET_W * f.s), h = (int)ceil(MK_ASSET_H * f.s);
  if (w < 1) w = 1;
  if (h < 1) h = 1;
  cairo_surface_t *s = cairo_image_surface_create(CAIRO_FORMAT_RGB24, w, h);
  cairo_t *c = cairo_create(s);
  /* the shared originals are read under the cache's mutex: two editors resampling at once would use the same cairo
   * surface from two threads */
  MkImages *im = A->shared ? &mk_cache : &A->own;
  if (A->shared) pthread_mutex_lock(&mk_cache_mx);
  if (!im->plate[id]) im->plate[id] = mk_load_png(A->dir, MK_PLATE_FILE[id]);
  cairo_surface_t *full = im->plate[id];
  if (full) mk_resample_plate(c, full, f);
  else { cairo_set_source_rgb(c, 0.09, 0.09, 0.1); cairo_paint(c); }   /* a missing plate: a flat placeholder, never a crash */
  if (MK81_UI_DROP_ORIGINALS && full) { cairo_surface_destroy(full); im->plate[id] = NULL; }
  if (A->shared) pthread_mutex_unlock(&mk_cache_mx);
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
 * work. Called from the UI's idle so a power toggle or a flip never waits for a load (tens of ms per plate) */
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
  mk_led_text(cr, MK_VARIANT_NAME[(unsigned)st->variant % MK_N_VARIANT], (b->x0 + b->x1) / 2, cy, h * 11.5 / 18.0, 1.0);
  double ax = b->x1 - 6.0 * mm, aw = 2.6 * mm;                 /* the ▼: the display is a drop-down */
  cairo_move_to(cr, ax - aw, cy - aw * 0.6); cairo_line_to(cr, ax + aw, cy - aw * 0.6); cairo_line_to(cr, ax, cy + aw * 0.7);
  cairo_close_path(cr);
  cairo_set_source_rgba(cr, MK_LED_R, MK_LED_G + 0.08, MK_LED_B + 0.06, 1.0); cairo_fill(cr);
}

/* the list opens under the LED's bezel, as wide as it, one row per variant */
/* the rows shrink if MK_N_VARIANT of them at the bezel's height would run past the frame's bottom */
static inline double mk_menu_row_h(void) {
  double rh = (MK_LED_BEZEL.y1 - MK_LED_BEZEL.y0) * 1.05, room = (MK_ASSET_H - 8.0 - (MK_LED_BEZEL.y1 + 4.0)) / (double)MK_N_VARIANT;
  return rh < room ? rh : room;
}
static inline double mk_menu_y0(void)    { return MK_LED_BEZEL.y1 + 4.0; }

static void mk_draw_menu(cairo_t *cr, const MkState *st) {
  double x0 = MK_LED_BEZEL.x0, x1 = MK_LED_BEZEL.x1, rh = mk_menu_row_h(), y0 = mk_menu_y0();
  double h = rh * MK_N_VARIANT;
  cairo_rectangle(cr, x0 - 3, y0 - 3, (x1 - x0) + 6, h + 6);
  cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.55); cairo_fill(cr);                  /* a soft drop shadow */
  cairo_rectangle(cr, x0, y0, x1 - x0, h);
  cairo_set_source_rgb(cr, 0.045, 0.043, 0.042); cairo_fill_preserve(cr);
  cairo_set_source_rgb(cr, 0.22, 0.22, 0.23); cairo_set_line_width(cr, 2.0); cairo_stroke(cr);
  for (int r = 0; r < MK_N_VARIANT; r++) {
    double ry = y0 + r * rh;
    if (r == st->variant) { cairo_rectangle(cr, x0 + 2, ry + 1, x1 - x0 - 4, rh - 2);
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
    if (k == 6) snprintf(out, n, "%+.1f dB", MK_LO_trim + (MK_HI_trim - MK_LO_trim) * v);   /* Output: trim, in dB */
    else        snprintf(out, n, "%.1f", 1.0 + 10.0 * v);                 /* the dial's own 1..11 */
    return;
  }
  switch (k) {
    case 1:                  /* Bias: mA per valve, 25 / 40 / 55 at 0 / 0.5 / 1 for EVERY variant (the bias law's
                              * axis, harness/nn_datos.py BIAS_AXIS via tools/gen_ultra_bias.py) */
      (void)variant;
      snprintf(out, n, "%.0f mA", 25.0 + 30.0 * v); break;
    case 3: {                                                              /* Return Level: 0..2 linear */
      double g = MK_LO_fx_return_lvl + (MK_HI_fx_return_lvl - MK_LO_fx_return_lvl) * v;
      if (g < 1e-4) snprintf(out, n, "-inf dB"); else snprintf(out, n, "%+.1f dB", 20.0 * log10(g));
      break; }
    default: snprintf(out, n, "%.0f %%", 100.0 * v); break;                /* Analog Noise, Cabinet Feel */
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
  double cx = (p->lx0 + p->lx1) / 2, cy = fr ? (p->ly0 + p->ly1) / 2 : p->ly0 + size * 0.36;   /* back: on the first line */
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
  /* the transform on entry: the identity in ui.c, an offset when a host draws the panel inside its own canvas. The LED
   * cache is pasted in THESE coordinates, not the device's (with the identity it would land off the LED there) */
  cairo_matrix_t entry;
  cairo_get_matrix(cr, &entry);
  cairo_set_source_rgb(cr, 0x16 / 255.0, 0x16 / 255.0, 0x18 / 255.0);   /* the host panel colour round the frame */
  cairo_paint(cr);
  cairo_surface_t *pl = mk_plate(A, mk_plate_id(st), win_w, win_h);
  if (A->ps >= f.s) {                                       /* resampled at the window's scale: drawn 1:1 */
    cairo_set_source_surface(cr, pl, f.ox, f.oy);
    cairo_paint(cr);
  } else {                                                  /* capped (a window above the originals' size): scaled here */
    cairo_save(cr);
    cairo_translate(cr, f.ox, f.oy);
    cairo_scale(cr, f.s / A->ps, f.s / A->ps);
    cairo_set_source_surface(cr, pl, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_GOOD);
    cairo_paint(cr);
    cairo_restore(cr);
  }
  cairo_save(cr);
  cairo_translate(cr, f.ox, f.oy);
  cairo_scale(cr, f.s, f.s);
  int lit = mk_lit(st);
  if (st->side == MK_FRONT) {
    /* the HQ letters: released / pressed, lit with the badge (power ON and not bypassed) */
    mk_draw_frame(cr, lit ? A->hq_lit : A->hq_dark, lit ? &MK_HQ_LIT : &MK_HQ_DARK, st->hq ? 1 : 0);
    for (int k = 0; k < MK_N_FRONT_KNOB; k++) {
      const MkStrip *g = lit ? &MK_FRONT_KNOB_LIT[k] : &MK_FRONT_KNOB_DARK[k];
      mk_draw_frame(cr, lit ? A->fk_lit[k] : A->fk_dark[k], g, mk_frame(st->front[k], g->n));
    }
    mk_draw_frame(cr, A->rocker, &MK_ROCKER, mk_frame(st->rocker, MK_ROCKER.n));
    if (st->power || st->led_hover || st->menu_open) {
      /* the LED from its cache: stroking the glyphs' glow is most of the front's drawing time per frame, and it only
       * changes with the variant or the window size */
      /* the cache sits on a whole-pixel origin and is drawn with the window's EXACT transform shifted to it, so it
       * lands on the very pixels a direct draw would (a cache drawn at the box's own origin would come out half a px off) */
      /* one pixel of margin round the LED box (+ 3 on the size) for the glow's antialiasing */
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
      cairo_set_matrix(cr, &entry);
      cairo_set_source_surface(cr, A->led, bx, by);
      cairo_paint(cr);
      cairo_restore(cr);
    }
    if (st->menu_open) mk_draw_menu(cr, st);
    if (st->show_val >= 0) mk_draw_value(cr, A, st);
  } else {
    mk_draw_frame(cr, A->back_level, &MK_BACK_LEVEL, st->power ? 0 : 1);   /* the LEVEL corner (no FX MIX knob) */
    for (int k = 0; k < MK_N_BACK_KNOB; k++) {
      const MkStrip *g = lit ? &MK_BACK_KNOB_LIT[k] : &MK_BACK_KNOB_DARK[k];
      mk_draw_frame(cr, lit ? A->bk_lit[k] : A->bk_dark[k], g, mk_frame(st->back[k], g->n));
    }
    for (int t = 0; t < MK_N_TOGGLE; t++) {
      if (mk_toggle_inert(st, t)) mk_draw_frame(cr, A->tog_era_c, &MK_TOGGLE_ERA_C, 0);   /* the bat at its centre */
      else mk_draw_frame(cr, A->tog[t], &MK_TOGGLE[t], st->toggle[t] ? 1 : 0);
    }
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
#define MK_HQ_HIT 6.0   /* asset px round the HQ letters' box (a script word has gaps a click would fall through) */

static inline int mk_hit(const MkState *st, double x, double y) {
  if (st->side == MK_FRONT) {
    if (st->menu_open) {
      double rh = mk_menu_row_h(), y0 = mk_menu_y0();
      if (x >= MK_LED_BEZEL.x0 && x <= MK_LED_BEZEL.x1 && y >= y0 && y < y0 + MK_N_VARIANT * rh)
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
    if (x >= MK_HQ_BOX.x0 - MK_HQ_HIT && x <= MK_HQ_BOX.x1 + MK_HQ_HIT && y >= MK_HQ_BOX.y0 - MK_HQ_HIT && y <= MK_HQ_BOX.y1 + MK_HQ_HIT)
      return MK_HIT_HQ;                                     /* the HQ letters (before the cloth they sit on) */
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
