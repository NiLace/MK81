/* ui.c — MK 81 : the LV2 UI (Pugl + Cairo), the photoreal panel (front and back).
 * The drawing and the click zones live in ui_photo.h (a pure module, also used by ui_preview.c); here: the Pugl
 * window, the events and the port writes.
 *
 * Design rules: a 30 fps cap in ui_idle (an uncapped UI can freeze the host), the plates resampled ONCE per window
 * size (ui_photo.h), and port_event updating every control so the panel never lies about the DSP (automation,
 * presets, a loaded session).
 *
 * Controls -> ports (lv2/mk81.ttl; the indices are API, see lv2/frozen_symbols.txt):
 *   FRONT  Presence 13 · Bass 7 · Middle 6 · Treble 5 · Master 8 · Pre-Amp `preamp_vol` 4 · Output `trim` 2 (dB)
 *          Power rocker 16 · HIGH/LOW jacks `low_input` 12 · LED drop-down `head` 20 + `us_spec` 11
 *   BACK   Analog Noise `hum` 17 + `hiss` 18 (together) · Bias 10 · Cabinet Feel `cab_load` 14 ·
 *          Return Level `fx_return_lvl` 27 (0-2) · FX Mix `fx_mix` 26 (0-100 %)
 *          Full Bypass `bypass` 15 · Preamp Bypass 21 · Power Bypass 22 · Era 19 (UP = 1976 = 1) · FX Loop 23
 *   The grille cloth (front) and the vent (back) turn the head round; a click anywhere else does nothing.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pugl/pugl.h>
#include <pugl/cairo.h>
#include <cairo/cairo.h>

#include <lv2/atom/atom.h>
#include <lv2/options/options.h>
#include <lv2/ui/ui.h>
#include <lv2/urid/urid.h>

#include "ui_fonts.h"
#include "ui_photo.h"
#include "ui_texts.h"

#define MK_UI_URI "https://nylarea.com/plugins/mk81#ui"

enum { P_TRIM = 2, P_PREAMP_VOL = 4, P_TREBLE = 5, P_MIDDLE = 6, P_BASS = 7, P_MASTER = 8, P_BIAS = 10,
       P_US_SPEC = 11, P_LOW = 12, P_PRESENCE = 13, P_CAB_LOAD = 14, P_BYPASS = 15, P_POWER = 16, P_HUM = 17,
       P_HISS = 18, P_ERA = 19, P_HEAD = 20, P_BYPASS_PREAMP = 21, P_BYPASS_POWER = 22, P_FX_LOOP = 23,
       P_FX_MIX = 26, P_FX_RETURN_LVL = 27 };

/* a knob's port and its range; the knob holds 0..1 (frame 0 = the minimum), the port the real unit */
typedef struct { int port, port2; double lo, hi, def; } MkKnobPort;
static const MkKnobPort FRONT_PORT[MK_N_FRONT_KNOB] = {     /* panel order, left to right (pack_assets.py) */
  { P_PRESENCE, -1, 0, 1, 0.0 }, { P_BASS, -1, 0, 1, 0.5 }, { P_MIDDLE, -1, 0, 1, 0.5 }, { P_TREBLE, -1, 0, 1, 0.5 },
  { P_MASTER, -1, 0, 1, 1.0 }, { P_PREAMP_VOL, -1, 0, 1, 0.7 }, { P_TRIM, -1, -24, 12, 0.0 } };
static const MkKnobPort BACK_PORT[MK_N_BACK_KNOB] = {
  { P_HUM, P_HISS, 0, 1, 0.0 }, { P_BIAS, -1, 0, 1, 0.5 }, { P_CAB_LOAD, -1, 0, 1, 0.0 },
  { P_FX_RETURN_LVL, -1, 0, 2, 1.0 }, { P_FX_MIX, -1, 0, 100, 100.0 } };
static const int TOGGLE_PORT[MK_N_TOGGLE] = { P_BYPASS, P_BYPASS_PREAMP, P_BYPASS_POWER, P_ERA, P_FX_LOOP };
static const int TOGGLE_DEF[MK_N_TOGGLE] = { 0, 0, 0, 1, 0 };

static inline double to_norm(const MkKnobPort *k, double p) { return (p - k->lo) / (k->hi - k->lo); }
static inline double to_port(const MkKnobPort *k, double v) { return k->lo + v * (k->hi - k->lo); }

typedef struct {
  LV2UI_Write_Function write;
  LV2UI_Controller     controller;
  LV2UI_Resize        *resize;
  PuglWorld           *world;
  PuglView            *view;

  MkState  st;
  MkAssets A;
  int      fonts_loaded;

  int      drag;              /* the knob being dragged (an MK_HIT_* knob code), -1 none */
  double   drag_y0, drag_v0, last_press_t;
  int      last_press_hit;
  double   px, py;            /* the pointer, in asset pixels */
  double   hum, hiss;         /* the two ports Analog Noise drives, as last heard from the host */
  int      hover_hit;         /* what the pointer rests on, for the help card */
  double   last_move_t, show_val_t;

  int      needs_redraw;
  int      win_w, win_h;
  double   last_paint, last_tick;
} MkUI;

static void redraw(MkUI *ui) { ui->needs_redraw = 1; }

static void write_port(MkUI *ui, int port, float v) {
  if (port >= 0) ui->write(ui->controller, (uint32_t)port, sizeof(float), 0, &v);
}

static double *knob_val(MkUI *ui, int hit, const MkKnobPort **kp) {
  if (hit >= MK_HIT_FRONT_KNOB && hit < MK_HIT_FRONT_KNOB + MK_N_FRONT_KNOB) {
    *kp = &FRONT_PORT[hit - MK_HIT_FRONT_KNOB]; return &ui->st.front[hit - MK_HIT_FRONT_KNOB];
  }
  if (hit >= MK_HIT_BACK_KNOB && hit < MK_HIT_BACK_KNOB + MK_N_BACK_KNOB) {
    *kp = &BACK_PORT[hit - MK_HIT_BACK_KNOB]; return &ui->st.back[hit - MK_HIT_BACK_KNOB];
  }
  return NULL;
}

static void set_knob(MkUI *ui, int hit, double v) {
  const MkKnobPort *kp; double *val = knob_val(ui, hit, &kp);
  if (!val) return;
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  if (v == *val) return;
  *val = v;
  write_port(ui, kp->port, (float)to_port(kp, v));
  write_port(ui, kp->port2, (float)to_port(kp, v));       /* Analog Noise drives hum AND hiss */
  if (kp->port == P_HUM) ui->hum = ui->hiss = v;
  redraw(ui);
}

/* the help card's text for what the pointer rests on: the port's name and ONE line come from the .ttl (ui_texts.h,
 * the same line a host's generic controls show); only the controls that are not one port carry their own */
static void build_tip(MkUI *ui, int h) {
  MkState *st = &ui->st;
  const char *title = "", *line = "";
  char val[48] = "";
  const MkText *t = NULL;
  if (h >= MK_HIT_FRONT_KNOB && h < MK_HIT_FRONT_KNOB + MK_N_FRONT_KNOB) {
    int k = h - MK_HIT_FRONT_KNOB; t = mk_text(FRONT_PORT[k].port);
    mk_knob_value(MK_FRONT, k, st->front[k], st->variant, val, sizeof val);
  } else if (h >= MK_HIT_BACK_KNOB && h < MK_HIT_BACK_KNOB + MK_N_BACK_KNOB) {
    int k = h - MK_HIT_BACK_KNOB;
    if (k == 0) { title = "Analog Noise"; line = "The hum and the hiss of a real amp (1 = as loud as the original)."; }
    else t = mk_text(BACK_PORT[k].port);
    mk_knob_value(MK_BACK, k, st->back[k], st->variant, val, sizeof val);
  } else if (h >= MK_HIT_TOGGLE && h < MK_HIT_TOGGLE + MK_N_TOGGLE) {
    int k = h - MK_HIT_TOGGLE; t = mk_text(TOGGLE_PORT[k]);
    if (TOGGLE_PORT[k] == P_ERA) snprintf(val, sizeof val, "%s", st->toggle[k] ? "1976" : "1988");
    else snprintf(val, sizeof val, "%s", st->toggle[k] ? "On" : "Off");
  } else switch (h) {
    case MK_HIT_ROCKER:    t = mk_text(P_POWER); snprintf(val, sizeof val, "%s", st->power ? "On" : "Off"); break;
    case MK_HIT_JACK_HIGH:
    case MK_HIT_JACK_LOW:  title = "Input"; line = "Plug into HIGH for full gain, or LOW for less gain and a softer response.";
                           snprintf(val, sizeof val, "Cable in %s", st->low ? "LOW" : "HIGH"); break;
    case MK_HIT_LED:       title = "Model"; line = "Picks the head (100 W or 50 W) and the power valves (EL34 or 6550).";
                           snprintf(val, sizeof val, "%s", MK_VARIANT_NAME[st->variant & 3]); break;
    case MK_HIT_FLIP:      title = st->side == MK_FRONT ? "Back panel" : "Front panel";
                           line = st->side == MK_FRONT ? "Click to turn the amp round and see its back panel."
                                                       : "Click to turn the amp back to its front panel."; break;
    default: return;
  }
  if (t) { title = t->name; line = t->line; }
  snprintf(st->tip_title, sizeof st->tip_title, "%s", title);
  snprintf(st->tip_line, sizeof st->tip_line, "%s", line);
  snprintf(st->tip_value, sizeof st->tip_value, "%s", val);
}

static void hide_tip(MkUI *ui) { if (ui->st.tip) { ui->st.tip = 0; redraw(ui); } }

static void to_asset(MkUI *ui, double wx, double wy, double *ax, double *ay) {
  MkFit f = mk_fit(ui->win_w, ui->win_h);
  *ax = (wx - f.ox) / f.s; *ay = (wy - f.oy) / f.s;
}

static void on_press(MkUI *ui, double x, double y, double wy, int button) {
  if (button != 0) return;                                 /* Pugl numbers buttons from 0: the left one */
  double now = puglGetTime(ui->world);
  hide_tip(ui);
  int h = mk_hit(&ui->st, x, y);
  MkState *st = &ui->st;
  if (st->menu_open) {                                       /* any click closes the list; a row picks a variant */
    st->menu_open = 0; st->menu_hover = -1;
    if (h >= MK_HIT_MENU && h < MK_HIT_MENU + 4) {
      st->variant = h - MK_HIT_MENU;
      write_port(ui, P_HEAD, (float)mk_variant_head(st->variant));
      write_port(ui, P_US_SPEC, (float)mk_variant_us(st->variant));
    }
    redraw(ui); return;
  }
  const MkKnobPort *kp; double *val = knob_val(ui, h, &kp);
  if (val) {
    if (h == ui->last_press_hit && now - ui->last_press_t < 0.35) {   /* double click: back to the port's default */
      set_knob(ui, h, to_norm(kp, kp->def)); ui->last_press_t = 0; return;
    }
    ui->last_press_hit = h; ui->last_press_t = now;
    ui->drag = h; ui->drag_y0 = wy; ui->drag_v0 = *val;
    st->show_val = h; ui->show_val_t = now; redraw(ui);        /* the name shows the value while it turns */
    return;
  }
  ui->last_press_hit = h; ui->last_press_t = now;
  if (h >= MK_HIT_TOGGLE && h < MK_HIT_TOGGLE + MK_N_TOGGLE) {
    int t = h - MK_HIT_TOGGLE; st->toggle[t] = !st->toggle[t];
    write_port(ui, TOGGLE_PORT[t], (float)st->toggle[t]);
    if (TOGGLE_PORT[t] == P_BYPASS) st->bypass = st->toggle[t];
    redraw(ui); return;
  }
  switch (h) {
    case MK_HIT_ROCKER:    st->power = !st->power; write_port(ui, P_POWER, (float)st->power); break;
    case MK_HIT_JACK_HIGH: if (st->low)  { st->low = 0; write_port(ui, P_LOW, 0.0f); } break;
    case MK_HIT_JACK_LOW:  if (!st->low) { st->low = 1; write_port(ui, P_LOW, 1.0f); } break;
    case MK_HIT_LED:       st->menu_open = 1; st->menu_hover = -1; break;
    case MK_HIT_FLIP:      st->side = !st->side; st->led_hover = 0; st->show_val = -1; break;   /* the grille / the vent */
    default: return;
  }
  redraw(ui);
}

static void on_motion(MkUI *ui, double x, double y, double wy, int fine) {
  ui->px = x; ui->py = y;
  MkState *st = &ui->st;
  if (ui->drag >= 0) {
    /* INCREMENTAL, in WINDOW pixels: 150 px of vertical travel = the whole range at any window size; Shift = ten
     * times finer, and pressing it mid-drag carries on from where the knob is (an absolute anchor would make it jump) */
    double dv = (ui->drag_y0 - wy) / 150.0 * (fine ? 0.1 : 1.0);
    ui->drag_y0 = wy;
    ui->drag_v0 += dv; ui->drag_v0 = ui->drag_v0 < 0 ? 0 : ui->drag_v0 > 1 ? 1 : ui->drag_v0;
    set_knob(ui, ui->drag, ui->drag_v0);
    ui->show_val_t = puglGetTime(ui->world);
    return;
  }
  {                                                          /* the help card: rest on a control to see it */
    int h = mk_hit(st, x, y);
    if (st->tip && (h != ui->hover_hit || hypot(x - st->tip_x, y - st->tip_y) > 8.0)) hide_tip(ui);
    ui->hover_hit = h; ui->last_move_t = puglGetTime(ui->world);
    if (!st->tip) { st->tip_x = x; st->tip_y = y; }
  }
  if (st->side != MK_FRONT) return;
  int hov = mk_in_box(x, y, &MK_LED_BEZEL);
  if (hov != st->led_hover) { st->led_hover = hov; if (!st->power) redraw(ui); }
  if (st->menu_open) {
    int h = mk_hit(st, x, y), r = (h >= MK_HIT_MENU && h < MK_HIT_MENU + 4) ? h - MK_HIT_MENU : -1;
    if (r != st->menu_hover) { st->menu_hover = r; redraw(ui); }
  }
}

static void on_scroll(MkUI *ui, double dy, int fine) {
  int h = mk_hit(&ui->st, ui->px, ui->py);
  const MkKnobPort *kp; double *val = knob_val(ui, h, &kp);
  if (val) {
    hide_tip(ui);
    set_knob(ui, h, *val + dy * (fine ? 0.002 : 0.02));
    ui->st.show_val = h; ui->show_val_t = puglGetTime(ui->world); redraw(ui);
  }
}

static PuglStatus on_event(PuglView *view, const PuglEvent *e) {
  MkUI *ui = (MkUI *)puglGetHandle(view);
  if (!ui) return PUGL_SUCCESS;
  double ax, ay;
  switch (e->type) {
    case PUGL_CONFIGURE:
      ui->win_w = (int)e->configure.width; ui->win_h = (int)e->configure.height; redraw(ui); break;
    case PUGL_EXPOSE: {
      cairo_t *cr = (cairo_t *)puglGetContext(view);
      if (cr) mk_draw(cr, &ui->A, &ui->st, ui->win_w, ui->win_h);
      break;
    }
    case PUGL_BUTTON_PRESS:
      to_asset(ui, e->button.x, e->button.y, &ax, &ay); on_press(ui, ax, ay, e->button.y, (int)e->button.button); break;
    case PUGL_BUTTON_RELEASE:
      if (ui->drag >= 0) ui->show_val_t = puglGetTime(ui->world);   /* the value lingers 1 s, then the name returns */
      ui->drag = -1; break;
    case PUGL_MOTION:
      to_asset(ui, e->motion.x, e->motion.y, &ax, &ay);
      on_motion(ui, ax, ay, e->motion.y, (e->motion.state & PUGL_MOD_SHIFT) != 0); break;
    case PUGL_KEY_PRESS:                                    /* Escape closes the LED's list */
      if (e->key.key == PUGL_KEY_ESCAPE && ui->st.menu_open) { ui->st.menu_open = 0; ui->st.menu_hover = -1; redraw(ui); }
      break;
    case PUGL_SCROLL:
      to_asset(ui, e->scroll.x, e->scroll.y, &ui->px, &ui->py);
      on_scroll(ui, e->scroll.dy, (e->scroll.state & PUGL_MOD_SHIFT) != 0); break;
    case PUGL_POINTER_OUT:
      if (ui->st.led_hover) { ui->st.led_hover = 0; redraw(ui); }
      ui->hover_hit = MK_HIT_NONE; hide_tip(ui);
      break;
    default: break;
  }
  return PUGL_SUCCESS;
}

static void defaults(MkUI *ui) {
  MkState *st = &ui->st;
  for (int k = 0; k < MK_N_FRONT_KNOB; k++) st->front[k] = to_norm(&FRONT_PORT[k], FRONT_PORT[k].def);
  for (int k = 0; k < MK_N_BACK_KNOB; k++) st->back[k] = to_norm(&BACK_PORT[k], BACK_PORT[k].def);
  for (int t = 0; t < MK_N_TOGGLE; t++) st->toggle[t] = TOGGLE_DEF[t];
  st->power = 1; st->rocker = 1.0; st->bypass = 0; st->low = 0; st->variant = 0; st->side = MK_FRONT;
  st->menu_hover = -1; st->show_val = -1; st->tip = 0;
  ui->hover_hit = MK_HIT_NONE;
}

static LV2UI_Handle instantiate(const LV2UI_Descriptor *d, const char *uri, const char *bundle,
                                LV2UI_Write_Function wf, LV2UI_Controller ctl, LV2UI_Widget *widget,
                                const LV2_Feature *const *features) {
  (void)d; (void)uri;
  MkUI *ui = (MkUI *)calloc(1, sizeof(MkUI));
  if (!ui) return NULL;
  ui->write = wf; ui->controller = ctl; ui->drag = -1; ui->last_press_hit = MK_HIT_NONE;
  char path[2048];
  const char *b = bundle ? bundle : ".";
  size_t n = strlen(b); int slash = n > 0 && b[n - 1] == '/';
  snprintf(path, sizeof path, "%s%sfonts", b, slash ? "" : "/");
  ui->fonts_loaded = bn_fonts_load(path);
  snprintf(path, sizeof path, "%s%sphoto", b, slash ? "" : "/");
  mk_assets_init(&ui->A, path);
  defaults(ui);

  PuglNativeView parent = 0;
  const LV2_Options_Option *opts = NULL; LV2_URID_Map *map = NULL;
  for (int i = 0; features && features[i]; i++) {
    if (!strcmp(features[i]->URI, LV2_UI__parent)) parent = (PuglNativeView)(uintptr_t)features[i]->data;
    else if (!strcmp(features[i]->URI, LV2_UI__resize)) ui->resize = (LV2UI_Resize *)features[i]->data;
    else if (!strcmp(features[i]->URI, LV2_OPTIONS__options)) opts = (const LV2_Options_Option *)features[i]->data;
    else if (!strcmp(features[i]->URI, LV2_URID__map)) map = (LV2_URID_Map *)features[i]->data;
  }
  /* HiDPI: the host's ui:scaleFactor sizes the window (the photos are 2x the logical size, so up to 2 stays sharp);
   * without it, 1 */
  double sf = 1.0;
  if (opts && map) {
    LV2_URID k = map->map(map->handle, LV2_UI__scaleFactor), tf = map->map(map->handle, LV2_ATOM__Float);
    for (const LV2_Options_Option *o = opts; o->key; o++)
      if (o->key == k && o->type == tf && o->value) sf = *(const float *)o->value;
  }
  if (!(sf >= 0.5 && sf <= 4.0)) sf = 1.0;
  int dw = (int)lround(MK_LOGICAL_W * sf), dh = (int)lround(MK_LOGICAL_H * sf);
  ui->world = puglNewWorld(PUGL_MODULE, 0);
  if (!ui->world) { mk_assets_free(&ui->A); if (ui->fonts_loaded) bn_fonts_free(); free(ui); return NULL; }
  puglSetWorldString(ui->world, PUGL_CLASS_NAME, "MK81");
  ui->view = puglNewView(ui->world);
  if (!ui->view) { puglFreeWorld(ui->world); mk_assets_free(&ui->A); if (ui->fonts_loaded) bn_fonts_free(); free(ui); return NULL; }
  puglSetHandle(ui->view, ui);
  puglSetEventFunc(ui->view, on_event);
  puglSetBackend(ui->view, puglCairoBackend());
  puglSetSizeHint(ui->view, PUGL_DEFAULT_SIZE, (PuglSpan)dw, (PuglSpan)dh);
  puglSetSizeHint(ui->view, PUGL_MIN_SIZE, (PuglSpan)(MK_LOGICAL_W / 2), (PuglSpan)(MK_LOGICAL_H / 2));
  puglSetSizeHint(ui->view, PUGL_FIXED_ASPECT, (PuglSpan)MK_LOGICAL_W, (PuglSpan)MK_LOGICAL_H);
  puglSetViewHint(ui->view, PUGL_RESIZABLE, PUGL_TRUE);
  ui->win_w = dw; ui->win_h = dh;
  if (parent) puglSetParent(ui->view, parent);
  if (puglRealize(ui->view) != PUGL_SUCCESS) {
    puglFreeView(ui->view); puglFreeWorld(ui->world); mk_assets_free(&ui->A);
    if (ui->fonts_loaded) bn_fonts_free();
    free(ui); return NULL;
  }
  puglShow(ui->view, PUGL_SHOW_RAISE);
  puglObscureView(ui->view);
  if (ui->resize) ui->resize->ui_resize(ui->resize->handle, dw, dh);
  *widget = (LV2UI_Widget)puglGetNativeView(ui->view);
  return ui;
}

static void cleanup(LV2UI_Handle h) {
  MkUI *ui = (MkUI *)h;
  if (ui->view) puglFreeView(ui->view);
  if (ui->world) puglFreeWorld(ui->world);
  mk_assets_free(&ui->A);
  if (ui->fonts_loaded) bn_fonts_free();
  free(ui);
}

static void port_event(LV2UI_Handle h, uint32_t port, uint32_t size, uint32_t format, const void *buf) {
  MkUI *ui = (MkUI *)h;
  if (format != 0 || size != sizeof(float)) return;
  float p = *(const float *)buf;
  MkState *st = &ui->st;
  for (int k = 0; k < MK_N_FRONT_KNOB; k++)
    if (FRONT_PORT[k].port == (int)port) {
      double v = to_norm(&FRONT_PORT[k], p); v = v < 0 ? 0 : v > 1 ? 1 : v;
      if (v != st->front[k]) { st->front[k] = v; redraw(ui); }
      return;
    }
  if ((int)port == P_HUM || (int)port == P_HISS) {
    /* Analog Noise drives hum AND hiss; a session or the generic UI can set them apart, so the knob shows the LARGER
     * of the two (turning it then sets both to that) — it never reads 0 over a hissing amp */
    double v = p < 0 ? 0 : p > 1 ? 1 : p;
    if ((int)port == P_HUM) ui->hum = v; else ui->hiss = v;
    double m = ui->hum > ui->hiss ? ui->hum : ui->hiss;
    if (m != st->back[0]) { st->back[0] = m; redraw(ui); }
    return;
  }
  for (int k = 1; k < MK_N_BACK_KNOB; k++)
    if (BACK_PORT[k].port == (int)port) {
      double v = to_norm(&BACK_PORT[k], p); v = v < 0 ? 0 : v > 1 ? 1 : v;
      if (v != st->back[k]) { st->back[k] = v; redraw(ui); }
      return;
    }
  for (int t = 0; t < MK_N_TOGGLE; t++)
    if (TOGGLE_PORT[t] == (int)port) {
      int v = p > 0.5f;
      if (v != st->toggle[t]) { st->toggle[t] = v; redraw(ui); }
      if (port == P_BYPASS && v != st->bypass) { st->bypass = v; redraw(ui); }
      return;
    }
  int v = p > 0.5f;
  switch ((int)port) {
    case P_POWER:   if (v != st->power) { st->power = v; redraw(ui); } break;
    case P_LOW:     if (v != st->low)   { st->low = v;   redraw(ui); } break;
    case P_HEAD:    { int nv = mk_variant_of(v, mk_variant_us(st->variant));
                      if (nv != st->variant) { st->variant = nv; redraw(ui); } break; }
    case P_US_SPEC: { int nv = mk_variant_of(mk_variant_head(st->variant), v);
                      if (nv != st->variant) { st->variant = nv; redraw(ui); } break; }
    default: break;
  }
}

static int ui_idle(LV2UI_Handle h) {
  MkUI *ui = (MkUI *)h;
  double now = puglGetTime(ui->world);
  double dt = ui->last_tick > 0 ? now - ui->last_tick : 0.0; ui->last_tick = now;
  /* the power rocker tilts over ~120 ms between its states */
  double target = ui->st.power ? 1.0 : 0.0;
  if (ui->st.rocker != target) {
    double step = dt / 0.12;
    ui->st.rocker += ui->st.rocker < target ? step : -step;
    if (fabs(ui->st.rocker - target) <= step || step <= 0) ui->st.rocker = target;
    redraw(ui);
  }
  /* the value goes back to the name 1 s after the knob stops */
  if (ui->st.show_val >= 0 && ui->drag < 0 && now - ui->show_val_t > 1.0) { ui->st.show_val = -1; redraw(ui); }
  /* the help card after the pointer rests 0.6 s on a control (not while dragging or with the list open) */
  if (!ui->st.tip && ui->drag < 0 && !ui->st.menu_open && ui->hover_hit != MK_HIT_NONE && ui->hover_hit < MK_HIT_MENU
      && ui->last_move_t > 0 && now - ui->last_move_t > 0.6) {
    build_tip(ui, ui->hover_hit);
    if (ui->st.tip_title[0]) { ui->st.tip = 1; redraw(ui); }
  }
  /* FRAME-RATE CAP: 30 fps is plenty for a knob drag; needs_redraw stays set, so
   * nothing is lost — it is only coalesced */
  if (ui->needs_redraw && (now - ui->last_paint) >= (1.0 / 30.0)) {
    ui->needs_redraw = 0; ui->last_paint = now;
    puglObscureView(ui->view);
  } else if (!ui->needs_redraw && ui->drag < 0 && now - ui->last_paint > 0.2) {
    mk_prewarm(&ui->A, &ui->st, ui->win_w, ui->win_h);     /* the plates one click away, one per idle, when quiet */
  }
  puglUpdate(ui->world, 0.0);
  return 0;
}

static int ui_resize_cb(LV2UI_Feature_Handle handle, int width, int height) {
  MkUI *ui = (MkUI *)handle;
  if (width < 1 || height < 1) return 1;
  /* resize the VIEW: PUGL_CONFIGURE then sets win_w/h from the real window (setting them here alone would draw
   * and hit-test for a size the X window does not have) */
  puglSetSizeHint(ui->view, PUGL_CURRENT_SIZE, (PuglSpan)width, (PuglSpan)height);
  redraw(ui);
  return 0;
}

static const void *extension_data(const char *uri) {
  static const LV2UI_Idle_Interface idle = { ui_idle };
  static const LV2UI_Resize resize = { NULL, ui_resize_cb };
  if (!strcmp(uri, LV2_UI__idleInterface)) return &idle;
  if (!strcmp(uri, LV2_UI__resize)) return &resize;
  return NULL;
}

static const LV2UI_Descriptor descriptor = { MK_UI_URI, instantiate, cleanup, port_event, extension_data };
LV2_SYMBOL_EXPORT const LV2UI_Descriptor *lv2ui_descriptor(uint32_t index) { return index == 0 ? &descriptor : NULL; }
