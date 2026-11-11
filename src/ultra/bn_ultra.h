/* bn_ultra.h - ULTRA: MK 81's whole amp solved exactly by nlcore's nodal amp solver (src/vendor/nlcore/nl_mna.c).
 * 15 generated systems: 5 product variants (the loop off) and, per variant, the effects card's two LEVEL settings (the
 * loop on). The plugin holds engines of bn_ultra_size() bytes (the largest system) and re-initialises one on a variant
 * switch, which runs through silence.
 * Variant index: 0 100 W UK, 1 100 W US, 2 50 W UK, 3 50 W US (= 2 x head + us), 4 the 100 W reissue.
 * The FX loop and the preamp bypass (a loop never changes the amp's sound): a SEND system (src/prepig/: the preamp +
 * phase inverter with the output valves' grids conducting, its rail and NFB from this engine) gives the send and the
 * dry, and the heard engine has C9's top (the master wiper into the PI) DETACHED and fed the loop's mix, its current
 * still drawn from its own wiper. Loop off: no send system, nothing detached.
 * Sources: x = the guitar (volts at the input jack), u1 = the B+ source (htraw, volts), u2 = the bias source (volts). */
#ifndef BN_ULTRA_H
#define BN_ULTRA_H
#include <stddef.h>

#define BN_ULTRA_NV 5   /* product variants: 100uk 100us 50uk 50us 100ri (the reissue, v = 4) */
/* the reissue effects card's systems (src/ultra/genfx<L>_<v>/), the loop ON: variant BN_ULTRA_FX(v, level) for
 * v = 0..4 (as above) and level 0 ("0") / 1 ("I"); their process() takes the card's return (IC1b's output against
 * VREF, volts) in `top`, and their tap carries the send node fx_c */
#define BN_ULTRA_NFX (2 * BN_ULTRA_NV)
#define BN_ULTRA_FX(v, level) (BN_ULTRA_NV + BN_ULTRA_NV * (level) + (v))

/* every field is a control POSITION 0..1; the generated system maps each to its component values (lowin and era are
 * read as switches at 0.5).
 * cab: the Cabinet Feel, the 4x12 load's R, L and C at run time; 0 = the flat 8 ohm.
 * era: the supply's era, 1 = the old (C23 50 uF, rchg 99.5 ohm), 0 = the new; the averaged rectifier is inside each
 * generated system (htraw-plates = rchg, C23) and this control moves both; the reissue and the 50 W heads ignore it */
typedef struct { double vol, treb, bass, mid, master, pres, lowin, cab, era; } BnUltraKnobs;

/* what the plugin reads besides the speaker: w2 the master wiper (power bypass, FX send), volts; ipl / isc the output
 * valves' summed plate and screen currents (the supply's load), amperes; fxc the card's send node, volts (card
 * variants; 0 otherwise) */
typedef struct { double w2, ipl, isc, fxc; } BnUltraTap;

size_t bn_ultra_size(void);                          /* bytes for any variant's engine (the largest) */
double bn_ultra_vht(int v);                          /* the variant's B+ at rest (its generated source value) */
/* 0 on success: the operating point at these knobs and sources (u1, u2), the product's Newton settings set */
int    bn_ultra_init(void* mem, int v, double fs, const BnUltraKnobs* k, double u1, double u2);
/* the controls move (a linear refactorisation: per chunk while a knob moves, never per sample); state kept */
void   bn_ultra_set_knobs(void* mem, int v, const BnUltraKnobs* k);
/* one sample at the engine's rate: x volts at the input jack, returns volts at the 8-ohm secondary; tap may be NULL.
 * top, volts: product variants, C9's top, read only while detached (bn_ultra_detach; attached, C9 hangs from the
 * wiper as drawn); card variants, the card's return. */
double bn_ultra_process(void* mem, int v, double x, double u1, double u2, double top, BnUltraTap* tap);
/* detach (1) / re-attach (0) C9's top from the master wiper: a matrix rebuild, the state kept */
void   bn_ultra_detach(void* mem, int v, int on);
/* the whole engine state, src -> dst (both bn_ultra_size() bytes, the same variant): the twin is born as a copy */
void   bn_ultra_copy(void* dst, const void* src);
/* the engine's counters since init (fallbacks: samples that hit the Newton cap) */
int    bn_ultra_fallbacks(const void* mem, int v);
int    bn_ultra_unconverged(const void* mem, int v); /* samples that left Newton above tol */
/* the output valves' plate and screen current at the engine's present point (after an init: its DC operating point) */
void   bn_ultra_currents(const void* mem, int v, double* ipl, double* isc);
/* the node voltages after the last sample (the generator's V layout: unknowns, ground, sources): read only */
const double* bn_ultra_nodes(const void* mem, int v);
/* WRITABLE views of the engine's state (the FX loop's send system hands its preamp + phase inverter back to the heard
 * engine when the loop goes idle; the card hand-over, bn_card_map.c): node voltages, the start extrapolation's two past
 * samples (unknowns only), capacitor histories. Whoever writes them follows with a rebuild (bn_ultra_detach / bn_ultra_set_knobs), which drops
 * the kept factorisation. */
typedef struct { double *V, *Vp1, *Vp2, *h; } BnUltraArrays;
void   bn_ultra_arrays(void* mem, int v, BnUltraArrays* a);
/* the engine's state for the plugin's silence freeze: its capacitors' trapezoidal histories (n of them) */
const double* bn_ultra_state(const void* mem, int v, int* n);
/* once, outside the audio thread: the valve models' tables */
void   bn_ultra_tables_init(void);
#endif
