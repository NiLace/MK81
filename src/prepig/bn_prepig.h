/* bn_prepig.h - ULTRA's FX-loop SEND generator: the preamp + phase inverter with the output valves' control grids
 * CONDUCTING (src/prepig/gen_<variant>/, tools/gen_prepig.sh; device bn_prepig_device.c), its inverter's rail xpi and
 * its feedback return t4 taken from the HEARD ULTRA engine one engine sample late. Its master wiper w2 is the loop's
 * send, the dry part of the seam and the preamp bypass's tap: a partial system, cheaper than a whole second amp.
 * Variant index: ULTRA's product index (0 100 W UK, 1 100 W US, 2 50 W UK, 3 50 W US, 4 the reissue). Every call reads
 * ULTRA's node voltages through the array bn_ultra_nodes() returns for the SAME variant. */
#ifndef BN_PREPIG_H
#define BN_PREPIG_H
#include <stddef.h>

#define BN_PREPIG_NV 5   /* as ULTRA's BN_ULTRA_NV */

typedef struct { double vol, treb, bass, mid, master, pres, lowin; } BnPrepigKnobs;   /* control positions 0..1 */

size_t bn_prepig_size(void);                          /* bytes for any variant's system (the largest) */
/* 0 on success: the matrices at fs and the knobs, an operating point with the sources of ULTRA's present nodes (xpi,
 * t4) and vbias, the product's Newton settings (ULTRA's). Not real-time cheap (a DC solve): at ULTRA's start only. */
int    bn_prepig_init(void* mem, int v, double fs, const BnPrepigKnobs* k, double vbias, const double* ultra_v);
/* the controls move (a linear refactorisation); state kept */
void   bn_prepig_set_knobs(void* mem, int v, const BnPrepigKnobs* k);
/* the send system takes the heard engine's present state: every node voltage and every capacitor history it shares
 * with ULTRA's system (bn_prepig_map.h), the output grids' passive far ends (tied to ground here) at zero current; the
 * start extrapolation from rest, no factorisation kept. ultra_v / ultra_h: bn_ultra_nodes() / bn_ultra_state(). */
void   bn_prepig_seed(void* mem, int v, const double* ultra_v, const double* ultra_h);
/* the reverse, when the loop goes idle: the heard engine takes the send system's present preamp + phase inverter (+ the
 * output grid networks): every node voltage and capacitor history the seed copies, written back into ULTRA's arrays
 * (bn_ultra_arrays), the start extrapolation of those nodes from rest. The output stage, the supply and the OT keep
 * the heard engine's own state. The caller then re-attaches C9 (a rebuild). */
void   bn_prepig_restore(const void* mem, int v, double* ultra_v, double* ultra_vp1, double* ultra_vp2, double* ultra_h);
/* one engine sample: the master wiper w2 (volts); xpi and t4 read from ultra_v (the heard engine's LAST sample) */
double bn_prepig_process(void* mem, int v, double x, double vbias, const double* ultra_v);
int    bn_prepig_fallbacks(const void* mem, int v);
int    bn_prepig_unconverged(const void* mem, int v);
/* once, outside the audio thread: the valve models' tables */
void   bn_prepig_tables_init(void);
#endif
