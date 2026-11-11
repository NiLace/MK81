/* bn_pentode_own.h — the project's OWN pentode (EL34: Koren's eq. 5 form + a
 * sigma split, parameters fitted by us to the EL34 data sheet).
 *
 * No third-party material. The derivation and the provenance of every
 * constant are in dsp/prototype/pentode_own.py; this is its C port, checked
 * against the prototype by tools/bn_test_pentode_own.c.
 *
 * Ports (pentode): v[0]=vg1k, v[1]=vg2k, v[2]=vak
 *                  i[0]=ig1,  i[1]=ig2,  i[2]=ia  (flowing in).
 * The control grid conducts with the project triode's law (Dempwolf & Zölzer,
 * Table 1 of the paper, RSD-1), the conductance scaled by this valve's cathode
 * perveance: the grid then clamps near +0.2 V at full drive and Ig2/Ia comes
 * out within the 0.20-0.25 the data sheet gives for maximum signal. See
 * bn_pentode_own.c.
 */
#ifndef BN_PENTODE_OWN_H
#define BN_PENTODE_OWN_H

void bn_pentode_own_currents(const double* v, double* i);
void bn_pentode_own(const double* v, double* i, double* J, void* user);
/* TABULATED variant: the same model through 1-D tables, ~5x cheaper; out of
 * range it falls back to the closed form. Parity: tools/bn_test_pentode_own.c. */
void bn_pentode_own_tables_init(void);
void bn_pentode_own_tab(const double* v, double* i, double* J, void* user);
void bn_pentode_own_tab_currents(const double* v, double* i);

#ifdef BN_GRID_TAB
void bn_pentode_own_grid_tab(double vg1, double* ig, double* dig);   /* the grid law alone: ig1 and dig1/dvg1 */
#endif

#endif /* BN_PENTODE_OWN_H */
