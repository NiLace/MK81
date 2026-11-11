/* bn_pentode_6550_own.h — the project's OWN 6550 pentode (the US edition). See
 * the .c and dsp/prototype/pentode_6550_own.py. Ports as in bn_pentode_own.h:
 * v[0]=vg1k v[1]=vg2k v[2]=vak -> i[0]=ig1 i[1]=ig2 i[2]=ia (flowing in). */
#ifndef BN_PENTODE_6550_OWN_H
#define BN_PENTODE_6550_OWN_H

void bn_pentode_6550_own_currents(const double* v, double* i);
void bn_pentode_6550_own(const double* v, double* i, double* J, void* user);

#ifdef BN_6550_TAB
void bn_pentode_6550_own_tables_init(void);
void bn_pentode_6550_own_tab(const double* v, double* i, double* J, void* user);
void bn_pentode_6550_own_tab_currents(const double* v, double* i);
#endif

#ifdef BN_GRID_TAB
void bn_pentode_6550_own_grid_tab(double vg1, double* ig, double* dig);   /* the grid law alone: ig1 and dig1/dvg1 */
#endif

#endif /* BN_PENTODE_6550_OWN_H */
