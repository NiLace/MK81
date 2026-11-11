/* bn_fullamp_own.h — the whole amp's valves as ONE COLLAPSED device, on the
 * project's OWN models (triode bn_tube_own, pentodes bn_pentode_own /
 * bn_pentode_6550_own): no third-party material.
 *
 * 18 ports = 6 triodes (2 each) + 2 EQUIVALENT pentodes (3 each, current and
 * Jacobian x npar because each stands for one side's valves in parallel).
 */
#ifndef BN_FULLAMP_OWN_H
#define BN_FULLAMP_OWN_H

/* Output valves PER SIDE: 2 = the 100 W head (the default), 1 = the 50 W head.
 * Set ONCE before instantiating, never from the audio thread. */
extern double bn_fullamp_npar;
/* 1 = the pentodes through their TABLES (bn_pentode_own_tab); 0 = the closed form. */
extern int bn_fullamp_pentode_tab;

/* The same two settings PER CALLER. A device called with `user` pointing at one of these takes its npar and
 * pentode_tab from it and never reads the globals above; `user` NULL keeps the globals. The plugin passes its own
 * per engine, so two instances, or two heads in one instance, cannot change each other's valve count. */
typedef struct {
    double npar;        /* valves per side: 2 the 100 W, 1 the 50 W */
    int    pentode_tab; /* 1 = the tabulated pentode (the 6550's table only with -DBN_6550_TAB) */
} BnFullampCfg;

void bn_fullamp_device_col_own(const double* v, double* i, double* J, void* user);
void bn_fullamp_currents_col_own(const double* v, double* i, void* user);

/* US EDITION: the same triodes, the project's 6550 pentodes. */
void bn_fullamp_device_col_own_6550(const double* v, double* i, double* J, void* user);
void bn_fullamp_currents_col_own_6550(const double* v, double* i, void* user);

#endif /* BN_FULLAMP_OWN_H */
