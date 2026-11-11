/* bn_fullamp_own.c — see bn_fullamp_own.h.
 *
 * The same port layout as bn_fullamp_device_col (bn_tube.c), with the project's
 * own valve models. If that layout changes, this one must change with it:
 *   ports  0..11 : 6 triodes (vgk, vak) each
 *   ports 12..17 : 2 equivalent pentodes (vg1k, vg2k, vak) each
 */
#include "bn_fullamp_own.h"

#include "bn_pentode_6550_own.h"
#include "bn_pentode_own.h"
#include "bn_tube_own.h"


/* Valves PER SIDE of the push-pull: 2 = 100 W (the default), 1 = 50 W. The
 * collapsed model merges each side into one equivalent valve, so the 50 W head
 * is EXACTLY this code with the factor at 1 — no new topology. Set ONCE before
 * instantiating, never from the audio thread. */
double bn_fullamp_npar = 2.0;
int bn_fullamp_pentode_tab = 0;

/* the caller's settings when it passes a BnFullampCfg, the globals otherwise (see bn_fullamp_own.h) */
static inline double cfg_npar(const void* user) { return user ? ((const BnFullampCfg*)user)->npar : bn_fullamp_npar; }
static inline int cfg_ptab(const void* user) { return user ? ((const BnFullampCfg*)user)->pentode_tab : bn_fullamp_pentode_tab; }

void bn_fullamp_device_col_own(const double* v, double* i, double* J, void* user)
{
    const double npar = cfg_npar(user);
    const int ptab = cfg_ptab(user);
    (void)ptab;
    bn_tube_own_tables_init();
    for (int t2 = 0; t2 < 6; ++t2) {
        double Jt[4];
        const int o = 2 * t2;
        bn_tube_own_tab(v + o, i + o, Jt, 0);
        J[(o + 0) * 18 + (o + 0)] = Jt[0];
        J[(o + 0) * 18 + (o + 1)] = Jt[1];
        J[(o + 1) * 18 + (o + 0)] = Jt[2];
        J[(o + 1) * 18 + (o + 1)] = Jt[3];
    }
    for (int p = 0; p < 2; ++p) {
        double Jp[9];
        const int o = 12 + 3 * p;
        if (ptab) { bn_pentode_own_tab(v + o, i + o, Jp, 0); }
        else                        { bn_pentode_own(v + o, i + o, Jp, 0); }
        for (int r = 0; r < 3; ++r) {
            i[o + r] *= npar;           /* valves in parallel */
            for (int cc = 0; cc < 3; ++cc) {
                J[(o + r) * 18 + (o + cc)] = npar * Jp[r * 3 + cc];
            }
        }
    }
}

void bn_fullamp_currents_col_own(const double* v, double* i, void* user)
{
    const double npar = cfg_npar(user);
    const int ptab = cfg_ptab(user);
    (void)ptab;
    bn_tube_own_tables_init();
    for (int t2 = 0; t2 < 6; ++t2) {
        bn_tube_own_tab_currents(v + 2 * t2, i + 2 * t2);
    }
    for (int p = 0; p < 2; ++p) {
        const int o = 12 + 3 * p;
        if (ptab) { bn_pentode_own_tab_currents(v + o, i + o); }
        else                        { bn_pentode_own_currents(v + o, i + o); }
        /* ⛔ npar, NOT a fixed 2.0: the device function above scales by it, and a
         * fixed 2.0 here would give Newton's residual twice the pentode current its
         * Jacobian sees on the 50 W head. */
        i[o] *= npar; i[o + 1] *= npar; i[o + 2] *= npar;
    }
}

/* --- US edition: identical, with the project's 6550 pentode ------------- */
void bn_fullamp_device_col_own_6550(const double* v, double* i, double* J, void* user)
{
    const double npar = cfg_npar(user);
    const int ptab = cfg_ptab(user);
    (void)ptab;
    bn_tube_own_tables_init();
    for (int t2 = 0; t2 < 6; ++t2) {
        double Jt[4];
        const int o = 2 * t2;
        bn_tube_own_tab(v + o, i + o, Jt, 0);
        J[(o + 0) * 18 + (o + 0)] = Jt[0];
        J[(o + 0) * 18 + (o + 1)] = Jt[1];
        J[(o + 1) * 18 + (o + 0)] = Jt[2];
        J[(o + 1) * 18 + (o + 1)] = Jt[3];
    }
    for (int p = 0; p < 2; ++p) {
        double Jp[9];
        const int o = 12 + 3 * p;
#ifdef BN_6550_TAB
        if (ptab) { bn_pentode_6550_own_tab(v + o, i + o, Jp, 0); } else
#endif
        bn_pentode_6550_own(v + o, i + o, Jp, 0);
        for (int r = 0; r < 3; ++r) {
            i[o + r] *= npar;
            for (int cc = 0; cc < 3; ++cc) {
                J[(o + r) * 18 + (o + cc)] = npar * Jp[r * 3 + cc];
            }
        }
    }
}

void bn_fullamp_currents_col_own_6550(const double* v, double* i, void* user)
{
    const double npar = cfg_npar(user);
    const int ptab = cfg_ptab(user);
    (void)ptab;
    bn_tube_own_tables_init();
    for (int t2 = 0; t2 < 6; ++t2) {
        bn_tube_own_tab_currents(v + 2 * t2, i + 2 * t2);
    }
    for (int p = 0; p < 2; ++p) {
        const int o = 12 + 3 * p;
#ifdef BN_6550_TAB
        if (ptab) { bn_pentode_6550_own_tab_currents(v + o, i + o); } else
#endif
        bn_pentode_6550_own_currents(v + o, i + o);
        /* ⛔ npar, NOT a fixed 2.0: the device function above scales by it, and a
         * fixed 2.0 here would give Newton's residual twice the pentode current its
         * Jacobian sees on the 50 W head. */
        i[o] *= npar; i[o + 1] *= npar; i[o + 2] *= npar;
    }
}
