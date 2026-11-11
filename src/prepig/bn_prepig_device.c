/* bn_prepig_device.c - the device of ULTRA's FX-loop SEND systems: the preamp + phase inverter with the output valves'
 * control grids conducting (src/prepig/gen_<variant>/, tools/gen_prepig.sh). No new model:
 * 14 ports: the six triodes over ports 0..11 (MK 81's tabulated triode, 2x2 blocks: grid-cathode, plate-cathode), then
 * the two equivalent output-valve control grids (g45, g67 to ground), 1x1 blocks. Each grid's law is the whole amp's
 * own, called, not copied: i[0] / J[0] of bn_pentode_own (or its table, as the instance's settings say) and of
 * bn_pentode_6550_own for the export edition, which are functions of vg1k alone (the screen and plate arguments are
 * set to an in-range rest value and do not enter ig1), times the valves per side (BnFullampCfg.npar).
 * The solver calls these through NL_MNA_DEVICE / NL_MNA_DEVICE_CURRENTS (tools/build_ultra_objs.sh): the _6550 pair
 * for the US variants. */
#include "bn_fullamp_own.h"
#include "bn_pentode_6550_own.h"
#include "bn_pentode_own.h"
#include "bn_tube_own.h"

#define NP 14

void bn_prepig_device(const double* v, double* i, double* J, void* user);
void bn_prepig_currents(const double* v, double* i, void* user);
void bn_prepig_device_6550(const double* v, double* i, double* J, void* user);
void bn_prepig_currents_6550(const double* v, double* i, void* user);

static void triodes(const double* v, double* i, double* J)
{
    bn_tube_own_tables_init();
    for (int t = 0; t < 6; ++t) {
        double Jt[4];
        const int o = 2 * t;
        bn_tube_own_tab(v + o, i + o, Jt, 0);
        J[(o + 0) * NP + (o + 0)] = Jt[0];
        J[(o + 0) * NP + (o + 1)] = Jt[1];
        J[(o + 1) * NP + (o + 0)] = Jt[2];
        J[(o + 1) * NP + (o + 1)] = Jt[3];
    }
}

/* one grid port: the pentode called at (vg1k, rest screen, rest plate); only its grid current and dig1/dvg1 are used */
static void grid(int us, const BnFullampCfg* c, double vg, double* ig, double* dig)
{
#ifdef BN_GRID_TAB
    /* the grid law read directly from its table, not a whole pentode evaluated for its ig1 */
#ifdef BN_6550_TAB
    if (us && c->pentode_tab) { bn_pentode_6550_own_grid_tab(vg, ig, dig); *ig *= c->npar; if (dig) { *dig *= c->npar; } return; }
#endif
    if (!us && c->pentode_tab) { bn_pentode_own_grid_tab(vg, ig, dig); *ig *= c->npar; if (dig) { *dig *= c->npar; } return; }
#endif
    const double v3[3] = { vg, 400.0, 400.0 };
    double i3[3], J3[9];
    if (us) { bn_pentode_6550_own(v3, i3, J3, 0); }
    else if (c->pentode_tab) { bn_pentode_own_tab(v3, i3, J3, 0); }
    else { bn_pentode_own(v3, i3, J3, 0); }
    *ig = c->npar * i3[0];
    if (dig) { *dig = c->npar * J3[0]; }
}

static void dev(int us, const double* v, double* i, double* J, void* user)
{
    const BnFullampCfg* c = (const BnFullampCfg*)user;
    triodes(v, i, J);
    for (int g = 12; g < 14; ++g) { grid(us, c, v[g], &i[g], &J[g * NP + g]); }
}

static void cur(int us, const double* v, double* i, void* user)
{
    const BnFullampCfg* c = (const BnFullampCfg*)user;
    bn_tube_own_tables_init();
    for (int t = 0; t < 6; ++t) { bn_tube_own_tab_currents(v + 2 * t, i + 2 * t); }
    for (int g = 12; g < 14; ++g) { grid(us, c, v[g], &i[g], 0); }
}

void bn_prepig_device(const double* v, double* i, double* J, void* user) { dev(0, v, i, J, user); }
void bn_prepig_currents(const double* v, double* i, void* user) { cur(0, v, i, user); }
void bn_prepig_device_6550(const double* v, double* i, double* J, void* user) { dev(1, v, i, J, user); }
void bn_prepig_currents_6550(const double* v, double* i, void* user) { cur(1, v, i, user); }
