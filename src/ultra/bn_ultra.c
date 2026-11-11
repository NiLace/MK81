/* bn_ultra.c - ULTRA's dispatch over its 15 systems (see bn_ultra.h). */
#include <string.h>

#include "bn_ultra.h"
#include "bn_pentode_own.h"
#include "bn_tube_own.h"
#include "bn_pentode_6550_own.h"
#include "nl_dev_valve.h"   /* the ULTRA engine's valves: nlcore's (src/vendor/nlcore) */

#define BN_DECL(v) size_t bn_ultra_##v##_size(void); double bn_ultra_##v##_vht(void); \
    int bn_ultra_##v##_init(void*, double, const BnUltraKnobs*, double, double); \
    void bn_ultra_##v##_set_knobs(void*, const BnUltraKnobs*); \
    double bn_ultra_##v##_process(void*, double, double, double, double, BnUltraTap*); \
    void bn_ultra_##v##_detach(void*, int); \
    int bn_ultra_##v##_fallbacks(const void*); \
    int bn_ultra_##v##_unconverged(const void*); \
    void bn_ultra_##v##_currents(const void*, double*, double*); \
    const double* bn_ultra_##v##_state(const void*, int*); \
    const double* bn_ultra_##v##_nodes(const void*); \
    void bn_ultra_##v##_arrays(void*, BnUltraArrays*);
BN_DECL(100uk) BN_DECL(100us) BN_DECL(50uk) BN_DECL(50us) BN_DECL(100ri)   /* the product's systems (100ri: the reissue) */
BN_DECL(fx0_100uk) BN_DECL(fx0_100us) BN_DECL(fx0_50uk) BN_DECL(fx0_50us) BN_DECL(fx0_100ri)   /* the card's systems, LEVEL "0" */
BN_DECL(fxI_100uk) BN_DECL(fxI_100us) BN_DECL(fxI_50uk) BN_DECL(fxI_50us) BN_DECL(fxI_100ri)   /* ... LEVEL "I" */

typedef struct {
    size_t (*size)(void);
    double (*vht)(void);
    int    (*init)(void*, double, const BnUltraKnobs*, double, double);
    void   (*set_knobs)(void*, const BnUltraKnobs*);
    double (*process)(void*, double, double, double, double, BnUltraTap*);
    void   (*detach)(void*, int);
    int    (*fallbacks)(const void*);
    int    (*unconverged)(const void*);
    void   (*currents)(const void*, double*, double*);
    const double* (*state)(const void*, int*);
    const double* (*nodes)(const void*);
    void   (*arrays)(void*, BnUltraArrays*);
} BnUltraOps;
#define BN_OPS(v) { bn_ultra_##v##_size, bn_ultra_##v##_vht, bn_ultra_##v##_init, bn_ultra_##v##_set_knobs, \
                    bn_ultra_##v##_process, bn_ultra_##v##_detach, bn_ultra_##v##_fallbacks, \
                    bn_ultra_##v##_unconverged, bn_ultra_##v##_currents, bn_ultra_##v##_state, \
                    bn_ultra_##v##_nodes, bn_ultra_##v##_arrays }
static const BnUltraOps OPS[BN_ULTRA_NV + BN_ULTRA_NFX] = { BN_OPS(100uk), BN_OPS(100us), BN_OPS(50uk), BN_OPS(50us), BN_OPS(100ri),
    BN_OPS(fx0_100uk), BN_OPS(fx0_100us), BN_OPS(fx0_50uk), BN_OPS(fx0_50us), BN_OPS(fx0_100ri),
    BN_OPS(fxI_100uk), BN_OPS(fxI_100us), BN_OPS(fxI_50uk), BN_OPS(fxI_50us), BN_OPS(fxI_100ri) };

static int clampv(int v) { return v < 0 ? 0 : v >= BN_ULTRA_NV + BN_ULTRA_NFX ? BN_ULTRA_NV + BN_ULTRA_NFX - 1 : v; }

size_t bn_ultra_size(void)
{
    size_t m = 0;
    for (int v = 0; v < BN_ULTRA_NV + BN_ULTRA_NFX; ++v) { if (OPS[v].size() > m) { m = OPS[v].size(); } }
    return m;
}
double bn_ultra_vht(int v) { return OPS[clampv(v)].vht(); }
int bn_ultra_init(void* mem, int v, double fs, const BnUltraKnobs* k, double u1, double u2)
{ return OPS[clampv(v)].init(mem, fs, k, u1, u2); }
void bn_ultra_set_knobs(void* mem, int v, const BnUltraKnobs* k) { OPS[clampv(v)].set_knobs(mem, k); }
double bn_ultra_process(void* mem, int v, double x, double u1, double u2, double top, BnUltraTap* tap)
{ return OPS[clampv(v)].process(mem, x, u1, u2, top, tap); }
void bn_ultra_detach(void* mem, int v, int on) { OPS[clampv(v)].detach(mem, on); }
void bn_ultra_copy(void* dst, const void* src) { memcpy(dst, src, bn_ultra_size()); }
int bn_ultra_fallbacks(const void* mem, int v) { return OPS[clampv(v)].fallbacks(mem); }
int bn_ultra_unconverged(const void* mem, int v) { return OPS[clampv(v)].unconverged(mem); }
void bn_ultra_currents(const void* mem, int v, double* ipl, double* isc) { OPS[clampv(v)].currents(mem, ipl, isc); }
const double* bn_ultra_state(const void* mem, int v, int* n) { return OPS[clampv(v)].state(mem, n); }
const double* bn_ultra_nodes(const void* mem, int v) { return OPS[clampv(v)].nodes(mem); }
void bn_ultra_arrays(void* mem, int v, BnUltraArrays* a) { OPS[clampv(v)].arrays(mem, a); }
void bn_ultra_tables_init(void) { nl_valve_12ax7_tables_init(); nl_valve_el34_tables_init(); nl_valve_6550_tables_init();
    bn_tube_own_tables_init(); bn_pentode_own_tables_init();
#ifdef BN_6550_TAB
    bn_pentode_6550_own_tables_init();
#endif
}
