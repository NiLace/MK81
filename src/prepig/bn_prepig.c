/* bn_prepig.c - the dispatch over the five FX-loop send systems (see bn_prepig.h). */
#include "bn_prepig.h"
#include "bn_pentode_own.h"
#include "bn_pentode_6550_own.h"
#include "bn_tube_own.h"

#define BN_DECL(v) size_t bn_prepig_##v##_size(void); \
    int bn_prepig_##v##_init(void*, double, const BnPrepigKnobs*, double, const double*); \
    void bn_prepig_##v##_set_knobs(void*, const BnPrepigKnobs*); \
    void bn_prepig_##v##_seed(void*, const double*, const double*); \
    void bn_prepig_##v##_restore(const void*, double*, double*, double*, double*); \
    double bn_prepig_##v##_process(void*, double, double, const double*); \
    int bn_prepig_##v##_fallbacks(const void*); \
    int bn_prepig_##v##_unconverged(const void*);
BN_DECL(100uk) BN_DECL(100us) BN_DECL(50uk) BN_DECL(50us) BN_DECL(100ri)

typedef struct {
    size_t (*size)(void);
    int    (*init)(void*, double, const BnPrepigKnobs*, double, const double*);
    void   (*set_knobs)(void*, const BnPrepigKnobs*);
    void   (*seed)(void*, const double*, const double*);
    void   (*restore)(const void*, double*, double*, double*, double*);
    double (*process)(void*, double, double, const double*);
    int    (*fallbacks)(const void*);
    int    (*unconverged)(const void*);
} BnPrepigOps;
#define BN_OPS(v) { bn_prepig_##v##_size, bn_prepig_##v##_init, bn_prepig_##v##_set_knobs, bn_prepig_##v##_seed, bn_prepig_##v##_restore, \
                    bn_prepig_##v##_process, bn_prepig_##v##_fallbacks, bn_prepig_##v##_unconverged }
static const BnPrepigOps OPS[BN_PREPIG_NV] = { BN_OPS(100uk), BN_OPS(100us), BN_OPS(50uk), BN_OPS(50us), BN_OPS(100ri) };

static int clampv(int v) { return v < 0 ? 0 : v >= BN_PREPIG_NV ? BN_PREPIG_NV - 1 : v; }

size_t bn_prepig_size(void)
{
    size_t m = 0;
    for (int v = 0; v < BN_PREPIG_NV; ++v) { if (OPS[v].size() > m) { m = OPS[v].size(); } }
    return m;
}
int bn_prepig_init(void* mem, int v, double fs, const BnPrepigKnobs* k, double vbias, const double* ultra_v)
{ return OPS[clampv(v)].init(mem, fs, k, vbias, ultra_v); }
void bn_prepig_set_knobs(void* mem, int v, const BnPrepigKnobs* k) { OPS[clampv(v)].set_knobs(mem, k); }
void bn_prepig_seed(void* mem, int v, const double* ultra_v, const double* ultra_h) { OPS[clampv(v)].seed(mem, ultra_v, ultra_h); }
void bn_prepig_restore(const void* mem, int v, double* ultra_v, double* ultra_vp1, double* ultra_vp2, double* ultra_h)
{ OPS[clampv(v)].restore(mem, ultra_v, ultra_vp1, ultra_vp2, ultra_h); }
double bn_prepig_process(void* mem, int v, double x, double vbias, const double* ultra_v)
{ return OPS[clampv(v)].process(mem, x, vbias, ultra_v); }
int bn_prepig_fallbacks(const void* mem, int v) { return OPS[clampv(v)].fallbacks(mem); }
int bn_prepig_unconverged(const void* mem, int v) { return OPS[clampv(v)].unconverged(mem); }
/* every table the variants read: the triodes, the EL34 and the 6550 (the US send systems) */
void bn_prepig_tables_init(void)
{
    bn_tube_own_tables_init(); bn_pentode_own_tables_init();
#ifdef BN_6550_TAB   /* the 6550's tables exist only in the tabulated build (the product's), as in bn_ultra_tables_init */
    bn_pentode_6550_own_tables_init();
#endif
}
