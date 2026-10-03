/* bn_psu.c — see bn_psu.h. Semi-implicit integration (backward Euler by
 * chained substitution): stable with a control dt (1 ms) and 0.5 s taus. */
#include <stdlib.h>   /* getenv/atof: experiment lever, see bn_psu_init */
#include "bn_psu.h"

/* experiment levers, filled ONCE from the environment in bn_psu_init (0 = unset) */
static double bn_psu_ovr_rchg = 0.0, bn_psu_ovr_vsrc = 0.0;

void bn_psu_init(BnPsu* p)
{
    /* 2-point FIT against psu_step_ref: in steady state the averaged model
     * obeys vc23 = vsrc − rchg·I_total; rest (455.2 V at 0.213 A) and
     * fortissimo (422.6 V at 0.633 A) give rchg=77.6, vsrc=471.7. */
    p->old_way = 0; p->w50 = 0;
    /* The PRODUCT runs the original supply — RWIND 20 / VSEC 345, the supply every shipped network was trained
     * with. A research fit to a replacement mains-transformer datasheet (RWIND 32.43, VSEC 352.7) stays in
     * spice/ as research only; `make psu-ctrl` renders its references with the product's RWIND/VSEC.
     * vsrc keeps the rest point (455.2 V at 0.213 A). bn_psu_refresh recomputes both; these are the defaults. */
    p->vsrc = 455.2 + 77.6 * 0.213;
    p->rchg = 77.6;
    /* EXPERIMENT LEVER, same pattern as BN_TRIMS/BN_MAXIT: read ONCE in init,
     * never on the audio thread, and without it the behaviour is
     * byte-identical. It exists because `rchg` is the ONLY number that sets
     * the plugin's SAG DEPTH, and it is MEASURED that the factory operating
     * point does NOT constrain it: a whole family of (VSEC_RMS, RWIND) gives
     * the same B+ at rest and sags from 36 to 92 V. The lever lets that
     * family be swept against captures. */
    { const char* e = getenv("BN_PSU_RCHG"); if (e && *e) { bn_psu_ovr_rchg = atof(e); p->rchg = bn_psu_ovr_rchg; } }
    { const char* e = getenv("BN_PSU_VSRC"); if (e && *e) { bn_psu_ovr_vsrc = atof(e); p->vsrc = bn_psu_ovr_vsrc; } }
    p->c23 = 100e-6; p->c22 = 100e-6;
    p->c21c = 100e-6;   /* 50+50u on the phase-inverter node */
    p->c21b = 50e-6; p->c21a = 50e-6;
    /* choke T3: nominal 3 H, DCR ~105 ohm. Weak arbiter: catalogue + forum
     * figures. */
    p->lch = 3.0; p->rch = 105.0;
    p->rdrop = 20e3; p->r29 = 10e3; p->r8 = 10e3;
    /* start at the FACTORY operating point (Iss.1 24-4-81, 100 W master-volume
     * column); the values derived by power_checks are 465.3 / 462.6 / 333.2 / 294.5 */
    p->vc23 = 465.0; p->il = 0.02; p->vc22 = 462.0;
    p->vpi = 333.0; p->vx = 294.5; p->vxp = 285.0;
}

/* Era and head move the SAME two numbers, so they are recomputed TOGETHER
 * from a single place. Applied separately they would overwrite each other:
 * `set_era` would rewrite rchg and vsrc from scratch, erasing any earlier
 * head adjustment. */
void bn_psu_refresh(BnPsu* p)
{
    /* MEASURED on the supply netlist (see the header): the 50 uF "old way"
     * sags 41.8 V at 0.633 A against 33.0 for the 100 uF one.
     * rchg = sag/(I-I_rest). */
    p->c23  = p->old_way ? 50e-6 : 100e-6;
    /* the original supply (RWIND 20), see bn_psu_init. The research transformer fit gave 126 / 112. */
    p->rchg = p->old_way ? 99.5 : 77.6;
    /* rest point PINNED: vc23 = vsrc - rchg*I */
    double i_rest = BN_PSU_I_REST, v_rest = BN_PSU_V_REST;
    if (p->w50) {
        /* K scales the 100 W base (the ratio of the two mains-transformer DCRs, 50 W / 100 W), so the 50 W
         * runs on (99.5 / 77.6) x K. */
        p->rchg *= BN_PSU_K_RCHG_50W;
        i_rest  *= BN_PSU_K_I_50W;
        v_rest  *= BN_PSU_K_V_50W;
    }
    p->vsrc = v_rest + p->rchg * i_rest;
    /* the experiment lever overrides this when set.
     * ⛔ It is read ONCE, in bn_psu_init, and here only the cached value is applied: this function is
     * called by bn_psu_set_era()/set_head(), which run on the AUDIO THREAD when the host moves the era
     * or head port — and `getenv` walks `environ`. */
    if (bn_psu_ovr_rchg > 0.0) { p->rchg = bn_psu_ovr_rchg; }
    if (bn_psu_ovr_vsrc > 0.0) { p->vsrc = bn_psu_ovr_vsrc; }
}

void bn_psu_set_era(BnPsu* p, int old_way)
{
    p->old_way = old_way ? 1 : 0;
    bn_psu_refresh(p);
}

void bn_psu_set_head(BnPsu* p, int w50)
{
    p->w50 = w50 ? 1 : 0;
    bn_psu_refresh(p);
}

void bn_psu_step(BnPsu* p, double dt, double i_plates, double i_screens,
                 double i_pi, double i_x, double i_xp)
{
    /* SUBSTEPS: the rectifier has tau = rchg*c23 ≈ 7.8 ms — with a block dt
     * (up to ~85 ms) Euler blows up (measured: the plugin output 8e9).
     * Integration runs in substeps of <=1 ms. */
    if (dt > 1.1e-3) {
        const int n = (int)(dt / 1e-3) + 1;
        const double h = dt / n;
        for (int k = 0; k < n; ++k) {
            bn_psu_step(p, h, i_plates, i_screens, i_pi, i_x, i_xp);
        }
        return;
    }
    /* averaged rectifier: charges only if vsrc > vc23 */
    const double ichg = (p->vsrc > p->vc23) ? (p->vsrc - p->vc23) / p->rchg : 0.0;
    /* choke (semi-implicit in il) */
    const double il_new = (p->il + dt / p->lch * (p->vc23 - p->vc22))
                          / (1.0 + dt * p->rch / p->lch);
    p->vc23 += dt / p->c23 * (ichg - i_plates - il_new);
    const double idrop = (p->vc22 - p->vpi) / p->rdrop;
    p->vc22 += dt / p->c22 * (il_new - i_screens - idrop);
    /* The small-signal stages' currents are given at the factory point (Iss.1 24-4-81: PI 330 V, X 290 V,
     * XP 280 V) and scale WITH their own supply — a cathode-biased triode on a resistive load draws roughly in
     * proportion to its B+. Constant currents would keep the PI->X drop fixed while the exact netlist's shrinks
     * under load (36 -> 29 V): node X would sag 6-25 % too much. Tested offline on the exact circuit's own PI
     * rail (node_x_probe.py traces): X error rms 1.10-2.97 V with constant currents -> 0.30-0.85 V with this
     * law (exponents 1.5 and 2 are worse). Topology and values below the screens are fullamp.py's. */
    const double ipi = i_pi * (p->vpi / 330.0), ix = i_x * (p->vx / 290.0), ixp = i_xp * (p->vxp / 280.0);
    const double ir29 = (p->vpi - p->vx) / p->r29;
    p->vpi  += dt / p->c21c * (idrop - ipi - ir29);
    const double ir8 = (p->vx - p->vxp) / p->r8;
    p->vx   += dt / p->c21b * (ir29 - ix - ir8);
    p->vxp  += dt / p->c21a * (ir8 - ixp);
    p->il = il_new;
}
