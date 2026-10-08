/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_psu.h — the 100 W head's power supply at CONTROL RATE.
 *
 * Averaged ODE of the rectifier + filters, integrated at ~1 kHz. It reproduces
 * the sag ENVELOPE (criterion: <1 dB vs the full simulation); the 2·f_mains
 * ripple is left out on purpose. The RCHG/VSRC parameters are fitted against
 * harness/out/psu_step_ref.txt (bn_test_psu).
 * States: Vc23 (plates), I_choke, Vc22 (screens), Vpi, Vx, Vxp.
 *
 * ⭐ The CHAIN follows the manufacturer's FACTORY VOLTAGE TABLE (50 W & 100 W
 * preamp circuit diagram, Iss.1 24-4-81, 100 W master-volume column):
 *
 *     screens --[rdrop 20k]-- Vpi 330 V --[r29 10k]-- Vx 290 V --[r8 10k]-- Vxp 280 V
 *
 * The later Iss.6 schematic draws a SINGLE resistor (screens -> Vx) with the
 * phase inverter hanging off Vx, but that drawing does NOT reproduce the
 * manufacturer's own declared operating point: it gives Vx ~387 V against
 * the 290 V measured on a replica build and on a serviced head.
 * The phase inverter has its OWN rail, 40 V higher than the preamp's.
 */
#ifndef BN_PSU_H
#define BN_PSU_H

/* Operating point of the 100 W head's supply, from the 2-point fit against
 * psu_step_ref. */
#define BN_PSU_I_REST      0.213      /* A, total rest current */
#define BN_PSU_V_REST      455.2      /* V, rest B+ */
#define BN_PSU_I_FULL      0.633      /* A, fortissimo */
/* the three factors of the 50 W head, DERIVED from its own netlist. */
#define BN_PSU_K_RCHG_50W  1.945      /* replacement mains transformer datasheet, DCR 50 W / 100 W = 63.09/32.43 */
#define BN_PSU_K_I_50W     0.5145     /*  97.5 mA / 189.5 mA (ngspice, 40 mA/valve) */
#define BN_PSU_K_V_50W     0.9132     /* 424.9 V / 465.3 V (plates at rest) */

typedef struct {
    /* parameters */
    double vsrc;        /* effective peak after the bridge; 471.7 V from the
                         * 2-point fit */
    double rchg;        /* charging impedance of the averaged rectifier */
    double c23, c22, c21c, c21b, c21a;
    double lch, rch;
    double rdrop, r29, r8;
    /* the HEAD. The 50 W head has its own supply so that it does not sag
     * like the 100 W, which is exactly the trait that sets them apart when
     * playing. Both axes (era and head) are stored because BOTH move
     * rchg/vsrc and must be recomputed together: applied separately they
     * overwrite each other. */
    int    old_way, w50;
    /* states */
    double vc23, il, vc22, vpi, vx, vxp;
} BnPsu;

void bn_psu_init(BnPsu* p);
/* supply HEAD. 0 = 100 W (4 valves) · 1 = 50 W (2).
 * The three numbers are DERIVED from the 50 W head's own netlist, not chosen:
 *   - source impedance x1.945 (the DCR ratio of a replacement mains transformer's datasheet: 63.09 ohm for
 *     the 50 W version against 32.43 for the 100 W — the 50 W has TWICE as much, which is why
 *     it "squashes sooner")
 *   - rest current x0.5145 (97.5 mA against 189.5, measured in ngspice with
 *     the bias calibrated to 40 mA/valve in EACH netlist)
 *   - rest rail x0.9132 (plates 424.9 V against 465.3)
 * ⚠️ Trap when re-deriving them: `RP1` senses TWO valves on the 100 W and ONE
 * on the 50 W, so the mA/valve formula is NOT the same for both. */
/* recomputes the rest point after rchg/vsrc change. Declared here because `bn_psu.c` defines
 * it as global: without a prototype an implicit signature goes unchecked. */
void bn_psu_refresh(BnPsu* p);
void bn_psu_set_head(BnPsu* p, int w50);
/* supply ERA. old=0: "new way" (>=1986, Iss.4/6 schematics), C23 = 100 uF.
 * old=1: "old way" (early and pre-1986 units; the 1976 schematic draws
 * 100+100 uF IN SERIES = 50 uF).
 * The REAL COMPONENT (c23) is changed for the dynamics, and rchg as well,
 * because in the real rectifier less reservoir = deeper ripple = a different
 * conduction angle = more average drop, and the averaged model folds all of
 * that into rchg. Both values come from measuring the supply netlist with
 * C23 100u and 50u:
 *      100u -> sag 33.0 V   ·   50u -> sag 41.8 V   (at 0.633 A)
 * vsrc moves with rchg to keep the rest point PINNED (455.2 V at 0.213 A),
 * so the only thing that changes between eras is the SLOPE. */
void bn_psu_set_era(BnPsu* p, int old_way);
/* one control step of `dt` seconds with the block's mean currents */
/* i_pi = the PHASE INVERTER's draw from its own rail (not from Vx) */
void bn_psu_step(BnPsu* p, double dt, double i_plates, double i_screens,
                 double i_pi, double i_x, double i_xp);

#endif /* BN_PSU_H */
