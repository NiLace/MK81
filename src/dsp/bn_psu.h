/* SPDX-License-Identifier: AGPL-3.0-only */
/* Copyright (C) The NL Sounds contributors */
/* bn_psu.h — the amp's power supply at CONTROL RATE (100 W, 50 W, the reissue).
 *
 * Averaged ODE of the rectifier + filters, integrated in <= 1 ms steps. It
 * reproduces the sag ENVELOPE (within 1 dB of the full simulation); the
 * 2·f_mains ripple is left out on purpose. RCHG / VSRC are fitted to a full
 * simulation's load-step response (bn_test_psu).
 * States: Vc23 (plates), I_choke, Vc22 (screens), Vpi, Vx, Vxp.
 *
 * The CHAIN follows the factory voltage table of the master-volume preamp
 * schematic (the 100 W column):
 *
 *     screens --[rdrop 20k]-- Vpi 330 V --[r29 10k]-- Vx 290 V --[r8 10k]-- Vxp 280 V
 *
 * A later schematic draws a SINGLE resistor (screens -> Vx) with the
 * phase inverter hanging off Vx, but that drawing does NOT reproduce the
 * manufacturer's own declared operating point: it gives Vx ~387 V against
 * the 290 V measured on a replica build and on a serviced head.
 * The phase inverter has its OWN rail, 40 V higher than the preamp's.
 */
#ifndef BN_PSU_H
#define BN_PSU_H

/* Operating point of the 100 W head's supply, from the 2-point fit against the
 * full simulation. */
#define BN_PSU_I_REST      0.213      /* A, total rest current */
#define BN_PSU_V_REST      455.2      /* V, rest B+ */
#define BN_PSU_I_FULL      0.633      /* A, fortissimo */
/* the three factors of the 50 W head, DERIVED from its own netlist. */
#define BN_PSU_K_RCHG_50W  1.945      /* replacement mains transformer datasheet, DCR 50 W / 100 W = 63.09/32.43 */
/* the averaged rectifier per era (bn_psu_refresh): rchg behind vsrc and the reservoir C23. Named here because the
 * engine's generated systems carry the same rectifier INSIDE their netlist (their generator reads these) */
#define BN_PSU_RCHG_NEW    77.6       /* ohm, the new way (C23 100 uF) */
#define BN_PSU_RCHG_OLD    99.5       /* ohm, the old way (C23 50 uF, the series pair) */
#define BN_PSU_C23_NEW     100e-6
#define BN_PSU_C23_OLD     50e-6
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
    /* the reissue: the old way's 50 uF reservoir plus three DC loads its schematic draws - the 56K + 56K
     * balancing pairs across C20/C21 (reservoir) and C22/C23 (after the choke), and R33 47K 10 W from the screen
     * node to the effects card's +30 V zener rail. Same mains transformer as the 100 W head: the rest point is
     * NOT re-pinned, the extra load lowers B+ by itself. */
    int    ri;
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
/* supply ERA. old=0: "new way", C23 = 100 uF.
 * old=1: "old way" (the earlier units: 100+100 uF IN SERIES = 50 uF).
 * The REAL COMPONENT (c23) is changed for the dynamics, and rchg as well,
 * because in the real rectifier less reservoir = deeper ripple = a different
 * conduction angle = more average drop, and the averaged model folds all of
 * that into rchg. Both values come from measuring the supply netlist with
 * C23 100u and 50u:
 *      100u -> sag 33.0 V   ·   50u -> sag 41.8 V   (at 0.633 A)
 * vsrc moves with rchg to keep the rest point PINNED (455.2 V at 0.213 A),
 * so the only thing that changes between eras is the SLOPE. */
void bn_psu_set_era(BnPsu* p, int old_way);
/* the reissue's supply (its DC loads, the old way's reservoir); 0 = off */
void bn_psu_set_ri(BnPsu* p, int ri);
#define BN_PSU_RI_RBAL  112e3   /* R48 + R49 (and R42 + R43): 56K + 56K across each series pair of cans */
/* R33 47K 10 W from the screen node to the card's +30 V zener rail, as its resistor to GROUND with the same current at
 * the screen node's nominal 449 V: 47k x 449 / (449 - 30). The truth's netlist (dsp/prototype/power.py, ri) reads THIS
 * number, so the two cannot drift; the slope differs from the zener-fed branch by 7 % of a 50 k branch (negligible) */
#define BN_PSU_RI_R33EQ 50365.0
/* one control step of `dt` seconds with the cell's mean currents, in A;
 * i_pi = the PHASE INVERTER's draw from its own rail (not from Vx) */
void bn_psu_step(BnPsu* p, double dt, double i_plates, double i_screens,
                 double i_pi, double i_x, double i_xp);

#endif /* BN_PSU_H */
