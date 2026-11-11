/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_mna_valves.h @ c0e3536, sha256 ce074978f9d315b953ab50edff72ea6cee93a9852481675f8c9ad74513389732 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* nl_mna_valves.h - the amp solver's device (nlcore#13): the circuit's valves as port BLOCKS over nl_dev_valve.h, in the
 * generated circuit's port order (mks_preamp_circuit.h): MKS_N_TRIODES 12AX7 blocks of 2 ports first, then MKS_N_PENTODES
 * pentode blocks of 3 (the EL34, or the 6550 with NL_MNA_6550), each pentode standing for `npar` valves in parallel (its
 * current and Jacobian scaled). J is the MKS_N_PORTS-square port Jacobian; only the diagonal blocks are written (the
 * engine reads its blocks through bstart / bsize). MOVED from MK81's src/dsp/bn_fullamp_own.c (5ab77fa0: its 6-triode,
 * 2-pentode, 18-port layout, written there with literal sizes; the same operations in the same order here), so the move
 * is bit for bit. The settings are per caller (MK81#5: two instances or two heads must not share a valve count). */
#ifndef NL_MNA_VALVES_H
#define NL_MNA_VALVES_H

#include "nl_dev_valve.h"

typedef struct {
    double npar;        /* valves per side of the push-pull: 2 a 100 W head, 1 a 50 W head */
    int    pentode_tab; /* 1 = the pentodes by their tables (what MK81 ships), 0 = the closed form */
} NlMnaValveCfg;

#ifdef NL_MNA_6550
#define NL_MNA_PENTODE(v, i, J)       nl_valve_6550((v), (i), (J), 0)
#define NL_MNA_PENTODE_TAB(v, i, J)   nl_valve_6550_tab((v), (i), (J), 0)
#define NL_MNA_PENTODE_CUR(v, i)      nl_valve_6550_currents((v), (i))
#define NL_MNA_PENTODE_TAB_CUR(v, i)  nl_valve_6550_tab_currents((v), (i))
#define NL_MNA_PENTODE_INIT()         nl_valve_6550_tables_init()
#else
#define NL_MNA_PENTODE(v, i, J)       nl_valve_el34((v), (i), (J), 0)
#define NL_MNA_PENTODE_TAB(v, i, J)   nl_valve_el34_tab((v), (i), (J), 0)
#define NL_MNA_PENTODE_CUR(v, i)      nl_valve_el34_currents((v), (i))
#define NL_MNA_PENTODE_TAB_CUR(v, i)  nl_valve_el34_tab_currents((v), (i))
#define NL_MNA_PENTODE_INIT()         nl_valve_el34_tables_init()
#endif

/* the tables are built once per process: call nl_mna_valves_init at instantiate (never in the audio thread). The calls
 * below check every valve's tables again (a flag test each): MK81's device checked only the 12AX7's and relied on its
 * plugin building the pentodes' - a caller that forgot read zeros from the pentode tables SILENTLY (Newton then fell back
 * on every sample: test/mna's first runner) */
static inline void nl_mna_valves_init(void)
{
    nl_valve_12ax7_tables_init(); nl_valve_el34_tables_init(); nl_valve_6550_tables_init();
}

static inline void nl_mna_valves(const double* v, double* i, double* J, const NlMnaValveCfg* cfg)
{
    const double npar = cfg->npar;
    const int ptab = cfg->pentode_tab;
    nl_valve_12ax7_tables_init();
    if (ptab) { NL_MNA_PENTODE_INIT(); }
    for (int t2 = 0; t2 < MKS_N_TRIODES; ++t2) {
        double Jt[4];
        const int o = 2 * t2;
        nl_valve_12ax7_tab(v + o, i + o, Jt, 0);
        J[(o + 0) * MKS_N_PORTS + (o + 0)] = Jt[0];
        J[(o + 0) * MKS_N_PORTS + (o + 1)] = Jt[1];
        J[(o + 1) * MKS_N_PORTS + (o + 0)] = Jt[2];
        J[(o + 1) * MKS_N_PORTS + (o + 1)] = Jt[3];
    }
    for (int p = 0; p < MKS_N_PENTODES; ++p) {
        double Jp[9];
        const int o = 2 * MKS_N_TRIODES + 3 * p;
        if (ptab) { NL_MNA_PENTODE_TAB(v + o, i + o, Jp); }
        else      { NL_MNA_PENTODE(v + o, i + o, Jp); }
        for (int r = 0; r < 3; ++r) {
            i[o + r] *= npar;           /* npar valves in parallel */
            for (int cc = 0; cc < 3; ++cc) {
                J[(o + r) * MKS_N_PORTS + (o + cc)] = npar * Jp[r * 3 + cc];
            }
        }
    }
}

static inline void nl_mna_valves_currents(const double* v, double* i, const NlMnaValveCfg* cfg)
{
    const double npar = cfg->npar;
    const int ptab = cfg->pentode_tab;
    nl_valve_12ax7_tables_init();
    if (ptab) { NL_MNA_PENTODE_INIT(); }
    for (int t2 = 0; t2 < MKS_N_TRIODES; ++t2) {
        nl_valve_12ax7_tab_currents(v + 2 * t2, i + 2 * t2);
    }
    for (int p = 0; p < MKS_N_PENTODES; ++p) {
        const int o = 2 * MKS_N_TRIODES + 3 * p;
        if (ptab) { NL_MNA_PENTODE_TAB_CUR(v + o, i + o); }
        else      { NL_MNA_PENTODE_CUR(v + o, i + o); }
        /* npar, never a literal 2: a 50 W head runs 1, and the Jacobian above already scales by it */
        i[o] *= npar; i[o + 1] *= npar; i[o + 2] *= npar;
    }
}

#endif /* NL_MNA_VALVES_H */
