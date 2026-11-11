/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_dev_valve.h @ c0e3536, sha256 0b182272cce070881984977261824430f8e7c4f4f818391b25c403d1b7d7d822 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* nl_dev_valve.h - the shared engine's VALVES (nlcore#14, piece 1 of the amps on the core, nlcore#13): the laws MK81's exact
 * engine ships, moved here so every amp on the core starts from them. Each valve is a BLOCK of coupled ports evaluated at
 * once - its port voltages in, its port currents and the dense Jacobian of the block (row-major, J[r * n + c] = d i[r] /
 * d v[c]) out:
 *   12AX7 (ECC83) triode, n = 2: v = { vgk, vak }        i = { ig, ia }
 *   EL34 / 6550 pentode, n = 3:  v = { vg1k, vg2k, vak }  i = { ig1, ig2, ia }
 * Each law comes closed-form (nl_valve_<v>, nl_valve_<v>_currents) and tabulated (_tab, _tab_currents: what MK81 ships;
 * the closed form above the tables' range), the pentodes' control-grid law alone too (_grid_tab: MK81's FX-loop send
 * system). The tables are process-wide, built once by _tables_init (call it at instantiate, never in the audio thread);
 * a valve pair in parallel is the caller's scale on i and J. `user` is unused (MK81's device signature).
 * test/valve holds every function BIT-IDENTICAL to MK81's on a grid of port voltages. Off the nodal solver for now: the
 * block stamps are nlcore#14's next step. VENDORED: never edited in a plugin. */
#ifndef NL_DEV_VALVE_H
#define NL_DEV_VALVE_H

void nl_valve_12ax7_currents(const double* v, double* i);
void nl_valve_12ax7(const double* v, double* i, double* J, void* user);
void nl_valve_12ax7_tables_init(void);
void nl_valve_12ax7_tab(const double* v, double* i, double* J, void* user);
void nl_valve_12ax7_tab_currents(const double* v, double* i);

void nl_valve_el34_currents(const double* v, double* i);
void nl_valve_el34(const double* v, double* i, double* J, void* user);
void nl_valve_el34_tables_init(void);
void nl_valve_el34_tab(const double* v, double* i, double* J, void* user);
void nl_valve_el34_tab_currents(const double* v, double* i);
void nl_valve_el34_grid_tab(double vg1, double* ig, double* dig);

void nl_valve_6550_currents(const double* v, double* i);
void nl_valve_6550(const double* v, double* i, double* J, void* user);
void nl_valve_6550_tables_init(void);
void nl_valve_6550_tab(const double* v, double* i, double* J, void* user);
void nl_valve_6550_tab_currents(const double* v, double* i);
void nl_valve_6550_grid_tab(double vg1, double* ig, double* dig);

#endif /* NL_DEV_VALVE_H */
