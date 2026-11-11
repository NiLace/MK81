/* SPDX-License-Identifier: AGPL-3.0-only */
/* VENDORED from nlcore/nl_mna_device.h @ c0e3536, sha256 a2b8cac3aca793880d394d5aca55425ce7deb5050257ef7a98fc11b78f05c902 - never edit here: change the core in the nlcore repository, pass its gate (make test), re-vendor (nlcore tools/vendor.py) */
/* Copyright (C) The NL Sounds contributors */
/* mks_device.h - the MK Silver preamp's nonlinear ports (see mks_device.c). */
#ifndef MKS_DEVICE_H
#define MKS_DEVICE_H

#include "mks_preamp_circuit.h"

typedef struct {
    double fs;                       /* the engine's rate: the trapezoidal step of the junction charge */
    int    started;                  /* 0 until mks_device_start: the DC solve sees no charge current */
    double q[MKS_N_DIODES];          /* charge excess at the last converged sample */
    double ic[MKS_N_DIODES];         /* its trapezoidal current at the last converged sample */
    double vlast[MKS_N_DIODES];      /* each junction's voltage at its last evaluation (the limiting's reference) */
    double qlast[MKS_N_DIODES];      /* ... its charge excess there, and the slope of that charge (for the commit) */
    double clast[MKS_N_DIODES];
    int    limit;                    /* 1 = SPICE-style junction voltage limiting on (mks_device_limit) */
    int    limited;                  /* set when the last evaluation limited any junction: not a converged point */
    int    bstart[MKS_N_PORTS];      /* Jacobian blocks for the engine: 2x2 per triode, 1x1 per junction */
    int    bsize[MKS_N_PORTS];
} MksDevice;

void mks_device_init(MksDevice* d);
void mks_device(const double* v, double* i, double* J, void* user);          /* BnDeviceFn */
void mks_device_currents(const double* v, double* i, void* user);            /* BnDeviceCurrFn */
void mks_device_commit(const double* v, void* user);                         /* BnDK.device_commit */
void mks_device_start(MksDevice* d, const double* v, double fs);             /* at the operating point */

#endif
