/* bn_circuit.h — a circuit described as DATA.
 *
 * This is the piece that makes a multi-model plugin possible: an amp is a
 * table, not code. Adding another amp means filling in one of these
 * structures, not writing a new engine.
 *
 * The engine (bn_dk.c) builds the DK-method matrices in `instantiate`, which is
 * where memory may be allocated and matrices inverted; `run` is left with only
 * matrix-vector products and the Newton iteration of the nonlinear elements.
 */
#ifndef BN_CIRCUIT_H
#define BN_CIRCUIT_H

#include <stddef.h>

/* Reference node (ground). Nodes are numbered 1..n_nodes. */
#define BN_GND 0

typedef struct {
    int    a;        /* + node */
    int    b;        /* − node */
    double value;    /* ohms (R) or farads (C) */
} BnElement;

typedef struct {
    int a;
    int b;
} BnPort;

/* Group of COUPLED inductors (transformer): indices into the `inductors`
 * list + full L matrix (row-major, n×n) with the mutual inductances. */
#define BN_MUTUAL_MAX 24   /* the split output transformer of the 100 W heads couples 14 inductors; the 50 W head's, 20 */
typedef struct {
    int    n;
    int    idx[BN_MUTUAL_MAX];
    double L[BN_MUTUAL_MAX * BN_MUTUAL_MAX];
} BnMutual;

typedef struct {
    const char*      name;
    int              n_nodes;      /* not counting ground */
    const BnElement* resistors;
    int              n_resistors;
    const BnElement* capacitors;
    int              n_capacitors;
    const BnElement* inductors;      /* optional (may be 0) */
    int              n_inductors;
    const BnMutual*  mutuals;        /* optional (may be 0) */
    int              n_mutuals;
    const BnPort*    sources;      /* voltage sources (inputs) */
    int              n_sources;
    const BnPort*    nonlinear;    /* ports of the nonlinear devices */
    int              n_nonlinear;
    const BnPort*    outputs;
    int              n_outputs;
} BnCircuit;

#endif /* BN_CIRCUIT_H */
