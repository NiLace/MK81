#!/bin/sh
# build_ultra_objs.sh OUTDIR [CFLAGS...] - ULTRA's objects: each of the 15 generated systems (5 product variants, and
# the effects card's two LEVEL settings per variant) compiled with nlcore's nodal amp solver (src/vendor/nlcore/nl_mna.c)
# and the wrapper (src/ultra/bn_ultra_engine.c), its global symbols renamed per system so all link into one binary,
# plus the dispatch (src/ultra/bn_ultra.c); optionally the card's return and send halves (ULTRA_RET, ULTRA_SEND); and
# the FX loop's five SEND systems (src/prepig/gen_*/, tools/gen_prepig.sh) on the same solver, renamed bn_prepig_<v>_*,
# with the grid-conducting device (src/prepig/bn_prepig_device.c), their dispatch and device. Prints the object list.
# -I order: the system's generated directory FIRST: the solver's headers include the generated headers in quotes, so
# each system compiles against its own.
set -e
cd "$(dirname "$0")/.."
O=${1:?OUTDIR}; shift
mkdir -p "$O"
CF="-std=c11 -D_DEFAULT_SOURCE -fPIC -DMKS_DEVICE_MK81 $*"
SYMS="mks_mna_init mks_mna_process mks_mna_set_knobs mks_mna_set_emf_detached mks_pi_load_at mks_mna_set_theta mks_mna_handover mks_mna_set_fs mks_mna_settle"
OBJS=""
for v in 100uk 100us 50uk 50us 100ri; do
    REN=""; for s in $SYMS; do REN="$REN -D$s=bn_ultra_${v}_$s"; done
    INC="-I${ULTRA_GEN_ROOT:-src/ultra}/gen_$v -Isrc/vendor/nlcore -Isrc/ultra -Isrc/dsp"   # ULTRA_GEN_ROOT: another set of systems (measurement builds)
    "${CC:-gcc}" $CF $REN $INC -c src/vendor/nlcore/nl_mna.c -o "$O/ultra_${v}_mna.o"
    "${CC:-gcc}" $CF $REN -DBN_UV=$v $INC -c src/ultra/bn_ultra_engine.c -o "$O/ultra_${v}_wrap.o"
    OBJS="$OBJS $O/ultra_${v}_mna.o $O/ultra_${v}_wrap.o"
done
# the reissue card's systems (the loop ON, src/ultra/genfx<L>_<v>, harness/a8/fx165/gen_card_systems.sh), the same
# solver and wrapper, renamed bn_ultra_fx<L>_<v>_*
for L in 0 I; do
    for v in 100uk 100us 50uk 50us 100ri; do
        REN=""; for s in $SYMS; do REN="$REN -D$s=bn_ultra_fx${L}_${v}_$s"; done
        INC="-I${ULTRA_CARD_GEN_ROOT:-src/ultra}/genfx${L}_$v -Isrc/vendor/nlcore -Isrc/ultra -Isrc/dsp"   # ULTRA_CARD_GEN_ROOT: another set of card systems
        # the card systems' step cap grows x2 while Newton's target holds (MKS_CAP_GROW: far fewer fallbacks);
        # ULTRA_CARD_MNA_CF overrides it (empty: none) - the product's variants never see it
        "${CC:-gcc}" $CF ${ULTRA_CARD_MNA_CF--DMKS_CAP_GROW=2} $REN $INC -c src/vendor/nlcore/nl_mna.c -o "$O/ultra_fx${L}_${v}_mna.o"
        "${CC:-gcc}" $CF $REN -DBN_UV=fx${L}_$v $INC -c src/ultra/bn_ultra_engine.c -o "$O/ultra_fx${L}_${v}_wrap.o"
        OBJS="$OBJS $O/ultra_fx${L}_${v}_mna.o $O/ultra_fx${L}_${v}_wrap.o"
    done
done
# ULTRA_RET=1 (plugin.mk's ULTRA rules: the product carries the card): the card's RETURN half, from ULTRA_RET_DIR -
# the nodal return on nlcore's solver (harness/a8/fx165/ret) or the linear return (linret/c, the product's) - with
# the same flags as the rest. Other callers (the measuring harness) leave it unset and build their own.
if [ "${ULTRA_RET:-0}" = 1 ]; then
    R=${ULTRA_RET_DIR:-harness/a8/fx165/ret}; RC=${ULTRA_RET_CORE:-src/vendor/nlcore}
    SRCS="$R/ret_nodal_api.c $RC/nl_nodal.c $RC/nl_core.c"
    # a return directory with linret_circuit.h is the LINEAR return (src/fxcard/ret): one file, no solver
    [ -f "$R/linret_circuit.h" ] && SRCS="$R/ret_nodal_api.c"
    for f in $SRCS; do
        b=$(basename "$f" .c)
        "${CC:-gcc}" $CF -DNL_NODAL=1 -I$R -I$RC -c "$f" -o "$O/ret_$b.o"
        OBJS="$OBJS $O/ret_$b.o"
    done
fi
# ULTRA_SEND=1 (builds with -DBN_SEND_TWIN, the product among them): the card's SEND half from ULTRA_SEND_DIR - the
# linear send (linret/send, the product's), or the nodal send (harness/a8/fx165/send/) on nlcore's solver, every global
# symbol renamed cs_* so it links beside the return half's copies
if [ "${ULTRA_SEND:-0}" = 1 ] && [ -f "${ULTRA_SEND_DIR:-harness/a8/fx165/send}/linsend_circuit.h" ]; then
    # a send directory with linsend_circuit.h is the LINEAR send (src/fxcard/send): one file, no solver
    S=$ULTRA_SEND_DIR
    "${CC:-gcc}" $CF -I$S -Isrc/fxcard/ret -Isrc/vendor/nlcore -c $S/send_nodal_api.c -o "$O/snd_send_nodal_api.o"
    OBJS="$OBJS $O/snd_send_nodal_api.o"
elif [ "${ULTRA_SEND:-0}" = 1 ]; then
    S=${ULTRA_SEND_DIR:-harness/a8/fx165/send}
    SCF="$CF -DNL_NODAL=1 -I$S -Isrc/vendor/nlcore"
    for f in src/vendor/nlcore/nl_nodal.c src/vendor/nlcore/nl_core.c; do "${CC:-gcc}" $SCF -c "$f" -o "$O/sraw_$(basename $f .c).o"; done
    SREN=$(nm --defined-only -g "$O"/sraw_*.o | awk 'NF==3 {print $3}' | sort -u | sed 's/.*/-D&=cs_&/' | tr '\n' ' ')
    for f in src/vendor/nlcore/nl_nodal.c src/vendor/nlcore/nl_core.c $S/send_nodal_api.c; do
        b=$(basename "$f" .c); "${CC:-gcc}" $SCF $SREN -c "$f" -o "$O/snd_$b.o"; OBJS="$OBJS $O/snd_$b.o"
    done
fi
"${CC:-gcc}" $CF -Isrc/ultra -Isrc/dsp -Isrc/vendor/nlcore -c src/ultra/bn_ultra.c -o "$O/ultra_dispatch.o"
for v in 100uk 100us 50uk 50us 100ri; do
    REN=""; for s in $SYMS; do REN="$REN -D$s=bn_prepig_${v}_$s"; done
    # the FX loop's send systems: the same solver with their own device (the preamp and PI plus the EL34 / 6550 grids)
    case $v in *us) DEV="-DNL_MNA_DEVICE=bn_prepig_device_6550 -DNL_MNA_DEVICE_CURRENTS=bn_prepig_currents_6550";;
               *)   DEV="-DNL_MNA_DEVICE=bn_prepig_device -DNL_MNA_DEVICE_CURRENTS=bn_prepig_currents";; esac
    INC="-Isrc/prepig/gen_$v -Isrc/vendor/nlcore -Isrc/prepig -Isrc/dsp"
    "${CC:-gcc}" $CF $REN $DEV $INC -c src/vendor/nlcore/nl_mna.c -o "$O/prepig_${v}_mna.o"
    "${CC:-gcc}" $CF $REN -DBN_GV=$v $INC -c src/prepig/bn_prepig_engine.c -o "$O/prepig_${v}_wrap.o"
    OBJS="$OBJS $O/prepig_${v}_mna.o $O/prepig_${v}_wrap.o"
done
"${CC:-gcc}" $CF -Isrc/prepig -Isrc/dsp -c src/prepig/bn_prepig.c -o "$O/prepig_dispatch.o"
"${CC:-gcc}" $CF -Isrc/dsp -c src/prepig/bn_prepig_device.c -o "$O/prepig_device.o"
for f in 12ax7 el34 6550; do   # the valve laws (once: their tables are process-wide)
    "${CC:-gcc}" $CF -Isrc/vendor/nlcore -c src/vendor/nlcore/nl_valve_$f.c -o "$O/nl_valve_$f.o"
    OBJS="$OBJS $O/nl_valve_$f.o"
done
echo $OBJS "$O/ultra_dispatch.o" "$O/prepig_dispatch.o" "$O/prepig_device.o"
