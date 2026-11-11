# MK 81 — build the LV2 bundle (Linux, x86-64).
#
#   make            build/mk81.lv2 (`make plugin` does the same)
#   make install    to ~/.lv2 (atomic: safe while a host has the plugin loaded)
#   make test       the bundle exports its entry points and carries every file it loads
#   make clean
#
# Needs gcc (C11, with -flto), and the development headers of LV2, Cairo, X11, Xcursor, Xrandr and FreeType.
# Generic x86-64: no AVX2 or other newer instruction is required, so the bundle runs on any 64-bit Intel or AMD CPU.
#
# The amplifier is one circuit solved sample by sample: each version's nodal system (src/ultra/gen_<version>/), the
# effects loop's send systems (src/prepig/gen_<version>/) and the loop's card (src/ultra/genfx*_<version>/) are each
# compiled with the nodal solver (src/vendor/nlcore/) into their own objects by tools/build_ultra_objs.sh, then
# linked with the plugin (src/mk81.c) and its stages (src/dsp/).

CC      ?= cc
CFLAGS  := -O3 -funroll-loops -fPIC -Wall -Wextra -std=c11
LDFLAGS := -shared -lm
BUILD   := build
BUNDLE  := $(BUILD)/mk81.lv2
INSTALL_DIR ?= $(HOME)/.lv2

# the solver's settings and the valves' tables, as the plugin is built and measured
ENGINE_DEFS := -DBN_PT_TAIL_TAB -DBN_6550_TAB -DBN_GRID_TAB -DMKS_THETA_RT -DBN_ULTRA_THETA=0.52 -DNL_VALVE_PART_FALLBACK=1 \
               -DMKS_RTOL=3e-4 -DMKS_RETRY_CAP=200.0 -DMKS_CYCLE_ABORT -DMKS_PLATE_FTB=0.25 -DMKS_WALK_RECAP=101.0 \
               -DMKS_FXZ_ITOL=1e-9 -DMKS_FXZ_TAB -DBN_CARD_NEWTON=2 -DBN_FXZ_WARM -DBN_CARD_LATE -DBN_CARD_MIDFADE \
               -DBN_SEND_TWIN -DBN_FX_PAIR=4 -DBN_CARD_LS0 -DBN_FXC_CONFIRM -DBN_FXC_WIN_C=0.1 -DBN_FXC_WIN_F=0.1 \
               -DBN_FXC_KEEP -DMKS_GAM_COLS -DMKS_STOP_UNCAPPED -DMKS_RESCUE=3 -DMKS_SAMPLE_BUDGET=1200 -flto
# the effects loop's card: its return and its send, each one exact linear block with its zener pair
CARD_RET  := src/fxcard/ret
CARD_SEND := src/fxcard/send
CARD_ENV  := ULTRA_RET=1 ULTRA_SEND=1 ULTRA_RET_DIR=$(CARD_RET) ULTRA_RET_CORE=src/vendor/nlcore ULTRA_SEND_DIR=$(CARD_SEND)
OBJS_CF   := $(CFLAGS) $(ENGINE_DEFS) -fvisibility=hidden
WRAP_CF   := $(CFLAGS) $(ENGINE_DEFS) -fvisibility=hidden -DBN_ULTRA -I src/ultra -I src/dsp \
             -DBN_FX_CARD -I $(CARD_RET) -I $(CARD_SEND)
# the link order is the one the plugin is built and measured with
DSP_SRC   := src/mk81.c src/dsp/bn_tube_own.c src/dsp/bn_pentode_own.c src/dsp/bn_oversample.c src/dsp/bn_psu.c \
             src/dsp/bn_fx_comp.c src/dsp/bn_fx_pair.c
DSP_TAIL  := src/dsp/bn_fullamp_own.c src/dsp/bn_pentode_6550_own.c src/ultra/bn_card_map.c
ENGINE_SRC := $(wildcard src/ultra/*.c src/ultra/*.h src/ultra/gen*/*.h src/prepig/*.c src/prepig/*.h src/prepig/gen_*/*.h \
                         src/vendor/nlcore/* src/dsp/*.h $(CARD_RET)/* $(CARD_SEND)/* src/fxcard/*.h)
OBJ_DIR   := $(BUILD)/engine_obj

PUGL_SRC := deps/pugl/src/common.c deps/pugl/src/internal.c deps/pugl/src/x11.c deps/pugl/src/x11_cairo.c
GUI_CF   := -Ideps/pugl/include $(shell pkg-config --cflags cairo x11 freetype2)
GUI_LIBS := $(shell pkg-config --libs cairo x11 freetype2) -lXcursor -lXrandr -lm -pthread
FONTS    := Doto-Round-Bold.ttf OFL-Doto.txt Oswald.ttf OFL-Oswald.txt

.PHONY: all plugin install test clean
plugin: all
all: $(BUNDLE)/mk81.so $(BUNDLE)/mk81_ui.so $(BUNDLE)/manifest.ttl
	@cp lv2/mk81.ttl $(BUNDLE)/
	@rm -rf $(BUNDLE)/fonts $(BUNDLE)/photo; mkdir -p $(BUNDLE)/fonts $(BUNDLE)/photo
	@cp $(FONTS:%=gui/fonts/%) $(BUNDLE)/fonts/
	@cp gui/photo/*.png $(BUNDLE)/photo/
	@echo "bundle ready: $(BUNDLE)"

# the engine's objects: every system with its own copy of the solver, its symbols renamed so all link into one binary
$(OBJ_DIR)/objs.txt: tools/build_ultra_objs.sh $(ENGINE_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CARD_ENV) sh tools/build_ultra_objs.sh $(OBJ_DIR) $(OBJS_CF) > $@.tmp && mv $@.tmp $@

$(BUNDLE)/mk81.so: $(DSP_SRC) $(DSP_TAIL) $(ENGINE_SRC) $(OBJ_DIR)/objs.txt
	@mkdir -p $(BUNDLE)
	$(CC) $(WRAP_CF) $(DSP_SRC) $$(cat $(OBJ_DIR)/objs.txt) $(DSP_TAIL) $(LDFLAGS) -flto -o $@

$(BUNDLE)/mk81_ui.so: gui/ui.c gui/ui_fonts.c $(wildcard gui/*.h) $(PUGL_SRC)
	@mkdir -p $(BUNDLE)
	$(CC) -O2 -fPIC -fvisibility=hidden -Wall -Wextra $(GUI_CF) -o $@ gui/ui.c gui/ui_fonts.c $(PUGL_SRC) -shared $(GUI_LIBS)

$(BUNDLE)/manifest.ttl: lv2/manifest.ttl.in
	@mkdir -p $(BUNDLE)
	sed 's/@LIB_EXT@/.so/' $< > $@

# Atomic install. A plain `cp` rewrites the .so in place and can crash a host that has it memory-mapped.
install: all
	@mkdir -p $(INSTALL_DIR)
	@rm -rf $(INSTALL_DIR)/mk81.lv2.tmp $(INSTALL_DIR)/mk81.lv2.old
	@cp -r $(BUNDLE) $(INSTALL_DIR)/mk81.lv2.tmp
	@if [ -d $(INSTALL_DIR)/mk81.lv2 ]; then mv $(INSTALL_DIR)/mk81.lv2 $(INSTALL_DIR)/mk81.lv2.old; fi
	@mv $(INSTALL_DIR)/mk81.lv2.tmp $(INSTALL_DIR)/mk81.lv2
	@rm -rf $(INSTALL_DIR)/mk81.lv2.old
	@echo "installed in $(INSTALL_DIR)/mk81.lv2"

# What a build can get wrong without an error: the entry points hidden by -fvisibility, a runtime file left out.
test: all
	@fail=0; \
	nm -D --defined-only $(BUNDLE)/mk81.so | grep -q ' lv2_descriptor$$' || { echo "FAIL: mk81.so does not export lv2_descriptor"; fail=1; }; \
	nm -D --defined-only $(BUNDLE)/mk81_ui.so | grep -q ' lv2ui_descriptor$$' || { echo "FAIL: mk81_ui.so does not export lv2ui_descriptor"; fail=1; }; \
	for f in manifest.ttl mk81.ttl $(FONTS:%=fonts/%); do [ -f $(BUNDLE)/$$f ] || { echo "FAIL: missing $$f"; fail=1; }; done; \
	[ $$(ls $(BUNDLE)/photo/*.png | wc -l) -gt 0 ] || { echo "FAIL: no photos in the bundle"; fail=1; }; \
	[ $$fail -eq 0 ] && echo "bundle OK: entry points exported, all runtime files present"; exit $$fail

clean:
	rm -rf $(BUILD)
