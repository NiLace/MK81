# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) The NL Sounds contributors
# MK 81 — build the LV2 bundle (Linux, x86-64).
#
#   make            build/mk81.lv2 (`make plugin` does the same)
#   make install    to ~/.lv2 (atomic: safe while a host has the plugin loaded)
#   make test       the bundle exports its entry points and carries every file it loads
#   make clean
#
# Needs a C11 compiler, python3, and the development headers of LV2, Cairo, X11, Xcursor, Xrandr and FreeType.
# Generic x86-64: no AVX2 or other newer instruction is required, so the bundle runs on any 64-bit Intel or AMD CPU.
# The power stage's weights live in nn/*.bnn2 (the form to edit them in); the two headers under build/gen are made
# from them at build time.

CC      ?= cc
CFLAGS  := -O3 -funroll-loops -fPIC -Wall -Wextra -std=c11
LDFLAGS := -shared -lm
BUILD   := build
GEN     := $(BUILD)/gen
BUNDLE  := $(BUILD)/mk81.lv2
INSTALL_DIR ?= $(HOME)/.lv2

DSP_SRC := src/dsp/bn_dk.c src/dsp/bn_tube_own.c src/dsp/bn_oversample.c src/dsp/bn_psu.c src/dsp/bn_nn_lstm.c \
           src/dsp/bn_nn_rate.c src/dsp/bn_fx_comp.c
PUGL_SRC := deps/pugl/src/common.c deps/pugl/src/internal.c deps/pugl/src/x11.c deps/pugl/src/x11_cairo.c
GUI_CF   := -Ideps/pugl/include $(shell pkg-config --cflags cairo x11 freetype2)
GUI_LIBS := $(shell pkg-config --libs cairo x11 freetype2) -lXcursor -lXrandr -lm
FONTS    := Doto-Round-Bold.ttf OFL-Doto.txt Oswald.ttf OFL-Oswald.txt

.PHONY: all plugin install test clean
plugin: all
all: $(BUNDLE)/mk81.so $(BUNDLE)/mk81_ui.so $(BUNDLE)/manifest.ttl
	@cp lv2/mk81.ttl $(BUNDLE)/
	@rm -rf $(BUNDLE)/fonts $(BUNDLE)/photo; mkdir -p $(BUNDLE)/fonts $(BUNDLE)/photo
	@cp $(FONTS:%=gui/fonts/%) $(BUNDLE)/fonts/
	@cp gui/photo/*.png $(BUNDLE)/photo/
	@echo "bundle ready: $(BUNDLE)"

$(GEN)/bn_nn_weights.h: tools/gen_nn_blob.py $(wildcard nn/*.bnn2)
	@mkdir -p $(GEN)
	python3 tools/gen_nn_blob.py $@

$(GEN)/bn_nn_rest.h: tools/gen_nn_rest.c $(GEN)/bn_nn_weights.h src/dsp/bn_nn_lstm.c src/dsp/bn_nn_lstm.h
	$(CC) -O3 -funroll-loops -std=c11 -w -I src/dsp -I $(GEN) -o $(BUILD)/gen_rest $< src/dsp/bn_nn_lstm.c -lm
	./$(BUILD)/gen_rest $@

$(BUNDLE)/mk81.so: src/mk81.c $(DSP_SRC) $(wildcard src/dsp/*.h) $(GEN)/bn_nn_weights.h $(GEN)/bn_nn_rest.h
	@mkdir -p $(BUNDLE)
	$(CC) $(CFLAGS) -fvisibility=hidden -I src/dsp -I $(GEN) src/mk81.c $(DSP_SRC) $(LDFLAGS) -o $@

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
