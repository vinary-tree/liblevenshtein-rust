# docs/diagrams/Makefile — incremental diagram rendering.
#
# `render.sh` is the source of truth for HOW each diagram is rendered; this
# Makefile only provides fast, dependency-aware *incremental* rebuilds (rerender
# a single SVG when its source is newer) plus convenience targets. CI should call
# `make check`, which delegates to `render.sh --check`.
#
# Targets:
#   make            # render every out-of-date SVG (incremental)
#   make all        # force-render every diagram (delegates to render.sh)
#   make check      # fail if any committed SVG is out of date with its source
#   make list       # list every source and its target SVG
#   make <dir>      # render one subsystem dir, e.g. `make automata`
#
SHELL := /usr/bin/env bash
RENDER := ./render.sh

# Every diagram source, and the SVG each one maps to.
SRCS := $(shell find . -type f \( -name '*.puml' -o -name '*.dsl' -o -name '*.d2' \
                                  -o -name '*.dot' -o -name '*.mmd' -o -name '*.pikchr' \
                                  -o -name '*.asy' \) | sort)
SVGS := $(foreach s,$(SRCS),$(basename $(s)).svg)

.PHONY: all check list clean help
.DEFAULT_GOAL := incremental

# Incremental: rebuild only SVGs whose source is newer (timestamp-based).
.PHONY: incremental
incremental: $(SVGS)

# Pattern rule: an SVG depends on its like-named source of any supported type.
%.svg: %.puml   ; $(RENDER) "$<"
%.svg: %.dsl    ; $(RENDER) "$<"
%.svg: %.d2     ; $(RENDER) "$<"
%.svg: %.dot    ; $(RENDER) "$<"
%.svg: %.mmd    ; $(RENDER) "$<"
%.svg: %.pikchr ; $(RENDER) "$<"
%.svg: %.asy    ; $(RENDER) "$<"

all:   ; $(RENDER)
check: ; $(RENDER) --check
list:  ; $(RENDER) --list

# Render one subsystem directory (e.g. `make automata`).
architectures automata traversal dictionary-structures concurrency distance \
phonetic time-series contextual cache serialization grep bindings _legend:
	$(RENDER) "/$@/"

help:
	@$(RENDER) --help
