SHELL = /bin/sh
########################################################################################
# Makefile for the installation of DAMASK
########################################################################################
.PHONY: all
all: spectral marc processing

.PHONY: spectral
spectral:
	$(MAKE) DAMASK_spectral.exe -C code

.PHONY: FEM
FEM:
	$(MAKE) DAMASK_FEM.exe -C code

.PHONY: marc
marc:
	@./installation/mods_MarcMentat/apply_DAMASK_modifications.sh ${MAKEFLAGS}

.PHONY: tidy
tidy:
	@$(MAKE) tidy -C code >/dev/null

.PHONY: clean
clean:
	@$(MAKE) cleanDAMASK -C code >/dev/null

.PHONY: install
install:
	@./installation/symlink_Code.py ${MAKEFLAGS}
	@./installation/symlink_Processing.py ${MAKEFLAGS}

