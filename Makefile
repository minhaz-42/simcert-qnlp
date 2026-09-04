CONDA ?= conda
AUDIT_ENV ?= qnlp
LAMBEQ_ENV ?= qnlp-lambeq

.PHONY: help env-audit env-lambeq lock install-dev test lint train audit demo figures tables paper reproduce clean

help:
	@echo "SimCert make targets:"
	@echo "  env-audit    create the audit conda env (qnlp, Py3.12)"
	@echo "  env-lambeq   create the lambeq producer conda env (qnlp-lambeq, Py3.11)"
	@echo "  lock         freeze exact versions into envs/*.lock.txt"
	@echo "  install-dev  pip install -e . into the audit env"
	@echo "  test         run the pytest suite in the audit env"
	@echo "  train        train the model zoo (Hydra multirun)"
	@echo "  audit        run the simulability audit over trained models"
	@echo "  figures      regenerate all figures from committed results/"
	@echo "  paper        build the PDF"
	@echo "  reproduce    figures + paper from committed results (no training)"

env-audit:
	cd envs && $(CONDA) env create -f environment-audit.yml

env-lambeq:
	cd envs && $(CONDA) env create -f environment-lambeq.yml

lock:
	$(CONDA) run -n $(AUDIT_ENV)  pip freeze > envs/requirements-audit.lock.txt
	$(CONDA) run -n $(LAMBEQ_ENV) pip freeze > envs/requirements-lambeq.lock.txt

install-dev:
	$(CONDA) run -n $(AUDIT_ENV) pip install -e .

test:
	$(CONDA) run -n $(AUDIT_ENV) pytest

lint:
	$(CONDA) run -n $(AUDIT_ENV) ruff check src tests

MODEL ?= vqc_text
DATASET ?= mc
SEED ?= 1

train:
	$(CONDA) run -n $(AUDIT_ENV) python -m simcert.runner mode=train model=$(MODEL) dataset=$(DATASET) seed=$(SEED)

audit:
	$(CONDA) run -n $(AUDIT_ENV) python -m simcert.runner mode=audit model=$(MODEL) dataset=$(DATASET) seed=$(SEED)

demo:
	$(CONDA) run -n $(AUDIT_ENV) python -m simcert.runner mode=both model=$(MODEL) dataset=$(DATASET) seed=$(SEED)

# Every figure and table is derived from results/, so a partial regeneration silently
# ships a stale one: repro/scaling/witnesses all move when a model gains seeds, and
# regenerating only chi_curves is how a paper ends up disagreeing with its own data.
FIG_SCRIPTS := fig_chi_curves fig_control fig_objects fig_repro fig_scaling fig_witnesses
TABLE_SCRIPTS := make_tables make_repro_table

# The DisCoCat string diagrams in figures/objects/ are rendered by
# scripts/render_diagram_objects.py in the LAMBEQ env (the two envs cannot coexist), and
# are committed as images. Run that separately if a diagram needs regenerating.
figures:
	@for f in $(FIG_SCRIPTS); do \
		echo "--> $$f"; \
		$(CONDA) run -n $(AUDIT_ENV) python figures/scripts/$$f.py || exit 1; \
	done

tables:
	@for t in $(TABLE_SCRIPTS); do \
		echo "--> $$t"; \
		$(CONDA) run -n $(AUDIT_ENV) python paper/scripts/$$t.py || exit 1; \
	done

# build_paper.sh, not a bare tectonic call: it builds the anonymous submission and the
# de-anonymised preprint from one source and fails if the blind PDF carries the author
# name. A plain `tectonic main.tex` skips that gate.
paper: figures tables
	scripts/build_paper.sh

reproduce: figures tables paper   # regenerate every figure + table + both PDFs from results/
