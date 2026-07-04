CONDA ?= conda
AUDIT_ENV ?= qnlp
LAMBEQ_ENV ?= qnlp-lambeq

.PHONY: help env-audit env-lambeq lock install-dev test lint train audit figures paper reproduce clean

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

figures:
	$(CONDA) run -n $(AUDIT_ENV) python figures/scripts/fig_chi_curves.py
	$(CONDA) run -n $(AUDIT_ENV) python figures/scripts/fig_cert_table.py

paper:
	cd paper && latexmk -pdf main.tex

reproduce: figures paper
