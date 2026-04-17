PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: install train compare-eval-loss clean

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

train:
	$(ACTIVATE) && python src/train_slm.py --config configs/default_training.yaml

compare-eval-loss:
	$(ACTIVATE) && python src/compare_eval_loss.py --config configs/default_training.yaml

clean:
	rm -rf $(VENV) artifacts
