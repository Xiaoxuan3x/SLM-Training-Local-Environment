PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: install create-test-split train compare-eval-loss export-merged run-model eval-ragas eval-ragas-test clean

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

create-test-split:
	$(ACTIVATE) && python src/create_test_split.py

train:
	$(ACTIVATE) && python src/train_slm.py --config configs/default_training.yaml

compare-eval-loss:
	$(ACTIVATE) && python src/compare_eval_loss.py --config configs/default_training.yaml

export-merged:
	$(ACTIVATE) && python src/export_merged_model.py --config configs/default_training.yaml

run-model:
	$(ACTIVATE) && python src/run_model.py --max-new-tokens 800 --use-classifier --allow-downloads

eval-ragas:
	$(ACTIVATE) && python src/evaluate_ragas.py --max-samples 50

eval-ragas-test:
	$(ACTIVATE) && python src/evaluate_ragas.py --data data/test_dataset.jsonl --max-samples 50

clean:
	rm -rf $(VENV) artifacts
