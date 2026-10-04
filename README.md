# spotter-freight-rate-ml-assessment
Machine Learning solution for Spotter Freight Rate Prediction assessment.

# Spotter Freight Rate Prediction

Machine Learning solution for the Spotter Machine Learning Engineer assessment.

## Overview

The goal is to predict freight `posted_rate` for the validation loads using the provided development dataset.

## Project Files

```text
README.md
requirements.txt
score.py
train_model.py
validation_predictions.csv
```

- `train_model.py` — main machine-learning training, feature engineering, validation, and prediction code.
- `score.py` — provided validation script used to check the final prediction files.
- `requirements.txt` — Python dependencies.
- `validation_predictions.csv` — final predictions for the 12,000 validation loads.

## Setup

Python 3.10+ is recommended.

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Model

Place the assessment-provided data files in a local `data/` directory:

```text
data/
├── train_test.csv
├── validation.csv
└── december_chart_inputs.csv
```

Then run:

```bash
python train_model.py --data-dir data --output-dir .
```

This generates:

```text
validation_predictions.csv
```

## Validate the Predictions

Run the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

A successful validation confirms that the final prediction file contains the required 12,000 validation predictions with the expected IDs and positive predicted rates.

## Final Prediction Format

`validation_predictions.csv` contains exactly:

```text
load_id,predicted_rate
```

with one prediction for each of the 12,000 validation loads.

