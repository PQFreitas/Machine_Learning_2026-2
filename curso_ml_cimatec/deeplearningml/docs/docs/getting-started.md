Getting started
===============

Install the project dependencies and download/validate the heart disease dataset:

```bash
uv sync
make data
```

The original CSV is saved to `data/raw/heart.csv` and a validated copy to
`data/processed/heart.csv`. Train and save the Keras model with:

```bash
make train
```

The trained model is saved to `models/heart_classifier.keras`. To make
predictions, place a CSV containing the 13 input columns (without `target`) at
`data/processed/test_features.csv`, then run:

```bash
make predict
```

The resulting probabilities are written to
`data/processed/test_predictions.csv`. Categorical lookups and numeric
normalization are part of the saved model, so inference uses the same
preprocessing learned from the training data.
