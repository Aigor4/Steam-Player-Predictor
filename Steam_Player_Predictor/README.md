# Steam Player Predictor

A small end-to-end ML project that predicts a Steam game's 24-hour peak player count from its price, review score, and genre. Built to practice a full pipeline — data cleaning, model training/comparison, evaluation, and a simple interactive UI — rather than a one-off notebook.

## Overview

- **Input:** price (USD), review score (%), primary genre
- **Output:** predicted 24h peak player count
- **Data:** ~1000 Steam games, 12 raw columns (`data/steam.csv`)
- **Models compared:** Linear Regression, Decision Tree, Random Forest (best one by R² is kept)

## Project structure

```
steam-player-predictor/
├── app.py                  # Streamlit UI for interactive predictions
├── data/
│   ├── steam.csv            # raw dataset
│   └── Clean_Steam.csv      # output of the ETL step
├── models/
│   ├── model.pkl             # best trained model
│   └── feature_columns.pkl   # feature column order, needed to build prediction rows
├── src/
│   ├── etl.py                # extract / transform / load
│   ├── train.py               # trains & compares 3 models, saves the best one
│   ├── evaluate.py            # plots: R² comparison, actual vs predicted, feature importance
│   └── predict.py             # CLI script for a single prediction
└── tests/
    └── test_etl.py            # unit tests for the transform step
```

## Setup

```bash
pip install -r requirements.txt
```

*(or manually: `pandas`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `pytest`)*

## Usage

Run the pipeline from `src/`:

```bash
cd src
python etl.py        # cleans data/steam.csv -> data/Clean_Steam.csv
python train.py       # trains 3 models, saves the best to models/model.pkl
python evaluate.py    # generates comparison plots
python predict.py     # CLI: enter price/review score/genre, get a prediction
```

Or launch the interactive app from the project root:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest tests/
```

## Model performance

| Model              | R²     | MSE           |
|--------------------|--------|---------------|
| Linear Regression  | -0.14  | 233,087,926   |
| Decision Tree      | -2.17  | 650,679,678   |
| Random Forest      | -0.87  | 384,666,939   |

*(Best model — highest R² — is the one persisted to `models/model.pkl`.)*

## Limitations & possible improvements

- All three models currently score **below** a naive mean-baseline (negative R²) — price/review score/genre alone aren't strong predictors of peak concurrent players. The dataset likely has much stronger signals that aren't used yet, e.g. `Total_Reviews` or `Estimated_Owners`.
- Peak player counts are heavily right-skewed (a handful of huge titles dominate); a log-transform of the target could help.
- No hyperparameter tuning has been done on the tree-based models yet.
- Genre is currently a single primary category; `All_Tags` has richer multi-label info that isn't used.

## Tech stack

Python, pandas, scikit-learn, Streamlit, matplotlib, pytest
