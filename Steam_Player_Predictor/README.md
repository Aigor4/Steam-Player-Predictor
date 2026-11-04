# Steam Player Predictor

A small end-to-end ML project that predicts a Steam game's 24-hour peak player count from its price, review score, number of reviews, and genre. Built to practice a full pipeline — data cleaning, model training/comparison, evaluation, and a simple interactive UI — rather than a one-off notebook.

## Overview

- **Input:** price (USD), review score (%), number of reviews, primary genre
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
python predict.py     # CLI: enter price/review score/number of reviews/genre, get a prediction
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

The target is trained as `log1p(24h_Peak_Players)`. R² is shown both on that log scale and after converting predictions back to player counts.

| Model              | R² (log) | R² (players) | MSE (log) |
|--------------------|----------|--------------|-----------|
| Linear Regression  | 0.28     | -0.07        | 2.71      |
| Decision Tree      | 0.44     | -0.82        | 2.10      |
| Random Forest      | 0.69     | 0.12         | 1.17      |

*(Best model — highest R² on the log scale — is the one persisted to `models/model.pkl`.)*

## Limitations & possible improvements

- The number of reviews is only known after release, so the model describes existing games and can't forecast an unreleased title. Without it, R² stays around 0 even with the log target: price, review score and genre carry almost no signal.
- R² in player counts stays low because a handful of huge titles dominate the squared error.
- `Total_Reviews` looks wrong for some of the biggest games (Dota 2: 14,369, Apex Legends: 1,577, Marvel Rivals: 0). These rows are kept as they are.
- `Estimated_Owners` is exactly `Total_Reviews * 30`, so it adds no new information and isn't used.
- No hyperparameter tuning has been done on the tree-based models yet.
- Genre is currently a single primary category; `All_Tags` has richer multi-label info that isn't used.

## Tech stack

Python, pandas, scikit-learn, Streamlit, matplotlib, pytest
