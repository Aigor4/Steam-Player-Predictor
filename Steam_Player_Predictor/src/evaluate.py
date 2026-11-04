import numpy as np
from etl import extract, transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def main():
    df = extract("../data/steam.csv")
    clean = transform(df)
    clean["Log_Reviews"] = np.log1p(clean["Total_Reviews"])

    genre_cols = [col for col in clean.columns if col.startswith('Primary_Genre_')]
    Features = ["Price_USD", "Review_Score_Pct", "Log_Reviews"] + genre_cols
    X = clean[Features]
    y = np.log1p(clean["24h_Peak_Players"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_r2 = -float("inf")
    results = {}
    model_names = []
    r2_scores = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"r2" : r2, "mse" : mean_squared_error(y_test, y_pred), "model" : model_name}

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

        model_names.append(model_name)
        r2_scores.append(r2)

    plt.bar(model_names, r2_scores)
    plt.xlabel("Model")
    plt.ylabel("R2 (log scale)")
    plt.title("Which model is the best")
    plt.show()

    x1 = y_test
    arr = np.array(x1)
    y_pred = best_model.predict(X_test)
    plt.scatter(arr, y_pred, color="blue", alpha=0.5)
    lims = [min(arr.min(), y_pred.min()), max(arr.max(), y_pred.max())]
    plt.plot(lims, lims, color="red", label="Perfect prediction (y=x)")
    plt.xlabel("Real Values (log)")
    plt.ylabel("Prediction Values (log)")
    plt.title("Real vs Prediction Values")
    plt.legend()
    plt.show()

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = best_model.coef_
    importance_series = pd.Series(importances, index=Features)
    importance_series = importance_series.sort_values(ascending=False)
    plt.barh(importance_series.index, importance_series.values)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance")
    plt.show()

if __name__ == "__main__": main()
