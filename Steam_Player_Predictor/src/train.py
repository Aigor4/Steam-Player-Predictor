import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from etl import extract, transform
import numpy as np

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

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_players = r2_score(np.expm1(y_test), np.expm1(y_pred))
    print(f"{model_name} r2_score: {r2:.2f} r2_players: {r2_players:.2f} mean_squared_error: {mean_squared_error(y_test, y_pred):.2f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

joblib.dump(best_model, "../models/model.pkl")
joblib.dump(Features, "../models/feature_columns.pkl")

