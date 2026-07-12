import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
features_columns = joblib.load("models/feature_columns.pkl")

st.title("Steam Player Predictor")

price = st.number_input("Price", min_value=0, value=0)
review = st.number_input("Review", min_value=0, max_value=100, value=0)
genres = [col.replace("Primary_Genre_", "") for col in features_columns if col.startswith('Primary_Genre_') ]
genre = st.selectbox("Genre", genres)
przycisk = st.button("Predict")
if przycisk:
    row = {col: 0 for col in features_columns}
    row["Price_USD"] = price
    row["Review_Score_Pct"] = review
    kolumna_gatunku = f"Primary_Genre_{genre}"
    if kolumna_gatunku not in row:
        print("Nie ma takiego gatunku gry")
        exit()
    else:
        row[kolumna_gatunku] = 1
    df = pd.DataFrame([row])
    df = df[features_columns]
    wynik = model.predict(df)
    st.write(f"Predicted number of players: {wynik[0]:.2f}")

st.caption("This is only demonstrating how app working")