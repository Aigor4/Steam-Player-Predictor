import joblib
import pandas as pd

model = joblib.load("../models/model.pkl")
features_columns = joblib.load("../models/feature_columns.pkl")

cena = float(input("podaj cene gry: "))
ocena = float(input("podaj ocene gry: "))
gatunek = input("podaj gatunek gry: ")

row = {col: 0 for col in features_columns}

row["Price_USD"] = cena
row["Review_Score_Pct"] = ocena
kolumna_gatunku = f"Primary_Genre_{gatunek}"

if kolumna_gatunku not in row:
    print("Nie ma takiego gatunku gry")
    exit()
else:
    row[kolumna_gatunku] = 1

df = pd.DataFrame([row])
df = df[features_columns]
wynik = model.predict(df)
print(f"przewidywana liczba graczy: {wynik[0]:.2f}")
