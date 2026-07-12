import pandas as pd


def extract(filename):
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)
    return df

def transform(df):
    df = df[df["24h_Peak_Players"] > 0]
    df = df.fillna({
        "Price_USD": df["Price_USD"].median(),
        "Review_Score_Pct": df["Review_Score_Pct"].median(),
        "Primary_Genre": "Unknown",
    })
    df = pd.get_dummies(df, columns=["Primary_Genre"])
    return df

def load(filename, file_path):
    filename.to_csv(file_path, index=False)
    return filename

def main():
    df = extract("../data/steam.csv")
    transformed = transform(df)
    file_path = "../data/Clean_Steam.csv"
    clean = load(transformed, file_path)
    print(clean)

if __name__ == "__main__":
    main()

