import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from etl import transform

def test_filters_zero_peak_players():
    d = {
        "col1": [1, 2],
        "24h_Peak_Players": [100, 0],
        "Price_USD": [9.99, 4.99],
        "Review_Score_Pct": [85, 70],
        "Primary_Genre": ["Action", "Indie"],
    }
    df = pd.DataFrame(data=d)
    transformed = transform(df)
    assert len(transformed) == 1

def test_NaN_Price():
    d = {
        "col1": [1, 2],
        "24h_Peak_Players": [100, 20],
        "Price_USD": [9.99, np.nan],
        "Review_Score_Pct": [85, 70],
        "Primary_Genre": ["Action", "Indie"],
    }
    df = pd.DataFrame(data=d)
    transformed = transform(df)
    assert transformed["Price_USD"].iloc[1] == 9.99
