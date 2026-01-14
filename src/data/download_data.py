import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT_PATH = os.path.join("data", "raw", "breast_cancer.csv")

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    ds = fetch_ucirepo(id=17)

    X = ds.data.features
    y = ds.data.targets

    # y souvent contient "Diagnosis" (B/M)
    if isinstance(y, pd.DataFrame):
        target_col = y.columns[0]
        y = y[target_col]
    else:
        target_col = "Diagnosis"
        y = pd.Series(y, name=target_col)

    df = X.copy()
    df[target_col] = y
    df.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH)
    print("Columns:", df.columns.tolist()[:5], "...")

if __name__ == "__main__":
    main()
