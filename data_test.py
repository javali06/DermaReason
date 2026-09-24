import pandas as pd

df = pd.read_csv(
    r"C:\Projects\DermaReason\dataset\metadata\train.csv"
)

print(df["label"].value_counts())