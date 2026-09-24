import pandas as pd

df = pd.read_csv("dataset/metadata/HAM10000_metadata.csv")
print(len(df))
print(df["dx"].value_counts())

import matplotlib.pyplot as plt

df["dx"].value_counts().plot(
    kind="bar"
)

plt.show()