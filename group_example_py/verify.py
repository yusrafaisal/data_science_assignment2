import pandas as pd
df = pd.read_csv("preds.csv")
print(df.head())
print(df.shape)           # should be (500, 2)
print(df.isnull().sum())  # should be 0 nulls
print(df["prob_early_readmit"].between(0,1).all())  # should be True