import pandas as pd
df = pd.read_csv("../diabetes_dev.csv")
df.drop(columns=["readmitted"]).head(500).to_csv("sample_features.csv", index=False)