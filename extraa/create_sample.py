import pandas as pd
df = pd.read_csv("C:/Users/User/OneDrive/Desktop/ds_assignment2/diabetes_dev.csv")
df.drop(columns=["readmitted"]).head(500).to_csv("sample_features.csv", index=False)