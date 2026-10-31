import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("C:/Users/User/OneDrive/Desktop/Package/Package/diabetes_dev.csv")
print("Shape:", df.shape)
print(df.head(3))


# ── 2. Basic info ─────────────────────────────────────────────────────────────
print("\n--- Column types ---")
print(df.dtypes)

print("\n--- Missing values (? treated as missing) ---")
# The dataset uses "?" for missing, not NaN
# Replace ? with NaN so pandas counts them properly
df.replace("?", np.nan, inplace=True)
print(df.isnull().sum()[df.isnull().sum() > 0])


# ── 3. Target distribution ────────────────────────────────────────────────────
print("\n--- Target: readmitted ---")
print(df["readmitted"].value_counts())
print(df["readmitted"].value_counts(normalize=True).round(3))

# For our task: <30 = 1, everything else = 0
df["early_readmit"] = (df["readmitted"] == "<30").astype(int)
print("\nEarly readmit (class 1) rate:", df["early_readmit"].mean().round(3))

df["readmitted"].value_counts().plot(kind="bar", color=["steelblue", "salmon", "gray"])
plt.title("Readmission distribution")
plt.xlabel("readmitted")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.show()


# ── 4. Patient duplicates ─────────────────────────────────────────────────────
print("\n--- Patient duplicates ---")
print("Total rows:", len(df))
print("Unique patients:", df["patient_nbr"].nunique())
print("Patients with >1 encounter:", (df["patient_nbr"].value_counts() > 1).sum())
# This is WHY we need patient-grouped split


# ── 5. Columns with too many missing values ───────────────────────────────────
print("\n--- Columns with >40% missing ---")
missing_pct = df.isnull().mean()
print(missing_pct[missing_pct > 0.4])
# weight and payer_code are usually very sparse — likely to drop


# ── 6. Categorical columns overview ──────────────────────────────────────────
cat_cols = ["race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id",
            "max_glu_serum", "A1Cresult"]

print("\n--- Unique values in categorical columns ---")
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique | sample: {df[col].dropna().unique()[:5]}")


# ── 7. Diagnosis columns ──────────────────────────────────────────────────────
print("\n--- Diagnosis columns (diag_1, diag_2, diag_3) ---")
print("diag_1 unique codes:", df["diag_1"].nunique())
print("diag_2 unique codes:", df["diag_2"].nunique())
print("diag_3 unique codes:", df["diag_3"].nunique())
print("\nSample diag_1 values:", df["diag_1"].dropna().unique()[:10])
# ICD-9 codes — hundreds of values, need to group into categories


# ── 8. ICD-9 grouping function ────────────────────────────────────────────────
# Group diagnosis codes into broad categories (standard approach)
def group_icd9(code):
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    # V and E codes
    if code.startswith("V"):
        return "Other"
    if code.startswith("E"):
        return "Injury"
    try:
        c = float(code)
    except ValueError:
        return "Other"
    if 390 <= c <= 459 or c == 785:
        return "Circulatory"
    elif 460 <= c <= 519 or c == 786:
        return "Respiratory"
    elif 520 <= c <= 579 or c == 787:
        return "Digestive"
    elif 250 <= c <= 250.99:
        return "Diabetes"
    elif 800 <= c <= 999:
        return "Injury"
    elif 710 <= c <= 739:
        return "Musculoskeletal"
    elif 580 <= c <= 629 or c == 788:
        return "Genitourinary"
    elif 140 <= c <= 239:
        return "Neoplasms"
    else:
        return "Other"

df["diag_1_group"] = df["diag_1"].apply(group_icd9)
df["diag_2_group"] = df["diag_2"].apply(group_icd9)
df["diag_3_group"] = df["diag_3"].apply(group_icd9)

print("\ndiag_1 groups:")
print(df["diag_1_group"].value_counts())


# ── 9. Numeric features distribution ─────────────────────────────────────────
num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses"]

print("\n--- Numeric features summary ---")
print(df[num_cols].describe().round(2))

df[num_cols].hist(bins=20, figsize=(12, 8), color="steelblue", edgecolor="white")
plt.suptitle("Numeric features distribution")
plt.tight_layout()
plt.savefig("numeric_distributions.png")
plt.show()


# ── 10. Medication columns ────────────────────────────────────────────────────
med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]

print("\n--- Medication value counts (first 3 meds) ---")
for col in med_cols[:3]:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Many meds are nearly always "No" — could combine into a count
df["n_meds_changed"] = (df[med_cols].isin(["Up", "Down"])).sum(axis=1)
df["n_meds_steady"] = (df[med_cols] == "Steady").sum(axis=1)
print("\nMed changes per patient (first 5):")
print(df["n_meds_changed"].value_counts().head())


# ── 11. Columns to drop ───────────────────────────────────────────────────────
print("\n--- Columns we will likely DROP ---")
# weight: >95% missing
# payer_code: not predictive + lots of missing
# encounter_id, patient_nbr: IDs, not features (assignment rule too)
to_drop = ["weight", "payer_code", "encounter_id", "patient_nbr",
           "diag_1", "diag_2", "diag_3",   # replaced by _group versions
           "readmitted"]                     # target, not feature
print(to_drop)


# ── 12. Discharge disposition — drop dead/hospice rows ───────────────────────
print("\n--- Discharge disposition values ---")
print(df["discharge_disposition_id"].value_counts().head(10))
# IDs 11,13,14,19,20,21 = expired/hospice — can't be readmitted, drop them
dead_ids = [11, 13, 14, 19, 20, 21]
df_clean = df[~df["discharge_disposition_id"].isin(dead_ids)].copy()
print(f"\nRows before dropping dead/hospice: {len(df)}")
print(f"Rows after: {len(df_clean)}")


# ── 13. Early readmit rate by age ─────────────────────────────────────────────
print("\n--- Early readmit rate by age ---")
print(df_clean.groupby("age")["early_readmit"].mean().round(3))

df_clean.groupby("age")["early_readmit"].mean().sort_index().plot(
    kind="bar", color="salmon", edgecolor="black")
plt.title("Early readmit rate by age group")
plt.ylabel("rate")
plt.tight_layout()
plt.savefig("readmit_by_age.png")
plt.show()


# ── 14. Early readmit rate by race ───────────────────────────────────────────
print("\n--- Early readmit rate by race ---")
print(df_clean.groupby("race")["early_readmit"].mean().round(3))


# ── 15. Correlation of numeric features with target ───────────────────────────
print("\n--- Correlation with early_readmit ---")
corr = df_clean[num_cols + ["early_readmit"]].corr()["early_readmit"].drop("early_readmit")
print(corr.sort_values(ascending=False).round(3))


# ── 16. Summary: what we'll do in preprocessing ──────────────────────────────
print("""
=== EDA SUMMARY — Preprocessing Plan ===

1. Replace '?' with NaN
2. Drop: weight, payer_code (too many missing), encounter_id, patient_nbr (IDs)
3. Drop rows where discharge = dead/hospice (11,13,14,19,20,21)
4. Group diag_1/2/3 ICD-9 codes into broad categories
5. Medication columns: keep as-is OR encode as {No=0, Steady=1, Up=2, Down=3}
   + add n_meds_changed and n_meds_steady as new features
6. Fill remaining NaNs:
   - Numeric: median
   - Categorical: 'Unknown'
7. Encode categoricals with OrdinalEncoder (handles unseen in sklearn >= 1.3)
8. Patient-grouped train/val split (GroupShuffleSplit on patient_nbr)
   to avoid data leakage
""")
