"""Training script - diabetes early readmission prediction.

Steps implemented:
  1) Load diabetes_dev.csv.
  2) Build target: early readmission = (readmitted == "<30").
  3) Drop stays ending in death/hospice (discharge_disposition_id in 11,13,14,19-21).
  4) Use PATIENT-GROUPED validation (GroupShuffleSplit on patient_nbr).
     Never let the same patient appear in both train and validation folds.
     Fit imputation/encoding/scaling on training folds only.
  5) Save the full fitted pipeline so predict.py can load it:
        joblib.dump(pipe, "model/pipeline.joblib")
  6) Save chosen threshold in threshold.json (dev data only).
"""
import os, json, argparse
import pandas as pd, joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              brier_score_loss, f1_score,
                              precision_score, recall_score,
                              classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Full ID mappings from IDS_mapping.csv ─────────────────────────────────────
ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

DISCHARGE_DISPOSITION_MAP = {
    1:  "Discharged to home",
    2:  "Transferred to short term hospital",
    3:  "Transferred to SNF",
    4:  "Transferred to ICF",
    5:  "Transferred to inpatient care",
    6:  "Home with home health service",
    7:  "Left AMA",
    8:  "Home under IV provider care",
    9:  "Admitted as inpatient",
    10: "Neonate transferred",
    11: "Expired",
    12: "Still patient or outpatient",
    13: "Hospice home",
    14: "Hospice medical facility",
    15: "Transferred to swing bed",
    16: "Transferred for outpatient services",
    17: "Referred for outpatient services",
    18: "NULL",
    19: "Expired at home hospice",
    20: "Expired in medical facility hospice",
    21: "Expired place unknown hospice",
    22: "Transferred to rehab facility",
    23: "Transferred to long term care",
    24: "Transferred to Medicaid nursing facility",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Transferred to federal facility",
    28: "Transferred to psychiatric hospital",
    29: "Transferred to Critical Access Hospital",
    30: "Transferred to other health care institution",
}

ADMISSION_SOURCE_MAP = {
    1:  "Physician Referral",
    2:  "Clinic Referral",
    3:  "HMO Referral",
    4:  "Transfer from hospital",
    5:  "Transfer from SNF",
    6:  "Transfer from health care facility",
    7:  "Emergency Room",
    8:  "Court/Law Enforcement",
    9:  "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer from Home Health Agency",
    19: "Readmission to Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital same facility",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice",
}

AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]

# ── Features used by the model ────────────────────────────────────────────────
# Dropped: weight (97% missing), max_glu_serum (95% missing),
#          A1Cresult (83% missing), payer_code (40% missing, low signal)
#          encounter_id, patient_nbr (IDs - not predictive)
#          diag_1/2/3 raw (replaced by _group columns)
#          admission_type_id, discharge_disposition_id, admission_source_id
#          (replaced by readable mapped columns)


CATEGORICAL_COLS = [
    "race",             
    "gender",            
    "admission_type",    
    "discharge_disposition",  
    "admission_source",  
    "medical_specialty",
    "diag_1_group",      
    "diag_2_group",      
    "diag_3_group",      
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

NUMERIC_COLS = [
    "age_num",           
    "time_in_hospital",  
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "n_meds_changed",    # engineered
    "n_meds_steady",     # engineered
    "a1c_tested",       
    "glucose_tested",   
    "a1c_abnormal",     
]

# Final column order matching original dataset (minus dropped columns)
FEATURE_COLS = [
    "race", "gender", "age_num",
    "admission_type", "discharge_disposition", "admission_source",
    "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1_group", "diag_2_group", "diag_3_group", "number_diagnoses",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
    "n_meds_changed", "n_meds_steady",  # engineered features at end
    "a1c_tested", "glucose_tested", "a1c_abnormal",
]


# ── ICD-9 grouping ────────────────────────────────────────────────────────────
def group_icd9(code):
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
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



def clean(df: pd.DataFrame) -> pd.DataFrame:
    """All cleaning + feature engineering. Used in both train.py and predict.py."""
    df = df.copy()

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # ICD-9 diagnosis grouping (700+ codes → 9 categories)
    df["diag_1_group"] = df["diag_1"].apply(group_icd9)
    df["diag_2_group"] = df["diag_2"].apply(group_icd9)
    df["diag_3_group"] = df["diag_3"].apply(group_icd9)

    # Medication summary features
    df["n_meds_changed"] = (df[MED_COLS].isin(["Up", "Down"])).sum(axis=1)
    df["n_meds_steady"]  = (df[MED_COLS] == "Steady").sum(axis=1)

    # Map raw ID columns to full descriptions from IDS_mapping.csv
    df["admission_type"]      = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
    df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP).fillna("Unknown")
    df["admission_source"]    = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown")

    # Age to ordinal integer (preserves order: [0-10)=0 ... [90-100)=9)
    df["age_num"] = df["age"].map({a: i for i, a in enumerate(AGE_ORDER)})

    # Fill missing medical_specialty explicitly before pipeline sees it
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

    df["a1c_tested"]      = df["A1Cresult"].notna().astype(int)
    df["glucose_tested"]  = df["max_glu_serum"].notna().astype(int)
    df["a1c_abnormal"]    = (df["A1Cresult"].isin([">7", ">8"])).astype(int)

    return df[FEATURE_COLS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/User/OneDrive/Desktop/ds_assignment2/diabetes_dev.csv")
    args = ap.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    print("Loaded:", df.shape)

    # ── 2. Build target ───────────────────────────────────────────────────────
    df.replace("?", np.nan, inplace=True)
    df["early_readmit"] = (df["readmitted"] == "<30").astype(int)
    print("Positive class rate:", df["early_readmit"].mean().round(3))

    # ── 3. Drop dead / hospice rows ───────────────────────────────────────────
    dead_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(dead_ids)].copy()
    print("After dropping dead/hospice:", df.shape)

    # ── 4. Feature matrix + patient-grouped split ─────────────────────────────
    X      = clean(df)
    y      = df["early_readmit"]
    groups = df["patient_nbr"]
    print("Feature matrix:", X.shape)

    # Save cleaned data for inspection
    X_saved = X.copy()
    X_saved["early_readmit"] = y.values
    X_saved["patient_nbr"]   = groups.values
    X_saved.to_csv(os.path.join(HERE, "diabetes_clean.csv"), index=False)
    print("Cleaned data saved: diabetes_clean.csv")

    # Patient-grouped split — same patient never in both train and val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Train positive rate: {y_train.mean().round(3)}")
    print(f"Val positive rate:   {y_val.mean().round(3)}")

    # ── Preprocessing pipeline (fit on train only) ────────────────────────────
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,         # unseen categories → -1, won't crash
            encoded_missing_value=-2
        )),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ], remainder="drop")


    # ── Compare 3 models ──────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=15,
            min_samples_leaf=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=500,      # up from 300
            learning_rate=0.03,    # slower learning to match more trees
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=20,   # prevent overfitting
            random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:, 1]

        ap  = average_precision_score(y_val, probs)
        auc = roc_auc_score(y_val, probs)
        bs  = brier_score_loss(y_val, probs)
        results[name] = {"pipeline": pipe, "probs": probs, "AP": ap, "AUC": auc, "Brier": bs}
        print(f"\n{name}  AP={ap:.4f}  AUC={auc:.4f}  Brier={bs:.4f}")

    # ── Pick best by Average Precision (primary metric per assignment) ─────────
    best_name = max(results, key=lambda k: results[k]["AP"])
    best      = results[best_name]
    print(f"\nBest model: {best_name}  (AP={best['AP']:.4f})")

    # ── Threshold scan on val set ─────────────────────────────────────────────
    probs_val = best["probs"]
    best_f1, best_thresh = 0, 0.5

    print("\nThreshold scan:")
    for t in np.arange(0.05, 0.55, 0.01):
        preds = (probs_val >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        p  = precision_score(y_val, preds, zero_division=0)
        r  = recall_score(y_val, preds, zero_division=0)
        print(f"  t={t:.2f}  F1={f1:.3f}  Prec={p:.3f}  Rec={r:.3f}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\nChosen threshold: {best_thresh:.2f}  (F1={best_f1:.3f})")
    preds_final = (probs_val >= best_thresh).astype(int)
    print(classification_report(y_val, preds_final, target_names=["No early", "Early <30"]))

    # ── Subgroup analysis ─────────────────────────────────────────────────────
    val_df = df.iloc[val_idx].copy()
    val_df["prob"] = probs_val

    print("\n--- Subgroup: Age ---")
    for grp in sorted(val_df["age"].unique()):
        sub = val_df[val_df["age"] == grp]
        if len(sub) < 20: continue
        print(f"  {grp}: n={len(sub)}, AP={average_precision_score(sub['early_readmit'], sub['prob']):.3f}")

    print("\n--- Subgroup: Race ---")
    for grp in val_df["race"].dropna().unique():
        sub = val_df[val_df["race"] == grp]
        if len(sub) < 20: continue
        print(f"  {grp}: n={len(sub)}, AP={average_precision_score(sub['early_readmit'], sub['prob']):.3f}")

    print("\n--- Subgroup: Gender ---")
    for grp in val_df["gender"].dropna().unique():
        sub = val_df[val_df["gender"] == grp]
        if len(sub) < 20: continue
        print(f"  {grp}: n={len(sub)}, AP={average_precision_score(sub['early_readmit'], sub['prob']):.3f}")

    # ── 5. Save pipeline ──────────────────────────────────────────────────────
    pipeline_path = os.path.join(MODEL_DIR, "pipeline.joblib")
    joblib.dump(best["pipeline"], pipeline_path)
    print(f"\nModel saved: {pipeline_path}")

    # ── 6. Save threshold ─────────────────────────────────────────────────────
    threshold_path = os.path.join(HERE, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": round(float(best_thresh), 2)}, f, indent=2)
    print(f"Threshold saved: {threshold_path}")

    # Save feature column names for predict.py
    meta_path = os.path.join(MODEL_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS}, f, indent=2)
    print(f"Feature meta saved: {meta_path}")