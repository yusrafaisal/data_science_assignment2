"""Predict module - diabetes early readmission prediction.

REQUIRED INTERFACE (do not change the signatures / CLI below):
  run by instructor:   python predict.py --input <features.csv> --output <preds.csv>
  input  : raw rows, same columns as diabetes_dev.csv, WITHOUT the "readmitted" column
  output : CSV with exactly two columns, same row order and count as input:
           encounter_id, prob_early_readmit   (one probability in [0,1] per row)
"""
import os, argparse
import numpy as np, pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "pipeline.joblib")

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

NUMERIC = [
    "age_num", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "n_meds_changed", "n_meds_steady",
    "a1c_tested",        
    "glucose_tested",    
    "a1c_abnormal",         
]

CATEGORICAL = [
    "race", "gender",
    "admission_type", "discharge_disposition", "admission_source",
    "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
]

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
    "n_meds_changed", "n_meds_steady",
    "a1c_tested", "glucose_tested", "a1c_abnormal",             
]


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
    """Raw rows -> exact input structure the fitted pipeline expects."""
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

    # Map raw ID columns to full descriptions
    df["admission_type"]        = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
    df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP).fillna("Unknown")
    df["admission_source"]      = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown")

    # Age bracket to ordinal integer (preserves order)
    df["age_num"] = df["age"].map({a: i for i, a in enumerate(AGE_ORDER)})

    # Fill missing medical_specialty
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

    df["a1c_tested"]     = df["A1Cresult"].notna().astype(int)
    df["glucose_tested"] = df["max_glu_serum"].notna().astype(int)
    df["a1c_abnormal"]   = (df["A1Cresult"].isin([">7", ">8"])).astype(int)

    return df[FEATURE_COLS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return 1-D array of early-readmission probabilities, same length/order as df."""
    pipe = joblib.load(MODEL_PATH)
    return np.clip(pipe.predict_proba(clean(df))[:, 1], 0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    out = pd.DataFrame({"encounter_id": df["encounter_id"],
                        "prob_early_readmit": predict_proba(df)})
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"Predictions saved to {a.output}  ({len(out)} rows)")