# group_14 — Diabetes Early Readmission Prediction

- Language / version: Python 3.14.5
- Team members: Yusra Faisal (yf08967), Maria Aqeel (ma09199)
- How to run prediction:
    python predict.py --input <features.csv> --output <preds.csv>
- Model / approach: Gradient Boosting Classifier (scikit-learn) with a full
  sklearn Pipeline. Preprocessing includes: replacing "?" with NaN, grouping
  ICD-9 diagnosis codes into 9 broad categories, mapping admission/discharge/
  source ID columns to readable strings, converting age brackets to ordinal
  integers, engineering n_meds_changed and n_meds_steady count features,
  a1c_tested and glucose_tested missingness flags, and a1c_abnormal
  for poorly controlled diabetes detection ,median imputation for numerics, 
  constant imputation + OrdinalEncoder for
  categoricals (unknown_value=-1 handles unseen categories gracefully).
- Validation approach: Patient-grouped split using GroupShuffleSplit on
  patient_nbr (80/20). No patient appears in both train and validation sets,
  preventing data leakage from repeat encounters.
- Threshold: 0.15 (chosen by F1 maximization on validation set, saved in
  threshold.json)
- Dependencies: pinned in requirements.txt