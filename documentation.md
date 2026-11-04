# Diabetes Early Readmission — Project Documentation

## Table of Contents
1. Problem Statement
2. Dataset Overview
3. EDA Findings
4. Preprocessing Decisions
5. Feature Engineering
6. Train/Validation Split
7. Preprocessing Pipeline
8. Model Selection
9. Threshold Selection
10. Subgroup / Fairness Analysis
11. What Gets Saved and Why
12. How predict.py Works

---

## 1. Problem Statement

When a diabetic patient is discharged from hospital, they sometimes return within 30 days. This is called **early readmission** and is costly for both the patient and the hospital. The goal is to build a machine learning model that, given information about a hospital visit, predicts the **probability** that the patient will be readmitted within 30 days.

This is a **binary classification** problem:
- **Class 1 (positive):** `readmitted == "<30"` — patient came back within 30 days
- **Class 0 (negative):** `readmitted == "NO"` or `readmitted == ">30"` — patient did not return early

The model outputs a **probability between 0 and 1**, not just a yes/no label.

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| File | diabetes_dev.csv |
| Total rows | 81,281 |
| Total columns | 50 |
| Unique patients | 57,214 |
| Patients with >1 visit | 13,376 |

Each row is one **hospital encounter** (visit), not one patient. The same patient can appear multiple times across different rows.

---

## 3. EDA Findings

### 3.1 Target Distribution

```
NO   → 43,913 rows (54%) — never readmitted
>30  → 28,315 rows (35%) — readmitted after 30 days
<30  →  9,053 rows (11%) — readmitted within 30 days ← our positive class
```

**Key finding:** Only 11% of rows are positive. This is called **class imbalance**. A naive model that always predicts "No" would be 89% accurate but completely useless — it would never catch a single high-risk patient. We need to handle this imbalance explicitly.

---

### 3.2 Missing Values

The dataset uses `"?"` to represent missing values instead of leaving the cell blank. Pandas does not recognize `"?"` as missing, so we must replace it with `NaN` first.

```python
df.replace("?", np.nan, inplace=True)
```

After doing this, the missing value counts are:

| Column | Missing | % Missing | Decision |
|---|---|---|---|
| weight | 78,716 | 97% | Drop — almost no data |
| max_glu_serum | 77,006 | 95% | Drop — too sparse |
| A1Cresult | 67,661 | 83% | Drop — too sparse |
| medical_specialty | 39,866 | 49% | Keep — fill with "Unknown" |
| payer_code | 32,172 | 40% | Drop — low predictive value |
| race | 1,860 | 2% | Keep — fill with "Unknown" |
| diag_2 | 286 | <1% | Keep — handle in grouping |
| diag_3 | 1,118 | 1% | Keep — handle in grouping |

**Why drop columns with >80% missing?**
When a column is missing for 95% of rows, imputing it with a median or mode just means the model sees the same value for almost every row. The column adds almost no real information — just noise. It is cleaner and more honest to drop it.

**Why keep medical_specialty despite 49% missing?**
The specialty of the treating physician (Internal Medicine, Cardiology, etc.) is still clinically meaningful for the 51% of rows where it exists. "Unknown" becomes its own category and the model can still learn from it.

---

### 3.3 Patient Duplicates — The Leakage Problem

```
Total rows:        81,281
Unique patients:   57,214
Patients >1 visit: 13,376
```

13,376 patients appear more than once. If we do a **random train/test split**, the same patient could appear in both training and validation. The model would have already "seen" that patient's information during training, making validation results unrealistically optimistic. This is called **data leakage**.

**Solution:** Use a patient-grouped split — all visits of a given patient go entirely to either train or validation, never both.

---

### 3.4 Dead / Hospice Patients

Some rows have `discharge_disposition_id` values that indicate the patient died or was sent to hospice:

| ID | Meaning |
|---|---|
| 11 | Expired |
| 13 | Hospice / home |
| 14 | Hospice / medical facility |
| 19 | Expired at home (Medicaid hospice) |
| 20 | Expired in medical facility (Medicaid hospice) |
| 21 | Expired, place unknown (Medicaid hospice) |

These patients **cannot be readmitted** because they are dead or in end-of-life care. Keeping them in the dataset would confuse the model — it would try to predict readmission for patients who had zero chance of returning. We drop them:

```python
dead_ids = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(dead_ids)]
```

This removes **1,941 rows**, leaving **79,340 rows**.

---

### 3.5 Diagnosis Columns

Three diagnosis columns exist: `diag_1` (primary), `diag_2`, `diag_3`. These contain **ICD-9 medical codes** — specific numeric codes for diseases and conditions.

```
diag_1: 695 unique codes
diag_2: 723 unique codes
diag_3: 759 unique codes
```

700+ unique values is too many categories for a model to learn from effectively. With so many categories, most codes appear only a handful of times — not enough data for the model to learn any pattern.

**Solution:** Group ICD-9 codes into 9 broad clinical categories based on their numeric ranges (standard medical groupings):

| Range | Category |
|---|---|
| 390–459, 785 | Circulatory (heart, blood vessels) |
| 460–519, 786 | Respiratory (lungs, breathing) |
| 520–579, 787 | Digestive (stomach, intestines) |
| 250–250.99 | Diabetes |
| 800–999 | Injury |
| 710–739 | Musculoskeletal (bones, joints) |
| 580–629, 788 | Genitourinary (kidneys, bladder) |
| 140–239 | Neoplasms (cancers, tumors) |
| V codes | Other |
| E codes | Injury (external causes) |
| Everything else | Other |

Result for diag_1:
```
Circulatory     → 24,301 (most common — heart issues in diabetic patients)
Other           → 14,591
Respiratory     → 11,497
Digestive       →  7,519
Diabetes        →  6,997
Injury          →  5,559
Genitourinary   →  4,130
Musculoskeletal →  3,962
Neoplasms       →  2,708
Missing         →     17
```

---

### 3.6 Raw ID Columns

Three columns contained raw integer IDs that were meaningless as numbers:
- `admission_type_id` — how the patient was admitted
- `discharge_disposition_id` — where the patient went after discharge
- `admission_source_id` — where the referral came from

We replaced each with its full text description from `IDS_mapping.csv`. For example:
- `admission_type_id = 1` → `"Emergency"`
- `discharge_disposition_id = 1` → `"Discharged to home"`
- `admission_source_id = 7` → `"Emergency Room"`

We did NOT group these into broader buckets (like we initially tried) because the full descriptions from the mapping file are specific and meaningful. Grouping would lose clinical information — for example, "Transferred to SNF" vs "Transferred to rehab facility" are very different situations.

---

### 3.7 Strongest Predictors (from correlation with target)

```
number_inpatient      0.163  ← strongest by far
number_emergency      0.061
number_diagnoses      0.054
time_in_hospital      0.047
num_medications       0.040
num_lab_procedures    0.024
number_outpatient     0.023
num_procedures       -0.009
```

`number_inpatient` — how many times the patient was admitted in the prior year — is by far the strongest single signal. Frequent hospital visitors are high risk. This makes clinical sense.

---

### 3.8 Medication Columns

23 medication columns exist, each with values: `No`, `Steady`, `Up`, `Down`.

```
metformin:   No=65,339  Steady=14,630  Up=860   Down=452
repaglinide: No=80,030  Steady=1,129   Up=88    Down=34
nateglinide: No=80,739  Steady=516     Up=17    Down=9
```

Most medications are `"No"` for nearly every patient. This is very sparse data. We keep all 23 medication columns as-is (the model can still learn from them), and additionally create two summary features:

- `n_meds_changed` — count of medications where dosage went Up or Down (was the regimen actively being adjusted?)
- `n_meds_steady` — count of medications staying Steady (how stable is the treatment?)

These summary features capture the overall pattern of medication management without requiring the model to learn from 23 sparse columns independently.

---

### 3.9 Age Column

Age is stored as string brackets: `[0-10)`, `[10-20)`, ..., `[90-100)`. These have a natural order but as strings the model cannot understand that `[70-80)` is older than `[30-40)`.

We convert to ordinal integers:
```
[0-10)   → 0
[10-20)  → 1
[20-30)  → 2
...
[90-100) → 9
```

This preserves the ordering so the model can use age as a continuous-ish variable.

---

## 4. Preprocessing Decisions — Summary

| Column | Original | Action | Reason |
|---|---|---|---|
| encounter_id | ID | Drop | Not predictive, assignment rule |
| patient_nbr | ID | Drop as feature | Not predictive, assignment rule |
| weight | 97% missing | Drop | Too sparse |
| max_glu_serum | 95% missing | → glucose_tested binary flag | Extract signal from missingness |
| A1Cresult | 83% missing | → a1c_tested, a1c_abnormal flags | Extract signal from missingness and value |
| payer_code | 40% missing | Drop | Low signal |
| age | String bracket | → age_num (0–9) | Preserve ordering |
| diag_1/2/3 | 700+ ICD codes | → _group (9 categories) | Reduce cardinality |
| admission_type_id | Integer ID | → text description | Make meaningful |
| discharge_disposition_id | Integer ID | → text description | Make meaningful |
| admission_source_id | Integer ID | → text description | Make meaningful |
| medical_specialty | 49% missing | Fill NaN → "Unknown" | Still useful |
| 23 med columns | Sparse No/Up/Down | Keep + add n_meds_changed, n_meds_steady | Capture dosage pattern |
| A1Cresult, max_glu_serum | 83–95% missing | → a1c_tested, glucose_tested, a1c_abnormal | Signal from missingness |
| race, gender | Small % missing | Fill NaN → "Unknown" in pipeline | Handled by imputer |

---

## 5. Feature Engineering

Five new columns created from existing data:

**n_meds_changed:** Number of medications where dosage was actively changed (Up or Down). A high number means the clinical team was actively adjusting the patient's regimen, which may indicate unstable control.

**n_meds_steady:** Number of medications staying at Steady dose. A high number means stable treatment.

```python
df["n_meds_changed"] = (df[MED_COLS].isin(["Up", "Down"])).sum(axis=1)
df["n_meds_steady"]  = (df[MED_COLS] == "Steady").sum(axis=1)
```

**a1c_tested:** Binary flag — was an A1C test even ordered during this visit? `A1Cresult` is 83% missing, but the missingness itself is informative. Patients whose A1C was not tested may reflect different care pathways or less thorough monitoring.

**glucose_tested:** Binary flag — was a glucose serum test ordered? `max_glu_serum` is 95% missing, same logic as above.

**a1c_abnormal:** Binary flag — was the A1C result abnormal (>7 or >8), indicating poorly controlled diabetes? This directly captures whether the patient's diabetes was under control at discharge.

```python
df["a1c_tested"]     = df["A1Cresult"].notna().astype(int)
df["glucose_tested"] = df["max_glu_serum"].notna().astype(int)
df["a1c_abnormal"]   = (df["A1Cresult"].isin([">7", ">8"])).astype(int)
```

These three features let us extract signal from columns that are otherwise too sparse to use directly. Rather than dropping them entirely or imputing them, we turn their missingness and values into binary indicators the model can learn from.

---

## 6. Train/Validation Split

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=patient_nbr))
```

**Why GroupShuffleSplit?**
`GroupShuffleSplit` with `groups=patient_nbr` guarantees that all visits of a given patient go entirely to either train or validation — never split across both. This prevents data leakage from patient duplicates.

**Why not regular train_test_split?**
Regular splitting is random by row. Since 13,376 patients appear more than once, a random split would almost certainly put the same patient in both sets. Validation would look great but the model would fail on truly new patients.

**Split result:**
```
Train: ~63,500 rows (80%)
Val:   ~15,800 rows (20%)
Train positive rate: 11.4%
Val positive rate:   11.1%
```

The positive rates are similar in both sets, confirming the split is representative.

---

## 7. Preprocessing Pipeline (sklearn)

We use sklearn's `Pipeline` and `ColumnTransformer` to bundle all preprocessing with the model into one object.

**Why a Pipeline?**
If we preprocess outside the pipeline, we'd have to manually apply the same transformations to new data in `predict.py`. A pipeline bundles everything together — when we call `pipe.predict_proba(raw_df)`, it automatically runs all preprocessing steps first. This also prevents a common mistake called **train-test contamination** — the imputer and encoder are fit only on training data, not on validation data.

**Numeric columns:** `SimpleImputer(strategy="median")`
Fill any remaining NaN values with the median of that column from the training set. We use median (not mean) because numeric columns like `number_inpatient` are right-skewed — most values are 0 or 1, but a few are as high as 16. The mean would be pulled up by outliers; the median is more representative.

**Categorical columns:**
1. `SimpleImputer(strategy="constant", fill_value="Unknown")` — fill NaN with the string "Unknown", which becomes its own category
2. `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` — convert string categories to integers. The key setting is `handle_unknown="use_encoded_value"` with `unknown_value=-1` — if the held-out data contains a category the model has never seen (e.g. a rare diagnosis code), it gets encoded as -1 instead of crashing.

**Why OrdinalEncoder instead of OneHotEncoder?**
Tree-based models (Random Forest, Gradient Boosting) work perfectly well with ordinal encoding — they split on values, so the numeric representation doesn't imply any ordering they can't handle. OneHotEncoder would create hundreds of extra columns from the diagnosis groups and medical specialty, slowing training with no benefit.

---

## 8. Model Selection

We trained three models and compared them on the validation set.

### Model 1: Logistic Regression
```python
LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
```
- Simplest possible model — draws a linear decision boundary
- `class_weight="balanced"` — automatically gives more weight to the minority class (readmitted=1) to compensate for the 11% imbalance
- `max_iter=2000` — increased from default 1000 to ensure full convergence
- Used as our **baseline** — if fancier models can't beat this, something is wrong with them
- Pros: fast, interpretable, easy to explain
- Cons: assumes a linear relationship between features and outcome — unlikely to capture complex medical patterns

### Model 2: Random Forest
```python
RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=15,
                       min_samples_leaf=10, random_state=42, n_jobs=-1)
```
- Builds 200 decision trees, each trained on a random subset of rows and features
- Final prediction is the average probability across all 200 trees
- `class_weight="balanced"` — handles imbalance
- `max_depth=15` — allows deeper trees for more expressive patterns
- `min_samples_leaf=10` — prevents overfitting on rare patient groups
- `n_jobs=-1` — uses all CPU cores to train in parallel (faster)
- Pros: handles non-linear relationships, robust to outliers, less prone to overfitting than a single tree
- Cons: slower than logistic regression, less interpretable

### Model 3: Gradient Boosting (Final Model)
```python
GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=4,
                           subsample=0.8, min_samples_leaf=20, random_state=42)
```
- Builds decision trees **sequentially** — each new tree specifically corrects the errors made by all previous trees
- `n_estimators=500` — more trees allow finer correction of errors
- `learning_rate=0.03` — slower learning rate paired with more trees gives better generalization
- `max_depth=4` — moderately deep trees capture complex patterns without overfitting
- `subsample=0.8` — each tree trained on 80% of rows (random sampling), reduces overfitting
- `min_samples_leaf=20` — prevents trees from fitting to tiny, unrepresentative patient groups
- Pros: typically best accuracy on structured tabular data, good probability calibration
- Cons: slower to train, no built-in class_weight (imbalance handled by probability output and threshold)

### Results from our run:

| Model | Average Precision | ROC-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression | 0.1990 | 0.6599 | 0.2294 |
| Random Forest | 0.2067 | 0.6672 | 0.2024 |
| **Gradient Boosting** | **0.2266** | **0.6728** | **0.0942** |

**Winner: Gradient Boosting** — best on all three metrics.

**Why use Average Precision as the primary metric?**
The assignment specifies this as the primary evaluation metric. AP summarizes the precision-recall curve across all possible thresholds. For imbalanced problems, AP is much more informative than accuracy or even AUC — it directly measures how well the model ranks high-risk patients above low-risk ones.

**Why is Brier score important?**
Brier score measures how well-calibrated the probabilities are (lower is better). If the model says "40% chance of readmission", the Brier score tells us whether roughly 40% of those patients actually were readmitted. Good calibration matters here because we're using the probability output to make clinical decisions.

---

## 9. Threshold Selection

The model outputs a probability between 0 and 1. To make an actual prediction (flag the patient or not), we need a decision threshold. Any patient whose probability exceeds the threshold gets flagged as high-risk.

**Why not just use 0.5?**
With only 11% positive class, the model rarely outputs probabilities above 0.5. Using 0.5 as threshold would flag almost nobody — missing the entire point of the system.

**How we chose the threshold:**
We scan every threshold from 0.05 to 0.54 in steps of 0.01, and pick the one with the highest **F1 score** on the validation set.

```python
for t in np.arange(0.05, 0.55, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
```

**Why F1?**
F1 is the harmonic mean of Precision and Recall:
- **Precision** = of all patients we flagged, what fraction actually got readmitted?
- **Recall** = of all patients who actually got readmitted, what fraction did we catch?

In a medical context, **missing a high-risk patient (low Recall) is worse** than over-flagging someone who turns out fine (low Precision). F1 balances both. We could also argue for prioritising Recall even more, but F1 gives a reasonable starting point.

**Result from our run:**
```
Chosen threshold: 0.15  (best F1 = 0.280)

At threshold 0.15:
  Early <30:  Precision=0.21  Recall=0.40  F1=0.28
  No early:   Precision=0.92  Recall=0.82  F1=0.86
  Accuracy: 77%
```

We catch **40% of actual high-risk patients**. Precision of 0.21 means roughly 1 in 5 patients we flag actually gets readmitted — acceptable in healthcare where the cost of missing a high-risk patient is much higher than giving extra follow-up to someone who didn't need it. The threshold shifted from 0.13 to 0.15 compared to the baseline model, reflecting better-calibrated probability outputs from the tuned GBM.

---

## 10. Subgroup / Fairness Analysis

We check whether the model performs equally across patient subgroups. This is required by the assignment and is important ethically — a model that works well on average but fails for specific groups (elderly patients, a particular race) is not clinically fair.

We report **Average Precision** per subgroup because it accounts for both ranking quality and class imbalance within the subgroup.

### By Age:
```
[0-10)    n=28,    AP=0.067  ← very few samples, unreliable
[10-20)   n=127,   AP=0.232
[20-30)   n=214,   AP=0.405
[30-40)   n=611,   AP=0.348
[40-50)   n=1478,  AP=0.357
[50-60)   n=2700,  AP=0.198
[60-70)   n=3506,  AP=0.210
[70-80)   n=4121,  AP=0.199
[80-90)   n=2582,  AP=0.224
[90-100)  n=387,   AP=0.178
```
Model performs best for ages 20–50, weakest for very young and elderly. The very young group (n=28) is too small to draw conclusions from. Elderly patients (70+) make up the majority of the dataset but show consistently lower AP — the model is less reliable for this group despite them being the most common patients.

### By Race:
```
AfricanAmerican  n=3001,  AP=0.230
Caucasian        n=11809, AP=0.223
Hispanic         n=328,   AP=0.415
Asian            n=88,    AP=0.330
Other            n=178,   AP=0.213
```
Hispanic and Asian show higher AP but have very small sample sizes (328 and 88) — interpret with caution, results may not be statistically reliable. AfricanAmerican vs Caucasian gap is small (0.230 vs 0.223) — no major fairness concern between the two largest groups.

### By Gender:
```
Female  n=8301,  AP=0.217
Male    n=7453,  AP=0.239
```
Small difference — not a major concern.

---

## 11. What Gets Saved and Why

| File | Contents | Why |
|---|---|---|
| `model/pipeline.joblib` | Full fitted pipeline (preprocessor + model) | So predict.py can load and use it on new data |
| `threshold.json` | `{"threshold": 0.15}` | Instructor applies this to predictions on held-out set |
| `model/feature_meta.json` | Lists of numeric and categorical column names | Reference for predict.py |
| `diabetes_clean.csv` | Cleaned and engineered feature matrix | For inspection and notebook use only |

**Why joblib for saving the model?**
joblib is the standard way to save sklearn pipelines. It serializes the entire fitted pipeline including the encoder's learned categories and the model's learned weights. When predict.py loads it, it gets exactly the same preprocessing and model that was trained.

---

## 12. How predict.py Works

`predict.py` is what the instructor runs on the hidden held-out data. It must:
1. Accept raw input in the same format as `diabetes_dev.csv` (without the `readmitted` column)
2. Apply all the same preprocessing
3. Load the saved pipeline
4. Return one probability per row

```python
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    pipe = joblib.load(MODEL_PATH)   # load saved pipeline
    X = preprocess(df)               # same preprocess() as train.py
    probs = pipe.predict_proba(X)[:, 1]  # probability of class 1
    return np.clip(probs, 0, 1)      # guarantee [0,1] range
```

The key is that `preprocess()` is identical in both `train.py` and `predict.py` — same column drops, same ICD grouping, same ID mapping, same age conversion. This ensures the held-out data goes through exactly the same transformations as the training data.

`np.clip(probs, 0, 1)` is a safety net — probabilities should already be in [0,1] but we clip just in case of any floating point edge cases, as the assignment requires no NaNs and no values outside [0,1].

---

## Quick Reference: Column Drops

| Dropped | Reason |
|---|---|
| encounter_id | ID — not a feature |
| patient_nbr | ID — not a feature |
| weight | 97% missing |
| max_glu_serum | 95% missing |
| A1Cresult | 83% missing |
| payer_code | 40% missing, low signal |
| diag_1, diag_2, diag_3 | Replaced by diag_1_group, diag_2_group, diag_3_group |
| admission_type_id | Replaced by admission_type (text) |
| discharge_disposition_id | Replaced by discharge_disposition (text) |
| admission_source_id | Replaced by admission_source (text) |
| age | Replaced by age_num (0–9) |
| readmitted | This is the target, not a feature |