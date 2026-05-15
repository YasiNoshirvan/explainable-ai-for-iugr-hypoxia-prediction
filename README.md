# Explainable AI for IUGR and Fetal Hypoxia Prediction 🧬🤖

This repository contains a machine learning and explainable AI pipeline for predicting **intrauterine growth restriction (IUGR)** and fetal hypoxia risk using non-invasive prenatal Doppler ultrasound and maternal clinical features.

The project was developed in a medical AI research context and focuses on reproducible modelling, clinically meaningful evaluation, and interpretable decision support while preserving data confidentiality.

---

## 🌍 Overview

Intrauterine Growth Restriction (IUGR) is a high-risk pregnancy condition associated with impaired fetal growth and possible fetal hypoxia. Early identification of fetal compromise is important for clinical monitoring and timely intervention.

This project explores whether machine learning models can support IUGR and fetal hypoxia risk prediction using routinely available non-invasive features such as Doppler ultrasound indices and maternal demographic variables.

The pipeline includes:

- Data quality checks
- Feature engineering
- Gestational-age-aware modelling
- Classical machine learning models
- Deep learning extension
- ROC-based model evaluation
- Clinical threshold comparison
- Explainability using SHAP

---

## 🎯 Objectives

The main objectives of this project are:

- Build a reproducible AI pipeline for IUGR and fetal hypoxia risk prediction.
- Use non-invasive prenatal Doppler and maternal clinical features.
- Compare interpretable and non-linear machine learning models.
- Apply gestational-age-based stratification to reduce physiological bias.
- Optimize thresholds for clinically relevant sensitivity.
- Compare AI models with conventional Doppler threshold rules.
- Explain model predictions using Logistic Regression coefficients, odds ratios, and SHAP.

---

## 🧪 Dataset

The original dataset contains retrospective prenatal Doppler ultrasound and maternal clinical variables.

Due to clinical data protection, research confidentiality, and future data-publication plans, the original dataset is **not included** in this repository.

A small synthetic demo dataset may be included only to demonstrate the expected input format and pipeline execution.

### Expected Input Columns

| Column | Description |
|---|---|
| `CaseID` | Case identifier; excluded from modelling |
| `Status` | Target label: `IUGR` or `Normal` |
| `Gestational_Age` | Gestational age at examination |
| `Maternal_Age` | Maternal age |
| `PI_MCA` | Middle cerebral artery pulsatility index |
| `PI_UA` | Umbilical artery pulsatility index |
| `PI_MCA_UA` | MCA/UA Doppler ratio |

---

## 🛠️ Methodology

The workflow follows a structured machine learning pipeline:

1. Load and validate tabular clinical data
2. Standardize column names and class labels
3. Check missing values and duplicates
4. Create engineered Doppler-based features
5. Split data using stratified validation
6. Build gestational-age groups
7. Train and evaluate ML models
8. Calibrate probabilities when needed
9. Tune thresholds for sensitivity-oriented screening
10. Generate performance tables and ROC curves
11. Compare AI models with clinical threshold rules
12. Apply explainability methods

---

## 🧩 Feature Engineering

The pipeline creates clinically inspired derived features:

| Feature | Formula |
|---|---|
| `PI_Diff` | `PI_MCA - PI_UA` |
| `PI_MCA_UA_Product` | `PI_MCA * PI_UA` |
| `GA_MCA_Interaction` | `Gestational_Age * PI_MCA` |
| `UA_Adjusted` | `PI_UA / Gestational_Age` |

These features aim to capture interactions between fetal vascular resistance and gestational development.

---

## 🤖 Models

The project compares four modelling strategies:

| Model | Role |
|---|---|
| Logistic Regression | Interpretable clinical baseline |
| Random Forest | Non-linear ensemble model |
| XGBoost | Gradient boosting model with sensitivity-oriented tuning |
| Multi-Layer Perceptron | Deep learning extension |

---

## 📊 Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- AUC-ROC
- Confusion matrix

Out-of-fold predictions from stratified cross-validation are used for pooled ROC analysis where applicable.

Sensitivity is particularly important in this clinical setting because missed high-risk cases may have serious consequences.

---

## 🏥 Clinical Threshold Comparison

The pipeline compares ML models with conventional Doppler-based rules such as:

- `PI_UA > threshold`
- `PI_MCA < threshold`
- `PI_MCA_UA < threshold`
- Combined rule-based screening criteria

This comparison helps evaluate whether AI-based models provide additional predictive value over simple clinical thresholds.

---

## 🧠 Explainability

Explainability is included to improve clinical transparency.

The repository supports:

- Logistic Regression coefficients
- Odds ratios
- SHAP global feature importance
- SHAP beeswarm plots
- SHAP waterfall plots for individual predictions

These tools help identify which Doppler and maternal features contribute most strongly to the predicted IUGR/hypoxia risk.

---

## 📂 Project Structure

```text
explainable-ai-for-iugr-hypoxia-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── ga_stratification.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── clinical_rules.py
│
├── notebooks/
│   └── iugr_hypoxia_pipeline_demo.ipynb
│
├── configs/
│   └── config.yaml
│
├── demo_data/
│   ├── synthetic_sample.csv
│   └── README.md
│
├── outputs/
│   ├── tables/
│   └── figures/
│
└── docs/
    ├── methodology.md
    ├── reproducibility.md
    └── confidentiality_notice.md



```
---

## 🔒 Confidentiality and Data Availability

This repository is a public and sanitized version of a medical AI research workflow.

The original clinical dataset is not released in this repository because it contains sensitive medical information and is subject to research confidentiality and future data-publication plans.

This repository includes only:

- Generalized source code
- Reproducible pipeline structure
- Methodological documentation
- Synthetic/demo data examples

No patient-level data, identifiable records, protected clinical files, or restricted internal documents are included.

---

## ⚠️ Medical Disclaimer

This project is intended for research and educational purposes only.

The models and code in this repository are not intended for direct clinical diagnosis, treatment decisions, or medical device use without external validation, regulatory review, and clinical approval.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

---

## 🚀 Future Work

Potential future extensions include:

- External validation on independent clinical cohorts
- Prospective evaluation
- Publication of de-identified datasets
- Improved calibration strategies
- Additional fetal monitoring indicators
- Integration into clinical decision-support workflows

---

## 📬 Contact

**Yasaman Noshirvanbaboli**  
GitHub: [YasiNoshirvan](https://github.com/YasiNoshirvan)
