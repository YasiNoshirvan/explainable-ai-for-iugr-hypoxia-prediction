# Explainable AI for IUGR and Fetal Hypoxia Prediction 🧬🤖

This repository contains a machine learning and explainable AI pipeline for predicting **intrauterine growth restriction (IUGR)** and fetal hypoxia risk using non-invasive prenatal Doppler ultrasound and maternal clinical features.

The project focuses on developing reproducible AI workflows for clinical risk prediction while preserving patient privacy and research confidentiality.

---

## 🌍 Overview

Intrauterine Growth Restriction (IUGR) is a high-risk pregnancy condition associated with impaired fetal growth and potential fetal hypoxia. Early identification of fetal compromise is essential for clinical monitoring and timely intervention.

This project investigates whether machine learning models can support the prediction of IUGR-related hypoxia risk using non-invasive prenatal features such as:

- Gestational Age
- Maternal Age
- Middle Cerebral Artery Pulsatility Index (PI_MCA)
- Umbilical Artery Pulsatility Index (PI_UA)
- MCA/UA Doppler ratio

The pipeline combines classical machine learning, deep learning, gestational-age-aware modeling, and explainability methods.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a reproducible AI pipeline for IUGR and fetal hypoxia risk prediction.
- Use non-invasive Doppler ultrasound and maternal clinical features.
- Compare classical ML models and a neural network model.
- Apply gestational-age-based stratification to reduce physiological bias.
- Evaluate models using clinically relevant metrics.
- Explain model predictions using SHAP and interpretable logistic regression.
- Compare AI-based predictions with conventional Doppler threshold rules.

---

## 🧪 Dataset

The original dataset includes retrospective prenatal Doppler ultrasound and maternal clinical variables.

Due to clinical data protection, research confidentiality, and future data publication plans, the original dataset is **not included** in this repository.

A small synthetic demo dataset may be provided only to demonstrate the expected data format and pipeline execution.

### Expected Input Features

| Feature | Description |
|---|---|
| `Gestational_Age` | Gestational age at examination |
| `Maternal_Age` | Maternal age |
| `PI_MCA` | Middle cerebral artery pulsatility index |
| `PI_UA` | Umbilical artery pulsatility index |
| `PI_MCA_UA` | MCA/UA Doppler ratio |
| `Status` | Binary clinical outcome: `IUGR` or `Normal` |

---

## 🛠️ Methodology

The project follows a structured machine learning workflow:

1. Data loading and quality checks
2. Missing value and duplicate inspection
3. Feature engineering
4. Gestational-age-based stratification
5. Model training and cross-validation
6. Probability calibration and threshold optimization
7. Performance evaluation
8. Clinical rule comparison
9. Explainability using SHAP

---

## 🧩 Feature Engineering

Several clinically motivated features can be generated from the original Doppler variables:

- Difference between MCA and UA pulsatility indices
- Product of MCA and UA indices
- Gestational-age interaction terms
- UA index adjusted by gestational age

These features are designed to capture vascular resistance patterns and developmental context.

---

## 🤖 Machine Learning Models

The pipeline supports the following models:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable clinical baseline |
| Random Forest | Non-linear ensemble model |
| XGBoost | Gradient boosting model for high predictive performance |
| Multi-Layer Perceptron | Deep learning extension |

---

## 📊 Evaluation Metrics

Models are evaluated using clinically meaningful classification metrics:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- AUC-ROC
- Confusion Matrix

Sensitivity is especially important in this context because missing high-risk IUGR cases may have serious clinical implications.

---

## 🧠 Explainability

Explainability is a central part of this project.

The pipeline includes:

- Logistic Regression coefficients
- Odds ratios
- SHAP global feature importance
- SHAP beeswarm plots
- SHAP waterfall plots for individual-level explanations

The goal is to make model behavior more transparent and clinically interpretable.

---

## 🏥 Clinical Rule Comparison

AI models are compared with conventional Doppler threshold-based rules, such as:

- `PI_UA > threshold`
- `PI_MCA < threshold`
- `PI_MCA_UA < threshold`

This comparison helps assess whether machine learning models can improve over traditional rule-based decision support.

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
