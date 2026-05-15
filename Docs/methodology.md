# Methodology

## 1. Data Preparation

The pipeline starts by loading a tabular dataset containing non-invasive prenatal Doppler ultrasound and maternal clinical variables.

The required columns are:

- `CaseID`
- `Status`
- `Gestational_Age`
- `Maternal_Age`
- `PI_MCA`
- `PI_UA`
- `PI_MCA_UA`

The `Status` column is standardized into two classes:

- `IUGR`
- `Normal`

The preprocessing stage includes:

- Column name normalization
- Duplicate removal
- Label standardization
- Numeric type conversion
- Missing-value inspection
- Class distribution summary

## 2. Feature Engineering

The pipeline creates additional Doppler-based features:

- `PI_Diff = PI_MCA - PI_UA`
- `PI_MCA_UA_Product = PI_MCA * PI_UA`
- `GA_MCA_Interaction = Gestational_Age * PI_MCA`
- `UA_Adjusted = PI_UA / Gestational_Age`

These features are designed to capture clinically meaningful relationships between fetal vascular resistance and gestational development.

## 3. Gestational-Age Stratification

Doppler indices change physiologically during pregnancy. To reduce gestational-age-related bias, the pipeline includes gestational-age-based grouping.

Quantile-based GA bins are generated and bins containing only one class are merged with adjacent groups.

This allows model evaluation within more physiologically comparable subgroups.

## 4. Model Development

The following models are implemented:

- Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron

Logistic Regression is used as an interpretable baseline. Random Forest and XGBoost are used to capture non-linear relationships. The MLP is included as a deep learning extension.

## 5. Model Evaluation

Models are evaluated using stratified cross-validation and out-of-fold predictions.

The main metrics are:

- Accuracy
- Precision
- Sensitivity
- Specificity
- AUC-ROC

Sensitivity is emphasized because the target application involves screening for high-risk fetal conditions.

## 6. Clinical Threshold Comparison

Model predictions are compared with conventional Doppler threshold rules:

- High UA PI
- Low MCA PI
- Low MCA/UA ratio
- Combined Doppler rule

This provides a clinically interpretable benchmark for model performance.

## 7. Explainability

Explainability is performed using:

- Logistic Regression coefficients
- Odds ratios
- SHAP global feature importance
- SHAP individual-level explanations

The goal is to improve transparency and support communication with clinical collaborators.
