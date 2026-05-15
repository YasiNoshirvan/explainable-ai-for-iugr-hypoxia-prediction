# Methodology

## 1. Data Preparation

The pipeline begins with loading a tabular clinical dataset containing non-invasive prenatal Doppler ultrasound features and maternal demographic variables.

Required columns include:

- `Gestational_Age`
- `Maternal_Age`
- `PI_MCA`
- `PI_UA`
- `PI_MCA_UA`
- `Status`

The target variable is standardized into two classes:

- `IUGR`
- `Normal`

The preprocessing stage includes:

- Column name normalization
- Duplicate removal
- Required-column validation
- Numeric type conversion
- Missing-value inspection
- Class distribution summary

## 2. Feature Engineering

Additional features can be generated from the original Doppler variables:

- `PI_Diff = PI_MCA - PI_UA`
- `PI_MCA_UA_Product = PI_MCA * PI_UA`
- `GA_MCA_Interaction = Gestational_Age * PI_MCA`
- `UA_Adjusted = PI_UA / Gestational_Age`

These features aim to capture clinically meaningful relationships between fetal vascular resistance and gestational development.

## 3. Gestational-Age Stratification

Because Doppler indices change physiologically during pregnancy, the pipeline includes gestational-age-based grouping.

Quantile-based GA bins are created, and bins containing only one class are merged with adjacent bins to preserve class balance.

Each GA group can be modeled separately to reduce gestational-age-related bias.

## 4. Model Development

The following models are implemented:

- Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron

Logistic Regression is used as an interpretable clinical baseline. Ensemble models are used to capture nonlinear relationships. The MLP provides a deep learning extension.

## 5. Evaluation

Models are evaluated using stratified cross-validation and out-of-fold predictions.

Metrics include:

- Accuracy
- Precision
- Sensitivity
- Specificity
- AUC-ROC

Sensitivity is prioritized because the clinical task involves identifying high-risk fetal conditions.

## 6. Explainability

Explainability is performed using:

- Logistic Regression coefficients
- Odds ratios
- SHAP summary plots
- SHAP waterfall plots

These methods help explain both global feature importance and individual predictions.

## 7. Clinical Rule Comparison

Model predictions are compared with conventional Doppler threshold rules, such as:

- Increased UA PI
- Decreased MCA PI
- Decreased MCA/UA ratio

This comparison provides a clinical baseline for evaluating the added value of machine learning.
