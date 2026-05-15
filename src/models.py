import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier


def make_logistic_regression():
    """
    Interpretable baseline model.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )


def make_random_forest():
    """
    Random Forest model with conservative hyperparameters to reduce overfitting.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=12,
        min_samples_split=12,
        max_features=2,
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def make_xgboost(scale_pos_weight=1.0):
    """
    XGBoost classifier with regularization.
    """
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_lambda=10.0,
        reg_alpha=2.0,
        gamma=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def make_calibrated_classifier(base_estimator, cv_splits=3, method="sigmoid"):
    """
    Version-safe wrapper for CalibratedClassifierCV.
    """
    calibration_cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    try:
        return CalibratedClassifierCV(
            estimator=base_estimator,
            cv=calibration_cv,
            method=method,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_estimator,
            cv=calibration_cv,
            method=method,
        )


def compute_scale_pos_weight(y_train):
    """
    Compute scale_pos_weight for XGBoost.
    """
    y_train = np.asarray(y_train).astype(int)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    if positives == 0:
        return 1.0

    return negatives / positives


def build_mlp(input_dim, l2=1e-4, dropout=0.4, learning_rate=1e-3):
    """
    Build a simple MLP model.

    TensorFlow is imported inside the function to keep classical ML usage lightweight.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2),
            ),
            layers.Dropout(dropout),
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2),
            ),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model
