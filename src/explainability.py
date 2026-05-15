from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logistic_coefficients_odds_ratios(X, y, output_path=None):
    """
    Train Logistic Regression and return coefficients and odds ratios.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    coef_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Coefficient": model.coef_[0],
            "Odds_Ratio": np.exp(model.coef_[0]),
        }
    ).sort_values("Odds_Ratio", ascending=False)

    if output_path:
        coef_df.to_csv(output_path, index=False)

    return coef_df, model, scaler


def shap_summary_plots(model, X_train_scaled, X_test_scaled, output_dir):
    """
    Generate SHAP summary plots for an interpretable model.
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    beeswarm_path = output_dir / "figure_shap_summary_beeswarm.png"
    bar_path = output_dir / "figure_shap_summary_bar.png"

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "beeswarm": str(beeswarm_path),
        "bar": str(bar_path),
    }
