import numpy as np
import pandas as pd

from evaluation import compute_metrics


def apply_clinical_rules(
    df,
    pi_ua_high=1.5,
    pi_mca_low=1.0,
    pi_mca_ua_low=1.0,
):
    """
    Apply conventional Doppler threshold rules.
    """
    out = df.copy()

    out["rule_ua_high"] = np.where(out["PI_UA"] > pi_ua_high, 1, 0)
    out["rule_mca_low"] = np.where(out["PI_MCA"] < pi_mca_low, 1, 0)
    out["rule_mca_ua_low"] = np.where(out["PI_MCA_UA"] < pi_mca_ua_low, 1, 0)

    out["rule_combined"] = np.where(
        (out["PI_UA"] > pi_ua_high)
        | (out["PI_MCA"] < pi_mca_low)
        | (out["PI_MCA_UA"] < pi_mca_ua_low),
        1,
        0,
    )

    return out


def evaluate_clinical_rules(df, y_true):
    """
    Evaluate all rule-based predictions.
    """
    rules = {
        "UA PI > threshold": "rule_ua_high",
        "MCA PI < threshold": "rule_mca_low",
        "MCA/UA ratio < threshold": "rule_mca_ua_low",
        "Combined Doppler rule": "rule_combined",
    }

    rows = []

    for rule_name, column in rules.items():
        metrics = compute_metrics(y_true, df[column], y_prob=df[column])
        row = {"Rule": rule_name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)
