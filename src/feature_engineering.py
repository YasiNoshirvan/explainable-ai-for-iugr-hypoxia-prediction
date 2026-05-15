ORIGINAL_FEATURES = [
    "Gestational_Age",
    "Maternal_Age",
    "PI_MCA",
    "PI_UA",
    "PI_MCA_UA",
]

ENGINEERED_FEATURES = [
    "PI_Diff",
    "PI_MCA_UA_Product",
    "GA_MCA_Interaction",
    "UA_Adjusted",
]


def add_engineered_features(df):
    """
    Create clinically inspired Doppler-based engineered features.
    """
    df = df.copy()

    df["PI_Diff"] = df["PI_MCA"] - df["PI_UA"]
    df["PI_MCA_UA_Product"] = df["PI_MCA"] * df["PI_UA"]
    df["GA_MCA_Interaction"] = df["Gestational_Age"] * df["PI_MCA"]
    df["UA_Adjusted"] = df["PI_UA"] / df["Gestational_Age"]

    return df


def build_final_dataset(df, use_engineered=True, shuffle=True, random_state=42):
    """
    Build the final modelling dataset.
    """
    df = add_engineered_features(df)

    feature_cols = ORIGINAL_FEATURES.copy()
    if use_engineered:
        feature_cols += ENGINEERED_FEATURES

    final_dataset = df[feature_cols + ["Status"]].copy()

    if shuffle:
        final_dataset = final_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return final_dataset, feature_cols
