import pandas as pd


NUMERIC_COLUMNS = [
    "Gestational_Age",
    "Maternal_Age",
    "PI_MCA",
    "PI_UA",
    "PI_MCA_UA",
]


def standardize_status_labels(df, label_col="Status"):
    """
    Standardize class labels to IUGR and Normal.
    """
    df = df.copy()

    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"IUGR": "IUGR", "NORMAL": "Normal"})
    )

    if df[label_col].isna().any():
        raise ValueError(
            "Unrecognized labels found in Status. "
            "Expected labels are 'IUGR' and 'Normal'."
        )

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def enforce_numeric_columns(df, numeric_cols=None):
    """
    Convert selected columns to numeric values and detect invalid entries.
    """
    df = df.copy()
    numeric_cols = numeric_cols or NUMERIC_COLUMNS

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().any().any():
        missing_report = df[numeric_cols].isnull().sum()
        raise ValueError(
            "NaN values detected after numeric conversion:\n"
            f"{missing_report}"
        )

    return df


def preprocess_dataset(df):
    """
    Apply basic preprocessing:
    - remove duplicates
    - standardize labels
    - enforce numeric columns
    """
    df, removed_duplicates = remove_duplicates(df)
    df = standardize_status_labels(df)
    df = enforce_numeric_columns(df)

    return df, {"removed_duplicates": removed_duplicates}


def print_quality_summary(df):
    """
    Print a compact data quality summary.
    """
    print("\n=== DATA QUALITY SUMMARY ===")
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nStatus distribution:\n", df["Status"].value_counts())
