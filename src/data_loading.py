from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "CaseID",
    "Status",
    "Gestational_Age",
    "Maternal_Age",
    "PI_MCA",
    "PI_UA",
    "PI_MCA_UA",
}


def find_data_file(candidates):
    """
    Return the first existing file path from a list of candidate paths.
    """
    for path in candidates:
        path = Path(path)
        if path.exists():
            return path

    raise FileNotFoundError(
        "No input dataset was found. Please provide a valid CSV or Excel file path."
    )


def load_dataset(file_path, sheet_name=None):
    """
    Load a tabular dataset from CSV or Excel.

    The original clinical dataset is not included in this repository.
    This function can be used with a local authorized dataset or the synthetic demo dataset.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        excel_file = pd.ExcelFile(file_path)
        selected_sheet = sheet_name or ("Sheet1" if "Sheet1" in excel_file.sheet_names else excel_file.sheet_names[0])
        df = pd.read_excel(file_path, sheet_name=selected_sheet)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    df.columns = [str(col).strip() for col in df.columns]
    validate_required_columns(df)

    return df


def validate_required_columns(df):
    """
    Validate that all required columns are present.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
