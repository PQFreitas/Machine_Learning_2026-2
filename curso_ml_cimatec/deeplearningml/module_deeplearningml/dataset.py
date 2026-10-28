from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from module_deeplearningml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
TARGET_FEATURE_NAME = "target"
NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]
CATEGORICAL_FEATURE_NAMES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "ca",
    "thal",
]
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
REQUIRED_COLUMNS = FEATURE_NAMES + [TARGET_FEATURE_NAME]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if dataframe.empty:
        raise ValueError("Dataset must contain at least one row.")
    if dataframe[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    targets = set(dataframe[TARGET_FEATURE_NAME].unique())
    if not targets.issubset({0, 1}):
        raise ValueError("The target column must contain only 0 and 1.")

    return dataframe


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "heart.csv",
    output_path: Path = PROCESSED_DATA_DIR / "heart.csv",
) -> None:
    if not input_path.is_file():
        logger.info(f"Downloading heart dataset to {input_path}")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = pd.read_csv(DATA_URL)
        downloaded.to_csv(input_path, index=False)

    dataframe = load_dataset(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    logger.success(f"Dataset saved to {output_path} ({len(dataframe)} rows).")


if __name__ == "__main__":
    app()
