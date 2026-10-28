from pathlib import Path

import keras
from loguru import logger
import pandas as pd
import typer

from module_deeplearningml.config import MODELS_DIR, PROCESSED_DATA_DIR
from module_deeplearningml.features import prepare_inputs

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "heart_classifier.keras",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    dataframe = pd.read_csv(features_path)
    inputs = prepare_inputs(dataframe)
    model = keras.models.load_model(model_path)
    probabilities = model.predict(inputs, verbose=0).reshape(-1)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target_probability": probabilities}).to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}.")


if __name__ == "__main__":
    app()
