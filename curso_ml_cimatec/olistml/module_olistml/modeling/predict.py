from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from module_olistml.config import MODELS_DIR, PROCESSED_DATA_DIR
from module_olistml.modeling.split import FEATURES

app = typer.Typer()


def load_model(model_path: Path):
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_delay_probability(model, features: pd.DataFrame) -> pd.Series:
    """Retorna a probabilidade de atraso para cada linha do conjunto de features."""
    ordered_features = features.reindex(columns=FEATURES)
    probabilities = model.predict_proba(ordered_features)[:, 1]
    return pd.Series(probabilities, index=ordered_features.index, name="delay_probability")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """Carrega um modelo treinado e salva a probabilidade de atraso por pedido."""
    logger.info(f"Carregando features de: {features_path}")
    features = pd.read_csv(features_path)

    logger.info(f"Carregando modelo de: {model_path}")
    model = load_model(model_path)

    probabilities = predict_delay_probability(model, features)
    output = features.copy()
    output["delay_probability"] = probabilities.to_numpy()
    output.to_csv(predictions_path, index=False)

    logger.success(f"Probabilidades salvas em: {predictions_path}")
    logger.info(output[["delay_probability"]].head())


if __name__ == "__main__":
    app()
