from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from module_olistml.config import INTERIM_DATA_DIR
from module_olistml.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
)
from module_olistml.modeling.split import FEATURES, TARGET

app = typer.Typer()


def get_model_factories():
    return {
        "Gradient Boosting": create_gradient_boosting_pipeline,
        "XGBoost": create_xgboost_pipeline,
        "LightGBM": create_lightgbm_pipeline,
    }


def select_best_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Busca o limiar que maximiza F1 para a classificação binária."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best["f1"]:
            best["threshold"] = float(threshold)
            best["precision"] = float(precision)
            best["recall"] = float(recall)
            best["f1"] = float(f1)

    return best


def cross_validate_models(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Executa validação cruzada estratificada para cada pipeline do projeto."""
    X = data[FEATURES]
    y = data[TARGET]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: dict[str, dict[str, float]] = {}

    for name, factory in get_model_factories().items():
        model = factory()
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        roc_auc = roc_auc_score(y, y_proba)
        best = select_best_threshold(y, y_proba)

        results[name] = {
            "roc_auc": round(float(roc_auc), 4),
            "threshold": round(float(best["threshold"]), 2),
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
            "f1": round(float(best["f1"]), 4),
        }

        logger.info(f"CV - {name}: ROC-AUC={results[name]['roc_auc']:.4f}, F1={results[name]['f1']:.4f}, threshold={results[name]['threshold']:.2f}")

    return results


@app.command()
def main(
    data_path: Path = INTERIM_DATA_DIR / "orders_dataset_refined.csv",
    n_splits: int = 5,
    random_state: int = 42,
):
    """Executa a validação cruzada para os modelos treinados no projeto."""
    logger.info(f"Carregando dataset para validação cruzada: {data_path}")
    data = pd.read_csv(data_path)

    metrics = cross_validate_models(data, n_splits=n_splits, random_state=random_state)
    logger.success("Validação cruzada concluída.")
    for model_name, model_metrics in metrics.items():
        logger.info(f"{model_name}: {model_metrics}")

    return metrics


if __name__ == "__main__":
    app()
