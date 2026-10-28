import pandas as pd
import pytest

from module_deeplearningml.dataset import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    load_dataset,
)
from module_deeplearningml.modeling.predict import main as predict
from module_deeplearningml.modeling.train import main as train


def sample_dataframe(rows: int = 10) -> pd.DataFrame:
    data = {
        name: [index + 40 for index in range(rows)]
        for name in NUMERIC_FEATURE_NAMES
    }
    data.update(
        {
            name: ["category_a" if index % 2 == 0 else "category_b" for index in range(rows)]
            for name in CATEGORICAL_FEATURE_NAMES
        }
    )
    data["target"] = [index % 2 for index in range(rows)]
    return pd.DataFrame(data)


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"age": [42]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_train_and_predict(tmp_path):
    dataframe = sample_dataframe()
    dataset_path = tmp_path / "heart.csv"
    model_path = tmp_path / "heart_classifier.keras"
    features_path = tmp_path / "features.csv"
    predictions_path = tmp_path / "predictions.csv"

    dataframe.to_csv(dataset_path, index=False)
    dataframe.drop(columns="target").head(3).to_csv(features_path, index=False)

    train(dataset_path, model_path, epochs=1)
    predict(features_path, model_path, predictions_path)

    predictions = pd.read_csv(predictions_path)
    assert list(predictions.columns) == ["target_probability"]
    assert len(predictions) == 3
    assert predictions["target_probability"].between(0, 1).all()
