import keras
from keras import layers
import numpy as np
import pandas as pd

from module_deeplearningml.dataset import (
    CATEGORICAL_FEATURE_NAMES,
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
)


def prepare_inputs(dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
    missing_columns = sorted(set(FEATURE_NAMES) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing columns: {', '.join(missing_columns)}")
    if dataframe[FEATURE_NAMES].isna().any().any():
        raise ValueError("Input data contains missing feature values.")

    inputs = {
        name: dataframe[name].to_numpy(dtype="float32").reshape(-1, 1)
        for name in NUMERIC_FEATURE_NAMES
    }
    inputs.update(
        {
            name: dataframe[name].astype(str).to_numpy(dtype=object)
            for name in CATEGORICAL_FEATURE_NAMES
        }
    )
    return inputs


def create_model(training_dataframe: pd.DataFrame) -> keras.Model:
    prepared_training_data = prepare_inputs(training_dataframe)
    model_inputs = {}
    encoded_features = []

    for name in NUMERIC_FEATURE_NAMES:
        feature_input = layers.Input(shape=(1,), dtype="float32", name=name)
        normalizer = layers.Normalization(name=f"{name}_normalization")
        normalizer.adapt(prepared_training_data[name])
        model_inputs[name] = feature_input
        encoded_features.append(normalizer(feature_input))

    for name in CATEGORICAL_FEATURE_NAMES:
        vocabulary = sorted(training_dataframe[name].astype(str).unique().tolist())
        feature_input = layers.Input(shape=(), dtype="string", name=name)
        lookup = layers.StringLookup(
            vocabulary=vocabulary,
            output_mode="one_hot",
            name=f"{name}_lookup",
        )
        model_inputs[name] = feature_input
        encoded_features.append(lookup(feature_input))

    all_features = layers.concatenate(encoded_features)
    hidden = layers.Dense(32, activation="relu")(all_features)
    hidden = layers.Dropout(0.5)(hidden)
    output = layers.Dense(1, activation="sigmoid")(hidden)

    model = keras.Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
