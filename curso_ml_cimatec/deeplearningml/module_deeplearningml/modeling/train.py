from pathlib import Path

from loguru import logger
import typer

from module_deeplearningml.config import MODELS_DIR, PROCESSED_DATA_DIR
from module_deeplearningml.dataset import TARGET_FEATURE_NAME, load_dataset
from module_deeplearningml.features import create_model, prepare_inputs

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "heart.csv",
    model_path: Path = MODELS_DIR / "heart_classifier.keras",
    epochs: int = 50,
) -> None:
    dataframe = load_dataset(input_path)
    validation_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    training_dataframe = dataframe.drop(validation_dataframe.index)
    if training_dataframe.empty or validation_dataframe.empty:
        raise ValueError("Dataset must have enough rows for training and validation.")

    model = create_model(training_dataframe)
    logger.info(
        f"Training with {len(training_dataframe)} rows; "
        f"validating with {len(validation_dataframe)} rows."
    )
    model.fit(
        prepare_inputs(training_dataframe),
        training_dataframe[TARGET_FEATURE_NAME].to_numpy(dtype="float32"),
        validation_data=(
            prepare_inputs(validation_dataframe),
            validation_dataframe[TARGET_FEATURE_NAME].to_numpy(dtype="float32"),
        ),
        epochs=epochs,
        batch_size=32,
        verbose=2,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.success(f"Trained model saved to {model_path}.")


if __name__ == "__main__":
    app()
