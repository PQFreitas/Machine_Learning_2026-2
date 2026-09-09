from module_olistml.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olistml.dataset import (
    load_data,
    create_dataset,
    save_dataset,
)

from module_olistml.features import create_features

from module_olistml.modeling.split import split_data

from module_olistml.modeling.train import (
    train_model,
)

from module_olistml.modeling.evaluate import (
    evaluate_model,
)

from module_olistml.modeling.cross_validation import (
    cross_validate_models,
)

from module_olistml.modeling.predict import predict

from loguru import logger


def main():

    logger.info(
        "Iniciando preparação do dataset..."
    )

    orders, items, customers = load_data(
        orders_path=(
            RAW_DATA_DIR
            / "olist_orders_dataset.csv"
        ),
        items_path=(
            RAW_DATA_DIR
            / "olist_order_items_dataset.csv"
        ),
        costumers_path=(
            RAW_DATA_DIR
            / "olist_customers_dataset.csv"
        ),
    )

    data = create_dataset(
        orders,
        items,
        customers,
    )

    data = create_features(data)

    save_dataset(
        data,
        INTERIM_DATA_DIR
        / "orders_dataset_refined.csv",
    )

    # =============================================
    # TRAIN / TEST
    # =============================================

    X_train, X_test, y_train, y_test = (
        split_data(data)
    )

    # =============================================
    # CROSS VALIDATION
    # Seleciona modelo + threshold
    # =============================================

    (
        best_model_name,
        best_threshold,
    ) = cross_validate_models(
        X_train,
        y_train,
    )

    logger.info(
        f"Modelo escolhido: "
        f"{best_model_name}"
    )

    logger.info(
        f"Threshold escolhido: "
        f"{best_threshold:.2f}"
    )

    # =============================================
    # TREINAMENTO FINAL
    # =============================================

    model = train_model(
        model_name=best_model_name,
        threshold=best_threshold,
        X_train=X_train,
        y_train=y_train,
        model_path=(
            MODELS_DIR
            / "best_model.joblib"
        ),
        metadata_path=(
            MODELS_DIR
            / "metadata.json"
        ),
    )

    # =============================================
    # TESTE FINAL
    # =============================================

    evaluate_model(
        model=model,
        model_name=best_model_name,
        X_test=X_test,
        y_test=y_test,
        threshold=best_threshold,
    )

    predictions = predict(
        model=model,
        X=X_test,
        threshold=best_threshold,
    )

    predictions_path = (
        INTERIM_DATA_DIR
        / "test_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=True,
    )

    logger.success(
        f"Predições salvas em: {predictions_path}"
    )

    logger.success(
        "Pipeline executado com sucesso."
    )


if __name__ == "__main__":
    main()