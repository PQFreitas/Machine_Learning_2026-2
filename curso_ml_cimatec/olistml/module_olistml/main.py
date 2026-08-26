from module_olistml.config import RAW_DATA_DIR, INTERIM_DATA_DIR

from module_olistml.dataset import load_data, create_dataset, save_dataset
from module_olistml.features import create_features
from module_olistml.modeling.split import split_data
from module_olistml.modeling.train import train_models
from module_olistml.modeling.evaluate import evaluate_models

from loguru import logger

def main():
    logger.info("Iniciando a preparação do dataset.")
    
    orders, items, costumers = load_data(
        orders_path= RAW_DATA_DIR / "olist_orders_dataset.csv",
        items_path= RAW_DATA_DIR / "olist_order_items_dataset.csv",
        costumers_path= RAW_DATA_DIR / "olist_customers_dataset.csv")

    data = create_dataset(orders, items, costumers)
    data = create_features(data)
    
    # Salvando resultado
    save_dataset(data, INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    # Dividindo o dataset em treino e teste
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Treinando os modelos
    trained_models = train_models(X_train, y_train)
    
    # Avaliando os modelos treinados
    evaluate_models(trained_models, X_test, y_test)
    

if __name__ == "__main__":
    main()
