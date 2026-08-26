from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Modelos para dados tabulares (estado da arte)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

NUMERIC_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state",
]

def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Nas colunas mnuméricas
            ("numeric", "passthrough", NUMERIC_FEATURES),
            
            # Nas colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), 
             CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria o pipeline do pré-processamento e treinamento do modelo
    """

    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100,   # número de árvores na floresta.
        learning_rate=0.1,  # taxa de aprendizado do modelo.
        max_depth=3,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

def create_xgboost_pipeline() -> Pipeline:
    preprocessor = create_preprocessor()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


def create_lightgbm_pipeline() -> Pipeline:
    preprocessor = create_preprocessor()

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )




