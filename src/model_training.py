"""Plantilla: predicción del precio de casas en California.

Completa las partes marcadas con TODO. No cambies la estructura general.
Autores: <nombre1>, <nombre2>
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

TARGET = "MedHouseVal"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 3
MODEL_PATH = "modelo_california.pkl"
TREE_IMAGE_PATH = "arbol_decision.png"


def check_nulls(df: pd.DataFrame):
    """Muestra por pantalla el número de valores nulos por columna."""
    print("Número de valores nulos por columna:")
    print(df.isnull().sum())


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Trata los valores nulos del DataFrame (descártalos con
    dropna() o rellénalos con fillna(), según lo que decidas) y
    devuelve el DataFrame resultante."""
    df = df.dropna()
    print(f"Filas restantes después de eliminar nulos: {len(df)}")
    return df


def plot_decision_tree(model):
    """Dibuja el árbol entrenado y guarda la imagen en un PNG."""
    plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        filled=True,
        rounded=True,
        fontsize=8,
        feature_names=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        class_names=None,
    )
    plt.title("Árbol de decisión: precio de viviendas en California")
    plt.savefig(TREE_IMAGE_PATH, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Árbol de decisión guardado en: {TREE_IMAGE_PATH}")


def compute_errors(y_true, y_pred, print_errors: bool = True) -> tuple[float, float]:
    """Calcula el MAE y el MSE entre los valores reales y las predicciones.

    Si `print_errors` es True (por defecto), los muestra por pantalla.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    if print_errors:
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
    return mae, mse


def save_model(model, path: str = MODEL_PATH):
    """Guarda el modelo entrenado utilizando joblib."""
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def validate_data():
    """Exploración y validación de los datos."""
    df = fetch_california_housing(as_frame=True).frame
    check_nulls(df)
    df = handle_nulls(df)

    print(f"Dimensiones del conjunto de datos: {df.shape}")
    return df


def build_model():
    """Carga los datos, entrena el modelo, muestra sus errores y lo devuelve."""
    df = fetch_california_housing(as_frame=True).frame

    check_nulls(df)
    df = handle_nulls(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    compute_errors(y_test, y_pred)

    return model


def plot_data():
    """Visualización: dibuja el árbol de decisión del modelo entrenado."""
    model = build_model()
    plot_decision_tree(model)


def main():
    """Entrena y guarda el modelo entrenado."""
    model = build_model()
    save_model(model)


if __name__ == "__main__":
    validate_data()
    plot_data()
    main()
