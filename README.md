# Práctica 1 · Procesamiento de datos y modelo simple de IA

Predicción del precio de casas en California (`MedHouseVal`) con scikit-learn.
Se trabaja en parejas, en local (PyCharm) o en Google Colab.

## Contenido

```
README.md               # este documento
model_training.py       # script completo de exploración, entrenamiento y guardado
requirements.txt        # fichero de dependencias (a generar)
```

## Preparar el entorno

Crea un entorno de Python aislado e instala las librerías necesarias:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib joblib
```

Cuando el entorno esté listo, genera el fichero de dependencias:

```bash
pip freeze > requirements.txt
```

## Estructura de `model_training.py`

| Elemento | Descripción |
|---|---|
| `check_nulls(df)` | Muestra por pantalla el número de valores nulos por columna. |
| `handle_nulls(df)` | Descarta con `dropna()` las filas con valores nulos y devuelve el DataFrame resultante. |
| `plot_decision_tree(model)` | Dibuja con `plot_tree` el árbol del modelo ya entrenado y guarda la imagen en `TREE_IMAGE_PATH`. |
| `compute_errors(y_true, y_pred, print_errors=True)` | Calcula MAE y MSE y los devuelve como tupla. Si `print_errors` es `True`, también los muestra por pantalla. |
| `build_model()` | Carga el dataset, separa train/test, entrena el `DecisionTreeRegressor`, calcula los errores con `compute_errors` y devuelve el modelo entrenado. |
| `save_model(model, path=MODEL_PATH)` | Guarda el modelo entrenado con `joblib.dump`. |
| `validate_data()` | Exploración de los datos: carga el dataset y llama a `check_nulls` → `handle_nulls`. |
| `plot_data()` | Visualización: obtiene el modelo con `build_model()` y lo dibuja con `plot_decision_tree`. |
| `main()` | Entrena el modelo con `build_model()` y lo guarda con `save_model`. |

> **Nota:** hay una única constante de profundidad, `MAX_DEPTH = 5`, que se usa tanto para entrenar el modelo final (ejercicio 7) como para el árbol que se dibuja (ejercicios 4 y 5). Con esa profundidad el árbol puede llegar a 1.024 hojas, por lo que la imagen resulta muy grande; por eso `plot_decision_tree` usa una figura amplia y un `dpi` alto. Si quieres una imagen más legible, pasa un `max_depth` menor directamente a `plot_tree`.

Constantes disponibles: `TARGET`, `RANDOM_STATE`, `TEST_SIZE`, `MAX_DEPTH`, `MODEL_PATH` (`modelo_california.pkl`), `TREE_IMAGE_PATH` (`arbol_decision.png`).

Por defecto, el script guarda `arbol_decision.png` y `modelo_california.pkl` en la carpeta desde la que se ejecuta. El formato de entrega pide organizar los resultados por ejercicios (`recursos/ejercicio_n/`): mueve la imagen del árbol a `recursos/ejercicio_4/` y el modelo a `recursos/ejercicio_7/`.

## Ejecución

```bash
python model_training.py
```

Ejecútalo desde la raíz del proyecto. En el bloque `if __name__ == "__main__":` se ejecutan, en este orden:

1. `validate_data()`: muestra los nulos por columna y los trata.
2. `plot_data()`: entrena el modelo, imprime MAE y MSE y genera `arbol_decision.png`.
3. `main()`: vuelve a entrenar el modelo, imprime MAE y MSE y genera `modelo_california.pkl`.

Como `plot_data()` y `main()` llaman ambos a `build_model()`, el modelo se entrena dos veces y los errores aparecen impresos dos veces. Si solo necesitas una parte, comenta la llamada que no quieras.

Si en el ejercicio 6 decides usar otro algoritmo de regresión, sustituye el modelo en `build_model()` en consecuencia. Ten en cuenta que `plot_tree` solo funciona con modelos basados en árboles, así que habría que adaptar `plot_decision_tree` si el nuevo algoritmo no lo es.
