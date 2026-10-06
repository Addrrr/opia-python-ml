import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing


california = fetch_california_housing(as_frame=True)
X, y = california.data, california.target # y = MedHouseVal (en cientos de miles de $)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = DecisionTreeRegressor(max_depth=8, random_state=0)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predicciones))

joblib.dump(modelo, "modelo_california.pkl")
