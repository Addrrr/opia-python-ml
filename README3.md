# Predicción del precio de casas en California (servicio web)

Este proyecto despliega un modelo de aprendizaje automático como servicio web. El modelo se ha entrenado con el dataset California Housing y estima el valor medio de las viviendas de una zona a partir de 8 variables. El proyecto tiene dos partes que se despliegan como servicios independientes en Render.

La primera es una API construida con FastAPI (`app.py`), que carga el modelo y expone el endpoint `POST /predict`. Recibe un JSON con las 8 variables (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude y Longitude) y devuelve la predicción en la clave `MedHouseVal`. Ese valor está en cientos de miles de dólares: 4.53 equivale a $453,000.

La segunda es una aplicación Streamlit (`streamlit_app.py`) que actúa como cliente de la API. El usuario solo introduce el ingreso medio de la zona y el número medio de habitaciones; el resto de variables se rellenan con valores por defecto (la mediana aproximada del dataset). Antes de enviar la petición, la app valida que están las 8 variables y, si falta alguna, no envía nada y muestra cuáles faltan. Cuando recibe la respuesta, convierte el valor a dólares y lo muestra al usuario junto con el valor crudo de la API.

La app controla los errores de comunicación (sin conexión, URL incorrecta o timeout) y las respuestas de error de la API (422, 500...) con mensajes claros para el usuario. Los detalles técnicos se guardan en el log.

Autores: Mariano, Adrián

## Documentación de la API

### Endpoint
`POST /predict`

### Ejemplo de petición
```json
{
  "MedInc": 3.5,
  "HouseAge": 29.0,
  "AveRooms": 5.2,
  "AveBedrms": 1.05,
  "Population": 1166.0,
  "AveOccup": 2.8,
  "Latitude": 34.26,
  "Longitude": -118.49
}
```

### Ejemplo de respuesta correcta (200)
```json
{"MedHouseVal": 1.86}
```
El valor está en cientos de miles de dólares (1.86 = $186,688).

### Errores
| Código | Significado | Cuerpo |
|---|---|---|
| 422 | Falta un campo o tiene un tipo incorrecto | JSON: `{"detail": [...]}` |
| 500 | Error interno del servidor | Texto `Internal Server Error` (no es JSON) |

### Ejemplo con curl
```bash
curl -X POST https://opia-python-ml.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 29.0, "AveRooms": 5.2, "AveBedrms": 1.05, "Population": 1166.0, "AveOccup": 2.8, "Latitude": 34.26, "Longitude": -118.49}'
```

## Contenido del repositorio

- `app.py`: API FastAPI que carga el modelo y expone `POST /predict`.
- `streamlit_app.py`: aplicación Streamlit (interfaz, validación, llamada a la API y control de errores).
- `modelo_california.pkl`: modelo entrenado.
- `requirements.txt`: dependencias del proyecto.
- `README.md`: este documento.

## Pasos del despliegue

1. Subir el repositorio a GitHub con todos los ficheros anteriores.
2. En Render, crear un **Web Service** para la API conectado al repositorio:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Comprobar que la API responde en `/docs` y copiar su URL pública.
4. En `streamlit_app.py`, poner esa URL en `API_URL` (sin barra final).
5. Crear un segundo **Web Service** en Render para la app Streamlit:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
6. Esperar a que ambos servicios estén en estado *Live*. En el plan gratuito pueden tardar más de un minuto en arrancar la primera vez.

Ejecución en local (con la API en marcha):
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## URL final

- API (FastAPI): <https://opia-python-ml.onrender.com>
- Aplicación (Streamlit): <https://opia-python-ml-1.onrender.com>
