from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import pandas as pd
import os
import sys
import webbrowser
from src.preprocesamiento_api import ClienteEntrada
from src.datos import VARIABLES_SELECCIONADAS

app = FastAPI(
    title="API de Predicción de Suscripción - Bank Marketing",
    description="Indica si conviene llamar a un cliente (sí/no) para ofrecerle un depósito a plazo. "
                "La predicción se hace antes de la llamada, por eso no recibe 'duration'.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", include_in_schema=False)
def redirigir_a_docs():
    """
    Redirige automáticamente la ruta raíz al panel interactivo de Swagger UI.
    """
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
def error_de_validacion(request: Request, exc: RequestValidationError):
    """
    Devuelve el código 400 (en lugar del 422 de FastAPI) con mensajes claros en español.
    """
    mensajes = []
    for error in exc.errors():
        campo = ".".join(str(p) for p in error.get("loc", []) if p != "body") or "solicitud"
        tipo = error.get("type", "")
        if tipo == "missing":
            mensajes.append(f"Falta el campo '{campo}'.")
        elif tipo == "extra_forbidden":
            mensajes.append(f"El campo '{campo}' no está permitido. Campos válidos: {', '.join(VARIABLES_SELECCIONADAS)}.")
        elif tipo == "enum":
            opciones = str(error.get("ctx", {}).get("expected", "")).replace(" or ", " o ")
            mensajes.append(f"Valor no válido en '{campo}'. Opciones: {opciones}.")
        else:
            mensajes.append(f"Dato no válido en '{campo}'.")
    return JSONResponse(status_code=400, content={"error": "Datos de entrada no válidos", "detalles": mensajes})

RUTA_PROMOVIDO = os.path.join("modelos", "modelo_promovido.json")
RUTA_PREPROCESADOR = os.path.join("modelos", "preprocesador.pkl")

modelo = None
tipo_modelo = None
preprocesador = None
promovido = None

@app.on_event("startup")
def cargar_modelo():
    """
    Evento de inicio de FastAPI. Lee modelos/modelo_promovido.json y carga en memoria RAM
    (una sola vez) el modelo ganador, su preprocesador y su umbral. Luego abre
    automáticamente el frontend local o el panel de documentación en el navegador.
    """
    global modelo, tipo_modelo, preprocesador, promovido
    try:
        if os.path.exists(RUTA_PROMOVIDO):
            with open(RUTA_PROMOVIDO, encoding="utf-8") as f:
                promovido = json.load(f)
            preprocesador = joblib.load(RUTA_PREPROCESADOR)
            if promovido["archivo"].endswith(".keras"):
                from tensorflow import keras
                modelo = keras.models.load_model(promovido["archivo"])
                tipo_modelo = "keras"
            else:
                modelo = joblib.load(promovido["archivo"])
                tipo_modelo = "sklearn"
            print(f">>> Modelo ganador ({promovido['modelo']}) cargado exitosamente en RAM.")
        else:
            print(">>> ADVERTENCIA: No se encontró ningún modelo promovido. Ejecute: python entrenar.py")

        ruta_frontend = os.path.abspath("index.html")
        if "pytest" in sys.modules:
            pass  # durante las pruebas automáticas no se abre el navegador
        elif os.path.exists(ruta_frontend):
            webbrowser.open(f"file:///{ruta_frontend}")
            print(">>> Abriendo interfaz gráfica (index.html) automáticamente...")
        else:
            webbrowser.open("http://127.0.0.1:8000/")
            
    except Exception as e:
        print(f">>> Error crítico al cargar el modelo: {e}")

@app.get("/salud", tags=["Estado"])
def health_check():
    """
    Verifica el estado operativo actual del servicio web y el estado de carga del modelo.
    """
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "tipo_modelo": tipo_modelo,
        "version_modelo": promovido["modelo"] if promovido else None,
        "variables": VARIABLES_SELECCIONADAS,
    }

@app.post("/predecir", tags=["Predicción"])
def predecir_suscripcion(cliente: ClienteEntrada):
    """
    Aplica a los datos del cliente el mismo preprocesador del entrenamiento, obtiene
    el puntaje del modelo (sin modificarlo) y lo compara con el umbral elegido en
    validación según el costo de los errores. Responde "sí" (conviene llamar) o "no".
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no se encuentra disponible. Ejecute 'python entrenar.py' y reinicie la API."
        )

    try:
        df_cliente = pd.DataFrame([cliente.model_dump(mode="json")])[VARIABLES_SELECCIONADAS]
        df_cliente_procesado = preprocesador.transform(df_cliente)

        if tipo_modelo == "sklearn":
            puntaje = float(modelo.predict_proba(df_cliente_procesado)[0][1])
        else:
            puntaje = float(modelo.predict(df_cliente_procesado, verbose=0)[0][0])

        umbral = float(promovido["umbral"])
        pred_clase = 1 if puntaje >= umbral else 0

        return {
            "respuesta": "sí" if pred_clase == 1 else "no",
            "predice_suscripcion": bool(pred_clase),
            "etiqueta": "yes" if pred_clase == 1 else "no",
            "puntaje": round(puntaje, 4),
            "umbral": umbral,
            "version_modelo": promovido["modelo"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error en el procesamiento de los datos de entrada: {str(e)}"
        )