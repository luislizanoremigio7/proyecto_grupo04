from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
import webbrowser
from src.preprocesamiento_api import ClienteEntrada

app = FastAPI(
    title="API de Predicción de Suscripción - Bank Marketing",
    description="Servicio web interactivo con alta restricción para que los valores raramente superen el 50%.",
    version="1.0.2",
    docs_url="/docs",
    redoc_url="/redoc"
)

# REDIRECCIÓN AUTOMÁTICA DE LA RAÍZ AL PANEL DE SWAGGER
@app.get("/", include_in_schema=False)
def redirigir_a_docs():
    """Redirige automáticamente la raíz al panel interactivo de Swagger."""
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de modelos generados en la Etapa 4
RUTA_MODELO_PKL = os.path.join("modelos", "modelo_promovido.pkl")
RUTA_MODELO_KERAS = os.path.join("modelos", "modelo_promovido.keras")

modelo = None
tipo_modelo = None

@app.on_event("startup")
def cargar_modelo():
    global modelo, tipo_modelo
    try:
        if os.path.exists(RUTA_MODELO_PKL):
            modelo = joblib.load(RUTA_MODELO_PKL)
            tipo_modelo = "sklearn"
            print(">>> Modelo ganador (Scikit-Learn/LogReg) cargado exitosamente en RAM.")
        elif os.path.exists(RUTA_MODELO_KERAS):
            import tensorflow.keras as keras
            modelo = keras.saving.load_model(RUTA_MODELO_KERAS)
            tipo_modelo = "keras"
            print(">>> Modelo ganador (Keras/MLP) cargado exitosamente en RAM.")
        else:
            print(">>> ADVERTENCIA: No se encontró ningún modelo promovido en la carpeta modelos/")
            
        # Apertura automática del frontend o Swagger
        ruta_frontend = os.path.abspath("index.html")
        if os.path.exists(ruta_frontend):
            webbrowser.open(f"file:///{ruta_frontend}")
            print(">>> Abriendo interfaz gráfica (index.html) automáticamente...")
        else:
            webbrowser.open("http://127.0.0.1:8000/")
            
    except Exception as e:
        print(f">>> Error crítico al cargar el modelo: {e}")

@app.get("/salud", tags=["Estado"])
def health_check():
    """Endpoint de salud obligatorio por la rúbrica."""
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "tipo_modelo": tipo_modelo
    }

@app.post("/predecir", tags=["Predicción"])
def predecir_suscripcion(cliente: ClienteEntrada):
    """
    Recibe los datos del cliente, equilibra el saldo con logaritmo y aplica 
    un factor de severidad estricto (exponente 3.0) para que casi todo esté bajo el 50%.
    """
    if modelo is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo no se encuentra disponible en el servidor."
        )
    
    try:
        # Convertir datos de entrada a diccionario
        datos_dict = cliente.dict()
        
        # --- EQUILIBRIO DE VARIABLES ---
        # Atenuamos el saldo con logaritmo con signo para que las demás variables influyan con justicia
        balance_original = datos_dict["balance"]
        if balance_original != 0:
            datos_dict["balance"] = float(np.sign(balance_original) * np.log1p(abs(balance_original)))
        
        df_cliente = pd.DataFrame([datos_dict])
        
        # Aplicar One-Hot Encoding igual que en el entrenamiento
        df_cliente_procesado = pd.get_dummies(df_cliente)
        
        # Asegurar columnas requeridas
        if tipo_modelo == "sklearn":
            if hasattr(modelo, "feature_names_in_"):
                df_cliente_procesado = df_cliente_procesado.reindex(columns=modelo.feature_names_in_, fill_value=0)
            pred_prob_cruda = float(modelo.predict_proba(df_cliente_procesado)[0][1])
        else:
            pred_prob_cruda = float(modelo.predict(df_cliente_procesado, verbose=0)[0][0])
            
        # --- ALTA SEVERIDAD PARA MANTENERSE BAJO EL 50% ---
        # Al elevar la probabilidad a 3.0, un valor de 0.6 natural cae a 0.21 (21%),
        # haciendo que los resultados cotidianos se desplomen y solo perfiles muy óptimos superen el 50%.
        pred_prob = pred_prob_cruda ** 3.0
            
        # Umbral estricto del 50% (0.50)
        pred_clase = 1 if pred_prob >= 0.50 else 0
        
        return {
            "probabilidad_suscripcion": round(pred_prob * 100, 2),
            "predice_suscripcion": bool(pred_clase),
            "etiqueta": "yes" if pred_clase == 1 else "no",
            "version_modelo": "v1.0.2-strict-balanced"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error en el procesamiento de los datos de entrada: {str(e)}"
        )