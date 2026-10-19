# 🏦 Proyecto de Predicción de Suscripción a Depósito a Plazo Fijo - Bank Marketing

Este proyecto forma parte de la asignatura de **Tecnologías Emergentes** (FIS-UNCP). Implementa una solución end-to-end orientada a machine learning que abarca desde la limpieza y exploración de datos, la construcción y evaluación comparativa de modelos (redes neuronales y clasificadores clásicos), hasta la puesta en producción mediante una API REST en **FastAPI**, un chatbot interactivo por consola y pruebas de calidad automáticas.

---

## 📐 Estructura del Proyecto

```text
proyecto_grupo04/
├── data/                       # Archivos de datos de entrenamiento y prueba
├── modelos/                    # Modelos guardados y versionados (.pkl / .keras)
├── src/                        # Código fuente modular
│   ├── datos.py                # Carga, limpieza y exploración de datos (Etapa 2)
│   ├── modelo.py               # Definición y entrenamiento de modelos (Etapa 3)
│   ├── evaluacion.py           # Métricas, versionado y fichas técnicas (Etapa 4)
│   ├── api.py                  # Servidor REST con FastAPI (Etapa 5)
│   ├── preprocesamiento_api.py # Esquema Pydantic y validación de entradas
│   └── chatbot.py              # Asistente virtual interactivo por consola (Etapa 6)
├── tests/                      # Suite de pruebas automáticas con Pytest (Etapa 7)
│   ├── test_datos.py           # Pruebas de esquema de datos e integridad
│   ├── test_api.py             # Pruebas de endpoints del API
│   └── test_chatbot.py         # Pruebas de mapeos y slots del chatbot
├── index.html                  # Interfaz de pruebas para la API
├── requirements.txt            # Lista de dependencias del entorno
└── README.md                   # Documentación y guía de reproducción
```

---

## 🛠️ Requisitos Previos e Instalación

### 1. Clonar el repositorio y ubicarse en la raíz
```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto_grupo04
```

### 2. Crear y activar el entorno virtual
* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
* **En Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar las dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Guía de Reproducción Paso a Paso

### Paso 1: Procesamiento de Datos (Etapa 2)
Ejecuta el script de carga, limpieza y división libre de fuga de información:
```bash
python -m src.datos
```

### Paso 2: Entrenar y Evaluar Modelos (Etapas 3 y 4)
Para entrenar la línea base (Regresión Logística / Scikit-Learn) y la Red Neuronal (Keras/TensorFlow), compararlas y guardar la mejor versión promovida en `modelos/`:
```bash
python -m src.modelo
python -m src.evaluacion
```

---

## 🌐 Ejecución de Servicios y Aplicaciones

### 1. Iniciar la API REST (Etapa 5)
Inicia el servidor backend utilizando Uvicorn:
```bash
uvicorn src.api:app --reload --port 8000
```
* **Documentación Interactiva (Swagger):** Abre tu navegador en `http://127.0.0.1:8000/docs`
* **Endpoint de Salud:** `http://127.0.0.1:8000/salud`
* **Endpoint de Predicción:** `POST http://127.0.0.1:8000/predecir`

### 2. Iniciar el Chatbot Interactivo (Etapa 6)
En una **nueva ventana de terminal** (asegurándote de mantener corriendo la API y con el entorno `.venv` activo), ejecuta:
```bash
python -m src.chatbot
```
*El chatbot te guiará paso a paso mediante 15 preguntas de recolección de datos (*slots*), gestionará reintentos y derivará a un asesor humano en caso de fallo u opción explícita.*

---

## 🧪 Pruebas de Calidad Automáticas (Etapa 7)

Para ejecutar la suite completa de pruebas unitarias que verifican los componentes de datos, API y chatbot:

```bash
pytest
```

*(Las pruebas verifican esquemas de datos con Pydantic, la respuesta de los endpoints HTTP con `TestClient` y la consistencia de los diccionarios de traducción en el chatbot).*

---

## 📌 Tecnologías Utilizadas
* **Lenguaje:** Python 3.12+
* **Ciencia de Datos & ML:** Pandas, NumPy, Scikit-Learn, TensorFlow / Keras, Joblib
* **Backend & API:** FastAPI, Uvicorn, Pydantic, HTTPX
* **CLI & Chatbot:** Rich, Requests
* **Testing:** Pytest