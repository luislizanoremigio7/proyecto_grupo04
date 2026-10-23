# 🏦 Proyecto de Predicción de Suscripción a Depósito a Plazo Fijo - Bank Marketing

Este proyecto forma parte de la asignatura de **Tecnologías Emergentes** (FIS-UNCP). Implementa una solución end-to-end orientada a machine learning que abarca desde la limpieza y exploración de datos, la construcción y evaluación comparativa de modelos (redes neuronales y clasificadores clásicos), hasta la puesta en producción mediante una API REST en **FastAPI**, un chatbot interactivo por consola y pruebas de calidad automáticas.

---

## 📐 Estructura del Proyecto

```text
proyecto_grupo04/
├── data/                       # bank-full.csv (original) y particiones train/val/test
├── modelos/                    # Modelos (.pkl / .keras), preprocesador, una ficha JSON por versión
│                               # y modelo_promovido.json (versión que usa la API)
├── src/                        # Código fuente modular
│   ├── datos.py                # Carga, limpieza y exploración de datos (Etapa 2)
│   ├── modelo.py               # Definición y entrenamiento de modelos (Etapa 3)
│   ├── evaluacion.py           # Métricas, versionado y fichas técnicas (Etapa 4)
│   ├── api.py                  # Servidor REST con FastAPI (Etapa 5)
│   ├── preprocesamiento_api.py # Esquema Pydantic y validación de entradas
│   └── chatbot.py              # Asistente virtual interactivo por consola (Etapa 6)
├── tests/                      # Suite de pruebas automáticas con Pytest (Etapa 7)
│   ├── test_datos.py           # Pruebas de datos, preprocesamiento, umbral y métricas
│   ├── test_api.py             # Pruebas de endpoints del API (éxito y errores 400)
│   └── test_chatbot.py         # Pruebas de mapeos, slots y fallback del chatbot
├── notebooks/                  # Exploración, modelado y evaluación (ejecutados)
├── docs/                       # Diagrama de flujo del chatbot
├── entrenar.py                 # Ejecuta las Etapas 2, 3 y 4 y guarda los modelos
├── index.html                  # Interfaz de pruebas para la API
├── informe.pdf                 # Informe técnico
├── requirements.txt            # Lista de dependencias del entorno
└── README.md                   # Documentación y guía de reproducción
```

---

## 🛠️ Requisitos Previos e Instalación

### 1. Clonar el repositorio y ubicarse en la raíz
```bash
git clone https://github.com/luislizanoremigio7/proyecto_grupo04.git
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

### Paso único: Entrenar, evaluar y versionar (Etapas 2, 3 y 4)
```bash
python entrenar.py
```
Este script:
1. Carga `data/bank-full.csv`, elimina `duration` (solo se conoce después de la llamada) y los duplicados, y divide en 70/15/15 estratificado con semilla 42.
2. Calcula el ranking de correlación con la variable objetivo (Pearson para numéricas y V de Cramér para categóricas) y usa las 5 variables más asociadas: `poutcome`, `month`, `contact`, `housing` y `job`.
3. Ajusta el preprocesador (One-Hot) solo con entrenamiento y entrena el modelo trivial, la regresión logística, la MLP v1 y la MLP v2.
4. Elige en validación el umbral de cada modelo minimizando el costo de los errores (falso negativo = 5, falso positivo = 1) y el modelo ganador (menor costo).
5. Evalúa el ganador una sola vez en prueba y guarda una ficha JSON por versión y `modelos/modelo_promovido.json`.

Con la misma semilla, dos ejecuciones dan las mismas métricas.

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

Ejemplo de entrada (solo las 5 variables; cualquier otro campo se rechaza):
```json
{"poutcome": "success", "month": "oct", "contact": "cellular", "housing": "no", "job": "management"}
```
Respuesta: `respuesta` ("sí" o "no"), `predice_suscripcion`, `etiqueta`, `puntaje` (salida del modelo), `umbral` y `version_modelo`. La respuesta es "sí" cuando `puntaje >= umbral`.

Si falta un campo, un valor no es válido o se envía un campo no permitido (como `duration`), la API responde **400** con un mensaje en español.

### 2. Iniciar el Chatbot Interactivo (Etapa 6)
En una **nueva ventana de terminal** (asegurándote de mantener corriendo la API y con el entorno `.venv` activo), ejecuta:
```bash
python -m src.chatbot
```
*El chatbot pregunta paso a paso las 5 variables del modelo (*slots*). Si no entiende una respuesta, avisa el número de intento; al tercer intento fallido, o si se escribe "humano", deriva a un asesor. El resultado es **SÍ, conviene llamar** o **NO, no priorizar por ahora**.*

---

## 🧪 Pruebas de Calidad Automáticas (Etapa 7)

Para ejecutar la suite completa de pruebas unitarias que verifican los componentes de datos, API y chatbot:

```bash
python -m pytest
```

*(Las pruebas verifican la limpieza sin fuga, la partición estratificada, el preprocesador ajustado solo con entrenamiento, el umbral por costo, la respuesta de la API en casos de éxito y de error 400, y el chatbot: mapeos, slots y límite de 3 intentos).*

---

## 📌 Tecnologías Utilizadas
* **Lenguaje:** Python 3.12+
* **Ciencia de Datos & ML:** Pandas, NumPy, SciPy, Scikit-Learn, TensorFlow / Keras, Joblib, Matplotlib, Seaborn
* **Backend & API:** FastAPI, Uvicorn, Pydantic, HTTPX
* **CLI & Chatbot:** Rich, Requests
* **Testing:** Pytest