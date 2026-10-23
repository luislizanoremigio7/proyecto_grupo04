"""Módulo de carga, limpieza y partición de datos para Bank Marketing (Grupo 4).

Este script gestiona el preprocesamiento del dataset asegurando el aislamiento
estricto entre entrenamiento, validación y prueba, evitando la fuga de datos.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

SEMILLA = 42

# Variables elegidas con el ranking de correlación (ver ranking_correlacion y el notebook 02).
VARIABLES_SELECCIONADAS = ["poutcome", "month", "contact", "housing", "job"]


def cargar_y_limpiar_datos(ruta_csv: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset de Bank Marketing, limpia nulos y elimina la fuga de datos.

    Args:
        ruta_csv: Ruta local hacia el archivo CSV (bank-full.csv).

    Returns:
        X: DataFrame con las 15 columnas originales (sin codificar todavía).
        y: Serie con la variable objetivo codificada (0 y 1).
    """
    path = Path(ruta_csv)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")

    df = pd.read_csv(path, sep=";")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Al quitar 'duration' aparecen filas repetidas; se eliminan antes de particionar
    # para que un mismo cliente no quede en entrenamiento y en prueba a la vez.
    df = df.drop_duplicates().reset_index(drop=True)

    if "y" in df.columns:
        df["y"] = df["y"].apply(lambda val: 1 if str(val).strip().lower() == "yes" else 0)
    
    y = df["y"]
    X = df.drop(columns=["y"])

    # La codificación One-Hot ya no se hace aquí: la hace el preprocesador de
    # src/modelo.py, ajustado solo con entrenamiento y guardado para la API.
    return X, y


def preparar_particiones(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    Aplica estratificación (stratify=y) debido al desbalance de clases (~12% positivos).
    No escala los datos: el escalado y la codificación se ajustan después, solo con
    el conjunto de entrenamiento (ver construir_preprocesador en src/modelo.py).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEMILLA, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1764, random_state=SEMILLA, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def v_de_cramer(x: pd.Series, y: pd.Series) -> float:
    """Calcula la V de Cramér entre dos variables categóricas (0 = ninguna relación, 1 = total)."""
    tabla = pd.crosstab(x, y)
    if min(tabla.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))


def ranking_correlacion(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Ordena las variables según su relación con la variable objetivo (solo entrenamiento).

    - Numéricas: valor absoluto de la correlación de Pearson con y.
    - Categóricas: V de Cramér con y. No se codifican como 0, 1, 2... porque eso
      inventa un orden que no existe (por ejemplo, entre los meses).
    """
    filas = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            valor = abs(X_train[col].corr(y_train))
            medida = "Pearson |r|"
        else:
            valor = v_de_cramer(X_train[col], y_train)
            medida = "V de Cramer"
        filas.append({"variable": col, "medida": medida, "asociacion": round(float(valor), 4)})
    ranking = pd.DataFrame(filas).sort_values("asociacion", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    return ranking
