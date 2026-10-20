"""Módulo de carga, limpieza y partición de datos para Bank Marketing (Grupo 4).

Este script gestiona el preprocesamiento del dataset asegurando el aislamiento
estricto entre entrenamiento, validación y prueba, evitando la fuga de datos.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cargar_y_limpiar_datos(ruta_csv: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset de Bank Marketing, limpia nulos y elimina la fuga de datos.

    Args:
        ruta_csv: Ruta local hacia el archivo CSV (bank-full.csv).

    Returns:
        X: DataFrame con las características procesadas.
        y: Serie con la variable objetivo codificada (0 y 1).
    """
    path = Path(ruta_csv)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")

    df = pd.read_csv(path, sep=";")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    if "y" in df.columns:
        df["y"] = df["y"].apply(lambda val: 1 if str(val).strip().lower() == "yes" else 0)
    
    y = df["y"]
    X = df.drop(columns=["y"])

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def preparar_particiones(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    Aplica estratificación (stratify=y) debido al desbalance de clases (~12% positivos).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1764, random_state=42, stratify=y_train_val
    )

    scaler = StandardScaler()
    
    columnas_numericas = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    cols_a_escalar = [col for col in columnas_numericas if col in X_train.columns]

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    if cols_a_escalar:
        X_train_scaled[cols_a_escalar] = scaler.fit_transform(X_train[cols_a_escalar])
        X_val_scaled[cols_a_escalar] = scaler.transform(X_val[cols_a_escalar])
        X_test_scaled[cols_a_escalar] = scaler.transform(X_test[cols_a_escalar])

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler