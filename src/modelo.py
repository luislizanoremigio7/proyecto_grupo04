"""Módulo de modelado: Línea base (trivial y simple) y Redes Neuronales (MLP) (Etapa 3).

Implementa la línea base completa (modelo trivial y modelo simple) y las dos 
versiones de Redes Neuronales exigidas en la rúbrica para Bank Marketing.
"""

import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

COLUMNAS_NUMERICAS = ["age", "balance", "day", "campaign", "pdays", "previous"]


def construir_preprocesador(variables: list) -> ColumnTransformer:
    """Crea el preprocesador para las variables elegidas (sin ajustar).

    Las categóricas pasan por One-Hot (las categorías nuevas se ignoran) y las
    numéricas, si hubiera, por StandardScaler. Se ajusta SOLO con entrenamiento
    y se guarda en modelos/preprocesador.pkl para que la API lo reutilice.
    """
    numericas = [c for c in variables if c in COLUMNAS_NUMERICAS]
    categoricas = [c for c in variables if c not in COLUMNAS_NUMERICAS]
    return ColumnTransformer(
        [
            ("numericas", StandardScaler(), numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ],
        remainder="drop",
    )


def fijar_semilla(semilla: int = 42) -> None:
    """Fija la semilla de Python, NumPy y TensorFlow para que el entrenamiento sea reproducible."""
    keras.utils.set_random_seed(semilla)
    tf.config.experimental.enable_op_determinism()


def calcular_pesos_clase(y_train) -> dict:
    """Pesos de clase 'balanced': equivocarse con la clase minoritaria (suscribe) pesa más."""
    clases = np.array([0, 1])
    pesos = compute_class_weight("balanced", classes=clases, y=np.asarray(y_train))
    return {0: float(pesos[0]), 1: float(pesos[1])}


def entrenar_modelo_trivial(X_train, y_train) -> DummyClassifier:
    """Entrena un modelo trivial (DummyClassifier) como línea base de peor caso.

    Predice siempre la clase mayoritaria (no suscribe) para servir de referencia
    absoluta de que los modelos siguientes realmente aprenden.
    """
    trivial = DummyClassifier(strategy="most_frequent", random_state=42)
    trivial.fit(X_train, y_train)
    return trivial


def entrenar_baseline_simple(X_train, y_train) -> LogisticRegression:
    """Entrena un modelo simple (Regresión Logística) como línea base estadística.

    Utiliza class_weight='balanced' para mitigar el desbalance inicial de clases.
    """
    baseline = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    baseline.fit(X_train, y_train)
    return baseline


def construir_mlp_v1(input_dim: int) -> keras.Model:
    """Versión 1 de Redes Neuronales: MLP con una capa oculta.

    Args:
        input_dim: Número de características de entrada.

    Returns:
        Modelo Keras secuencial compilado.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Clasificación binaria (sí/no)
        ],
        name="MLP_Version_1",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ],
    )
    return model


def construir_mlp_v2(input_dim: int) -> keras.Model:
    """Versión 2 de Redes Neuronales: MLP más profunda con Dropout (Regularización).

    Incorpora dos capas ocultas y prevención de sobreajuste (Overfitting).
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),  # Apaga el 30% de neuronas aleatoriamente
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),  # Clasificación binaria
        ],
        name="MLP_Version_2",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ],
    )
    return model


def entrenar_red(modelo: keras.Model, X_train, y_train, X_val, y_val):
    """Entrena una red con pesos de clase y parada temprana (patience=5).

    El conjunto de validación solo se usa para la parada temprana. Antes de
    construir la red se debe llamar a fijar_semilla() para que sea reproducible.
    Devuelve el historial de entrenamiento.
    """
    parada_temprana = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    historial = modelo.fit(
        X_train, np.asarray(y_train),
        epochs=80,
        batch_size=64,
        validation_data=(X_val, np.asarray(y_val)),
        class_weight=calcular_pesos_clase(y_train),
        callbacks=[parada_temprana],
        verbose=0,
    )
    return historial
