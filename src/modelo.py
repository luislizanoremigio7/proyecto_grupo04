"""Módulo de modelado: Línea base (trivial y simple) y Redes Neuronales (MLP) (Etapa 3).

Implementa la línea base completa (modelo trivial y modelo simple) y las dos 
versiones de Redes Neuronales exigidas en la rúbrica para Bank Marketing.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers


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
    baseline = LogisticRegression(random_state=42, max_iter=1000)
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