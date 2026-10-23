import sys
import os

# Añade la ruta raíz al path para asegurar la importación correcta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import pytest
from src.preprocesamiento_api import ClienteEntrada
from src.datos import cargar_y_limpiar_datos, preparar_particiones
from src.modelo import construir_preprocesador
from src.evaluacion import calcular_metricas, elegir_umbral

RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"

def test_estructura_datos_cliente():
    """Valida que el esquema de datos de entrada (ClienteEntrada) acepte los 5 datos esperados."""
    datos_prueba = {
        "poutcome": "unknown",
        "month": "may",
        "contact": "cellular",
        "housing": "yes",
        "job": "management"
    }

    # Validar que Pydantic procese los datos de los slots/datos sin errores
    cliente = ClienteEntrada(**datos_prueba)
    assert cliente.job == "management"
    assert cliente.month == "may"

@pytest.mark.skipif(not RUTA_CSV.exists(), reason="falta data/bank-full.csv")
def test_limpieza_sin_duration_ni_duplicados():
    """'duration' se elimina (fuga) y no quedan filas duplicadas."""
    X, y = cargar_y_limpiar_datos(RUTA_CSV)
    assert "duration" not in X.columns
    assert not X.assign(y=y).duplicated().any()

@pytest.mark.skipif(not RUTA_CSV.exists(), reason="falta data/bank-full.csv")
def test_particion_estratificada_sin_solapamiento():
    """Los tres conjuntos no comparten filas y mantienen ~11.7% de positivos."""
    X, y = cargar_y_limpiar_datos(RUTA_CSV)
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_particiones(X, y)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_train.index) & set(X_val.index)
    for parte in (y_train, y_val, y_test):
        assert abs(parte.mean() - y.mean()) < 0.01

@pytest.mark.skipif(not RUTA_CSV.exists(), reason="falta data/bank-full.csv")
def test_preprocesador_se_ajusta_solo_con_entrenamiento():
    """El escalador aprende la media solo de entrenamiento (sin fuga hacia validación/prueba)."""
    X, y = cargar_y_limpiar_datos(RUTA_CSV)
    X_train, X_val, X_test, *_ = preparar_particiones(X, y)
    prep = construir_preprocesador(["age", "job"]).fit(X_train)
    media = prep.named_transformers_["numericas"].mean_[0]
    assert np.isclose(media, X_train["age"].mean())

def test_umbral_minimiza_el_costo():
    """El umbral elegido da un costo menor o igual que el umbral fijo de 0.5."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.45, 0.8, 0.3, 0.2, 0.6, 0.1, 0.1, 0.2])
    umbral = elegir_umbral(y, prob)
    costo_elegido = calcular_metricas(y, (prob >= umbral).astype(int), prob)["costo_total"]
    costo_05 = calcular_metricas(y, (prob >= 0.5).astype(int), prob)["costo_total"]
    assert costo_elegido <= costo_05

def test_pr_auc_del_trivial_no_es_cero():
    """Un modelo que siempre dice 'no' tiene PR-AUC igual a la proporción de positivos."""
    y = np.array([1, 0, 0, 0])
    assert calcular_metricas(y, np.zeros(4, dtype=int))["pr_auc"] == 0.25
