"""Módulo de evaluación estricta, cálculo de métricas y versionado JSON (Etapa 4).

Implementa las métricas para desbalance de clases y la exportación de fichas técnicas.
"""

from datetime import datetime
import json
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)


def calcular_metricas(y_true, y_pred, y_prob=None) -> dict:
    """Calcula las métricas clave orientadas a la clase minoritaria (suscripción)."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    pr_auc = 0.0
    if y_prob is not None:
        pr_auc = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm,
    }


def guardar_ficha_tecnica(
    nombre_modelo: str,
    hiperparametros: dict,
    metricas_val: dict,
    metricas_test: dict | None = None,
    ruta_salida: str | Path = "modelos/ficha_tecnica.json",
) -> None:
    """Guarda la ficha técnica en formato JSON registrando metadatos y resultados."""
    path_salida = Path(ruta_salida)
    path_salida.parent.mkdir(parents=True, exist_ok=True)

    ficha = {
        "timestamp": datetime.now().isoformat(),
        "modelo": nombre_modelo,
        "dataset": "Bank Marketing UCI (bank-full.csv)",
        "hiperparametros": hiperparametros,
        "metricas_validacion": metricas_val,
        "metricas_prueba_final": metricas_test if metricas_test else "No evaluado en prueba",
    }

    with open(path_salida, "w", encoding="utf-8") as f:
        json.dump(ficha, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Ficha técnica registrada exitosamente en: {path_salida}")