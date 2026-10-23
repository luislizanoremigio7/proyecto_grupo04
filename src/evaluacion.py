"""Módulo de evaluación estricta, cálculo de métricas y versionado JSON (Etapa 4).

Implementa las métricas para desbalance de clases y la exportación de fichas técnicas.
"""

from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

# Costo de cada error: no llamar a quien sí suscribiría (FN) cuesta más que una llamada de más (FP).
COSTO_FN = 5
COSTO_FP = 1


def calcular_metricas(y_true, y_pred, y_prob=None) -> dict:
    """Calcula las métricas clave orientadas a la clase minoritaria (suscripción)."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Si el modelo no da puntajes (modelo trivial), se usa su predicción. Así el
    # PR-AUC del trivial es la proporción de positivos (~0.117) y no 0.
    pr_auc = average_precision_score(y_true, y_prob if y_prob is not None else y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    fp, fn = cm[0][1], cm[1][0]
    costo = COSTO_FN * fn + COSTO_FP * fp

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm,
        "costo_total": int(costo),
        "costo_por_cliente": round(costo / len(y_true), 4),
    }


def elegir_umbral(y_true, y_prob) -> float:
    """Elige en VALIDACIÓN el umbral que minimiza el costo total (FN = 5, FP = 1)."""
    y_true = np.asarray(y_true)
    mejor_umbral, mejor_costo = 0.5, None
    for umbral in np.round(np.arange(0.01, 1.0, 0.01), 2):
        y_pred = (np.asarray(y_prob) >= umbral).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        costo = COSTO_FN * fn + COSTO_FP * fp
        if mejor_costo is None or costo < mejor_costo:
            mejor_umbral, mejor_costo = float(umbral), costo
    return mejor_umbral


def metricas_por_grupo(grupo: pd.Series, y_true, y_pred) -> dict:
    """Calcula recall y precisión por grupo (por ejemplo, por ocupación) para revisar la equidad."""
    tabla = pd.DataFrame({"grupo": np.asarray(grupo), "y": np.asarray(y_true), "pred": np.asarray(y_pred)})
    resultado = {}
    for nombre, g in tabla.groupby("grupo"):
        resultado[str(nombre)] = {
            "clientes": int(len(g)),
            "suscriben": int(g["y"].sum()),
            "recall": round(float(recall_score(g["y"], g["pred"], zero_division=0)), 4),
            "precision": round(float(precision_score(g["y"], g["pred"], zero_division=0)), 4),
        }
    return resultado


def guardar_ficha_tecnica(
    nombre_modelo: str,
    hiperparametros: dict,
    metricas_val: dict,
    metricas_test: dict | None = None,
    ruta_salida: str | Path = "modelos/ficha_tecnica.json",
    umbral: float | None = None,
    datos: dict | None = None,
    variables: list | None = None,
    extra: dict | None = None,
) -> None:
    """Guarda la ficha técnica en formato JSON registrando metadatos y resultados."""
    path_salida = Path(ruta_salida)
    path_salida.parent.mkdir(parents=True, exist_ok=True)

    ficha = {
        "timestamp": datetime.now().isoformat(),
        "modelo": nombre_modelo,
        "dataset": "Bank Marketing UCI (bank-full.csv)",
        "datos": datos,
        "variables": variables,
        "hiperparametros": hiperparametros,
        "umbral": {"valor": umbral, "criterio": f"minimo costo en validacion (FN={COSTO_FN}, FP={COSTO_FP})"},
        "metricas_validacion": metricas_val,
        "metricas_prueba_final": metricas_test if metricas_test else "No evaluado en prueba",
    }
    if extra:
        ficha.update(extra)

    with open(path_salida, "w", encoding="utf-8") as f:
        json.dump(ficha, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Ficha técnica registrada exitosamente en: {path_salida}")


def promover_modelo(nombre_modelo: str, archivo: str, umbral: float, motivo: str,
                    ruta_salida: str | Path = "modelos/modelo_promovido.json") -> None:
    """Registra qué versión usa la API, con su archivo, su umbral y el motivo de la elección."""
    promovido = {
        "timestamp": datetime.now().isoformat(),
        "modelo": nombre_modelo,
        "archivo": archivo,
        "umbral": umbral,
        "motivo": motivo,
    }
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(promovido, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Versión promovida registrada en: {ruta_salida}")
