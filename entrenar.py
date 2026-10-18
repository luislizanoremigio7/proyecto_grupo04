"""Script principal de orquestación para ejecutar las Etapas 2, 3 y 4 del proyecto."""

from pathlib import Path
from src.datos import cargar_y_limpiar_datos, preparar_particiones
from src.modelo import (
    entrenar_modelo_trivial,
    entrenar_baseline_simple,
    construir_mlp_v1,
    construir_mlp_v2,
)
from src.evaluacion import calcular_metricas, guardar_ficha_tecnica


def main():
    print("=== ETAPA 2: CARGA Y PREPROCESAMIENTO DE DATOS ===")
    ruta_csv = Path("data/bank-full.csv")
    X, y = cargar_y_limpiar_datos(ruta_csv)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preparar_particiones(X, y)
    print(f"[OK] Datos divididos: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    print("\n=== ETAPA 3: ENTRENAMIENTO DE MODELOS ===")
    
    # 1. Modelo Trivial
    print("1. Entrenando Modelo Trivial (Dummy)...")
    trivial = entrenar_modelo_trivial(X_train, y_train)
    y_pred_t_val = trivial.predict(X_val)
    metricas_trivial = calcular_metricas(y_val, y_pred_t_val)

    # 2. Modelo Simple (Regresión Logística)
    print("2. Entrenando Modelo Simple (Regresión Logística)...")
    simple = entrenar_baseline_simple(X_train, y_train)
    y_pred_s_val = simple.predict(X_val)
    y_prob_s_val = simple.predict_proba(X_val)[:, 1]
    metricas_simple = calcular_metricas(y_val, y_pred_s_val, y_prob_s_val)

    # 3. Red Neuronal MLP V1
    print("3. Entrenando MLP Versión 1 (1 capa oculta)...")
    mlp_v1 = construir_mlp_v1(X_train.shape[1])
    mlp_v1.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=0)
    y_prob_v1_val = mlp_v1.predict(X_val, verbose=0).flatten()
    y_pred_v1_val = (y_prob_v1_val >= 0.5).astype(int)
    metricas_v1 = calcular_metricas(y_val, y_pred_v1_val, y_prob_v1_val)

    # 4. Red Neuronal MLP V2
    print("4. Entrenando MLP Versión 2 (2 capas ocultas + Dropout)...")
    mlp_v2 = construir_mlp_v2(X_train.shape[1])
    mlp_v2.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=0)
    y_prob_v2_val = mlp_v2.predict(X_val, verbose=0).flatten()
    y_pred_v2_val = (y_prob_v2_val >= 0.5).astype(int)
    metricas_v2 = calcular_metricas(y_val, y_pred_v2_val, y_prob_v2_val)

    print("\n=== ETAPA 4: SELECCIÓN Y EVALUACIÓN FINAL EN PRUEBA ===")
    
    # Registro de candidatos
    candidatos = {
        "Modelo_Trivial": (metricas_trivial, trivial, {}),
        "Modelo_Simple_RegrLog": (metricas_simple, simple, {"class_weight": "balanced", "max_iter": 1000}),
        "MLP_Version_1": (metricas_v1, mlp_v1, {"capas": [32], "activacion": "relu"}),
        "MLP_Version_2": (metricas_v2, mlp_v2, {"capas": [64, 32], "dropout": [0.3, 0.2]}),
    }

    # SELECCIÓN: Basada estrictamente en el F1-Score sobre VALIDACIÓN
    nombre_ganador = max(candidatos, key=lambda k: candidatos[k][0]["f1_score"])
    metricas_ganador_val, modelo_ganador, params_ganador = candidatos[nombre_ganador]

    print(f"\n[GANADOR EN VALIDACIÓN]: {nombre_ganador}")
    print(f"Métricas de Validación: {metricas_ganador_val}")

    # EVALUACIÓN ÚNICA EN TEST (Solo para la versión elegida)
    print("\nEjecutando evaluación única sobre el conjunto de PRUEBA (Test)...")
    if "MLP" in nombre_ganador:
        y_prob_test = modelo_ganador.predict(X_test, verbose=0).flatten()
        y_pred_test = (y_prob_test >= 0.5).astype(int)
    else:
        y_pred_test = modelo_ganador.predict(X_test)
        y_prob_test = modelo_ganador.predict_proba(X_test)[:, 1] if hasattr(modelo_ganador, "predict_proba") else None

    metricas_test_final = calcular_metricas(y_test, y_pred_test, y_prob_test)
    print(f"Métricas Finales de Prueba (Test): {metricas_test_final}")

    # GUARDAR FICHA TÉCNICA JSON
    guardar_ficha_tecnica(
        nombre_modelo=nombre_ganador,
        hiperparametros=params_ganador,
        metricas_val=metricas_ganador_val,
        metricas_test=metricas_test_final,
        ruta_salida="modelos/ficha_tecnica.json"
    )


if __name__ == "__main__":
    main()