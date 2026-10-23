"""Script principal de orquestación para ejecutar las Etapas 2, 3 y 4 del proyecto."""

from pathlib import Path
import joblib
from src.datos import (
    cargar_y_limpiar_datos,
    preparar_particiones,
    ranking_correlacion,
    VARIABLES_SELECCIONADAS,
    SEMILLA,
)
from src.modelo import (
    entrenar_modelo_trivial,
    entrenar_baseline_simple,
    construir_mlp_v1,
    construir_mlp_v2,
    construir_preprocesador,
    fijar_semilla,
    entrenar_red,
)
from src.evaluacion import (
    calcular_metricas,
    elegir_umbral,
    metricas_por_grupo,
    guardar_ficha_tecnica,
    promover_modelo,
)


def main():
    """Ejecuta el flujo completo de carga de datos, entrenamiento, selección de modelos,

    evaluación en prueba y almacenamiento de las versiones y del modelo promovido.
    """
    print("=== ETAPA 2: CARGA Y PREPROCESAMIENTO DE DATOS ===")
    ruta_csv = Path("data/bank-full.csv")
    X, y = cargar_y_limpiar_datos(ruta_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_particiones(X, y)
    print(f"[OK] Datos divididos: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    ranking = ranking_correlacion(X_train, y_train)
    print("\n[INFO] Ranking de correlación con y (solo entrenamiento):")
    print(ranking.head(8).to_string())
    print(f"[OK] Variables usadas por los modelos: {VARIABLES_SELECCIONADAS}")

    # El preprocesador se ajusta SOLO con entrenamiento (evita la fuga de datos).
    preprocesador = construir_preprocesador(VARIABLES_SELECCIONADAS)
    Xp_train = preprocesador.fit_transform(X_train)
    Xp_val = preprocesador.transform(X_val)
    Xp_test = preprocesador.transform(X_test)
    print(f"[OK] Columnas de entrada tras One-Hot: {Xp_train.shape[1]}")

    print("\n=== ETAPA 3: ENTRENAMIENTO DE MODELOS ===")

    print("1. Entrenando Modelo Trivial (Dummy)...")
    trivial = entrenar_modelo_trivial(Xp_train, y_train)
    y_pred_t_val = trivial.predict(Xp_val)
    metricas_trivial = calcular_metricas(y_val, y_pred_t_val)

    print("2. Entrenando Modelo Simple (Regresión Logística)...")
    simple = entrenar_baseline_simple(Xp_train, y_train)
    y_prob_s_val = simple.predict_proba(Xp_val)[:, 1]

    print("3. Entrenando MLP Versión 1 (1 capa oculta)...")
    fijar_semilla(SEMILLA)
    mlp_v1 = construir_mlp_v1(Xp_train.shape[1])
    historial_v1 = entrenar_red(mlp_v1, Xp_train, y_train, Xp_val, y_val)
    y_prob_v1_val = mlp_v1.predict(Xp_val, verbose=0).flatten()

    print("4. Entrenando MLP Versión 2 (2 capas ocultas + Dropout)...")
    fijar_semilla(SEMILLA)
    mlp_v2 = construir_mlp_v2(Xp_train.shape[1])
    historial_v2 = entrenar_red(mlp_v2, Xp_train, y_train, Xp_val, y_val)
    y_prob_v2_val = mlp_v2.predict(Xp_val, verbose=0).flatten()

    # Umbral de cada modelo elegido en VALIDACIÓN según el costo de los errores.
    umbral_s = elegir_umbral(y_val, y_prob_s_val)
    umbral_v1 = elegir_umbral(y_val, y_prob_v1_val)
    umbral_v2 = elegir_umbral(y_val, y_prob_v2_val)
    metricas_simple = calcular_metricas(y_val, (y_prob_s_val >= umbral_s).astype(int), y_prob_s_val)
    metricas_v1 = calcular_metricas(y_val, (y_prob_v1_val >= umbral_v1).astype(int), y_prob_v1_val)
    metricas_v2 = calcular_metricas(y_val, (y_prob_v2_val >= umbral_v2).astype(int), y_prob_v2_val)

    candidatos = {
        "Modelo_Trivial": (metricas_trivial, None, {"estrategia": "most_frequent"}, None),
        "Modelo_Simple_RegrLog": (metricas_simple, umbral_s, {"class_weight": "balanced", "max_iter": 1000}, None),
        "MLP_Version_1": (metricas_v1, umbral_v1,
                          {"capas": [32], "activacion": "relu", "optimizador": "adam", "epocas_max": 80,
                           "batch_size": 64, "parada_temprana": {"monitor": "val_loss", "patience": 5},
                           "class_weight": "balanced", "semilla": SEMILLA},
                          len(historial_v1.history["loss"])),
        "MLP_Version_2": (metricas_v2, umbral_v2,
                          {"capas": [64, 32], "dropout": [0.3, 0.2], "optimizador": "adam", "epocas_max": 80,
                           "batch_size": 64, "parada_temprana": {"monitor": "val_loss", "patience": 5},
                           "class_weight": "balanced", "semilla": SEMILLA},
                          len(historial_v2.history["loss"])),
    }

    print("\n[INFO] Resultados en validación (cada modelo con su umbral):")
    for nombre, (metricas, umbral, _, _) in candidatos.items():
        print(f"{nombre:22s} umbral={umbral}  costo/cliente={metricas['costo_por_cliente']}  "
              f"recall={metricas['recall']}  precision={metricas['precision']}  pr_auc={metricas['pr_auc']}")

    # Criterio explícito: gana el modelo con MENOR costo por cliente en validación.
    nombre_ganador = min(candidatos, key=lambda k: candidatos[k][0]["costo_por_cliente"])
    metricas_ganador_val, umbral_ganador, params_ganador, _ = candidatos[nombre_ganador]
    motivo = f"Menor costo por cliente en validación ({metricas_ganador_val['costo_por_cliente']})."

    print(f"\n[GANADOR EN VALIDACIÓN]: {nombre_ganador}")
    print(f"Métricas de Validación: {metricas_ganador_val}")

    print("\nEjecutando evaluación única sobre el conjunto de PRUEBA (Test)...")
    modelos = {"Modelo_Simple_RegrLog": simple, "MLP_Version_1": mlp_v1, "MLP_Version_2": mlp_v2}
    modelo_ganador = modelos[nombre_ganador]
    if "MLP" in nombre_ganador:
        y_prob_test = modelo_ganador.predict(Xp_test, verbose=0).flatten()
    else:
        y_prob_test = modelo_ganador.predict_proba(Xp_test)[:, 1]
    y_pred_test = (y_prob_test >= umbral_ganador).astype(int)

    metricas_test_final = calcular_metricas(y_test, y_pred_test, y_prob_test)
    print(f"Métricas Finales de Prueba (Test): {metricas_test_final}")

    # Equidad: ¿el error cambia entre ocupaciones? (job es la variable demográfica que usa el modelo)
    equidad_test = metricas_por_grupo(X_test["job"], y_test, y_pred_test)

    Path("modelos").mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Guardando versiones de modelos en la carpeta 'modelos/'...")
    mlp_v1.save("modelos/mlp_v1.keras")
    mlp_v2.save("modelos/mlp_v2.keras")
    joblib.dump(simple, "modelos/modelo_simple_logreg.pkl")
    joblib.dump(preprocesador, "modelos/preprocesador.pkl")

    datos_info = {
        "archivo": str(ruta_csv),
        "filas_tras_limpieza": int(len(X)),
        "semilla": SEMILLA,
        "train": int(len(y_train)), "validacion": int(len(y_val)), "prueba": int(len(y_test)),
        "columna_excluida": "duration (se conoce despues de la llamada)",
    }
    archivos_fichas = {
        "Modelo_Trivial": "modelos/ficha_trivial.json",
        "Modelo_Simple_RegrLog": "modelos/ficha_logreg.json",
        "MLP_Version_1": "modelos/ficha_mlp_v1.json",
        "MLP_Version_2": "modelos/ficha_mlp_v2.json",
    }
    for nombre, (metricas, umbral, params, epocas) in candidatos.items():
        extra = {"promovido": nombre == nombre_ganador}
        if epocas is not None:
            extra["epocas_entrenadas"] = epocas
        if nombre == nombre_ganador:
            extra["equidad_prueba_por_ocupacion"] = equidad_test
        guardar_ficha_tecnica(
            nombre_modelo=nombre,
            hiperparametros=params,
            metricas_val=metricas,
            metricas_test=metricas_test_final if nombre == nombre_ganador else None,
            ruta_salida=archivos_fichas[nombre],
            umbral=umbral,
            datos=datos_info,
            variables=VARIABLES_SELECCIONADAS,
            extra=extra,
        )

    archivos_modelos = {
        "Modelo_Simple_RegrLog": "modelos/modelo_simple_logreg.pkl",
        "MLP_Version_1": "modelos/mlp_v1.keras",
        "MLP_Version_2": "modelos/mlp_v2.keras",
    }
    promover_modelo(nombre_ganador, archivos_modelos[nombre_ganador], umbral_ganador, motivo)
    print(f"[EXITO] Modelo ganador promovido: {nombre_ganador} ({archivos_modelos[nombre_ganador]})")


if __name__ == "__main__":
    main()
