"""Carga, limpieza y partición del dataset Bank Marketing (bank-full).
Todo el preprocesamiento que "aprende" de los datos (escalador y codificador)
se ajusta SOLO con entrenamiento, para evitar fuga de información.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
SEMILLA = 42
RUTA_POR_DEFECTO = "data/bank-full.csv"
NUMERICAS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAS = [
"job", "marital", "education", "default", "housing",
"loan", "contact", "month", "poutcome",
]
# duration solo se conoce DESPUÉS de la llamada: se explora, pero no predice.
EXCLUIDAS = ["duration"]
NUM_PRED = [c for c in NUMERICAS if c not in EXCLUIDAS]
PREDICTORAS = NUM_PRED + CATEGORICAS
CLASE = "y"
CLASE_POSITIVA = "yes"
def cargar_datos(ruta=RUTA_POR_DEFECTO):
    """Lee el .csv (separador ';') y devuelve un DataFrame."""
    return pd.read_csv(ruta, sep=";")

def revisar_calidad(df):
    """Devuelve un diccionario con los problemas de calidad que se buscan."""
    return {
        "filas": len(df),
        "columnas": list(df.columns),
        "nulos": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "balance": df[CLASE].value_counts(normalize=True).round(4).to_dict(),
        "conteo_clases": df[CLASE].value_counts().to_dict(),
        "edad_no_positiva": int((df["age"] <= 0).sum()),
        "dia_fuera_de_1_31": int(((df["day"] < 1) | (df["day"] > 31)).sum()),
        "duracion_cero": int((df["duration"] <= 0).sum()),
        "campaign_no_positiva": int((df["campaign"] <= 0).sum()),
        "deposito_a_plazo": {
        c: int((df[c] == "unknown").sum())
        for c in CATEGORICAS if (df[c] == "unknown").any()
        },
        "pdays_sin_contacto_previo": int((df["pdays"] == -1).sum()),
    }

def contar_atipicos(df):
    """Cuenta valores fuera de 1.5*IQR por variable (solo informa, no elimina)."""
    resultado = {}
    for col in NUMERICAS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        resultado[col] = int(mask.sum())
    return resultado

def limpiar(df):
    """Elimina filas duplicadas (ANTES de particionar)."""
    return df.drop_duplicates().reset_index(drop=True)

def particionar(df, semilla=SEMILLA, prop_val=0.15, prop_test=0.15):
    """Partición estratificada en entrenamiento, validación y prueba.
    Devuelve (X_tr, y_tr), (X_va, y_va), (X_te, y_te) SIN transformar, y el
    preprocesador (escalador + codificador) ajustado solo con entrenamiento.
    """
    X, y = df[PREDICTORAS], df[CLASE]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=prop_val + prop_test, stratify=y, random_state=semilla
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=prop_test / (prop_val + prop_test),
        stratify=y_tmp, random_state=semilla,
    )
    preprocesador = ColumnTransformer([
    ("num", StandardScaler(), NUM_PRED),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ]).fit(X_tr) # SOLO entrenamiento
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), preprocesador

def transformar(preprocesador, X):
    """Aplica un preprocesador ya ajustado y conserva nombres de columnas e índice."""
    return pd.DataFrame(
        preprocesador.transform(X),
        columns=preprocesador.get_feature_names_out(),
        index=X.index,
    )