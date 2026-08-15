# =============================================================
# PROYECTO: Clasificación Multiclase del Tipo de Arma
#           en Homicidios Intencionales en Ecuador
# =============================================================
# Estructura:
#   01_preprocesamiento.py  ← este archivo
#   02_modelos.py           ← entrenamiento y evaluación
#   03_comparacion.py       ← tabla comparativa y gráficos
#   04_app.py               ← interfaz Streamlit
# =============================================================

# ─────────────────────────────────────────────────────────────
# 01_preprocesamiento.py
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Configuración ─────────────────────────────────────────────
ARCHIVO_2014_2025 = "data/raw/mdi_homicidiosintencionales_pm_2014_2025.xlsx"
ARCHIVO_2026      = "data/raw/mdi_homicidiosintencionalse_pm_2026_enero_junio.xlsx"
HOJA              = "1. Homicidios Intencionales"
RUTA_SALIDA       = "data/processed/"
RANDOM_STATE      = 42

os.makedirs(RUTA_SALIDA, exist_ok=True)
os.makedirs("modelos/", exist_ok=True)

# ── 1. Carga y unión de datasets ──────────────────────────────
print("Cargando datasets...")
df1 = pd.read_excel(ARCHIVO_2014_2025, sheet_name=HOJA)
df2 = pd.read_excel(ARCHIVO_2026,      sheet_name=HOJA)
df  = pd.concat([df1, df2], ignore_index=True)
print(f"  Dataset combinado: {df.shape[0]:,} registros, {df.shape[1]} columnas")

# ── 2. Filtrar solo homicidios intencionales ──────────────────
df = df[df["tipo_muerte"].str.upper().isin(["ASESINATO", "HOMICIDIO"])]
print(f"  Tras filtrar tipo_muerte: {df.shape[0]:,} registros")

# ── 3. Eliminar fecha centinela (2025-12-31) ──────────────────
df["fecha_infraccion"] = pd.to_datetime(df["fecha_infraccion"], errors="coerce")
df = df[df["fecha_infraccion"].dt.year != 2025].copy()  # centinela detectada antes
df = df[df["fecha_infraccion"].notna()].copy()
print(f"  Tras limpiar fechas inválidas: {df.shape[0]:,} registros")

# ── 4. Variable objetivo: binarizar columna 'arma' ───────────
# SUSTANCIAS tiene solo ~3 casos → fusionar con OTROS
df["target"] = df["arma"].str.upper().str.strip()
df["target"] = df["target"].replace("SUSTANCIAS", "OTROS")

# Verificar distribución
print("\nDistribución de clases (variable objetivo):")
print(df["target"].value_counts())

# Codificar target a enteros
clases = ["ARMA DE FUEGO", "ARMA BLANCA", "ARMA CONTUNDENTE", "CONSTRICTORA", "OTROS"]
target_encoder = LabelEncoder()
target_encoder.classes_ = np.array(clases)
df["y"] = target_encoder.transform(df["target"])

# ── 5. Ingeniería de features temporales ─────────────────────
df["mes"]           = df["fecha_infraccion"].dt.month
df["dia_semana"]    = df["fecha_infraccion"].dt.dayofweek  # 0=lunes, 6=domingo

# Franja horaria desde hora_infraccion
def hora_a_franja(h):
    try:
        if isinstance(h, str):
            hora = int(h.split(":")[0])
        else:
            hora = h.hour
        if   0 <= hora < 6:  return "MADRUGADA"
        elif 6 <= hora < 12: return "MANANA"
        elif 12 <= hora < 18: return "TARDE"
        else:                return "NOCHE"
    except:
        return "NO_REGISTRADO"

df["franja_horaria"] = df["hora_infraccion"].apply(hora_a_franja)

# ── 6. Selección de features ──────────────────────────────────
# Columnas a usar (NO incluir arma, tipo_arma, target — son la respuesta)
# instruccion tiene 75% nulos → descartada
# profesion_registro_civil tiene 17% nulos → descartada
FEATURES_CAT = [
    "provincia",
    "canton",
    "area_hecho",
    "tipo_lugar",
    "presunta_motivacion",
    "sexo",
    "franja_horaria",
]
FEATURES_NUM = [
    "mes",
    "dia_semana",
    "edad",
]

# ── 7. Tratamiento de nulos ───────────────────────────────────
# Categóricas: rellenar con "NO_REGISTRADO"
for col in FEATURES_CAT:
    df[col] = df[col].astype(str).str.strip().str.upper()
    df[col] = df[col].replace(
        ["NAN", "NONE", "SIN DATO", "SIN_DATO", "NO DETERMINADO", ""],
        "NO_REGISTRADO"
    )

# Numéricas: reemplazar "SIN_DATO" string con NaN, luego mediana
for col in FEATURES_NUM:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace("SIN_DATO", "", regex=False).str.strip(),
        errors="coerce"
    )

mediana_edad = df["edad"].median()
df["edad"] = df["edad"].fillna(mediana_edad)
print(f"\nMediana de edad usada para imputación: {mediana_edad}")

# ── 8. Codificación de variables categóricas ──────────────────
# Target encoding para canton (alta cardinalidad ~214 categorías)
# One-hot encoding para el resto de categóricas

# Target encoding manual para canton
target_means = df.groupby("canton")["y"].mean()
df["canton_encoded"] = df["canton"].map(target_means).fillna(target_means.mean())

# One-hot encoding para el resto
cat_para_ohe = [c for c in FEATURES_CAT if c != "canton"]
df_ohe = pd.get_dummies(df[cat_para_ohe], prefix=cat_para_ohe, drop_first=False)

# ── 9. Armar X final ──────────────────────────────────────────
X = pd.concat([
    df[FEATURES_NUM].reset_index(drop=True),
    pd.Series(df["canton_encoded"].values, name="canton_encoded"),
    df_ohe.reset_index(drop=True)
], axis=1).astype(float)

y = df["y"].values

print(f"\nShape de X: {X.shape}")
print(f"Columnas totales de features: {X.shape[1]}")

# ── 10. Normalización de features numéricas ───────────────────
scaler = StandardScaler()
X[FEATURES_NUM + ["canton_encoded"]] = scaler.fit_transform(
    X[FEATURES_NUM + ["canton_encoded"]]
)

# ── 11. División estratificada 70/15/15 ───────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nTamaños de particiones:")
print(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"\nDistribución en train (antes de SMOTE):")
for i, c in enumerate(clases):
    print(f"  {c}: {(y_train == i).sum()}")

# ── 12. SMOTE solo sobre entrenamiento ───────────────────────
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nDistribución en train (después de SMOTE):")
for i, c in enumerate(clases):
    print(f"  {c}: {(y_train_bal == i).sum()}")

# ── 13. Guardar artefactos ────────────────────────────────────
joblib.dump(scaler,         "modelos/scaler.pkl")
joblib.dump(target_encoder, "modelos/target_encoder.pkl")
joblib.dump(target_means,   "modelos/canton_target_means.pkl")

# Guardar columnas OHE para reproducirlas en inferencia
ohe_columns = list(df_ohe.columns)
joblib.dump(ohe_columns,    "modelos/ohe_columns.pkl")
joblib.dump(FEATURES_NUM,   "modelos/features_num.pkl")

# Guardar splits
np.save(f"{RUTA_SALIDA}X_train.npy",     X_train_bal)
np.save(f"{RUTA_SALIDA}y_train.npy",     y_train_bal)
np.save(f"{RUTA_SALIDA}X_val.npy",       X_val)
np.save(f"{RUTA_SALIDA}y_val.npy",       y_val)
np.save(f"{RUTA_SALIDA}X_test.npy",      X_test)
np.save(f"{RUTA_SALIDA}y_test.npy",      y_test)

print("\n✓ Preprocesamiento completado. Artefactos guardados en modelos/ y data/processed/")
print(f"  Dimensión final X_train balanceado: {X_train_bal.shape}")
