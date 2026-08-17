"""
ETAPA 3 - PREPARACION (split + imputacion + codificacion + escalado)
======================================================================

FUENTE: data/processed/dataset.xlsx (NO los datasets oficiales A/B).

Este dataset lo armo el usuario a partir de los datos limpios, pero para
balancear ARMA_CONTUNDENTE y OTRAS incluye filas que NO corresponden a
casos reales (se verifico por cruce exacto contra A_2014_2025_limpio.csv
y B_2026_limpio.csv):

    ARMA_FUEGO        99.2% de las filas son reales (submuestreo, ok)
    ARMA_BLANCA       99.0% de las filas son reales (submuestreo, ok)
    ARMA_CONTUNDENTE  22.7% reales, 77.3% NO encontradas en los datos oficiales
    OTRAS             31.1% reales, 68.9% NO encontradas en los datos oficiales

DECISION EXPLICITA DEL USUARIO (no mia): entrenar igual con este dataset
para ver el resultado. La defensa de esta decision queda pendiente para
el usuario. Se documenta aqui para que quede trazable.

POR QUE UN SPLIT ALEATORIO ESTRATIFICADO Y NO TEMPORAL:
El split temporal (2014-2023 train / 2024-2025 val / 2026 test) que se
penso para los datasets originales pierde su sentido aqui: las filas
sinteticas de ARMA_CONTUNDENTE/OTRAS tienen anios asignados de forma
arbitraria para lograr el balance, no reflejan una secuencia temporal
real. Por eso se usa un split aleatorio estratificado por 'y' (70/15/15),
semilla fija para reproducibilidad. Si mas adelante se entrena con los
datos oficiales (A/B, sin las filas fabricadas), se debe volver al split
temporal.

DECISIONES DE COLUMNAS PARA EL MODELADO (adicionales a las de Etapa 2):
  - arma_categoria: SE ELIMINA. Es el texto original del arma (6 categorias),
    practicamente el target sin colapsar. Dejarla seria leakage total.
  - coordenada_x, coordenada_y: SE ELIMINAN. Son (casi) unicas por caso,
    un modelo lineal podria memorizar ubicaciones puntuales en vez de
    generalizar. Ademas no son un input realista para la interfaz.
  - fecha: SE ELIMINA (datetime crudo). Ya esta descompuesta en
    anio/mes/dia_semana/es_fin_semana/hora/franja_horaria.
  - subzona, distrito: SE ELIMINAN por redundancia de granularidad con
    provincia/canton (mismo criterio ya aplicado en Etapa 2 a
    circuito/subcircuito).
  - presunta_motivacion, presun_motiva_observada: se CONSERVAN por ahora
    (variables "dudosas" de Etapa 2). Pendiente: medir impacto con/sin.

IMPUTACION:
  - Categoricas: NaN -> "DESCONOCIDO" (regla fija, no aprende nada, se
    puede aplicar antes del split sin leakage).
  - edad (numerica): NaN -> mediana calculada SOLO con el train (Etapa
    posterior al split, para no filtrar informacion de val/test).

CODIFICACION Y ESCALADO (aprenden parametros -> SOLO con train, luego
se aplica igual a val y test):
  - One-Hot Encoding para las categoricas.
  - StandardScaler para edad y anio.
  - es_fin_semana ya es binaria (0/1), se deja igual.

Salida:
  data/processed/train_raw.csv, val_raw.csv, test_raw.csv  (antes de codificar,
      por transparencia y para poder auditar el split)
  modelos/preprocesador.joblib   (ColumnTransformer ya ajustado con el train)
  data/processed/train_X.npy, val_X.npy, test_X.npy   (features ya codificadas)
  data/processed/train_y.npy, val_y.npy, test_y.npy
  resultados/preparacion/informe_preparacion.txt
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "preparacion"
DIR_MOD.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)

COLUMNAS_DATASET = [
    "zona", "subzona", "distrito", "provincia", "canton",
    "coordenada_y", "coordenada_x", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "edad", "sexo",
    "etnia", "estado_civil", "nacionalidad", "discapacidad",
    "arma_categoria", "y", "anio", "mes", "dia_semana", "es_fin_semana",
    "fecha", "hora", "franja_horaria",
]

COLS_ELIMINAR = ["arma_categoria", "coordenada_x", "coordenada_y", "fecha",
                  "subzona", "distrito", "anio", "mes"]

# POR QUE SE ELIMINA 'anio' (verificado, no solo sospechado):
#   En dataset.xlsx, ARMA_FUEGO tiene 0% de presencia en 2014-2023 y en 2026,
#   y esta casi toda concentrada en 2024-2025. Cualquier modelo que reciba
#   'anio' aprende el atajo "si el anio no es 2024/2025, no es arma de fuego"
#   en vez de un patron criminologico real. Se confirmo midiendo el impacto:
#   Random Forest con anio: F1 macro val 0.7833 | sin anio: 0.6240.
#   Ademas, aunque el dato fuera limpio, seguiria sin ser usable para
#   predecir anios futuros (extrapolacion fuera del rango visto en train).
#   Esta exclusion aplica a TODOS los modelos entrenados sobre dataset.xlsx,
#   no solo a Random Forest: el problema esta en el dato, no en el algoritmo.
#
# POR QUE SE ELIMINA 'mes' (misma logica, detectado despues):
#   La fabricacion de filas de dataset.xlsx tambien concentro las clases por
#   mes: ARMA_FUEGO tiene 0% de presencia en marzo y abril. Medido con Random
#   Forest: CON mes F1 macro val 0.6283 | SIN mes 0.4952. Esos 0.133 de
#   diferencia eran un atajo sin sentido causal ("si es marzo, no es arma de
#   fuego"), no señal criminologica.
#   Verificado que NO es un fenomeno real: en los datos oficiales del
#   Ministerio (A+B), ARMA_FUEGO se mantiene entre 76% y 81% en los 12 meses,
#   sin excepcion. La fuga es un artefacto de como se armo dataset.xlsx.
#   entrenamiento_v2 tambien la excluye, asi que ambas lineas usan el mismo
#   conjunto de features y son comparables.

COLS_CATEGORICAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "dia_semana",
    "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad"]
COL_BINARIA = ["es_fin_semana"]


class Reporte:
    def __init__(self, ruta):
        self.buf, self.ruta = [], ruta

    def p(self, *a):
        t = " ".join(str(x) for x in a)
        print(t)
        self.buf.append(t)

    def titulo(self, t, c="="):
        self.p("")
        self.p(c * 78)
        self.p(t)
        self.p(c * 78)

    def guardar(self):
        self.ruta.write_text("\n".join(self.buf), encoding="utf-8")


rep = Reporte(DIR_REP / "informe_preparacion.txt")


def cargar_dataset():
    ruta = DIR_PROC / "dataset.xlsx"
    df = pd.read_excel(ruta, dtype=str, header=None, skiprows=1)
    df.columns = COLUMNAS_DATASET
    return df


def main():
    rep.titulo("ETAPA 3 - PREPARACION", "#")
    rep.p(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")

    df = cargar_dataset()
    rep.p(f"\nEntrada: data/processed/dataset.xlsx -> {df.shape[0]:,} filas x {df.shape[1]} columnas")
    rep.p("ADVERTENCIA (documentada, decision del usuario): este dataset contiene")
    rep.p("filas fabricadas para balancear ARMA_CONTUNDENTE (77.3%) y OTRAS (68.9%).")
    rep.p("Ver docstring del script para el detalle de la verificacion.")

    # --- Tipos numericos ---
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["es_fin_semana"] = pd.to_numeric(df["es_fin_semana"], errors="coerce").astype(int)

    # --- Columnas eliminadas para el modelado ---
    df = df.drop(columns=COLS_ELIMINAR)
    rep.p(f"\nColumnas eliminadas para modelado: {COLS_ELIMINAR}")

    # --- Imputacion determinista de categoricas (regla fija, sin split) ---
    for c in COLS_CATEGORICAS:
        n_nulos = df[c].isna().sum()
        if n_nulos:
            df[c] = df[c].fillna("DESCONOCIDO")
            rep.p(f"  [{c}] {n_nulos} nulos -> 'DESCONOCIDO'")

    X = df[COLS_CATEGORICAS + COLS_NUMERICAS + COL_BINARIA]
    y = df["y"]

    rep.p(f"\nFeatures finales ({X.shape[1]}): {list(X.columns)}")
    rep.p(f"Target: y  -> clases: {sorted(y.unique())}")

    # --- Split estratificado 70/15/15 (aleatorio, ver justificacion arriba) ---
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)

    rep.titulo("SPLIT (aleatorio estratificado, semilla=42)")
    for nom, xx, yy in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        rep.p(f"\n  {nom}: {len(xx):,} filas ({100*len(xx)/len(X):.1f}%)")
        dist = yy.value_counts(normalize=True).mul(100).round(2)
        for k, v in dist.items():
            rep.p(f"       {k:<18} {v:5.2f}%")

    # --- Preprocesador: imputar edad (mediana de train) + escalar + one-hot ---
    pre_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    pre_categorico = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocesador = ColumnTransformer([
        ("num", pre_numerico, COLS_NUMERICAS),
        ("cat", pre_categorico, COLS_CATEGORICAS),
        ("bin", "passthrough", COL_BINARIA),
    ])

    rep.titulo("PREPROCESADOR (ajustado SOLO con train)")
    mediana_edad = X_train["edad"].median()
    rep.p(f"  Mediana de 'edad' en train (usada para imputar NaN): {mediana_edad}")
    rep.p(f"  Nulos de 'edad' en train: {X_train['edad'].isna().sum()} de {len(X_train)}")

    X_train_t = preprocesador.fit_transform(X_train)
    X_val_t = preprocesador.transform(X_val)
    X_test_t = preprocesador.transform(X_test)

    rep.p(f"\n  Dimension tras codificar (one-hot): {X_train_t.shape[1]} columnas")
    n_cats_onehot = preprocesador.named_transformers_["cat"]["onehot"].get_feature_names_out(COLS_CATEGORICAS)
    rep.p(f"  ({len(COLS_NUMERICAS)} numericas + {len(n_cats_onehot)} dummies + {len(COL_BINARIA)} binaria)")

    # --- Guardado ---
    for nom, xdf, ydf in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        xdf.assign(y=ydf).to_csv(DIR_PROC / f"{nom}_raw.csv", index=False)

    np.save(DIR_PROC / "train_X.npy", X_train_t)
    np.save(DIR_PROC / "val_X.npy", X_val_t)
    np.save(DIR_PROC / "test_X.npy", X_test_t)
    np.save(DIR_PROC / "train_y.npy", y_train.values)
    np.save(DIR_PROC / "val_y.npy", y_val.values)
    np.save(DIR_PROC / "test_y.npy", y_test.values)
    joblib.dump(preprocesador, DIR_MOD / "preprocesador.joblib")

    rep.titulo("FIN DE LA ETAPA 3", "#")
    rep.p("Guardado: data/processed/{train,val,test}_raw.csv")
    rep.p("Guardado: data/processed/{train,val,test}_X.npy y _y.npy")
    rep.p("Guardado: modelos/preprocesador.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
