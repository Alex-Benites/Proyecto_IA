"""
ENTRENAMIENTO V2 - PREPARACION
================================

DIFERENCIA CON V1 (la carpeta raiz del proyecto, que usa dataset.xlsx)
------------------------------------------------------------------------
V1 entrena sobre dataset.xlsx, que balancea las clases fabricando filas que
no corresponden a casos reales (77% de ARMA_CONTUNDENTE, 69% de OTRAS son
sinteticas - verificado por cruce exacto contra los datos oficiales).

V2 entrena SOLO con datos reales del Ministerio (A_2014_2025_limpio.csv +
B_2026_limpio.csv, salida de 02_limpieza.py), sin fabricar ninguna fila.
El desbalance real de clases se maneja con TECNICAS, no con datos falsos:

  VARIANTE A: los 43.975 casos reales completos, sin tocar. El desbalance
              (80% ARMA_FUEGO vs 3% ARMA_CONTUNDENTE) se compensa con
              class_weight='balanced' en el momento de entrenar cada modelo.
              No se descarta NINGUN caso real.

  VARIANTE B: se toma una muestra de 5.800 casos de ARMA_FUEGO (de los
              34.905 reales disponibles), y se conservan TODOS los casos
              reales de las otras 3 clases (5.521 + 2.369 + 1.180). Sin
              ningun ajuste adicional (sin class_weight), para aislar el
              efecto del recorte solo, y poder compararlo limpio contra el
              efecto de class_weight solo en la Variante A.

Las dos variantes tienen su PROPIO split 70/15/15 (no comparten val/test),
igual que V1 tiene el suyo independiente. Semilla 42 en ambas.

TAMBIEN SE ELIMINAN 'mes' Y 'anio' (ademas de las columnas de leakage ya
conocidas): en dataset.xlsx ambas resultaron ser fugas (mes: RF pasa de F1
0.4952 a 0.6283 solo por esa columna; anio: de 0.6283 a 0.7833). Aca, con
datos 100% reales, YA NO HAY fuga (verificado: en los datos oficiales
ARMA_FUEGO se mantiene 76-81% en todos los meses y crece de forma genuina
y gradual entre 2014 y 2025). Aun asi se excluyen por consistencia y para
que V1 y V2 sean comparables usando el mismo conjunto de features. Se puede
reincorporar 'anio' mas adelante si se decide investigar la tendencia real.

EL PREPROCESAMIENTO VIVE ACA, NO EN CADA MODELO
------------------------------------------------
Igual que V1 (03_preparacion.py), este script ajusta el ColumnTransformer
(imputar edad + escalar + one-hot) UNA SOLA VEZ por variante, usando
exclusivamente el train de esa variante, y guarda las matrices ya
transformadas en .npy. Asi:
  - Todos los modelos de una variante ven EXACTAMENTE la misma matriz
    (comparacion limpia entre algoritmos).
  - El preprocesador queda guardado en .joblib para la app final.
  - Ningun script de modelo puede "colar" sin querer estadisticas de val/test.

Salida:
  data/variante_{a,b}_{train,val,test}_raw.csv     (texto, para CatBoost nativo)
  data/variante_{a,b}_{train,val,test}_X.npy       (matriz one-hot + escalada)
  data/variante_{a,b}_{train,val,test}_y.npy       (etiquetas)
  data/variante_{a,b}_columnas.npy                 (nombre de cada columna de X)
  modelos/preprocesador_variante_{a,b}.joblib
  resultados/informe_preparacion_v2.txt
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
SEMILLA = 42

DIR_DATA = BASE / "data"
DIR_REP = BASE / "resultados"
DIR_MOD = BASE / "modelos"
DIR_DATA.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)
DIR_MOD.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]
N_ARMA_FUEGO_VARIANTE_B = 5800

# Mismas columnas de features que V1 (03_preparacion.py), MENOS 'anio' y 'mes'
COLS_ELIMINAR = ["arma_categoria", "coordenada_x", "coordenada_y", "fecha",
                  "subzona", "distrito", "anio", "mes"]
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


rep = Reporte(DIR_REP / "informe_preparacion_v2.txt")


def cargar_datos_reales():
    a = pd.read_csv(RAIZ / "data" / "processed" / "A_2014_2025_limpio.csv", dtype=str)
    b = pd.read_csv(RAIZ / "data" / "processed" / "B_2026_limpio.csv", dtype=str)
    df = pd.concat([a, b], ignore_index=True)
    df = df.drop(columns=COLS_ELIMINAR)

    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["es_fin_semana"] = pd.to_numeric(df["es_fin_semana"], errors="coerce").astype(int)
    df["dia_semana"] = pd.to_numeric(df["dia_semana"], errors="coerce").astype("Int64").astype(str)
    df["hora"] = pd.to_numeric(df["hora"], errors="coerce").astype("Int64").astype(str)

    for c in COLS_CATEGORICAS:
        df[c] = df[c].fillna("DESCONOCIDO")

    return df


def construir_preprocesador():
    """Identico al de V1: imputar edad con la mediana de TRAIN, escalar, one-hot."""
    pre_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    pre_categorico = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", pre_numerico, COLS_NUMERICAS),
        ("cat", pre_categorico, COLS_CATEGORICAS),
        ("bin", "passthrough", COL_BINARIA),
    ])


def guardar_split(df, nombre_variante):
    X = df[COLS_CATEGORICAS + COLS_NUMERICAS + COL_BINARIA]
    y = df["y"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)

    rep.p(f"\n  Split de {nombre_variante} (70/15/15, semilla={SEMILLA}):")
    for nom, xx, yy in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        rep.p(f"    {nom}: {len(xx):,} filas")
        dist = yy.value_counts()
        for k in CLASES:
            v = dist.get(k, 0)
            rep.p(f"         {k:<18} {v:6,}  ({100*v/len(xx):5.2f}%)")
        xx.assign(y=yy).to_csv(DIR_DATA / f"{nombre_variante}_{nom}_raw.csv", index=False)

    # --- Preprocesador: se AJUSTA solo con train, se APLICA a val y test ---
    pre = construir_preprocesador()
    mediana_edad = X_train["edad"].median()
    rep.p(f"\n    Preprocesador ajustado SOLO con train:")
    rep.p(f"      mediana de 'edad' usada para imputar: {mediana_edad}")
    rep.p(f"      nulos de 'edad' en train: {X_train['edad'].isna().sum()} de {len(X_train):,}")

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    nombres_cat = pre.named_transformers_["cat"]["onehot"].get_feature_names_out(COLS_CATEGORICAS)
    columnas = np.array(list(COLS_NUMERICAS) + list(nombres_cat) + list(COL_BINARIA))
    rep.p(f"      dimension tras one-hot: {X_train_t.shape[1]} columnas "
          f"({len(COLS_NUMERICAS)} numerica + {len(nombres_cat)} dummies + "
          f"{len(COL_BINARIA)} binaria)")

    for nom, xt, yy in [("train", X_train_t, y_train), ("val", X_val_t, y_val),
                        ("test", X_test_t, y_test)]:
        np.save(DIR_DATA / f"{nombre_variante}_{nom}_X.npy", xt)
        np.save(DIR_DATA / f"{nombre_variante}_{nom}_y.npy", yy.values)
    np.save(DIR_DATA / f"{nombre_variante}_columnas.npy", columnas)
    joblib.dump(pre, DIR_MOD / f"preprocesador_{nombre_variante}.joblib")


def main():
    rep.titulo("ENTRENAMIENTO V2 - PREPARACION (datos 100% reales)", "#")
    rep.p(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | semilla {SEMILLA}")
    rep.p("\nFuente: A_2014_2025_limpio.csv + B_2026_limpio.csv (datos oficiales,")
    rep.p("ninguna fila fabricada). Columnas eliminadas ademas de las de leakage")
    rep.p(f"ya conocidas: {['anio', 'mes']} (ver docstring: fuente de fuga en V1,")
    rep.p("aca se excluyen por consistencia aunque no hay evidencia de fuga real).")

    df = cargar_datos_reales()
    rep.p(f"\nTotal de casos reales disponibles: {len(df):,}")
    dist_total = df["y"].value_counts()
    for k in CLASES:
        rep.p(f"  {k:<18} {dist_total.get(k,0):6,}")

    # --- VARIANTE A: todos los casos reales, sin tocar ---
    rep.titulo("VARIANTE A: datos completos + class_weight='balanced' (al entrenar)")
    rep.p("  No se descarta NINGUN caso real. El desbalance se compensa en cada")
    rep.p("  modelo con su propio mecanismo de peso por clase (ver scripts de modelo).")
    guardar_split(df, "variante_a")

    # --- VARIANTE B: ARMA_FUEGO recortado a 5.800, resto completo ---
    rep.titulo(f"VARIANTE B: ARMA_FUEGO recortado a {N_ARMA_FUEGO_VARIANTE_B:,} (de "
              f"{dist_total['ARMA_FUEGO']:,} reales), sin ajuste adicional")
    fuego = df[df["y"] == "ARMA_FUEGO"].sample(
        n=N_ARMA_FUEGO_VARIANTE_B, random_state=SEMILLA)
    resto = df[df["y"] != "ARMA_FUEGO"]
    df_b = pd.concat([fuego, resto], ignore_index=True)
    rep.p(f"  Total Variante B: {len(df_b):,} filas")
    for k in CLASES:
        v = (df_b["y"] == k).sum()
        rep.p(f"    {k:<18} {v:6,}  ({100*v/len(df_b):5.2f}%)")
    guardar_split(df_b, "variante_b")

    rep.titulo("FIN DE LA PREPARACION V2", "#")
    rep.p("Guardado en entrenamiento_v2/data/:")
    rep.p("  variante_{a,b}_{train,val,test}_raw.csv  -> texto (lo usa CatBoost nativo)")
    rep.p("  variante_{a,b}_{train,val,test}_X.npy    -> matriz one-hot + escalada")
    rep.p("  variante_{a,b}_{train,val,test}_y.npy    -> etiquetas")
    rep.p("  variante_{a,b}_columnas.npy              -> nombre de cada columna de X")
    rep.p("Guardado en entrenamiento_v2/modelos/:")
    rep.p("  preprocesador_variante_{a,b}.joblib      -> para la app final")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
