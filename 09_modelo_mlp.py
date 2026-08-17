"""
MODELO 5: MLP (RED NEURONAL MULTICAPA / MULTI-LAYER PERCEPTRON)
==================================================================

POR QUE ESTE MODELO
  Es el siguiente escalon logico despues de la Regresion Logistica: la
  logistica solo traza fronteras LINEALES entre las 4 clases. Un MLP agrega
  una o mas capas ocultas con activaciones NO lineales, lo que le permite
  aprender interacciones entre variables (similar a lo que ya logra Random
  Forest con sus arboles, pero con otro mecanismo).

QUE ES UN MLP (para la defensa, sin caja negra)
  Es una cadena de capas totalmente conectadas entre la entrada y la salida:

      ENTRADA (una columna por feature codificada)
        -> CAPA OCULTA 1 (N neuronas, ReLU)
        -> [CAPA OCULTA 2 opcional]
        -> SALIDA (4 neuronas, softmax = probabilidad por clase)

  Cada neurona hace dos pasos:
    1. Combinacion lineal:  z = w1*x1 + w2*x2 + ... + wn*xn + b
       (exactamente lo mismo que hace LogisticRegression con su coef_)
    2. Activacion no lineal: a = ReLU(z) = max(0, z)
       Sin este paso, apilar capas seguiria siendo matematicamente una sola
       transformacion lineal (por mas capas que se agreguen). La no
       linealidad es lo unico que le da al MLP capacidad para separar
       clases con fronteras curvas, que la logistica no puede representar.

  La capa de salida usa softmax (la misma que usa la logistica multinomial),
  y la funcion de perdida es la misma entropia cruzada:
      L = - suma_i log( p(clase_verdadera_i) )

  COMO APRENDE: backpropagation + descenso de gradiente por LOTES (mini-
  batch, optimizador Adam). No se le pasa el train de una sola vez
  (full-batch, estable pero pesado) ni de a una fila (stochastic, rapido por
  paso pero muy ruidoso): se parte en lotes de tamanio fijo (batch_size, ver
  hiperparametros abajo). Una EPOCA = una pasada completa por el train =
  ceil(n_train / batch_size) actualizaciones de pesos.

POR QUE NO UN TRANSFORMER
  Los Transformers (BERT, GPT, etc.) estan pensados para SECUENCIAS, donde
  la auto-atencion aprende relaciones de orden/contexto entre elementos
  (ej. palabras de una oracion). Este dataset es tabular: cada fila es un
  caso con columnas fijas sin relacion secuencial entre si. Ademas, los
  Transformers necesitan ordenes de magnitud mas datos de los disponibles
  (si no, no convergen o memorizan de inmediato), y traen mucha mas
  maquinaria (atencion multi-cabeza, positional encoding, layer norm) que es
  mas dificil de defender sin caja negra que un MLP simple.

NO ES CAJA NEGRA
  - Arquitectura, funcion de perdida y optimizador documentados arriba.
  - hidden_layer_sizes se elige con un EXPERIMENTO EXPLICITO (barrido
    manual), no con GridSearchCV ni busqueda automatica.
  - batch_size se fija EXPLICITO y se justifica (no se deja 'auto' de
    sklearn, que decidiria esto en silencio).
  - No se usa el early_stopping automatico de sklearn (que separaria su
    propia validacion interna). En su lugar, la cantidad de epocas se
    decide con una curva train/val explicita contra el set de validacion
    real del proyecto (Experimento 2), igual que el early stopping manual
    (od_wait) que uso CatBoost.
  - alpha (regularizacion L2) tambien se barre explicitamente, SOLO si el
    modelo con el valor por defecto muestra overfitting (Experimento 3).
  - Interpretabilidad: el MLP no tiene coef_ ni feature_importances_, asi
    que se usa permutation_importance (agrupada por variable original, no
    dispersa en decenas de columnas one-hot) para ver que variables pesan.

==============================================================================
RECORRIDO QUE LLEVO A LA VERSION ACTUAL DE ESTE SCRIPT (documentado para que
quede trazable, no es la primera version que se corrio)
==============================================================================

1) 'mes' SE EXCLUYE - leakage confirmado, mismo mecanismo que 'anio'.
   Verificado con los datos oficiales SIN alterar (43.975 filas de
   A_2014_2025_limpio.csv + B_2026_limpio.csv): ARMA_FUEGO esta entre 76% y
   81% en TODOS los meses, sin excepcion -> el mes no tiene relacion real
   con el tipo de arma. Sin embargo, en dataset.xlsx las filas FABRICADAS
   para balancear ARMA_CONTUNDENTE/OTRAS quedaron concentradas por bloques
   de mes casi perfectos, y el submuestreo de ARMA_FUEGO/ARMA_BLANCA
   tampoco fue neutral respecto al mes. Medido: F1 macro val CON mes =
   0.6052, SIN mes = 0.4715 (mismo experimento, misma semilla). Pendiente
   de que el equipo confirme sacarlo tambien de 03_preparacion.py para los
   4 modelos.

2) 'hora' SE EXCLUYE, se conserva 'franja_horaria'. 02_limpieza.py
   construye 'franja_horaria' a partir de 'hora' (pd.cut en 4 bloques) ->
   son la MISMA informacion en dos formatos. Medido: sacar 'hora' ademas de
   'mes' no cambio el F1 (0.4715 -> 0.4705, diferencia no significativa):
   confirma que era redundante, no dañina ni valiosa.

3) LA DATA FABRICADA DE ARMA_CONTUNDENTE/OTRAS TIENE UN PROBLEMA DE FONDO,
   no solo en 'mes'. Se verifico cruzando dataset.xlsx contra los datos
   oficiales (misma tecnica que en el punto 1):
     - 21.8% de las filas fabricadas de ARMA_CONTUNDENTE son duplicados de
       "plantilla" (mismo lugar/motivo/perfil, solo cambia fecha/edad),
       contra 1.4% en las filas reales -> señal de que se clonaron pocos
       casos base en vez de generar casos nuevos representativos.
     - La data fabricada esta MUCHO menos diversificada que la real: ej.
       'presunta_motivacion'=DELINCUENCIA COMUN es 44.9% en filas reales de
       ARMA_CONTUNDENTE pero 97.8% en las fabricadas; 'lugar'=VIA PUBLICA es
       35.7% real vs 67.3% fabricada.
     - El modelo entrenado con la mezcla predice MEJOR en las filas
       fabricadas de ARMA_CONTUNDENTE (F1=0.35) que en las reales (F1=0.09):
       aprendio a reconocer el "molde" de fabricacion, no la clase real.
   Esto significa que 'mes' no es el unico problema del dataset, es el mas
   facil de medir. No se puede "arreglar" generando mejor data sintetica
   (el proyecto ya descarta SMOTE/oversampling por la misma razon: fabrica
   informacion que no existe). La alternativa real, sin fabricar nada, es
   entrenar solo con filas REALES.

4) DECISION TOMADA PARA ESTE SCRIPT: entrenar con SOLO filas reales
   (verificadas contra los datos oficiales), y ademas SUBMUESTREAR las
   clases mayoritarias (ARMA_FUEGO, ARMA_BLANCA) para igualar a la
   minoritaria real (ARMA_CONTUNDENTE, ~812 filas en train) en vez de
   dejar el desbalance real 4:1. Comparado (mismos hiperparametros):

       Config                                  F1 macro | Precision | Recall
       Mezclado (real+fabricado, como venia)     0.4728  |  0.4936   | 0.4961
       Solo real, SIN submuestrear (desbalance)  0.4917  |  0.5684   | 0.4965
       Solo real, SUBMUESTREADO balanceado       0.4800  |  0.4930   | 0.5077

   Ninguna de las 3 es "mejor" sin matices: desbalanceado da mas precision
   pero ARMA_CONTUNDENTE queda casi invisible (recall 0.03); submuestreado
   sacrifica algo de precision pero ARMA_CONTUNDENTE se vuelve detectable
   (recall 0.38). Se elige SUBMUESTREADO porque un modelo que ignora una de
   las 4 clases del problema no cumple el objetivo del proyecto, aunque su
   metrica promedio se vea mejor.

5) RE-OPTIMIZACION DE HIPERPARAMETROS para el dataset submuestreado (mucho
   mas chico: ~3.250 filas vs las 14.630 originales). Los hiperparametros
   elegidos para el dataset grande NO sirven para este: con tan pocas
   filas, una red de 128 neuronas sobreajusta muchisimo (brecha train-val
   de hasta 0.56 sin regularizar). Por eso este script vuelve a correr los
   3 experimentos (arquitectura, epocas, alpha) sobre ESTE dataset, no
   reusa los numeros del dataset anterior. Tambien se prueba (y se
   descarta, ver mas abajo) agrupar categorias raras y sacar variables de
   aporte casi nulo: ambas dan trade-offs entre clases, no una mejora neta.

RESULTADO: no se puede subir mucho mas el F1 macro (~0.48-0.50) con ningun
ajuste de preprocesamiento o hiperparametros probado hasta ahora. Es
evidencia de un techo real de los datos disponibles (pocas filas reales de
ARMA_CONTUNDENTE/OTRAS), no del modelo. Documentado para la reunion de
equipo, no es una limitacion escondida.

DEPENDENCIAS DE ESTE SCRIPT (ademas de 03_preparacion.py):
  Necesita que ANTES se haya corrido tambien 02_limpieza.py, porque usa
  data/processed/A_2014_2025_limpio.csv y B_2026_limpio.csv (los datos
  oficiales sin alterar) para distinguir filas reales de fabricadas.

ARCHIVO AUTONOMO: no modifica 03_preparacion.py, 05_graficos.py ni los
demas modelo*.py (solo LEE sus .joblib, si existen, para la comparativa
final). Arma su propio split+preprocesador a partir de dataset.xlsx
directamente (reproduce los mismos pasos de 03_preparacion.py, misma
semilla), porque necesita las columnas originales (arma_categoria, anio,
coordenadas) para poder cruzar contra los datos oficiales y detectar que
filas son reales - esas columnas ya no estan en train_raw.csv.

Salida:
  modelos/modelo5_mlp.joblib
  modelos/preprocesador_mlp.joblib
  resultados/modelos/informe_modelo5_mlp.txt
  Graficos/modelo5_mlp_fig0_arquitectura.png
  Graficos/modelo5_mlp_fig1_matriz_confusion.png
  Graficos/modelo5_mlp_fig4_dinamica_epocas.png
  Graficos/modelo5_mlp_fig5_alpha.png            (solo si hubo overfitting)
  Graficos/modelo5_mlp_fig6_importancia.png
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"
DIR_FIG = BASE / "Graficos"
DIR_REP.mkdir(parents=True, exist_ok=True)
DIR_FIG.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Columnas de dataset.xlsx tal cual las lee 03_preparacion.py (necesitamos
# TODAS, incluidas las que 03_preparacion.py despues elimina, porque hacen
# falta para cruzar contra los datos oficiales).
COLUMNAS_DATASET = [
    "zona", "subzona", "distrito", "provincia", "canton",
    "coordenada_y", "coordenada_x", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "edad", "sexo",
    "etnia", "estado_civil", "nacionalidad", "discapacidad",
    "arma_categoria", "y", "anio", "mes", "dia_semana", "es_fin_semana",
    "fecha", "hora", "franja_horaria",
]
COLS_ELIMINAR = ["arma_categoria", "coordenada_x", "coordenada_y", "fecha",
                  "subzona", "distrito", "anio"]  # identico a 03_preparacion.py

# Lista COMPLETA de columnas categoricas (17), igual que 03_preparacion.py.
COLS_CATEGORICAS_TODAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "mes", "dia_semana",
    "hora", "franja_horaria",
]
# Lista que USA este modelo: sin 'mes' (leakage) ni 'hora' (redundante).
COLS_CATEGORICAS = [c for c in COLS_CATEGORICAS_TODAS if c not in ("mes", "hora")]
COLS_NUMERICAS = ["edad"]
COL_BINARIA = ["es_fin_semana"]

# Columnas para cruzar dataset.xlsx contra los datos oficiales limpios y
# detectar filas reales vs fabricadas (ver punto 3 del recorrido arriba).
# Sin coordenadas (riesgo de precision de floats) ni 'fecha' (redundante
# con anio/mes/dia_semana, que si se usan).
COLS_CLAVE_REAL = [
    "zona", "subzona", "distrito", "provincia", "canton",
    "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "edad", "sexo",
    "etnia", "estado_civil", "nacionalidad", "discapacidad",
    "arma_categoria", "y", "anio", "mes", "dia_semana", "es_fin_semana",
    "hora", "franja_horaria",
]

# Estilo de graficos (misma paleta que el resto del proyecto)
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
TINTA, TINTA_SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SUPERFICIE = "#e1e0d9", "#fcfcfb"
COLOR_CLASE = {
    "ARMA_FUEGO": "#2a78d6", "ARMA_BLANCA": "#eb6834",
    "ARMA_CONTUNDENTE": "#1baf7a", "OTRAS": "#eda100",
}

# --- Hiperparametros del barrido (explicitos, ver docstring) ---
# batch_size=64 (no 128 como en el barrido anterior): con ~3.250 filas de
# train (dataset submuestreado, mucho mas chico que las 14.630 originales),
# 128 daria muy pocos lotes por epoca (~25); 64 da un numero mas razonable
# de actualizaciones (~51 por epoca).
BATCH_SIZE = 64
ALPHA_DEFECTO = 0.0001    # L2, valor por defecto de sklearn
MAX_ITER_BARRIDO = 300    # tope para el experimento de arquitectura
MAX_EPOCAS_CURVA = 200    # tope para la curva epoca a epoca (experimento 2)
UMBRAL_OVERFITTING = 0.08  # mismo umbral que usan los otros 3 modelos del equipo
ALPHAS_BARRIDO = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3]  # experimento 3, solo si hace falta
N_SEMILLAS_ROBUSTEZ = 10  # experimento 4: cuantas semillas de submuestreo promediar

# Arquitecturas candidatas, ordenadas de MENOS a MAS parametros entrenables
# (no de menos a mas neuronas: dos capas chicas pueden tener menos
# parametros que una capa grande; el orden real se recalcula con los pesos
# ya entrenados, ver experimento_arquitectura).
ARQUITECTURAS = [(16,), (32,), (64,), (64, 32), (128,), (128, 64)]


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


rep = Reporte(DIR_REP / "informe_modelo5_mlp.txt")


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    rep.p(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  F1 (macro)       : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def base_estilo(ax, titulo, xlabel="", ylabel=""):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, color=TINTA, fontsize=12, pad=14, loc="left", weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=TINTA_SEC, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TINTA_SEC, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def guardar_fig(fig, nombre):
    ruta = DIR_FIG / f"modelo5_mlp_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Grafico: Graficos/{ruta.name}")


def n_parametros(modelo):
    """Cuenta los pesos + sesgos realmente entrenados (no una formula a mano)."""
    return int(sum(c.size for c in modelo.coefs_) + sum(b.size for b in modelo.intercepts_))


# ---------------------------------------------------------------------------
# Carga de datos: reproduce el split de 03_preparacion.py, marca filas
# reales vs fabricadas cruzando contra los datos oficiales, filtra a solo
# reales, y submuestrea el train para balancear (ver punto 3-4 del docstring)
# ---------------------------------------------------------------------------
def _clave(df, cols):
    partes = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s = s.round(2)
        s = s.astype(str).str.strip().str.upper()
        s = s.replace({"NAN": "NULL", "<NA>": "NULL", "NONE": "NULL"})
        partes.append(s)
    return partes[0].str.cat(partes[1:], sep="||")


def cargar_split_real():
    """Reproduce el split de 03_preparacion.py (misma semilla) y filtra a
    SOLO filas verificadas como reales. NO submuestrea todavia: eso se hace
    aparte (submuestrear_y_codificar) porque se repite con varias semillas
    para promediar (ver EXPERIMENTO 4 - robustez)."""
    ruta_a = DIR_PROC / "A_2014_2025_limpio.csv"
    ruta_b = DIR_PROC / "B_2026_limpio.csv"
    if not (ruta_a.exists() and ruta_b.exists()):
        raise FileNotFoundError(
            "Faltan data/processed/A_2014_2025_limpio.csv y/o B_2026_limpio.csv. "
            "Este script necesita que se haya corrido 'python 02_limpieza.py' "
            "ademas de 03_preparacion.py (los usa para distinguir filas reales "
            "de fabricadas en dataset.xlsx, ver docstring).")

    ds = pd.read_excel(DIR_PROC / "dataset.xlsx", dtype=str, header=None, skiprows=1)
    ds.columns = COLUMNAS_DATASET
    for c in ["edad", "anio", "mes", "dia_semana", "es_fin_semana", "hora"]:
        ds[c] = pd.to_numeric(ds[c], errors="coerce")

    of = pd.concat([pd.read_csv(ruta_a), pd.read_csv(ruta_b)], ignore_index=True)
    claves_oficiales = set(_clave(of, COLS_CLAVE_REAL))
    ds["_es_real"] = _clave(ds, COLS_CLAVE_REAL).isin(claves_oficiales)

    # --- desde aqui, mismos pasos que 03_preparacion.py (misma semilla) ---
    ds = ds.drop(columns=COLS_ELIMINAR)
    for c in COLS_CATEGORICAS_TODAS:
        ds[c] = ds[c].fillna("DESCONOCIDO")

    X = ds[COLS_CATEGORICAS_TODAS + COLS_NUMERICAS + COL_BINARIA]
    y, real = ds["y"], ds["_es_real"]

    X_train, X_tmp, y_train, y_tmp, real_train, real_tmp = train_test_split(
        X, y, real, test_size=0.30, stratify=y, random_state=SEMILLA)
    X_val, _, y_val, _, real_val, _ = train_test_split(
        X_tmp, y_tmp, real_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)

    n_train_total, n_val_total = len(X_train), len(X_val)

    # --- filtrar a SOLO filas reales (train y val) ---
    X_train_real = X_train[real_train.values]
    y_train_real = y_train[real_train.values]
    X_val_real = X_val[real_val.values]
    y_val_real = y_val[real_val.values]

    info = {
        "n_train_total": n_train_total, "n_val_total": n_val_total,
        "n_train_real": len(X_train_real), "n_val_real": len(X_val_real),
        "conteo_real": y_train_real.value_counts(),
    }
    return X_train_real, y_train_real, X_val_real, y_val_real, info


def submuestrear_y_codificar(X_train_real, y_train_real, X_val_real, y_val_real,
                              semilla_submuestreo):
    """Submuestrea el train para balancear las 4 clases (val se deja con su
    desbalance real, para evaluar contra la distribucion real del mundo), y
    ajusta el preprocesador propio (sin mes/hora) SOLO con ese train
    balanceado. semilla_submuestreo es la unica parte de esto que varia
    entre corridas: determina QUE filas de las clases mayoritarias
    sobreviven al recorte."""
    conteo = y_train_real.value_counts()
    n_min = conteo.min()
    rng = np.random.default_rng(semilla_submuestreo)
    idx_partes = []
    for clase in y_train_real.unique():
        idx_clase = y_train_real[y_train_real == clase].index
        idx_partes.append(rng.choice(idx_clase, size=n_min, replace=False))
    idx_balanceado = np.concatenate(idx_partes)
    X_train_bal = X_train_real.loc[idx_balanceado]
    y_train_bal = y_train_real.loc[idx_balanceado]

    cols = COLS_CATEGORICAS + COLS_NUMERICAS + COL_BINARIA
    pre_num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler())])
    pre_cat = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preproc = ColumnTransformer([
        ("num", pre_num, COLS_NUMERICAS),
        ("cat", pre_cat, COLS_CATEGORICAS),
        ("bin", "passthrough", COL_BINARIA),
    ])
    X_train_t = preproc.fit_transform(X_train_bal[cols])
    X_val_t = preproc.transform(X_val_real[cols])

    return X_train_t, y_train_bal.values, X_val_t, y_val_real.values, preproc, n_min


# ---------------------------------------------------------------------------
# EXPERIMENTO 1 - arquitectura (hidden_layer_sizes)
# ---------------------------------------------------------------------------
def experimento_arquitectura(X_train, y_train, X_val, y_val):
    rep.titulo("EXPERIMENTO 1: ¿que arquitectura usar? (barrido manual)")
    rep.p("  Se prueba cada hidden_layer_sizes con los demas hiperparametros")
    rep.p(f"  fijos (activation=relu, solver=adam, batch_size={BATCH_SIZE}, alpha={ALPHA_DEFECTO},")
    rep.p(f"  max_iter={MAX_ITER_BARRIDO}) y se mide F1 en train y en validacion.")
    rep.p("  NO se usa GridSearchCV: el barrido es explicito y esta documentado.")
    rep.p("")
    rep.p(f"  {'arquitectura':>14} | {'parametros':>10} | {'iter usadas':>11} | "
          f"{'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 78)

    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for arq in ARQUITECTURAS:
            m = MLPClassifier(
                hidden_layer_sizes=arq, activation="relu", solver="adam",
                batch_size=BATCH_SIZE, alpha=ALPHA_DEFECTO,
                max_iter=MAX_ITER_BARRIDO, random_state=SEMILLA,
            )
            m.fit(X_train, y_train)
            f1_tr = f1_score(y_train, m.predict(X_train), average="macro", zero_division=0)
            f1_va = f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)
            n_par = n_parametros(m)
            etiqueta = "x".join(str(n) for n in arq)
            rep.p(f"  {etiqueta:>14} | {n_par:>10,} | {m.n_iter_:>11} | "
                  f"{f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
            filas.append({"arquitectura": arq, "etiqueta": etiqueta,
                          "n_parametros": n_par, "f1_train": f1_tr,
                          "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas).sort_values("n_parametros").reset_index(drop=True)

    # CRITERIO DE PARSIMONIA (mismo que Random Forest y CatBoost): la
    # arquitectura con MENOS PARAMETROS cuyo F1 val este dentro del 1% del
    # mejor. Entre redes de rendimiento equivalente, la mas chica generaliza
    # mejor y es mas defendible (menos parametros = menos riesgo de
    # memorizar el train).
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    candidatos = df.index[df["f1_val"] >= umbral].tolist()
    i_elegido = candidatos[0]
    arq_elegida = df.loc[i_elegido, "arquitectura"]

    rep.p("")
    rep.p(f"  Mejor F1 absoluto : {f1_mejor:.4f} "
          f"(arquitectura={df.loc[df['f1_val'].idxmax(), 'etiqueta']})")
    rep.p(f"  Umbral del 1%     : {umbral:.4f}")
    rep.p(f"  ELEGIDA (menos parametros dentro del umbral): "
          f"{df.loc[i_elegido, 'etiqueta']} "
          f"({df.loc[i_elegido, 'n_parametros']:,} parametros) -> "
          f"F1 val {df.loc[i_elegido, 'f1_val']:.4f}")

    graficar_arquitectura(df, i_elegido)
    return arq_elegida, df


def graficar_arquitectura(df, i_elegido):
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.fill_between(list(x), df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Entrenamiento", zorder=3)
    ax.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Validacion", zorder=3)
    ax.axvline(i_elegido, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(i_elegido, df["f1_train"].max() + 0.03,
            f"elegida: {df.loc[i_elegido, 'etiqueta']}", ha="center",
            fontsize=9, color=TINTA, weight="bold")

    base_estilo(ax, "Experimento: arquitectura (hidden_layer_sizes) vs generalizacion\n"
                    "Ordenado por cantidad de parametros entrenables, no por neuronas.",
                xlabel="Arquitectura (capas ocultas)", ylabel="F1 macro")
    ax.set_xticks(list(x))
    etiquetas = [f"{r.etiqueta}\n({r.n_parametros:,} par.)" for r in df.itertuples()]
    ax.set_xticklabels(etiquetas, fontsize=8)
    leg = ax.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar_fig(fig, "fig0_arquitectura.png")


# ---------------------------------------------------------------------------
# EXPERIMENTO 2 - cuantas epocas (curva train/val, "early stopping" manual)
# ---------------------------------------------------------------------------
def experimento_epocas(arq, X_train, y_train, X_val, y_val):
    rep.titulo("EXPERIMENTO 2: ¿cuantas epocas entrenar? (curva train/val)")
    rep.p(f"  Arquitectura fija: {arq}. Se entrena epoca a epoca (warm_start,")
    rep.p("  1 epoca por paso) y se mide F1 en train y val en CADA epoca.")
    rep.p("  Esto reemplaza al early_stopping automatico de sklearn (que")
    rep.p("  separaria su propia validacion interna): aqui se decide contra")
    rep.p("  el set de validacion REAL del proyecto, de forma explicita.")
    rep.p("")

    modelo = MLPClassifier(
        hidden_layer_sizes=arq, activation="relu", solver="adam",
        batch_size=BATCH_SIZE, alpha=ALPHA_DEFECTO,
        max_iter=1, warm_start=True, random_state=SEMILLA,
    )
    epocas, f1_tr, f1_va = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for paso in range(1, MAX_EPOCAS_CURVA + 1):
            modelo.fit(X_train, y_train)
            epocas.append(paso)
            f1_tr.append(f1_score(y_train, modelo.predict(X_train), average="macro", zero_division=0))
            f1_va.append(f1_score(y_val, modelo.predict(X_val), average="macro", zero_division=0))

    df = pd.DataFrame({"epoca": epocas, "f1_train": f1_tr, "f1_val": f1_va})
    df["brecha"] = df["f1_train"] - df["f1_val"]

    # Mismo criterio de parsimonia: la epoca MAS TEMPRANA cuyo F1 val este
    # dentro del 1% del mejor F1 val alcanzado en toda la curva. Entrenar
    # de mas no gana casi nada y solo agranda la brecha train-val.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    i_elegido = df.index[df["f1_val"] >= umbral].tolist()[0]
    epoca_elegida = int(df.loc[i_elegido, "epoca"])

    rep.p(f"  Mejor F1 val en la curva : {f1_mejor:.4f} "
          f"(epoca {int(df.loc[df['f1_val'].idxmax(), 'epoca'])})")
    rep.p(f"  Umbral del 1%            : {umbral:.4f}")
    rep.p(f"  ELEGIDA (epoca mas temprana dentro del umbral): epoca {epoca_elegida} "
          f"-> F1 val {df.loc[i_elegido, 'f1_val']:.4f}, brecha {df.loc[i_elegido, 'brecha']:.4f}")

    graficar_epocas(df, i_elegido)
    return epoca_elegida, df


# ---------------------------------------------------------------------------
# EXPERIMENTO 3 - alpha (regularizacion L2), SOLO si hay overfitting
# ---------------------------------------------------------------------------
def experimento_alpha(arq, epocas, X_train, y_train, X_val, y_val):
    """Barrido manual de alpha (L2). Se corre SOLO si el modelo con el alpha
    por defecto muestra overfitting (brecha train-val > UMBRAL_OVERFITTING).
    alpha mas alto = pesos mas chicos = red menos libre para memorizar."""
    rep.titulo("EXPERIMENTO 3: ¿que alpha usar? (barrido manual, por overfitting)")
    rep.p(f"  El modelo con alpha={ALPHA_DEFECTO} (por defecto) supero el umbral de")
    rep.p(f"  overfitting ({UMBRAL_OVERFITTING}), asi que se barre alpha con la")
    rep.p(f"  arquitectura ({arq}) y las epocas ({epocas}) ya elegidas, fijas.")
    rep.p("")
    rep.p(f"  {'alpha':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 42)

    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for a in ALPHAS_BARRIDO:
            m = MLPClassifier(hidden_layer_sizes=arq, activation="relu", solver="adam",
                              batch_size=BATCH_SIZE, alpha=a, max_iter=epocas,
                              random_state=SEMILLA)
            m.fit(X_train, y_train)
            f1_tr = f1_score(y_train, m.predict(X_train), average="macro", zero_division=0)
            f1_va = f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)
            rep.p(f"  {a:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
            filas.append({"alpha": a, "f1_train": f1_tr, "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas)

    # Mismo criterio de parsimonia que arquitectura/epocas, pero en la
    # direccion opuesta: acá "mas simple" = MAS regularizado (alpha mas
    # alto), asi que se elige el alpha MAS ALTO cuyo F1 val siga dentro del
    # 1% del mejor. Un alpha mayor reduce mas la brecha sin resignar F1 real.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    candidatos = df.index[df["f1_val"] >= umbral].tolist()
    i_elegido = candidatos[-1]   # el ULTIMO = el alpha mas alto (mas regularizado)
    alpha_elegido = float(df.loc[i_elegido, "alpha"])

    rep.p("")
    rep.p(f"  Mejor F1 val   : {f1_mejor:.4f} (alpha={df.loc[df['f1_val'].idxmax(), 'alpha']})")
    rep.p(f"  Umbral del 1%  : {umbral:.4f}")
    rep.p(f"  ELEGIDO (mas regularizado dentro del umbral): alpha={alpha_elegido} -> "
          f"F1 val {df.loc[i_elegido, 'f1_val']:.4f}, brecha {df.loc[i_elegido, 'brecha']:.4f}")

    graficar_alpha(df, i_elegido)
    return alpha_elegido, df


def graficar_alpha(df, i_elegido):
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(list(x), df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Entrenamiento", zorder=3)
    ax.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Validacion", zorder=3)
    ax.axvline(i_elegido, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(i_elegido, df["f1_train"].max() + 0.02,
            f"elegido: alpha={df.loc[i_elegido, 'alpha']}", ha="center",
            fontsize=9, color=TINTA, weight="bold")
    base_estilo(ax, "Experimento: alpha (regularizacion L2) vs overfitting\n"
                    "Mas alpha = pesos mas chicos = menos memorizacion.",
                xlabel="alpha", ylabel="F1 macro")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in df["alpha"]], fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar_fig(fig, "fig5_alpha.png")


def graficar_epocas(df, i_elegido):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.fill_between(df["epoca"], df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax.plot(df["epoca"], df["f1_train"], color=COLOR_TRAIN, linewidth=1.8,
            label="Entrenamiento", zorder=3)
    ax.plot(df["epoca"], df["f1_val"], color=COLOR_VAL, linewidth=1.8,
            label="Validacion", zorder=3)
    epoca_elegida = df.loc[i_elegido, "epoca"]
    ax.axvline(epoca_elegida, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(epoca_elegida, df["f1_train"].max() + 0.015,
            f"elegida: epoca {epoca_elegida}", ha="center", fontsize=9,
            color=TINTA, weight="bold")
    base_estilo(ax, "Como aprende la red, epoca a epoca\n"
                    "Curva de generalizacion real (no la interna de sklearn).",
                xlabel="Epoca (pasadas completas por train)", ylabel="F1 macro")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar_fig(fig, "fig4_dinamica_epocas.png")


# ---------------------------------------------------------------------------
# Matriz de confusion (estilo normalizado, igual al de 05_graficos.py)
# ---------------------------------------------------------------------------
def graficar_matriz(y_val, pred_val):
    cm = confusion_matrix(y_val, pred_val, labels=CLASES)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASES], fontsize=9)
    ax.set_yticklabels([c.replace("_", "\n") for c in CLASES], fontsize=9)
    ax.set_xlabel("Lo que el modelo PREDIJO", color=TINTA_SEC, fontsize=10)
    ax.set_ylabel("Lo que REALMENTE era", color=TINTA_SEC, fontsize=10)
    ax.set_title("Matriz de confusion - MLP (validacion, solo filas reales)\n"
                 "Diagonal = aciertos. Fuera de la diagonal = confusiones.",
                 color=TINTA, fontsize=12, pad=16, loc="left", weight="bold")
    ax.tick_params(colors=MUTED)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            color_txt = "#ffffff" if cm_pct[i, j] > 50 else TINTA
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%", ha="center",
                    va="center", color=color_txt, fontsize=10,
                    weight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% de la clase real", color=TINTA_SEC, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    guardar_fig(fig, "fig1_matriz_confusion.png")


# ---------------------------------------------------------------------------
# EXPERIMENTO 4 - robustez del submuestreo (varias semillas, se promedia)
# ---------------------------------------------------------------------------
def experimento_robustez(X_train_real, y_train_real, X_val_real, y_val_real, params):
    """El submuestreo descarta muchas filas de las clases mayoritarias
    (ARMA_FUEGO/ARMA_BLANCA): que filas sobreviven depende de la semilla.
    Reportar UNA sola corrida corre el riesgo de mostrar un numero con
    suerte/mala suerte. Se repite con N semillas distintas (mismos
    hiperparametros ya elegidos, fijos) y se reporta el promedio +/-
    desvio, que es el numero mas defendible."""
    rep.titulo("EXPERIMENTO 4: robustez del submuestreo (varias semillas)")
    rep.p(f"  Se repite el submuestreo+entrenamiento con {N_SEMILLAS_ROBUSTEZ} semillas")
    rep.p("  distintas (mismos hiperparametros ya elegidos: arquitectura, alpha,")
    rep.p("  epocas). El val NO cambia entre corridas, solo que filas de train")
    rep.p("  sobreviven al submuestreo. Esto mide cuanto varia el resultado por")
    rep.p("  el azar de cual submuestra toco, no reemplaza el modelo final.")
    rep.p("")
    rep.p(f"  {'semilla':>8} | {'F1 val':>9} | {'Precision':>9} | {'Recall':>9}")
    rep.p("  " + "-" * 44)

    resultados = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for s in range(N_SEMILLAS_ROBUSTEZ):
            Xtr, ytr, Xva, yva, _, _ = submuestrear_y_codificar(
                X_train_real, y_train_real, X_val_real, y_val_real, semilla_submuestreo=s)
            m = MLPClassifier(**params)
            m.fit(Xtr, ytr)
            p = m.predict(Xva)
            f1 = f1_score(yva, p, average="macro", zero_division=0)
            prec = precision_score(yva, p, average="macro", zero_division=0)
            rec = recall_score(yva, p, average="macro", zero_division=0)
            rep.p(f"  {s:>8} | {f1:9.4f} | {prec:9.4f} | {rec:9.4f}")
            resultados.append((f1, prec, rec))

    arr = np.array(resultados)
    rep.p("")
    rep.p(f"  F1 macro   : media={arr[:,0].mean():.4f}  desvio={arr[:,0].std():.4f}  "
          f"min={arr[:,0].min():.4f}  max={arr[:,0].max():.4f}")
    rep.p(f"  Precision  : media={arr[:,1].mean():.4f}  desvio={arr[:,1].std():.4f}")
    rep.p(f"  Recall     : media={arr[:,2].mean():.4f}  desvio={arr[:,2].std():.4f}")
    rep.p("")
    rep.p(f"  NUMERO A REPORTAR (mas defendible que una sola corrida): "
          f"F1 macro = {arr[:,0].mean():.4f} +/- {arr[:,0].std():.4f}")

    graficar_robustez(arr[:, 0])
    return arr[:, 0].mean(), arr[:, 0].std()


def graficar_robustez(f1_por_semilla):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(f1_por_semilla))
    media, desvio = f1_por_semilla.mean(), f1_por_semilla.std()

    ax.axhspan(media - desvio, media + desvio, color=COLOR_VAL, alpha=0.10, zorder=1)
    ax.axhline(media, color=COLOR_VAL, linestyle="--", linewidth=1.6, zorder=2,
               label=f"Media = {media:.4f}")
    ax.scatter(list(x), f1_por_semilla, color=COLOR_TRAIN, s=60, zorder=3,
              edgecolor=SUPERFICIE, linewidth=1.5, label="F1 por semilla de submuestreo")

    base_estilo(ax, "Robustez del submuestreo: F1 macro en 10 semillas distintas\n"
                    f"Media {media:.4f} +/- {desvio:.4f} (desvio chico = resultado estable).",
                xlabel="Semilla de submuestreo", ylabel="F1 macro (val)")
    ax.set_xticks(list(x))
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar_fig(fig, "fig7_robustez.png")


# ---------------------------------------------------------------------------
# Importancia por permutacion, AGRUPADA por variable original
# ---------------------------------------------------------------------------
def nombres_agrupados(preproc):
    """Reconstruye, en el mismo orden que las columnas del preprocesador, a
    que variable ORIGINAL pertenece cada columna one-hot (para agrupar la
    importancia por permutacion en vez de dispersarla en decenas de
    dummies)."""
    grupos = list(COLS_NUMERICAS)  # 1 columna: 'edad'
    onehot = preproc.named_transformers_["cat"]["onehot"]
    for col, categorias in zip(COLS_CATEGORICAS, onehot.categories_):
        grupos.extend([col] * len(categorias))
    grupos.extend(COL_BINARIA)     # 1 columna: 'es_fin_semana'
    return grupos


def experimento_importancia(modelo, X_val, y_val, preproc):
    rep.titulo("IMPORTANCIA DE VARIABLES (permutation_importance, val)")
    rep.p("  El MLP no expone coef_ ni feature_importances_. Se mide cuanto")
    rep.p("  EMPEORA el F1 macro al mezclar (permutar) al azar los valores de")
    rep.p("  una columna, manteniendo el resto igual: si el F1 cae mucho, esa")
    rep.p("  columna era importante para la prediccion. Se hace sobre VAL, no")
    rep.p("  train, para medir importancia real (no memorizacion).")
    rep.p("  Se agrupan las columnas one-hot por variable ORIGINAL (si no, la")
    rep.p("  importancia de 'provincia' quedaria dispersa en 24 columnas).")

    resultado = permutation_importance(
        modelo, X_val, y_val, scoring="f1_macro", n_repeats=10,
        random_state=SEMILLA, n_jobs=-1,
    )
    grupos = nombres_agrupados(preproc)
    imp = pd.Series(resultado.importances_mean, index=grupos).groupby(level=0).sum()
    imp = imp.sort_values(ascending=False)

    rep.p("")
    rep.p(imp.round(4).to_string())
    graficar_importancia(imp)
    return imp


def graficar_importancia(imp):
    top = imp.head(19).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    barras = ax.barh(range(len(top)), top.values, color="#2a78d6", zorder=3, height=0.7)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(1.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    base_estilo(ax, "Importancia de variables - MLP (permutation importance)\n"
                    "Agrupada por variable original, no por columna one-hot.",
                xlabel="Caida de F1 macro al permutar (val)")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(top.values):
        ax.text(v + max(top.values.max(), 1e-6) * 0.02, i, f"{v:.4f}", va="center",
                fontsize=8, color=TINTA_SEC)
    guardar_fig(fig, "fig6_importancia.png")


# ---------------------------------------------------------------------------
def main():
    rep.titulo("MODELO 5: MLP (RED NEURONAL MULTICAPA)", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")

    X_train_real, y_train_real, X_val_real, y_val_real, info = cargar_split_real()
    X_train, y_train, X_val, y_val, preproc, n_min = submuestrear_y_codificar(
        X_train_real, y_train_real, X_val_real, y_val_real, semilla_submuestreo=SEMILLA)

    rep.titulo("DATOS: solo filas REALES, train submuestreado balanceado")
    rep.p(f"  Split original (mezcla real+fabricado, igual que 03_preparacion.py):")
    rep.p(f"    train: {info['n_train_total']:,} filas | val: {info['n_val_total']:,} filas")
    rep.p(f"  Filas verificadas como REALES (cruzadas contra A_2014_2025_limpio.csv +")
    rep.p(f"  B_2026_limpio.csv):")
    rep.p(f"    train-real: {info['n_train_real']:,} ({100*info['n_train_real']/info['n_train_total']:.1f}%)")
    rep.p(f"    val-real  : {info['n_val_real']:,} ({100*info['n_val_real']/info['n_val_total']:.1f}%)")
    rep.p(f"  Balance de clases en train-real (antes de submuestrear):")
    for k, v in info["conteo_real"].items():
        rep.p(f"    {k:<18} {v:,}")
    rep.p(f"  Clase minoritaria: {n_min:,} filas -> se submuestrean TODAS las")
    rep.p(f"  clases a {n_min:,} (sin fabricar nada, solo usando menos filas reales")
    rep.p(f"  de las clases que sobran). Train final: {len(y_train):,} filas.")
    rep.p(f"  Val se deja con su desbalance real (no se toca): asi la evaluacion refleja")
    rep.p(f"  la distribucion real de clases, no una artificial.")
    rep.p(f"  NOTA: que filas de FUEGO/BLANCA sobreviven al recorte depende de la semilla")
    rep.p(f"  de submuestreo. Se usa semilla={SEMILLA} para el modelo que se guarda, pero")
    rep.p(f"  el EXPERIMENTO 4 mas abajo mide cuanto varia el resultado entre semillas.")
    rep.p(f"\n  Train (codificado): {X_train.shape}  |  Val (codificado): {X_val.shape}")
    rep.p(f"  Columnas categoricas usadas ({len(COLS_CATEGORICAS)}, sin 'mes' ni 'hora'): {COLS_CATEGORICAS}")

    rep.titulo("BATCH SIZE (concepto de mini-batch, explicito)")
    n_batches = -(-len(X_train) // BATCH_SIZE)  # ceil
    rep.p(f"  batch_size = {BATCH_SIZE} filas por lote (fijo, NO se deja el 'auto' de")
    rep.p(f"  sklearn). Con un train mucho mas chico que el original (submuestreado),")
    rep.p(f"  128 daria muy pocos lotes por epoca; {BATCH_SIZE} da un numero mas razonable.")
    rep.p(f"  {len(X_train):,} filas de train / {BATCH_SIZE} = {n_batches} lotes por epoca.")
    rep.p("  Cada lote produce UNA actualizacion de pesos (Adam). Una epoca =")
    rep.p(f"  pasar por los {n_batches} lotes una vez.")

    # --- Experimentos explicitos (re-optimizados para ESTE tamanio de dataset) ---
    arq, tabla_arq = experimento_arquitectura(X_train, y_train, X_val, y_val)
    epocas, tabla_epocas = experimento_epocas(arq, X_train, y_train, X_val, y_val)

    # --- Modelo final ---
    rep.titulo("MODELO FINAL (hiperparametros explicitos)")
    params = dict(
        hidden_layer_sizes=arq,      # elegida en el experimento 1
        activation="relu",           # evita el desvanecimiento de gradiente de sigmoid/tanh
        solver="adam",               # descenso de gradiente por mini-batches
        batch_size=BATCH_SIZE,       # ver seccion de batch size arriba
        alpha=ALPHA_DEFECTO,         # regularizacion L2, valor por defecto de sklearn
        max_iter=epocas,             # elegido en el experimento 2 (curva train/val)
        random_state=SEMILLA,
    )
    for k, v in params.items():
        rep.p(f"  {k} = {v}")
    rep.p("")
    rep.p("  Justificacion de cada uno:")
    rep.p("    hidden_layer_sizes : fijada por el experimento 1 (parsimonia).")
    rep.p("    activation=relu    : estandar, no satura el gradiente como tanh/sigmoid.")
    rep.p("    solver=adam        : descenso de gradiente por mini-batches, con momento")
    rep.p("                         adaptativo.")
    rep.p(f"    batch_size={BATCH_SIZE}      : tamanio de mini-batch, fijado y justificado arriba.")
    rep.p("    alpha              : regularizacion L2 sobre los pesos (evita que crezcan")
    rep.p("                         sin control). Valor por defecto de sklearn.")
    rep.p("    max_iter=epocas    : fijado por el experimento 2, NO por early_stopping")
    rep.p("                         automatico de sklearn.")
    rep.p("    class_weight       : no existe en MLPClassifier. El balance se logra")
    rep.p("                         submuestreando train (ver arriba), no con este parametro.")

    modelo = MLPClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(X_train, y_train)

    if modelo.n_iter_ >= params["max_iter"]:
        rep.p(f"\n  Uso las {modelo.n_iter_} epocas completas (el limite fijado).")
    else:
        rep.p(f"\n  Convergio en {modelo.n_iter_} epocas (de un maximo de {params['max_iter']}):")
        rep.p("  la perdida de entrenamiento dejo de mejorar antes del limite.")

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    rep.titulo("METRICAS")
    m_train = metricas(y_train, pred_train, "TRAIN")
    m_val = metricas(y_val, pred_val, "VALIDACION")

    rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING (alpha por defecto)")
    diff_f1 = m_train["f1_macro"] - m_val["f1_macro"]
    rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia     : {diff_f1:.4f}")
    if diff_f1 > UMBRAL_OVERFITTING:
        rep.p(f"  -> OVERFITTING (brecha > {UMBRAL_OVERFITTING}): memoriza train, no")
        rep.p("     generaliza igual a val. Se corre el experimento 3 (alpha) abajo.")
    elif m_train["f1_macro"] < 0.5:
        rep.p("  -> Posible UNDERFITTING (ni siquiera en train aprende bien).")
    else:
        rep.p("  -> Sin senales fuertes de over/underfitting: train y val estan cerca.")

    # --- Experimento 3 (condicional): solo si el modelo con alpha por
    # defecto quedo overfitteado. Si no hace falta, se sigue con ESTE modelo.
    alpha_final = ALPHA_DEFECTO
    if diff_f1 > UMBRAL_OVERFITTING:
        alpha_final, tabla_alpha = experimento_alpha(arq, epocas, X_train, y_train, X_val, y_val)

        if alpha_final != ALPHA_DEFECTO:
            rep.titulo("MODELO FINAL, REENTRENADO CON alpha AJUSTADO")
            rep.p(f"  alpha: {ALPHA_DEFECTO} (por defecto) -> {alpha_final} (experimento 3)")
            params["alpha"] = alpha_final
            modelo = MLPClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                modelo.fit(X_train, y_train)
            pred_train = modelo.predict(X_train)
            pred_val = modelo.predict(X_val)

            rep.titulo("METRICAS (con alpha ajustado)")
            m_train = metricas(y_train, pred_train, "TRAIN")
            m_val = metricas(y_val, pred_val, "VALIDACION")

            rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING (alpha ajustado)")
            diff_f1 = m_train["f1_macro"] - m_val["f1_macro"]
            rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
            rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
            rep.p(f"  Diferencia     : {diff_f1:.4f}")
            if diff_f1 > UMBRAL_OVERFITTING:
                rep.p("  -> Sigue habiendo overfitting, pero es lo mejor que se logra")
                rep.p("     dentro del umbral del 1% de F1 val (ver experimento 3).")
            else:
                rep.p("  -> Corregido: la brecha ya esta dentro de lo saludable.")

    # --- Experimento 4: robustez del submuestreo (varias semillas) ---
    f1_robusto_media, f1_robusto_desvio = experimento_robustez(
        X_train_real, y_train_real, X_val_real, y_val_real, params)

    rep.titulo("REPORTE POR CLASE (VALIDACION, solo filas reales)")
    rep.p(classification_report(y_val, pred_val, labels=CLASES, zero_division=0))

    rep.titulo("MATRIZ DE CONFUSION (VALIDACION, solo filas reales)")
    cm = confusion_matrix(y_val, pred_val, labels=CLASES)
    cm_df = pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                         columns=[f"pred_{c}" for c in CLASES])
    rep.p(cm_df.to_string())
    graficar_matriz(y_val, pred_val)

    # --- Importancia (permutation, agrupada) ---
    experimento_importancia(modelo, X_val, y_val, preproc)

    # --- Comparativa contra los modelos anteriores ---
    # OJO: logistica/RF/CatBoost se entrenaron con el set de columnas
    # COMPARTIDO (que todavia incluye 'mes'/'hora' Y las filas fabricadas),
    # asi que esta comparacion no es pareja hasta que el equipo confirme
    # los mismos cambios para los 4 modelos. Se deja como referencia.
    rep.titulo("COMPARATIVA CONTRA LOS MODELOS ANTERIORES (validacion)")
    rep.p("  (logistica/RF/CatBoost siguen usando 'mes'/'hora' Y filas fabricadas ->")
    rep.p("  no es una comparacion 100% pareja hasta que el equipo confirme los")
    rep.p("  mismos cambios para los 4 modelos. Se deja como referencia.)")
    rep.p(f"  MLP: se usa el F1 ROBUSTO del experimento 4 ({f1_robusto_media:.4f} +/- "
          f"{f1_robusto_desvio:.4f}), no el de una sola corrida, para la comparativa.")
    filas = {"MLP (real, submuestreado)": [m_val["accuracy"], f1_robusto_media]}

    ruta_l = DIR_MOD / "modelo1_logistica.joblib"
    ruta_rf = DIR_MOD / "modelo2_randomforest.joblib"
    if ruta_l.exists() or ruta_rf.exists():
        Xv_compartido = np.load(DIR_PROC / "val_X.npy")
        yv_compartido = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)
        for nombre, ruta in [("Regresion Logistica", ruta_l), ("Random Forest", ruta_rf)]:
            if ruta.exists():
                m = joblib.load(ruta)
                p = m.predict(Xv_compartido)
                filas[nombre] = [accuracy_score(yv_compartido, p),
                                 f1_score(yv_compartido, p, average="macro", zero_division=0)]
            else:
                rep.p(f"  ({ruta.name} no encontrado, se omite)")
    else:
        rep.p("  (modelo1_logistica.joblib / modelo2_randomforest.joblib no encontrados, se omiten)")

    ruta_cb = DIR_MOD / "modelo3_catboost.joblib"
    if ruta_cb.exists():
        va_completo = pd.read_csv(DIR_PROC / "val_raw.csv")
        m = joblib.load(ruta_cb)
        p = m.predict(va_completo[COLS_CATEGORICAS_TODAS + COLS_NUMERICAS + COL_BINARIA]).ravel()
        filas["CatBoost"] = [accuracy_score(va_completo["y"], p),
                             f1_score(va_completo["y"], p, average="macro", zero_division=0)]
    else:
        rep.p("  (modelo3_catboost.joblib no encontrado, se omite)")

    comp = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)"]).round(4)
    rep.p(comp.to_string())

    joblib.dump(modelo, DIR_MOD / "modelo5_mlp.joblib")
    joblib.dump(preproc, DIR_MOD / "preprocesador_mlp.joblib")

    rep.titulo("FIN MODELO 5", "#")
    rep.p(f"RESULTADO A REPORTAR: F1 macro = {f1_robusto_media:.4f} +/- {f1_robusto_desvio:.4f}")
    rep.p(f"(promedio de {N_SEMILLAS_ROBUSTEZ} semillas de submuestreo, experimento 4 -")
    rep.p(f" mas defendible que el numero de una sola corrida, {m_val['f1_macro']:.4f})")
    rep.p("Guardado: modelos/modelo5_mlp.joblib (instancia con semilla={})".format(SEMILLA))
    rep.p("Guardado: modelos/preprocesador_mlp.joblib (propio: sin mes/hora, ajustado")
    rep.p("          SOLO con filas reales submuestreadas balanceadas)")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
