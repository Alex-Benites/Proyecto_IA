"""
ETAPA 4 - MODELO 4: XGBOOST MULTICLASE
=======================================

Objetivo:
  Entrenar y comparar tres escenarios del mismo algoritmo XGBoost:

  A) XGBoost BASE:
     Usa los hiperparametros del booster practicamente por defecto.
     Solo se fijan los parametros necesarios para el problema multiclase
     y la semilla para reproducibilidad.

  B) XGBoost AJUSTE MANUAL:
     Modifica de forma razonada algunos hiperparametros importantes:
     cantidad de arboles, learning rate, profundidad, muestreo y regularizacion.

  C) XGBoost BUSQUEDA AUTOMATICA:
     Usa RandomizedSearchCV sobre el conjunto TRAIN para buscar una combinacion
     de hiperparametros que maximice F1 macro mediante validacion cruzada
     estratificada.

IMPORTANTE:
  - Este script NO vuelve a limpiar, imputar, codificar ni escalar.
  - Usa directamente la salida de 03_preparacion.py.
  - El conjunto TEST NO se toca aqui. Queda sellado para la evaluacion final.
  - La seleccion entre los tres escenarios se realiza usando VALIDACION.
  - El dataset de entrada ya esta balanceado artificialmente, por lo que
    NO se aplica class_weight ni sample_weight en este script.

Entradas:
  data/processed/train_X.npy
  data/processed/train_y.npy
  data/processed/val_X.npy
  data/processed/val_y.npy

Salidas:
  modelos/modelo4_xgb_default.joblib
  modelos/modelo4_xgb_manual.joblib
  modelos/modelo4_xgb_search.joblib
  modelos/modelo4_xgboost_mejor.joblib
  modelos/modelo4_xgboost_clases.joblib

  resultados/modelos/informe_modelo4_xgboost.txt
  resultados/modelos/comparacion_xgboost.csv
  resultados/modelos/xgboost_busqueda_resultados.csv
  resultados/modelos/matriz_xgb_default.png
  resultados/modelos/matriz_xgb_manual.png
  resultados/modelos/matriz_xgb_search.png
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "No se encontro XGBoost. Instalar con: pip install xgboost"
    ) from exc


# =============================================================================
# CONFIGURACION GENERAL
# =============================================================================

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"

DIR_MOD.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)

CLASES = [
    "ARMA_FUEGO",
    "ARMA_BLANCA",
    "ARMA_CONTUNDENTE",
    "OTRAS",
]

MAPA_CLASES = {clase: i for i, clase in enumerate(CLASES)}
MAPA_INVERSO = {i: clase for clase, i in MAPA_CLASES.items()}

N_CLASES = len(CLASES)

# La busqueda hace N_ITER_BUSQUEDA * CV_FOLDS entrenamientos.
# 25 * 5 = 125 ajustes de XGBoost.
N_ITER_BUSQUEDA = 25
CV_FOLDS = 5


# =============================================================================
# REPORTE
# =============================================================================

class Reporte:
    def __init__(self, ruta):
        self.buf = []
        self.ruta = ruta

    def p(self, *args):
        texto = " ".join(str(x) for x in args)
        print(texto)
        self.buf.append(texto)

    def titulo(self, texto, char="="):
        self.p("")
        self.p(char * 78)
        self.p(texto)
        self.p(char * 78)

    def guardar(self):
        self.ruta.write_text("\n".join(self.buf), encoding="utf-8")


rep = Reporte(DIR_REP / "informe_modelo4_xgboost.txt")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def codificar_y(y, nombre):
    """
    Convierte las etiquetas de texto a enteros 0..3.

    Se usa un mapeo fijo en lugar de LabelEncoder para conservar exactamente
    este orden:
      0 = ARMA_FUEGO
      1 = ARMA_BLANCA
      2 = ARMA_CONTUNDENTE
      3 = OTRAS
    """
    y = np.asarray(y).astype(str)

    desconocidas = sorted(set(y) - set(CLASES))
    if desconocidas:
        raise ValueError(
            f"{nombre} contiene clases no reconocidas: {desconocidas}"
        )

    return np.array([MAPA_CLASES[v] for v in y], dtype=np.int32)


def metricas(y_true, y_pred, y_proba, nombre):
    """Calcula las metricas principales para clasificacion multiclase."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    rec = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    perdida = log_loss(
        y_true,
        y_proba,
        labels=list(range(N_CLASES)),
    )

    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {acc:.4f}")
    rep.p(f"  Precision (macro): {prec:.4f}")
    rep.p(f"  Recall (macro)   : {rec:.4f}")
    rep.p(f"  F1 (macro)       : {f1:.4f}")
    rep.p(f"  Log Loss         : {perdida:.4f}")

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "log_loss": perdida,
    }


def graficar_matriz(y_true, y_pred, titulo, ruta):
    """Guarda la matriz de confusion del conjunto de validacion."""
    etiquetas = list(range(N_CLASES))
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(N_CLASES))
    ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES, rotation=45, ha="right")
    ax.set_yticklabels(CLASES)

    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    limite = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(N_CLASES):
        for j in range(N_CLASES):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > limite else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def evaluar_escenario(nombre, modelo, X_train, y_train, X_val, y_val, tiempo):
    """
    Evalua TRAIN y VALIDACION.

    TRAIN sirve para observar si el modelo esta memorizando.
    VALIDACION sirve para comparar los tres escenarios.
    """
    pred_train = modelo.predict(X_train)
    proba_train = modelo.predict_proba(X_train)

    pred_val = modelo.predict(X_val)
    proba_val = modelo.predict_proba(X_val)

    rep.titulo(f"RESULTADOS - {nombre}")

    m_train = metricas(
        y_train,
        pred_train,
        proba_train,
        "TRAIN",
    )

    m_val = metricas(
        y_val,
        pred_val,
        proba_val,
        "VALIDACION",
    )

    diff_f1 = m_train["f1_macro"] - m_val["f1_macro"]

    rep.p("\n  Chequeo train vs validacion:")
    rep.p(f"  F1 train         : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 validacion    : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia       : {diff_f1:.4f}")
    rep.p(f"  Tiempo entrenamiento: {tiempo:.2f} segundos")

    if diff_f1 > 0.08:
        rep.p("  -> Posible OVERFITTING.")
    elif m_train["f1_macro"] < 0.50:
        rep.p("  -> Posible UNDERFITTING.")
    else:
        rep.p("  -> Sin una senal fuerte de over/underfitting.")

    rep.p("\n  Reporte por clase - VALIDACION:")
    rep.p(
        classification_report(
            y_val,
            pred_val,
            labels=list(range(N_CLASES)),
            target_names=CLASES,
            zero_division=0,
        )
    )

    return {
        "modelo": nombre,
        "accuracy_train": m_train["accuracy"],
        "f1_train": m_train["f1_macro"],
        "accuracy_val": m_val["accuracy"],
        "precision_macro_val": m_val["precision_macro"],
        "recall_macro_val": m_val["recall_macro"],
        "f1_macro_val": m_val["f1_macro"],
        "log_loss_val": m_val["log_loss"],
        "gap_f1_train_val": diff_f1,
        "tiempo_segundos": tiempo,
    }


def mostrar_parametros_relevantes(modelo, titulo):
    """Imprime solo los parametros relevantes para explicar el experimento."""
    p = modelo.get_params()

    relevantes = [
        "booster",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "tree_method",
    ]

    rep.titulo(titulo)

    for nombre in relevantes:
        if nombre in p:
            rep.p(f"  {nombre:<20} = {p[nombre]}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    rep.titulo("ETAPA 4 - MODELO 4: XGBOOST MULTICLASE", "#")
    rep.p(
        f"Python {sys.version.split()[0]} | "
        f"scikit-learn {sklearn.__version__} | "
        f"xgboost {xgboost.__version__} | "
        f"semilla {SEMILLA}"
    )

    # -------------------------------------------------------------------------
    # 1. CARGA DE DATOS GENERADOS POR 03_preparacion.py
    # -------------------------------------------------------------------------

    rep.titulo("CARGA DE DATOS")

    archivos_necesarios = [
        DIR_PROC / "train_X.npy",
        DIR_PROC / "train_y.npy",
        DIR_PROC / "val_X.npy",
        DIR_PROC / "val_y.npy",
    ]

    faltantes = [str(p) for p in archivos_necesarios if not p.exists()]

    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos de la Etapa 3:\n" + "\n".join(faltantes)
        )

    X_train = np.load(DIR_PROC / "train_X.npy")
    y_train_txt = np.load(
        DIR_PROC / "train_y.npy",
        allow_pickle=True,
    )

    X_val = np.load(DIR_PROC / "val_X.npy")
    y_val_txt = np.load(
        DIR_PROC / "val_y.npy",
        allow_pickle=True,
    )

    y_train = codificar_y(y_train_txt, "train_y")
    y_val = codificar_y(y_val_txt, "val_y")

    rep.p(f"  Train: X={X_train.shape} | y={y_train.shape}")
    rep.p(f"  Val  : X={X_val.shape} | y={y_val.shape}")

    rep.p("\n  Mapeo de clases usado por XGBoost:")
    for clase, codigo in MAPA_CLASES.items():
        rep.p(f"    {codigo} -> {clase}")

    rep.p("\n  Distribucion TRAIN:")
    for i, clase in enumerate(CLASES):
        n = int((y_train == i).sum())
        porcentaje = 100 * n / len(y_train)
        rep.p(f"    {clase:<18}: {n:6d} ({porcentaje:5.2f}%)")

    rep.p("\n  NOTA: test_X.npy y test_y.npy NO se cargan.")
    rep.p("  El conjunto TEST permanece sellado para la evaluacion final.")

    # Guardamos el mapeo para la aplicacion/inferencia futura.
    joblib.dump(
        {
            "clases": CLASES,
            "mapa_clases": MAPA_CLASES,
            "mapa_inverso": MAPA_INVERSO,
        },
        DIR_MOD / "modelo4_xgboost_clases.joblib",
    )

    resultados = []

    # =========================================================================
    # ESCENARIO A - XGBOOST BASE / DEFAULT
    # =========================================================================

    rep.titulo("ESCENARIO A - XGBOOST BASE", "#")
    rep.p(
        "Se conservan los hiperparametros del booster en sus valores base."
    )
    rep.p(
        "Solo se define explicitamente la tarea multiclase y la semilla."
    )

    modelo_default = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASES,
        random_state=SEMILLA,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    mostrar_parametros_relevantes(
        modelo_default,
        "PARAMETROS EFECTIVOS - XGBOOST BASE",
    )

    inicio = time.perf_counter()
    modelo_default.fit(X_train, y_train)
    tiempo_default = time.perf_counter() - inicio

    ruta_default = DIR_MOD / "modelo4_xgb_default.joblib"
    joblib.dump(modelo_default, ruta_default)

    r_default = evaluar_escenario(
        "XGBoost Default",
        modelo_default,
        X_train,
        y_train,
        X_val,
        y_val,
        tiempo_default,
    )
    resultados.append(r_default)

    pred_default = modelo_default.predict(X_val)
    graficar_matriz(
        y_val,
        pred_default,
        "XGBoost Default - Validacion",
        DIR_REP / "matriz_xgb_default.png",
    )

    # =========================================================================
    # ESCENARIO B - AJUSTE MANUAL
    # =========================================================================

    rep.titulo("ESCENARIO B - XGBOOST AJUSTE MANUAL", "#")
    rep.p(
        "Configuracion mas conservadora para reducir sobreajuste y permitir"
    )
    rep.p(
        "que varios arboles pequenos aprendan progresivamente."
    )

    params_manual = dict(
        objective="multi:softprob",
        num_class=N_CLASES,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        gamma=0.10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=SEMILLA,
        n_jobs=-1,
    )

    modelo_manual = XGBClassifier(**params_manual)

    mostrar_parametros_relevantes(
        modelo_manual,
        "PARAMETROS - XGBOOST AJUSTE MANUAL",
    )

    inicio = time.perf_counter()
    modelo_manual.fit(X_train, y_train)
    tiempo_manual = time.perf_counter() - inicio

    ruta_manual = DIR_MOD / "modelo4_xgb_manual.joblib"
    joblib.dump(modelo_manual, ruta_manual)

    r_manual = evaluar_escenario(
        "XGBoost Manual",
        modelo_manual,
        X_train,
        y_train,
        X_val,
        y_val,
        tiempo_manual,
    )
    resultados.append(r_manual)

    pred_manual = modelo_manual.predict(X_val)
    graficar_matriz(
        y_val,
        pred_manual,
        "XGBoost Ajuste Manual - Validacion",
        DIR_REP / "matriz_xgb_manual.png",
    )

    # =========================================================================
    # ESCENARIO C - RANDOMIZED SEARCH
    # =========================================================================

    rep.titulo("ESCENARIO C - XGBOOST RANDOMIZED SEARCH", "#")
    rep.p(
        "La busqueda usa SOLO TRAIN y validacion cruzada estratificada."
    )
    rep.p(
        f"Combinaciones aleatorias: {N_ITER_BUSQUEDA} | "
        f"folds: {CV_FOLDS} | "
        f"ajustes aproximados: {N_ITER_BUSQUEDA * CV_FOLDS}"
    )
    rep.p("Metrica de seleccion: F1 macro.")

    modelo_busqueda_base = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASES,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEMILLA,

        # RandomizedSearchCV manejara el paralelismo externamente.
        # Esto evita paralelismo anidado excesivo.
        n_jobs=1,
    )

    espacio_busqueda = {
        "n_estimators": [150, 250, 350, 500, 700],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 2, 3, 5, 8],
        "gamma": [0.0, 0.05, 0.10, 0.30, 0.50],
        "subsample": [0.70, 0.80, 0.90, 1.00],
        "colsample_bytree": [0.70, 0.80, 0.90, 1.00],
        "reg_alpha": [0.0, 0.05, 0.10, 0.50, 1.00],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEMILLA,
    )

    busqueda = RandomizedSearchCV(
        estimator=modelo_busqueda_base,
        param_distributions=espacio_busqueda,
        n_iter=N_ITER_BUSQUEDA,
        scoring="f1_macro",
        cv=cv,
        random_state=SEMILLA,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    inicio = time.perf_counter()
    busqueda.fit(X_train, y_train)
    tiempo_search = time.perf_counter() - inicio

    rep.titulo("MEJOR CONFIGURACION ENCONTRADA POR RANDOMIZED SEARCH")
    rep.p(f"  Mejor F1 macro promedio en CV: {busqueda.best_score_:.4f}")
    rep.p(f"  Mejor indice                  : {busqueda.best_index_}")

    for k, v in sorted(busqueda.best_params_.items()):
        rep.p(f"  {k:<20} = {v}")

    # Guardar todos los resultados de la busqueda para auditoria.
    df_busqueda = pd.DataFrame(busqueda.cv_results_)
    columnas_utiles = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "params",
    ]
    df_busqueda[columnas_utiles].sort_values(
        "rank_test_score"
    ).to_csv(
        DIR_REP / "xgboost_busqueda_resultados.csv",
        index=False,
    )

    modelo_search = busqueda.best_estimator_

    # Ahora medimos el mejor de la busqueda contra el mismo conjunto VAL
    # utilizado por los otros dos escenarios.
    ruta_search = DIR_MOD / "modelo4_xgb_search.joblib"
    joblib.dump(modelo_search, ruta_search)

    r_search = evaluar_escenario(
        "XGBoost RandomizedSearch",
        modelo_search,
        X_train,
        y_train,
        X_val,
        y_val,
        tiempo_search,
    )
    resultados.append(r_search)

    pred_search = modelo_search.predict(X_val)
    graficar_matriz(
        y_val,
        pred_search,
        "XGBoost RandomizedSearch - Validacion",
        DIR_REP / "matriz_xgb_search.png",
    )

    # =========================================================================
    # COMPARACION DE LOS TRES ESCENARIOS
    # =========================================================================

    rep.titulo("COMPARACION FINAL DE LOS 3 ESCENARIOS", "#")

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(
        "f1_macro_val",
        ascending=False,
    ).reset_index(drop=True)

    rep.p(
        df_resultados.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    ruta_csv = DIR_REP / "comparacion_xgboost.csv"
    df_resultados.to_csv(ruta_csv, index=False)

    ganador = df_resultados.iloc[0]["modelo"]
    mejor_f1 = df_resultados.iloc[0]["f1_macro_val"]

    modelos_entrenados = {
        "XGBoost Default": modelo_default,
        "XGBoost Manual": modelo_manual,
        "XGBoost RandomizedSearch": modelo_search,
    }

    mejor_modelo = modelos_entrenados[ganador]

    ruta_mejor = DIR_MOD / "modelo4_xgboost_mejor.joblib"
    joblib.dump(mejor_modelo, ruta_mejor)

    rep.p("")
    rep.p(f"GANADOR EN VALIDACION: {ganador}")
    rep.p(f"F1 macro validacion : {mejor_f1:.4f}")
    rep.p(f"Guardado             : {ruta_mejor.relative_to(BASE)}")

    rep.p("")
    rep.p("IMPORTANTE:")
    rep.p("  El conjunto TEST NO fue utilizado para escoger este ganador.")
    rep.p("  La evaluacion sobre TEST debe hacerse en la etapa final, despues")
    rep.p("  de comparar tambien Regresion Logistica, Arbol, Random Forest y MLP.")

    rep.titulo("FIN MODELO 4 - XGBOOST", "#")
    rep.p(f"Guardado: {ruta_default.relative_to(BASE)}")
    rep.p(f"Guardado: {ruta_manual.relative_to(BASE)}")
    rep.p(f"Guardado: {ruta_search.relative_to(BASE)}")
    rep.p(f"Guardado: {ruta_mejor.relative_to(BASE)}")
    rep.p(f"Guardado: {ruta_csv.relative_to(BASE)}")

    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
