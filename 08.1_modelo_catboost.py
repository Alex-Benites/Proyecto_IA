"""
CATBOOST - 3 ESCENARIOS
=======================
1) Default
2) Ajuste manual
3) RandomizedSearchCV

Usa train_raw.csv y val_raw.csv generados por 03_preparacion.py.
CatBoost recibe las categoricas de forma nativa.
TEST no se utiliza aqui.
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

from catboost import CatBoostClassifier
import catboost

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


BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"

DIR_MOD.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)

TARGET = "y"
CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Todo excepto estas columnas se trata como categorico.
COLS_NUMERICAS = ["edad", "es_fin_semana"]

# Busqueda inicial razonable: 20 x 3 = 60 fits.
# Si luego quieres una busqueda mas exhaustiva:
# N_ITER_BUSQUEDA = 25
# CV_FOLDS = 5
N_ITER_BUSQUEDA = 20
CV_FOLDS = 3


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


rep = Reporte(DIR_REP / "informe_catboost_3_escenarios.txt")


def cargar_datos():
    ruta_train = DIR_PROC / "train_raw.csv"
    ruta_val = DIR_PROC / "val_raw.csv"

    faltantes = [p for p in [ruta_train, ruta_val] if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos de Etapa 3:\n" +
            "\n".join(str(x) for x in faltantes)
        )

    train = pd.read_csv(ruta_train)
    val = pd.read_csv(ruta_val)

    if TARGET not in train.columns or TARGET not in val.columns:
        raise ValueError("Los raw deben contener la columna 'y'.")

    X_train = train.drop(columns=[TARGET]).copy()
    y_train = train[TARGET].astype(str)
    X_val = val.drop(columns=[TARGET]).copy()
    y_val = val[TARGET].astype(str)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Las features de train y val no coinciden.")

    desconocidas = sorted((set(y_train) | set(y_val)) - set(CLASES))
    if desconocidas:
        raise ValueError(f"Clases inesperadas: {desconocidas}")

    return X_train, y_train, X_val, y_val


def preparar_catboost(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    numericas = [c for c in COLS_NUMERICAS if c in X_train.columns]
    categoricas = [c for c in X_train.columns if c not in numericas]

    rep.p(f"Columnas categoricas: {len(categoricas)}")
    rep.p(f"Columnas numericas: {numericas}")

    # Numericas: conversion + mediana SOLO de train.
    for c in numericas:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")

        n_train = int(X_train[c].isna().sum())
        n_val = int(X_val[c].isna().sum())

        mediana = X_train[c].median()
        if pd.isna(mediana):
            mediana = 0.0

        X_train[c] = X_train[c].fillna(mediana)
        X_val[c] = X_val[c].fillna(mediana)

        if n_train or n_val:
            rep.p(
                f"{c}: {n_train} NaN train, {n_val} NaN val "
                f"-> mediana train={mediana}"
            )

    # Categoricas: CatBoost requiere valores validos, no NaN float.
    for c in categoricas:
        X_train[c] = X_train[c].fillna("DESCONOCIDO").astype(str)
        X_val[c] = X_val[c].fillna("DESCONOCIDO").astype(str)

    return X_train, X_val, categoricas


def calcular_metricas(modelo, X, y, nombre):
    pred = np.asarray(modelo.predict(X)).reshape(-1).astype(str)
    proba = modelo.predict_proba(X)

    acc = accuracy_score(y, pred)
    prec = precision_score(
        y, pred, labels=CLASES, average="macro", zero_division=0
    )
    rec = recall_score(
        y, pred, labels=CLASES, average="macro", zero_division=0
    )
    f1 = f1_score(
        y, pred, labels=CLASES, average="macro", zero_division=0
    )
    loss = log_loss(y, proba, labels=list(modelo.classes_))

    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {acc:.4f}")
    rep.p(f"  Precision (macro): {prec:.4f}")
    rep.p(f"  Recall (macro)   : {rec:.4f}")
    rep.p(f"  F1 (macro)       : {f1:.4f}")
    rep.p(f"  Log Loss         : {loss:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_loss": loss,
        "pred": pred,
    }


def matriz_confusion(y, pred, titulo, ruta):
    cm = confusion_matrix(y, pred, labels=CLASES)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=45, ha="right")
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    limite = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > limite else "black"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def evaluar(nombre, modelo, X_train, y_train, X_val, y_val, tiempo, ruta_png):
    rep.titulo(f"RESULTADOS - {nombre}")

    train = calcular_metricas(modelo, X_train, y_train, "TRAIN")
    val = calcular_metricas(modelo, X_val, y_val, "VALIDACION")

    gap = train["f1"] - val["f1"]

    rep.p(f"\n  Brecha F1 train-val: {gap:.4f}")
    rep.p(f"  Tiempo: {tiempo:.2f} s")

    if gap > 0.08:
        rep.p("  -> Posible OVERFITTING.")
    elif train["f1"] < 0.50:
        rep.p("  -> Posible UNDERFITTING.")
    else:
        rep.p("  -> Sin senal fuerte de over/underfitting.")

    rep.p("\nReporte por clase - VALIDACION:")
    rep.p(
        classification_report(
            y_val, val["pred"], labels=CLASES, zero_division=0
        )
    )

    matriz_confusion(y_val, val["pred"], f"{nombre} - Validacion", ruta_png)

    return {
        "modelo": nombre,
        "accuracy_train": train["accuracy"],
        "f1_train": train["f1"],
        "accuracy_val": val["accuracy"],
        "precision_macro_val": val["precision"],
        "recall_macro_val": val["recall"],
        "f1_macro_val": val["f1"],
        "log_loss_val": val["log_loss"],
        "gap_f1_train_val": gap,
        "tiempo_segundos": tiempo,
    }


def imprimir_params_efectivos(modelo, titulo):
    rep.titulo(titulo)

    try:
        params = modelo.get_all_params()
    except Exception:
        params = modelo.get_params()

    claves = [
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bootstrap_type",
        "bagging_temperature",
        "border_count",
        "loss_function",
        "grow_policy",
        "random_seed",
    ]

    for k in claves:
        if k in params:
            rep.p(f"  {k:<20} = {params[k]}")


def importancia(modelo, columnas):
    valores = np.asarray(modelo.get_feature_importance())

    if len(valores) != len(columnas):
        return

    df = pd.DataFrame({
        "feature": columnas,
        "importancia": valores,
    }).sort_values("importancia", ascending=False)

    df.to_csv(DIR_REP / "importancia_catboost_mejor.csv", index=False)

    rep.titulo("IMPORTANCIA DE VARIABLES - GANADOR")
    rep.p(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    top = df.head(20).sort_values("importancia")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importancia"])
    ax.set_xlabel("Importancia")
    ax.set_title("CatBoost - Importancia de variables")
    fig.tight_layout()
    fig.savefig(DIR_REP / "importancia_catboost_mejor.png", dpi=150)
    plt.close(fig)


def main():
    rep.titulo("CATBOOST - 3 ESCENARIOS", "#")
    rep.p(
        f"Python {sys.version.split()[0]} | "
        f"scikit-learn {sklearn.__version__} | "
        f"catboost {catboost.__version__} | semilla {SEMILLA}"
    )

    X_train, y_train, X_val, y_val = cargar_datos()

    rep.titulo("CARGA DE DATOS")
    rep.p("Fuente: train_raw.csv / val_raw.csv")
    rep.p(f"Train: {X_train.shape} | Val: {X_val.shape}")

    X_train, X_val, cat_cols = preparar_catboost(X_train, X_val)

    rep.p("No se escala.")
    rep.p("Las categoricas se entregan a CatBoost como texto.")
    rep.p("TEST NO SE CARGA.")

    resultados = []

    # ============================================================
    # ESCENARIO 1: DEFAULT
    # ============================================================
    rep.titulo("ESCENARIO 1 - CATBOOST DEFAULT", "#")

    default = CatBoostClassifier(
        loss_function="MultiClass",
        cat_features=cat_cols,
        random_seed=SEMILLA,
        verbose=100,
        thread_count=-1,
        allow_writing_files=False,
    )

    t0 = time.perf_counter()
    default.fit(X_train, y_train)
    t_default = time.perf_counter() - t0

    imprimir_params_efectivos(default, "PARAMETROS EFECTIVOS - DEFAULT")
    joblib.dump(default, DIR_MOD / "catboost_01_default.joblib")

    resultados.append(
        evaluar(
            "CatBoost Default",
            default,
            X_train, y_train,
            X_val, y_val,
            t_default,
            DIR_REP / "matriz_catboost_default.png",
        )
    )

    # ============================================================
    # ESCENARIO 2: AJUSTE MANUAL
    # ============================================================
    rep.titulo("ESCENARIO 2 - CATBOOST AJUSTE MANUAL", "#")

    params_manual = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.5,
        "border_count": 128,
        "loss_function": "MultiClass",
        "cat_features": cat_cols,
        "random_seed": SEMILLA,
        "verbose": False,
        "allow_writing_files": False,
    }

    manual = CatBoostClassifier(**params_manual)

    rep.titulo("PARAMETROS MANUALES")
    for k, v in params_manual.items():
        if k not in ["cat_features", "verbose", "allow_writing_files"]:
            rep.p(f"  {k:<20} = {v}")

    t0 = time.perf_counter()
    manual.fit(X_train, y_train)
    t_manual = time.perf_counter() - t0

    joblib.dump(manual, DIR_MOD / "catboost_02_manual.joblib")

    resultados.append(
        evaluar(
            "CatBoost Manual",
            manual,
            X_train, y_train,
            X_val, y_val,
            t_manual,
            DIR_REP / "matriz_catboost_manual.png",
        )
    )

    # ============================================================
    # ESCENARIO 3: RANDOMIZED SEARCH
    # ============================================================
    rep.titulo("ESCENARIO 3 - CATBOOST RANDOMIZED SEARCH", "#")
    rep.p(
        f"{N_ITER_BUSQUEDA} configuraciones x {CV_FOLDS} folds "
        f"= aprox. {N_ITER_BUSQUEDA * CV_FOLDS} fits"
    )
    rep.p("La busqueda usa SOLO TRAIN.")
    rep.p("Scoring: f1_macro.")

    base_search = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=SEMILLA,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    espacio = {
        "iterations": [300, 500, 700, 900, 1200],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
        "depth": [4, 5, 6, 7, 8],
        "l2_leaf_reg": [3.0, 5.0, 7.0, 10.0, 15.0],
        "random_strength": [0.5, 1.0, 1.5, 2.0, 3.0],
        "bagging_temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
        "border_count": [32, 64, 128, 254],
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEMILLA,
    )

    search = RandomizedSearchCV(
        estimator=base_search,
        param_distributions=espacio,
        n_iter=N_ITER_BUSQUEDA,
        scoring="f1_macro",
        cv=cv,
        random_state=SEMILLA,
        n_jobs=1,

        verbose=2,
        refit=True,
        return_train_score=True,
    )

    t0 = time.perf_counter()
    search.fit(
        X_train,
        y_train,
        cat_features=cat_cols
    )
    t_search = time.perf_counter() - t0

    rep.titulo("MEJORES HIPERPARAMETROS")
    rep.p(f"Mejor F1 macro promedio CV: {search.best_score_:.4f}")
    for k, v in sorted(search.best_params_.items()):
        rep.p(f"  {k:<20} = {v}")

    cv_df = pd.DataFrame(search.cv_results_)
    cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "params",
    ]
    cv_df[cols].sort_values("rank_test_score").to_csv(
        DIR_REP / "catboost_random_search_resultados.csv",
        index=False,
    )

    best_search = search.best_estimator_
    joblib.dump(best_search, DIR_MOD / "catboost_03_search.joblib")

    resultados.append(
        evaluar(
            "CatBoost RandomizedSearch",
            best_search,
            X_train, y_train,
            X_val, y_val,
            t_search,
            DIR_REP / "matriz_catboost_search.png",
        )
    )

    # ============================================================
    # COMPARACION
    # ============================================================
    rep.titulo("COMPARACION FINAL - CATBOOST", "#")

    tabla = pd.DataFrame(resultados).sort_values(
        "f1_macro_val",
        ascending=False,
    ).reset_index(drop=True)

    rep.p(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tabla.to_csv(
        DIR_REP / "comparacion_catboost_3_escenarios.csv",
        index=False,
    )

    ganador = tabla.iloc[0]["modelo"]
    f1_ganador = tabla.iloc[0]["f1_macro_val"]

    modelos = {
        "CatBoost Default": default,
        "CatBoost Manual": manual,
        "CatBoost RandomizedSearch": best_search,
    }

    mejor = modelos[ganador]
    joblib.dump(mejor, DIR_MOD / "catboost_mejor.joblib")
    mejor.save_model(str(DIR_MOD / "catboost_mejor.cbm"))

    rep.p("")
    rep.p(f"GANADOR EN VALIDACION: {ganador}")
    rep.p(f"F1 macro validacion: {f1_ganador:.4f}")

    importancia(mejor, list(X_train.columns))

    rep.titulo("FIN", "#")
    rep.p("TEST permanecio sellado.")
    rep.p("Guardado: modelos/catboost_01_default.joblib")
    rep.p("Guardado: modelos/catboost_02_manual.joblib")
    rep.p("Guardado: modelos/catboost_03_search.joblib")
    rep.p("Guardado: modelos/catboost_mejor.joblib")
    rep.p("Guardado: modelos/catboost_mejor.cbm")
    rep.p("Guardado: resultados/modelos/comparacion_catboost_3_escenarios.csv")
    rep.guardar()

    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
