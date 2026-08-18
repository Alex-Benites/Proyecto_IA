"""
MODELO MLP - 3 ESCENARIOS
=========================
1) MLP Default
2) MLP Ajuste Manual
3) MLP RandomizedSearchCV

Usa train_X.npy / val_X.npy de la Etapa 3.
TEST no se carga ni se utiliza.
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier


BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"

DIR_MOD.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# 20 x 3 = 60 fits.
# Para prueba rapida: 5 x 3.
# Para busqueda mas fuerte: 25 x 5.
N_ITER_BUSQUEDA = 20
CV_FOLDS = 3


class Reporte:
    def __init__(self, ruta):
        self.ruta = ruta
        self.lineas = []

    def p(self, *args):
        txt = " ".join(str(x) for x in args)
        print(txt)
        self.lineas.append(txt)

    def titulo(self, txt, char="="):
        self.p("")
        self.p(char * 78)
        self.p(txt)
        self.p(char * 78)

    def guardar(self):
        self.ruta.write_text("\n".join(self.lineas), encoding="utf-8")


rep = Reporte(DIR_REP / "informe_mlp_3_escenarios.txt")


def cargar_datos():
    rutas = {
        "X_train": DIR_PROC / "train_X.npy",
        "y_train": DIR_PROC / "train_y.npy",
        "X_val": DIR_PROC / "val_X.npy",
        "y_val": DIR_PROC / "val_y.npy",
    }

    faltantes = [str(p) for p in rutas.values() if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos de la Etapa 3:\n" + "\n".join(faltantes)
        )

    X_train = np.load(rutas["X_train"], allow_pickle=False)
    X_val = np.load(rutas["X_val"], allow_pickle=False)
    y_train = np.load(rutas["y_train"], allow_pickle=True).astype(str)
    y_val = np.load(rutas["y_val"], allow_pickle=True).astype(str)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train y X_val deben ser matrices 2D.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train y validacion tienen distinto numero de features.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("X e y tienen distinta cantidad de filas.")

    desconocidas = sorted((set(y_train) | set(y_val)) - set(CLASES))
    if desconocidas:
        raise ValueError(f"Clases inesperadas: {desconocidas}")

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Los arrays X contienen NaN o infinitos.")

    return (
        X_train.astype(np.float64, copy=False),
        y_train,
        X_val.astype(np.float64, copy=False),
        y_val,
    )


def metricas(modelo, X, y_true, nombre):
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)

    m = {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(
            y_true, pred, labels=CLASES, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, pred, labels=CLASES, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, pred, labels=CLASES, average="macro", zero_division=0
        ),
        "log_loss": log_loss(y_true, proba, labels=list(modelo.classes_)),
        "pred": pred,
    }

    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {m['accuracy']:.4f}")
    rep.p(f"  Precision (macro): {m['precision_macro']:.4f}")
    rep.p(f"  Recall (macro)   : {m['recall_macro']:.4f}")
    rep.p(f"  F1 (macro)       : {m['f1_macro']:.4f}")
    rep.p(f"  Log Loss         : {m['log_loss']:.4f}")

    return m


def guardar_matriz(y_true, pred, titulo, ruta):
    cm = confusion_matrix(y_true, pred, labels=CLASES)

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
                color="white" if cm[i, j] > limite else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_curva(modelo, titulo, ruta):
    curva = getattr(modelo, "loss_curve_", None)
    if curva is None or len(curva) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(curva) + 1), curva)
    ax.set_xlabel("Iteracion / epoca")
    ax.set_ylabel("Loss de entrenamiento")
    ax.set_title(titulo)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def imprimir_parametros(modelo, titulo):
    rep.titulo(titulo)
    p = modelo.get_params()
    claves = [
        "hidden_layer_sizes", "activation", "solver", "alpha",
        "batch_size", "learning_rate", "learning_rate_init",
        "max_iter", "shuffle", "tol", "early_stopping",
        "validation_fraction", "n_iter_no_change", "random_state"
    ]
    for k in claves:
        rep.p(f"  {k:<22} = {p.get(k)}")


def entrenar(modelo, X, y, nombre):
    rep.p(f"\nIniciando entrenamiento: {nombre}")
    t0 = time.perf_counter()

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(X, y)

    tiempo = time.perf_counter() - t0

    conv = [
        w for w in avisos
        if issubclass(w.category, ConvergenceWarning)
    ]

    if conv:
        rep.p("\nADVERTENCIA DE CONVERGENCIA:")
        for w in conv:
            rep.p(f"  {w.message}")
    else:
        rep.p("\nNo se detectaron advertencias de convergencia.")

    return tiempo


def evaluar(
    nombre, modelo,
    X_train, y_train,
    X_val, y_val,
    tiempo,
    ruta_matriz,
    ruta_curva,
):
    rep.titulo(f"RESULTADOS - {nombre}")

    train = metricas(modelo, X_train, y_train, "TRAIN")
    val = metricas(modelo, X_val, y_val, "VALIDACION")

    gap = train["f1_macro"] - val["f1_macro"]

    rep.p("\n  Chequeo train vs validacion:")
    rep.p(f"  F1 train         : {train['f1_macro']:.4f}")
    rep.p(f"  F1 validacion    : {val['f1_macro']:.4f}")
    rep.p(f"  Diferencia       : {gap:.4f}")
    rep.p(f"  Epocas ejecutadas: {getattr(modelo, 'n_iter_', 'N/D')}")
    rep.p(f"  Tiempo           : {tiempo:.2f} segundos")

    if gap > 0.08:
        rep.p("  -> Posible OVERFITTING.")
    elif train["f1_macro"] < 0.50:
        rep.p("  -> Posible UNDERFITTING.")
    else:
        rep.p("  -> Sin una senal fuerte de over/underfitting.")

    if getattr(modelo, "early_stopping", False):
        score_int = getattr(modelo, "best_validation_score_", None)
        if score_int is not None:
            rep.p(
                f"  Mejor accuracy de validacion INTERNA: {score_int:.4f}"
            )
            rep.p("  Esa validacion interna sale SOLO de TRAIN.")

    rep.p("\n  Reporte por clase - VALIDACION:")
    rep.p(
        classification_report(
            y_val, val["pred"], labels=CLASES, zero_division=0
        )
    )

    guardar_matriz(y_val, val["pred"], f"{nombre} - Validacion", ruta_matriz)
    guardar_curva(modelo, f"{nombre} - Curva de perdida", ruta_curva)

    return {
        "modelo": nombre,
        "accuracy_train": train["accuracy"],
        "f1_train": train["f1_macro"],
        "accuracy_val": val["accuracy"],
        "precision_macro_val": val["precision_macro"],
        "recall_macro_val": val["recall_macro"],
        "f1_macro_val": val["f1_macro"],
        "log_loss_val": val["log_loss"],
        "gap_f1_train_val": gap,
        "epocas": getattr(modelo, "n_iter_", np.nan),
        "tiempo_segundos": tiempo,
    }


def main():
    rep.titulo("MODELO MLP - TRES ESCENARIOS", "#")
    rep.p(
        f"Python {sys.version.split()[0]} | "
        f"scikit-learn {sklearn.__version__} | semilla {SEMILLA}"
    )

    X_train, y_train, X_val, y_val = cargar_datos()

    rep.titulo("CARGA DE DATOS")
    rep.p(f"Train: X={X_train.shape} | y={y_train.shape}")
    rep.p(f"Val  : X={X_val.shape} | y={y_val.shape}")
    rep.p("Fuente: train_X.npy / val_X.npy")
    rep.p("Los datos ya vienen One-Hot encoded y escalados desde Etapa 3.")
    rep.p("No se vuelve a ajustar ningun preprocesamiento.")
    rep.p("TEST NO SE CARGA.")

    rep.p("\nDistribucion TRAIN:")
    for clase in CLASES:
        n = int(np.sum(y_train == clase))
        rep.p(f"  {clase:<18}: {n:6d} ({100*n/len(y_train):5.2f}%)")

    resultados = []

    # =====================================================================
    # ESCENARIO 1: DEFAULT
    # =====================================================================
    rep.titulo("ESCENARIO 1 - MLP DEFAULT", "#")
    rep.p(
        "Se mantienen los hiperparametros por defecto de MLPClassifier."
    )
    rep.p(
        "Solo se fija random_state=42 y verbose=True para reproducibilidad "
        "y progreso visible."
    )

    default = MLPClassifier(
        random_state=SEMILLA,
        verbose=True,
    )

    imprimir_parametros(default, "PARAMETROS EFECTIVOS - MLP DEFAULT")
    t_default = entrenar(default, X_train, y_train, "MLP Default")

    joblib.dump(default, DIR_MOD / "mlp_01_default.joblib")

    resultados.append(
        evaluar(
            "MLP Default", default,
            X_train, y_train, X_val, y_val, t_default,
            DIR_REP / "matriz_mlp_default.png",
            DIR_REP / "curva_mlp_default.png",
        )
    )

    # =====================================================================
    # ESCENARIO 2: MANUAL
    # =====================================================================
    rep.titulo("ESCENARIO 2 - MLP AJUSTE MANUAL", "#")
    rep.p(
        "Dos capas ocultas + ReLU + Adam + regularizacion L2 + "
        "early stopping interno."
    )
    rep.p(
        "El early stopping usa 15% de TRAIN; VAL externo no interviene."
    )

    manual = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=128,
        learning_rate_init=0.0005,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-4,
        random_state=SEMILLA,
        verbose=True,
    )

    imprimir_parametros(manual, "PARAMETROS - MLP AJUSTE MANUAL")
    t_manual = entrenar(manual, X_train, y_train, "MLP Manual")

    joblib.dump(manual, DIR_MOD / "mlp_02_manual.joblib")

    resultados.append(
        evaluar(
            "MLP Manual", manual,
            X_train, y_train, X_val, y_val, t_manual,
            DIR_REP / "matriz_mlp_manual.png",
            DIR_REP / "curva_mlp_manual.png",
        )
    )

    # =====================================================================
    # ESCENARIO 3: RANDOMIZED SEARCH
    # =====================================================================
    rep.titulo("ESCENARIO 3 - MLP RANDOMIZED SEARCH", "#")
    rep.p(
        f"{N_ITER_BUSQUEDA} configuraciones x {CV_FOLDS} folds "
        f"= aprox. {N_ITER_BUSQUEDA * CV_FOLDS} fits"
    )
    rep.p("La busqueda usa SOLO TRAIN.")
    rep.p("Scoring: f1_macro.")
    rep.p(
        "Cada candidato usa early stopping dentro de su fold de entrenamiento."
    )

    base_search = MLPClassifier(
        solver="adam",
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=500,
        random_state=SEMILLA,
        verbose=False,
    )

    espacio = {
        "hidden_layer_sizes": [
            (64,), (100,), (128,), (128, 64),
            (256, 128), (128, 64, 32)
        ],
        "activation": ["relu", "tanh"],
        "alpha": [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        "batch_size": [32, 64, 128, 256],
        "learning_rate_init": [0.0003, 0.0005, 0.001, 0.003],
        "n_iter_no_change": [10, 15, 20, 30],
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
        n_jobs=1,          # mas seguro en Windows / RAM
        verbose=2,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    rep.p("\nComienza RandomizedSearchCV...")
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    t_search = time.perf_counter() - t0

    rep.titulo("MEJOR CONFIGURACION ENCONTRADA")
    rep.p(f"Mejor F1 macro promedio CV: {search.best_score_:.4f}")
    rep.p(f"Mejor indice              : {search.best_index_}")

    for k, v in sorted(search.best_params_.items()):
        rep.p(f"  {k:<22} = {v}")

    df_cv = pd.DataFrame(search.cv_results_)
    columnas = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "params",
    ]

    df_cv[columnas].sort_values("rank_test_score").to_csv(
        DIR_REP / "mlp_random_search_resultados.csv",
        index=False,
    )

    best_search = search.best_estimator_
    joblib.dump(best_search, DIR_MOD / "mlp_03_search.joblib")

    resultados.append(
        evaluar(
            "MLP RandomizedSearch", best_search,
            X_train, y_train, X_val, y_val, t_search,
            DIR_REP / "matriz_mlp_search.png",
            DIR_REP / "curva_mlp_search.png",
        )
    )

    # =====================================================================
    # COMPARACION
    # =====================================================================
    rep.titulo("COMPARACION FINAL - 3 ESCENARIOS MLP", "#")

    tabla = pd.DataFrame(resultados).sort_values(
        "f1_macro_val", ascending=False
    ).reset_index(drop=True)

    rep.p(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    tabla.to_csv(
        DIR_REP / "comparacion_mlp_3_escenarios.csv",
        index=False,
    )

    ganador = tabla.iloc[0]["modelo"]
    mejor_f1 = float(tabla.iloc[0]["f1_macro_val"])

    modelos = {
        "MLP Default": default,
        "MLP Manual": manual,
        "MLP RandomizedSearch": best_search,
    }

    mejor = modelos[ganador]
    joblib.dump(mejor, DIR_MOD / "mlp_mejor.joblib")

    rep.p("")
    rep.p(f"GANADOR EN VALIDACION: {ganador}")
    rep.p(f"F1 macro validacion : {mejor_f1:.4f}")
    rep.p("Guardado             : modelos/mlp_mejor.joblib")

    rep.p("\nIMPORTANTE:")
    rep.p("  TEST NO fue utilizado para escoger este ganador.")
    rep.p(
        "  La comparacion definitiva con los otros algoritmos debe "
        "mantener TEST sellado hasta la etapa final."
    )

    rep.titulo("FIN - MODELO MLP", "#")
    rep.p("Guardado: modelos/mlp_01_default.joblib")
    rep.p("Guardado: modelos/mlp_02_manual.joblib")
    rep.p("Guardado: modelos/mlp_03_search.joblib")
    rep.p("Guardado: modelos/mlp_mejor.joblib")
    rep.p(
        "Guardado: resultados/modelos/comparacion_mlp_3_escenarios.csv"
    )

    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
