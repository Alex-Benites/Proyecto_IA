"""
ETAPA 4 - MODELO 1: REGRESION LOGISTICA MULTINOMIAL
=====================================================

Por que este modelo primero:
  - Es el mas simple e interpretable: cada clase tiene un vector de pesos,
    uno por columna codificada. Se puede explicar coeficiente por coeficiente
    en la defensa (sin caja negra).
  - Sirve de LINEA BASE: cualquier modelo mas complejo (arbol, random forest)
    debe superarlo para justificar su complejidad extra.

Que es "no caja negra" aqui:
  - sklearn.LogisticRegression no oculta el algoritmo: es la formula estandar
    de regresion logistica multinomial (softmax) resuelta por optimizacion
    numerica (lbfgs). No hay seleccion automatica de features, no hay
    busqueda automatica de hiperparametros en este primer entrenamiento:
    se usan los valores por defecto documentados abajo y se explica cada uno.
  - No se uso class_weight='balanced' porque el dataset de entrada
    (dataset.xlsx) YA esta balanceado artificialmente (~25% cada clase).
    Aplicar class_weight encima seria corregir un desbalance que ya no existe.

Datos: usa la salida de 03_preparacion.py (train/val ya separados,
ya imputados/codificados/escalados SOLO con estadisticas de train).
El set de test queda sellado (no se toca en esta etapa).

Metricas: accuracy, precision/recall/f1 (macro, porque nos interesa que
el modelo funcione igual de bien en las 4 clases, no solo en la mayoritaria),
matriz de confusion, reporte por clase, y comparacion train vs val para
detectar overfitting/underfitting.

Salida:
  modelos/modelo1_logistica.joblib
  resultados/modelos/informe_modelo1_logistica.txt
  resultados/modelos/matriz_confusion_modelo1.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"
DIR_REP.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]


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


rep = Reporte(DIR_REP / "informe_modelo1_logistica.txt")


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    rep.p(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  F1 (macro)       : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def graficar_matriz(y_true, y_pred, ruta):
    cm = confusion_matrix(y_true, y_pred, labels=CLASES)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=45, ha="right")
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion - Modelo 1 (Regresion Logistica) - Validacion")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    rep.titulo("ETAPA 4 - MODELO 1: REGRESION LOGISTICA MULTINOMIAL", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | semilla {SEMILLA}")

    X_train = np.load(DIR_PROC / "train_X.npy")
    y_train = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_val = np.load(DIR_PROC / "val_X.npy")
    y_val = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)

    rep.p(f"\nTrain: {X_train.shape}  |  Val: {X_val.shape}")

    rep.titulo("HIPERPARAMETROS (todos explicitos, sin busqueda automatica)")
    params = dict(
        max_iter=2000,       # suficientes iteraciones para asegurar convergencia real
        random_state=SEMILLA,
    )
    for k, v in params.items():
        rep.p(f"  {k} = {v}")
    rep.p("  solver = lbfgs (por defecto). Maneja softmax multinomial de forma nativa.")
    rep.p("  class_weight = None (dataset ya balanceado artificialmente, ver docstring).")

    modelo = LogisticRegression(**params)
    modelo.fit(X_train, y_train)

    if modelo.n_iter_[0] >= params["max_iter"]:
        rep.p(f"\n  !! ATENCION: no convergio en {params['max_iter']} iteraciones (n_iter_={modelo.n_iter_}).")
    else:
        rep.p(f"\n  Convergio en {modelo.n_iter_[0]} iteraciones (de un maximo de {params['max_iter']}).")

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    rep.titulo("METRICAS")
    m_train = metricas(y_train, pred_train, "TRAIN")
    m_val = metricas(y_val, pred_val, "VALIDACION")

    rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING")
    diff_f1 = m_train["f1_macro"] - m_val["f1_macro"]
    rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia     : {diff_f1:.4f}")
    if diff_f1 > 0.08:
        rep.p("  -> Posible OVERFITTING (el modelo memoriza train y no generaliza igual a val).")
    elif m_train["f1_macro"] < 0.5:
        rep.p("  -> Posible UNDERFITTING (ni siquiera en train el modelo aprende bien).")
    else:
        rep.p("  -> Sin señales fuertes de over/underfitting: train y val estan cerca.")

    rep.titulo("REPORTE POR CLASE (VALIDACION)")
    rep.p(classification_report(y_val, pred_val, labels=CLASES, zero_division=0))

    rep.titulo("MATRIZ DE CONFUSION (VALIDACION)")
    cm = confusion_matrix(y_val, pred_val, labels=CLASES)
    cm_df = pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES], columns=[f"pred_{c}" for c in CLASES])
    rep.p(cm_df.to_string())

    ruta_png = DIR_REP / "matriz_confusion_modelo1.png"
    graficar_matriz(y_val, pred_val, ruta_png)
    rep.p(f"\nGuardado grafico: {ruta_png.relative_to(BASE)}")

    joblib.dump(modelo, DIR_MOD / "modelo1_logistica.joblib")

    rep.titulo("FIN MODELO 1", "#")
    rep.p("Guardado: modelos/modelo1_logistica.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
