"""
ENTRENAMIENTO V2 - RANDOM FOREST: VARIANTE A (class_weight) vs VARIANTE B (recorte)
=====================================================================================

Entrena el MISMO algoritmo, con los MISMOS hiperparametros fijos, sobre las
dos variantes preparadas por 01_preparacion_v2.py, para aislar el efecto de
la TECNICA de manejo del desbalance (no del algoritmo, que es el mismo).

  VARIANTE A: 30.782 filas de train (100% reales), class_weight='balanced'.
              RandomForestClassifier soporta esto nativamente: cada clase
              recibe un peso = n_total / (n_clases * n_clase), asi que
              ARMA_CONTUNDENTE (826 casos) pesa ~9.3x mas que ARMA_FUEGO
              (24.433 casos) en la funcion de perdida de cada arbol. NINGUN
              caso se descarta ni se fabrica.

  VARIANTE B: 10.409 filas de train (100% reales), ARMA_FUEGO recortado a
              una muestra de 5.800 (de 34.905 disponibles). Sin class_weight:
              se aisla el efecto de "menos datos de la mayoria" solo.

Ambas variantes tienen su propio val/test (misma proporcion real dentro de
cada una, ya que ambas se separaron con el mismo criterio estratificado).

Se reutiliza el mismo experimento de profundidad y la misma regla de
parsimonia que en V1 (07_modelo_randomforest.py), porque el tamanio del
dataset cambio bastante (30.782 y 10.409 filas, contra 14.630 en V1) y la
profundidad optima podria no ser la misma.
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_DATA = BASE / "data"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados"
DIR_FIG = BASE / "Graficos"
DIR_MOD.mkdir(exist_ok=True)
DIR_FIG.mkdir(exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

COLS_CATEGORICAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "dia_semana",
    "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad"]
COL_BINARIA = ["es_fin_semana"]

PROFUNDIDADES = [4, 6, 8, 10, 12, 15, 20, 25, 30, None]

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
TINTA, TINTA_SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SUPERFICIE = "#e1e0d9", "#fcfcfb"


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


rep = Reporte(DIR_REP / "informe_modelo_randomforest_v2.txt")


def cargar_variante(nombre):
    """Lee las matrices ya preprocesadas por 01_preparacion_v2.py.

    El preprocesador (imputar edad + escalar + one-hot) se ajusta UNA vez
    alli, solo con el train de cada variante. Ningun script de modelo lo
    reconstruye por su cuenta: asi todos los algoritmos comparan sobre
    exactamente la misma matriz.
    """
    d = {}
    for parte in ("train", "val"):
        d[f"X_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_X.npy")
        d[f"y_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_y.npy", allow_pickle=True)
    d["columnas"] = np.load(DIR_DATA / f"{nombre}_columnas.npy", allow_pickle=True)
    d["preprocesador"] = joblib.load(DIR_MOD / f"preprocesador_{nombre}.joblib")
    return d


def experimento_profundidad(X_tr, y_tr, X_va, y_va, class_weight, etiqueta):
    rep.p(f"\n  Barrido de max_depth para {etiqueta} "
          f"(class_weight={class_weight}):")
    rep.p(f"  {'max_depth':>10} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 45)
    filas = []
    for d in PROFUNDIDADES:
        m = RandomForestClassifier(n_estimators=200, max_depth=d, min_samples_leaf=2,
                                   max_features="sqrt", class_weight=class_weight,
                                   random_state=SEMILLA, n_jobs=-1)
        m.fit(X_tr, y_tr)
        f1_tr = f1_score(y_tr, m.predict(X_tr), average="macro", zero_division=0)
        f1_va = f1_score(y_va, m.predict(X_va), average="macro", zero_division=0)
        et = "sin limite" if d is None else str(d)
        rep.p(f"  {et:>10} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
        filas.append({"depth": d, "etiqueta": et, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    i_elegido = df.index[df["f1_val"] >= umbral].tolist()[0]
    rep.p(f"  ELEGIDO: max_depth={df.loc[i_elegido,'etiqueta']} -> "
          f"F1 val {df.loc[i_elegido,'f1_val']:.4f}")
    return PROFUNDIDADES[i_elegido]


def entrenar_variante(nombre_variante, etiqueta, class_weight, color):
    rep.titulo(f"{etiqueta} (class_weight={class_weight})")
    d = cargar_variante(nombre_variante)
    X_tr, X_va = d["X_train"], d["X_val"]
    y_tr, y_va = d["y_train"], d["y_val"]
    pre = d["preprocesador"]
    rep.p(f"  train={X_tr.shape}  val={X_va.shape}")

    depth = experimento_profundidad(X_tr, y_tr, X_va, y_va, class_weight, etiqueta)

    modelo = RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=2,
                                    max_features="sqrt", class_weight=class_weight,
                                    random_state=SEMILLA, n_jobs=-1)
    modelo.fit(X_tr, y_tr)
    pred_tr, pred_va = modelo.predict(X_tr), modelo.predict(X_va)

    m = {
        "accuracy_val": accuracy_score(y_va, pred_va),
        "f1_val": f1_score(y_va, pred_va, average="macro", zero_division=0),
        "f1_train": f1_score(y_tr, pred_tr, average="macro", zero_division=0),
        "depth": depth,
    }
    rep.p(f"\n  MODELO FINAL (max_depth={depth}, n_estimators=300):")
    rep.p(f"    F1 macro train: {m['f1_train']:.4f}")
    rep.p(f"    F1 macro val  : {m['f1_val']:.4f}")
    rep.p(f"    Brecha        : {m['f1_train']-m['f1_val']:.4f}")
    rep.p(f"\n  Reporte por clase (val):")
    rep.p(classification_report(y_va, pred_va, labels=CLASES, zero_division=0))

    cm = confusion_matrix(y_va, pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    joblib.dump({"modelo": modelo, "preprocesador": pre},
               DIR_MOD / f"randomforest_{nombre_variante}.joblib")
    return m, y_va, pred_va


def graficar_comparacion(m_a, m_b):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(2)
    ancho = 0.32
    tr = [m_a["f1_train"], m_b["f1_train"]]
    va = [m_a["f1_val"], m_b["f1_val"]]
    b1 = ax.bar(x - ancho/2, tr, ancho, label="Train", color=COLOR_TRAIN, zorder=3)
    b2 = ax.bar(x + ancho/2, va, ancho, label="Validacion", color=COLOR_VAL, zorder=3)
    for barras in (b1, b2):
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
    for xi, (t, v) in enumerate(zip(tr, va)):
        ax.text(xi - ancho/2, t + 0.015, f"{t:.3f}", ha="center", fontsize=9, color=TINTA)
        ax.text(xi + ancho/2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=TINTA)

    ax.set_facecolor(SUPERFICIE)
    ax.set_title("Random Forest: class_weight (A) vs recorte de ARMA_FUEGO (B)\n"
                "Mismo algoritmo, misma cantidad de arboles, distinta tecnica de balance",
                color=TINTA, fontsize=12, pad=14, loc="left", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Variante A\nclass_weight='balanced'\n(30.782 filas train)",
                        "Variante B\nARMA_FUEGO recortado\n(10.409 filas train)"], fontsize=9)
    ax.set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "randomforest_v2_comparacion.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"\nGrafico: Graficos/{ruta.name}")


def main():
    rep.titulo("ENTRENAMIENTO V2 - RANDOM FOREST", "#")
    rep.p(f"Python {sys.version.split()[0]} | semilla {SEMILLA}")

    m_a, ya, pa = entrenar_variante("variante_a", "VARIANTE A", "balanced", COLOR_A)
    m_b, yb, pb = entrenar_variante("variante_b", "VARIANTE B", None, COLOR_B)

    rep.titulo("COMPARACION DIRECTA")
    comp = pd.DataFrame({
        "Variante A (class_weight)": [m_a["accuracy_val"], m_a["f1_val"], m_a["f1_train"]-m_a["f1_val"]],
        "Variante B (recorte)": [m_b["accuracy_val"], m_b["f1_val"], m_b["f1_train"]-m_b["f1_val"]],
    }, index=["Accuracy (val)", "F1 macro (val)", "Brecha overfitting"]).round(4)
    rep.p(comp.to_string())

    graficar_comparacion(m_a, m_b)

    rep.titulo("FIN", "#")
    rep.p("Guardado: modelos/randomforest_variante_a.joblib, randomforest_variante_b.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
