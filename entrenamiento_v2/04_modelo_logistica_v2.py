"""
ENTRENAMIENTO V2 - REGRESION LOGISTICA MULTINOMIAL (Variante A vs Variante B)
==============================================================================

QUE HACE EL MODELO (la formula, sin caja negra)
------------------------------------------------
Para cada clase k (ARMA_FUEGO, ARMA_BLANCA, ARMA_CONTUNDENTE, OTRAS) el
modelo aprende un vector de pesos w_k (uno por columna de X) y un sesgo b_k:

    z_k(x) = w_k . x + b_k
    P(y=k | x) = exp(z_k) / SUM_j exp(z_j)          <- softmax

Se entrena minimizando la entropia cruzada multinomial mas una penalizacion
L2 sobre los pesos:

    L(W) = - SUM_i log P(y_i | x_i) + (1 / (2C)) * ||W||^2

C es el hiperparametro que controla cuanta penalizacion se aplica:
  C chico  -> penalizacion FUERTE  -> pesos chicos, modelo mas simple/rigido
  C grande -> penalizacion DEBIL   -> pesos libres, modelo mas flexible
sklearn usa C (inverso de la regularizacion), no lambda: por eso va al reves
de lo que uno esperaria.

DIFERENCIA CON LA LOGISTICA DE V1
-----------------------------------
V1 (04_modelo_logistica.py) uso los valores por defecto de C sin justificarlo
y sin class_weight, porque dataset.xlsx ya venia balanceado ~25% por clase.
Aca hay dos cambios de fondo:

  1. Los datos son 100% reales y estan DESBALANCEADOS (79% ARMA_FUEGO en la
     Variante A). Por eso la Variante A usa class_weight='balanced', que
     multiplica el error de cada clase por n / (n_clases * n_clase). Sin eso,
     el optimo matematico de la funcion de perdida es predecir ARMA_FUEGO
     casi siempre: acierta 79% de accuracy pero con F1 macro pesimo.
     La Variante B no lo usa a proposito, para aislar el efecto del recorte.

  2. Se hace un EXPERIMENTO EXPLICITO sobre C en vez de aceptar el default,
     con la misma regla de parsimonia que se uso en Random Forest, CatBoost
     y MLP: entre todos los C cuyo F1 val esta dentro del 1% del mejor, se
     elige el MAS REGULARIZADO (C mas chico). Navaja de Occam: si dos modelos
     rinden igual, se prefiere el mas simple.

Entrada : data/variante_{a,b}_{train,val}_X.npy y _y.npy (los genera
          01_preparacion_v2.py; el preprocesador se ajusto SOLO con train)
Salida  : modelos/logistica_variante_{a,b}.joblib
          resultados/informe_modelo_logistica_v2.txt
          Graficos/logistica_v2_*.png
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

DIR_DATA = BASE / "data"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados"
DIR_FIG = BASE / "Graficos"
DIR_MOD.mkdir(exist_ok=True)
DIR_FIG.mkdir(exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Barrido de regularizacion. Cubre 4 ordenes de magnitud alrededor del
# default de sklearn (C=1.0), que queda justo en el medio.
CES = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
MAX_ITER = 3000

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
COLOR_CLASE = {"ARMA_FUEGO": "#2a78d6", "ARMA_BLANCA": "#eb6834",
               "ARMA_CONTUNDENTE": "#1baf7a", "OTRAS": "#eda100"}
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


rep = Reporte(DIR_REP / "informe_modelo_logistica_v2.txt")


def estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def cargar(nombre):
    d = {}
    for parte in ("train", "val"):
        d[f"X_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_X.npy")
        d[f"y_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_y.npy", allow_pickle=True)
    d["columnas"] = np.load(DIR_DATA / f"{nombre}_columnas.npy", allow_pickle=True)
    return d


def experimento_C(d, class_weight, etiqueta):
    """Barrido de C. Devuelve el C elegido y la tabla completa del barrido."""
    rep.p(f"\n  EXPERIMENTO 1: regularizacion C ({etiqueta})")
    rep.p(f"  {'C':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7} | "
          f"{'|w| medio':>10} | {'iter':>5}")
    rep.p("  " + "-" * 66)

    filas = []
    for c in CES:
        m = LogisticRegression(C=c, max_iter=MAX_ITER, class_weight=class_weight,
                               random_state=SEMILLA)
        m.fit(d["X_train"], d["y_train"])
        f1_tr = f1_score(d["y_train"], m.predict(d["X_train"]), average="macro", zero_division=0)
        f1_va = f1_score(d["y_val"], m.predict(d["X_val"]), average="macro", zero_division=0)
        w_medio = np.abs(m.coef_).mean()
        n_iter = int(m.n_iter_[0])
        rep.p(f"  {c:>8.2f} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr-f1_va:7.4f} | "
              f"{w_medio:10.4f} | {n_iter:5d}")
        filas.append({"C": c, "f1_train": f1_tr, "f1_val": f1_va,
                      "w_medio": w_medio, "n_iter": n_iter})

    df = pd.DataFrame(filas)
    mejor = df["f1_val"].max()
    umbral = mejor * 0.99
    i = df.index[df["f1_val"] >= umbral].tolist()[0]  # el primero = C mas chico
    c_elegido = CES[i]
    rep.p(f"\n  Mejor F1 val absoluto : {mejor:.4f} (C={df.loc[df['f1_val'].idxmax(),'C']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : C={c_elegido} -> F1 val {df.loc[i,'f1_val']:.4f}")
    rep.p(f"  Razon: es el C mas chico (mas regularizado, pesos mas chicos) que")
    rep.p(f"         se mantiene dentro del 1% del mejor resultado.")
    return c_elegido, df


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n    --- {nombre} ---")
    a = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep.p(f"    Accuracy         : {a:.4f}")
    rep.p(f"    Precision (macro): {p:.4f}")
    rep.p(f"    Recall (macro)   : {r:.4f}")
    rep.p(f"    F1 (macro)       : {f:.4f}")
    return {"accuracy": a, "f1_macro": f}


def entrenar_variante(nombre_variante, etiqueta, class_weight):
    rep.titulo(f"{etiqueta} (class_weight={class_weight})")
    d = cargar(nombre_variante)
    rep.p(f"  train: {d['X_train'].shape}   val: {d['X_val'].shape}")

    dist = pd.Series(d["y_train"]).value_counts()
    rep.p("\n  Distribucion real en train:")
    for k in CLASES:
        v = int(dist.get(k, 0))
        rep.p(f"    {k:<18} {v:6,}  ({100*v/len(d['y_train']):5.2f}%)")

    if class_weight == "balanced":
        n, n_cl = len(d["y_train"]), len(CLASES)
        rep.p("\n  Pesos que aplica class_weight='balanced'  [ n / (n_clases * n_clase) ]:")
        for k in CLASES:
            v = int(dist.get(k, 0))
            rep.p(f"    {k:<18} peso = {n / (n_cl * v):6.3f}")

    c_elegido, df_c = experimento_C(d, class_weight, etiqueta)

    rep.p(f"\n  HIPERPARAMETROS FINALES ({etiqueta}):")
    params = dict(C=c_elegido, max_iter=MAX_ITER, class_weight=class_weight,
                  random_state=SEMILLA)
    for k, v in params.items():
        rep.p(f"    {k} = {v}")
    rep.p(f"    solver = lbfgs        (DEFAULT de sklearn)")
    rep.p(f"    penalty = 'l2'        (DEFAULT de sklearn)")
    rep.p(f"    tol = 1e-4            (DEFAULT de sklearn)")
    rep.p(f"    fit_intercept = True  (DEFAULT de sklearn)")

    modelo = LogisticRegression(**params)
    modelo.fit(d["X_train"], d["y_train"])

    if modelo.n_iter_[0] >= MAX_ITER:
        rep.p(f"\n    !! NO convergio en {MAX_ITER} iteraciones (n_iter_={modelo.n_iter_[0]}).")
    else:
        rep.p(f"\n    Convergio en {modelo.n_iter_[0]} iteraciones (max {MAX_ITER}).")

    pred_tr = modelo.predict(d["X_train"])
    pred_va = modelo.predict(d["X_val"])

    m_tr = metricas(d["y_train"], pred_tr, "TRAIN")
    m_va = metricas(d["y_val"], pred_va, "VALIDACION")

    brecha = m_tr["f1_macro"] - m_va["f1_macro"]
    rep.p(f"\n  OVERFITTING / UNDERFITTING: brecha F1 train-val = {brecha:.4f}")
    if brecha > 0.08:
        rep.p("    -> Posible OVERFITTING.")
    elif m_tr["f1_macro"] < 0.5:
        rep.p("    -> Posible UNDERFITTING (ni en train aprende bien).")
    else:
        rep.p("    -> Sin senales fuertes de over/underfitting.")

    rep.p(f"\n  Reporte por clase (validacion):")
    rep.p(classification_report(d["y_val"], pred_va, labels=CLASES, zero_division=0))

    cm = confusion_matrix(d["y_val"], pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    joblib.dump({"modelo": modelo, "C": c_elegido, "class_weight": class_weight},
                DIR_MOD / f"logistica_{nombre_variante}.joblib")

    return {
        "etiqueta": etiqueta, "modelo": modelo, "datos": d, "C": c_elegido,
        "df_c": df_c, "cm": cm, "pred_val": pred_va,
        "f1_train": m_tr["f1_macro"], "f1_val": m_va["f1_macro"],
        "acc_val": m_va["accuracy"], "brecha": brecha,
    }


# ---------------------------------------------------------------- graficos
def fig_barrido_C(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, r in zip(axes, (ra, rb)):
        df = r["df_c"]
        ax.plot(df["C"], df["f1_train"], "o-", color=COLOR_TRAIN, label="Train", zorder=3)
        ax.plot(df["C"], df["f1_val"], "o-", color=COLOR_VAL, label="Validacion", zorder=3)
        ax.axvline(r["C"], color=MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.annotate(f"elegido C={r['C']}", xy=(r["C"], ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8.5, color=TINTA_SEC)
        ax.set_xscale("log")
        ax.set_xlabel("C  (mas a la izquierda = mas regularizado)",
                      color=TINTA_SEC, fontsize=9.5)
        ax.set_title(r["etiqueta"], color=TINTA, fontsize=11, loc="left", weight="bold")
        estilo(ax)
    axes[0].set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    fig.suptitle("Experimento 1: cuanta regularizacion conviene",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "logistica_v2_barrido_C.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def fig_matrices(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, r in zip(axes, (ra, rb)):
        cm = r["cm"]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASES)))
        ax.set_yticks(range(len(CLASES)))
        ax.set_xticklabels(CLASES, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticklabels(CLASES, fontsize=8.5)
        ax.set_xlabel("Prediccion", color=TINTA_SEC, fontsize=9.5)
        ax.set_ylabel("Real", color=TINTA_SEC, fontsize=9.5)
        ax.set_title(f"{r['etiqueta']}  (F1 macro val {r['f1_val']:.3f})",
                     color=TINTA, fontsize=11, loc="left", weight="bold")
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.5 else TINTA)
        ax.tick_params(colors=MUTED)
        ax.grid(False)
    fig.suptitle("Matriz de confusion en validacion (celda: casos y % de la fila real)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "logistica_v2_matrices_confusion.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def fig_comparacion(ra, rb):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.arange(2)
    ancho = 0.32
    tr = [ra["f1_train"], rb["f1_train"]]
    va = [ra["f1_val"], rb["f1_val"]]
    b1 = ax1.bar(x - ancho/2, tr, ancho, label="Train", color=COLOR_TRAIN, zorder=3)
    b2 = ax1.bar(x + ancho/2, va, ancho, label="Validacion", color=COLOR_VAL, zorder=3)
    for barras in (b1, b2):
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
    for xi, (t, v) in enumerate(zip(tr, va)):
        ax1.text(xi - ancho/2, t + 0.012, f"{t:.3f}", ha="center", fontsize=9, color=TINTA)
        ax1.text(xi + ancho/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=TINTA)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Variante A\nclass_weight='balanced'", "Variante B\nrecorte ARMA_FUEGO"],
                        fontsize=9)
    ax1.set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    ax1.set_title("Rendimiento global", color=TINTA, fontsize=11, loc="left", weight="bold")
    estilo(ax1)
    leg = ax1.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    xc = np.arange(len(CLASES))
    f1a = f1_score(ra["datos"]["y_val"], ra["pred_val"], labels=CLASES,
                   average=None, zero_division=0)
    f1b = f1_score(rb["datos"]["y_val"], rb["pred_val"], labels=CLASES,
                   average=None, zero_division=0)
    b1 = ax2.bar(xc - ancho/2, f1a, ancho, label="Variante A", color=COLOR_A, zorder=3)
    b2 = ax2.bar(xc + ancho/2, f1b, ancho, label="Variante B", color=COLOR_B, zorder=3)
    for barras in (b1, b2):
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
    for xi, (a_, b_) in enumerate(zip(f1a, f1b)):
        ax2.text(xi - ancho/2, a_ + 0.012, f"{a_:.2f}", ha="center", fontsize=8, color=TINTA)
        ax2.text(xi + ancho/2, b_ + 0.012, f"{b_:.2f}", ha="center", fontsize=8, color=TINTA)
    ax2.set_xticks(xc)
    ax2.set_xticklabels([c.replace("_", "\n") for c in CLASES], fontsize=8.5)
    ax2.set_ylabel("F1 de la clase (validacion)", color=TINTA_SEC, fontsize=10)
    ax2.set_title("Rendimiento por clase", color=TINTA, fontsize=11, loc="left", weight="bold")
    estilo(ax2)
    leg = ax2.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    fig.suptitle("Regresion Logistica V2: class_weight (A) vs recorte (B)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "logistica_v2_comparacion.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def fig_coeficientes(r, nombre_variante, n=12):
    """Los coeficientes SON el modelo: w_k por columna. Se muestran los mas
    extremos de cada clase (positivo = empuja hacia esa clase)."""
    modelo, cols = r["modelo"], r["datos"]["columnas"]
    orden = list(modelo.classes_)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, clase in zip(axes.ravel(), CLASES):
        w = modelo.coef_[orden.index(clase)]
        idx = np.argsort(w)
        sel = np.concatenate([idx[:n // 2], idx[-n // 2:]])
        vals, nombres = w[sel], cols[sel]
        colores = [COLOR_B if v < 0 else COLOR_CLASE[clase] for v in vals]
        ax.barh(range(len(vals)), vals, color=colores, zorder=3)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([str(s)[:42] for s in nombres], fontsize=7.5)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(clase, color=TINTA, fontsize=10.5, loc="left", weight="bold")
        ax.set_facecolor(SUPERFICIE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(f"Coeficientes mas extremos por clase - {r['etiqueta']} (C={r['C']})\n"
                 "positivo = esa caracteristica empuja hacia la clase | negativo = la aleja",
                 color=TINTA, fontsize=12.5, weight="bold", x=0.01, ha="left")
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / f"logistica_v2_coeficientes_{nombre_variante}.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    return ruta


def main():
    rep.titulo("ENTRENAMIENTO V2 - REGRESION LOGISTICA MULTINOMIAL", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")
    rep.p("Datos: matrices .npy generadas por 01_preparacion_v2.py")
    rep.p("(preprocesador ajustado SOLO con train de cada variante)")

    ra = entrenar_variante("variante_a", "VARIANTE A", "balanced")
    rb = entrenar_variante("variante_b", "VARIANTE B", None)

    rep.titulo("COMPARACION DIRECTA A vs B")
    comp = pd.DataFrame({
        "Variante A (class_weight)": [ra["C"], ra["acc_val"], ra["f1_val"], ra["brecha"]],
        "Variante B (recorte)": [rb["C"], rb["acc_val"], rb["f1_val"], rb["brecha"]],
    }, index=["C elegido", "Accuracy (val)", "F1 macro (val)", "Brecha train-val"]).round(4)
    rep.p(comp.to_string())

    rep.titulo("COMPARACION CONTRA LA LOGISTICA DE V1 (dataset.xlsx)")
    rep.p("  V1 (dataset.xlsx, con filas fabricadas, C=1.0 por defecto, sin class_weight):")
    rep.p("      Accuracy val 0.5939 | F1 macro val 0.5923 | brecha 0.027")
    rep.p(f"  V2 Variante A: Accuracy val {ra['acc_val']:.4f} | F1 macro val {ra['f1_val']:.4f} "
          f"| brecha {ra['brecha']:.4f}")
    rep.p(f"  V2 Variante B: Accuracy val {rb['acc_val']:.4f} | F1 macro val {rb['f1_val']:.4f} "
          f"| brecha {rb['brecha']:.4f}")
    rep.p("")
    rep.p("  NO son comparables como 'quien es mejor': V1 se evalua sobre un val")
    rep.p("  que tambien contiene filas fabricadas y clases artificialmente")
    rep.p("  equilibradas (25% cada una), asi que su F1 macro esta inflado. La")
    rep.p("  comparacion util es entre A y B, que si comparten fuente real.")

    rep.titulo("GRAFICOS")
    for ruta in (fig_barrido_C(ra, rb), fig_matrices(ra, rb), fig_comparacion(ra, rb),
                 fig_coeficientes(ra, "variante_a"), fig_coeficientes(rb, "variante_b")):
        rep.p(f"  Graficos/{ruta.name}")

    rep.titulo("FIN", "#")
    rep.p("Guardado: modelos/logistica_variante_a.joblib, logistica_variante_b.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
