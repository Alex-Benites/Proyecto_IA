"""
GRAFICOS DE DIAGNOSTICO (reutilizable para cualquier modelo)
=============================================================

Uso:
    python 05_graficos.py logistica
    python 05_graficos.py randomforest

Genera las figuras para entender QUE esta pasando dentro del modelo, no solo
el numero final. Cada figura responde una pregunta concreta:

  fig1_matriz_confusion.png   -> ¿Con que clase se confunde el modelo?
  fig2_metricas_por_clase.png -> ¿Que clase predice bien y cual mal?
  fig3_train_vs_val.png       -> ¿Hay overfitting?
  fig4_dinamica.png           -> ¿Como aprende? (depende del modelo:
                                 iteraciones en logistica, cantidad de
                                 arboles en random forest)
  fig5_curva_aprendizaje.png  -> ¿Mas datos mejorarian el modelo?
  fig6_importancia.png        -> ¿Que variables pesan mas?
                                 (interpretabilidad: lo contrario de una
                                 caja negra)

Las figuras se guardan en Graficos/ con el prefijo del modelo, para que los
resultados de un modelo no pisen los de otro.

Paleta: azul #2a78d6, naranja #eb6834, aqua #1baf7a, amarillo #eda100.
Orden fijo de colores por clase (nunca se recicla ni se reasigna).
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_FIG = BASE / "Graficos"
DIR_FIG.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Registro de modelos soportados. Al agregar un modelo nuevo, se anade aqui.
MODELOS = {
    "logistica": {
        "archivo": "modelo1_logistica.joblib",
        "prefijo": "modelo1_logistica",
        "titulo": "Regresion Logistica",
        "dinamica": "iteraciones",
    },
    "randomforest": {
        "archivo": "modelo2_randomforest.joblib",
        "prefijo": "modelo2_randomforest",
        "titulo": "Random Forest",
        "dinamica": "arboles",
    },
    "mlp": {
        "archivo": "modelo4_mlp.joblib",
        "prefijo": "modelo4_mlp",
        "titulo": "Red Neuronal (MLP)",
        # La curva de perdida por epoca la genera 09_modelo_mlp.py, que tiene
        # acceso a loss_curve_ del entrenamiento. Aqui no se regenera.
        "dinamica": None,
    },
}

# Paleta categorica en orden fijo: un color por clase, siempre el mismo.
COLOR_CLASE = {
    "ARMA_FUEGO":       "#2a78d6",  # azul
    "ARMA_BLANCA":      "#eb6834",  # naranja
    "ARMA_CONTUNDENTE": "#1baf7a",  # aqua
    "OTRAS":            "#eda100",  # amarillo
}
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"

TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SUPERFICIE = "#fcfcfb"

RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_seq", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6",
                 "#256abf", "#184f95", "#0d366b"])

PREFIJO = None   # se define en main() segun el modelo elegido
TITULO = None


def estilo(ax, titulo, xlabel="", ylabel=""):
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
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    nombre = f"{PREFIJO}_{nombre}"
    ruta = DIR_FIG / nombre
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    print(f"  guardado: Graficos/{nombre}")


# ---------------------------------------------------------------------------
# FIG 1 - Matriz de confusion (normalizada por fila = recall por clase)
# ---------------------------------------------------------------------------
def fig_matriz(y_val, pred_val):
    cm = confusion_matrix(y_val, pred_val, labels=CLASES)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_pct, cmap=RAMPA_AZUL, vmin=0, vmax=100)

    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASES], fontsize=9)
    ax.set_yticklabels([c.replace("_", "\n") for c in CLASES], fontsize=9)
    ax.set_xlabel("Lo que el modelo PREDIJO", color=TINTA_SEC, fontsize=10)
    ax.set_ylabel("Lo que REALMENTE era", color=TINTA_SEC, fontsize=10)
    ax.set_title(f"Matriz de confusion - {TITULO} (validacion)\n"
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
    guardar(fig, "fig1_matriz_confusion.png")


# ---------------------------------------------------------------------------
# FIG 2 - Precision / Recall / F1 por clase
# ---------------------------------------------------------------------------
def fig_metricas_clase(y_val, pred_val):
    prec = precision_score(y_val, pred_val, labels=CLASES, average=None, zero_division=0)
    rec = recall_score(y_val, pred_val, labels=CLASES, average=None, zero_division=0)
    f1 = f1_score(y_val, pred_val, labels=CLASES, average=None, zero_division=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, valores, nombre in zip(axes, [prec, rec, f1],
                                    ["Precision", "Recall", "F1-score"]):
        colores = [COLOR_CLASE[c] for c in CLASES]
        barras = ax.bar(range(len(CLASES)), valores, color=colores,
                        width=0.62, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        estilo(ax, nombre)
        ax.set_xticks(range(len(CLASES)))
        ax.set_xticklabels([c.replace("ARMA_", "").replace("_", " ") for c in CLASES],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.25, color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.025, f"{v:.2f}", ha="center", color=TINTA,
                    fontsize=9, weight="bold")

    axes[0].set_ylabel("Puntaje (0 a 1)", color=TINTA_SEC, fontsize=10)
    axes[2].text(len(CLASES) - 0.5, 0.27, "azar = 0.25", color=MUTED,
                 fontsize=8, ha="right")
    fig.suptitle(f"{TITULO} - Rendimiento por clase (validacion)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    guardar(fig, "fig2_metricas_por_clase.png")


# ---------------------------------------------------------------------------
# FIG 3 - Train vs Validacion (deteccion de overfitting)
# ---------------------------------------------------------------------------
def fig_train_vs_val(y_train, pred_train, y_val, pred_val):
    nombres = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
    tr = [accuracy_score(y_train, pred_train),
          precision_score(y_train, pred_train, average="macro", zero_division=0),
          recall_score(y_train, pred_train, average="macro", zero_division=0),
          f1_score(y_train, pred_train, average="macro", zero_division=0)]
    va = [accuracy_score(y_val, pred_val),
          precision_score(y_val, pred_val, average="macro", zero_division=0),
          recall_score(y_val, pred_val, average="macro", zero_division=0),
          f1_score(y_val, pred_val, average="macro", zero_division=0)]

    x = np.arange(len(nombres))
    ancho = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - ancho / 2, tr, ancho, label="Entrenamiento", color=COLOR_TRAIN, zorder=3)
    b2 = ax.bar(x + ancho / 2, va, ancho, label="Validacion", color=COLOR_VAL, zorder=3)
    for barras in (b1, b2):
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)

    estilo(ax, f"{TITULO}: entrenamiento vs validacion\n"
               "Si las barras azules son MUCHO mas altas -> overfitting.",
           ylabel="Puntaje (0 a 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=9)
    ax.set_ylim(0, 1.12)
    for xi, (t, v) in enumerate(zip(tr, va)):
        ax.text(xi - ancho / 2, t + 0.015, f"{t:.3f}", ha="center", fontsize=8.5, color=TINTA)
        ax.text(xi + ancho / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5, color=TINTA)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    brecha = tr[3] - va[3]
    ax.text(0.01, 0.02, f"Brecha F1 (train - val) = {brecha:.3f}   "
                        f"(< 0.08 se considera saludable)",
            transform=ax.transAxes, fontsize=9, color=TINTA_SEC)
    guardar(fig, "fig3_train_vs_val.png")


# ---------------------------------------------------------------------------
# FIG 4a - Dinamica de aprendizaje: REGRESION LOGISTICA (iteracion a iteracion)
# ---------------------------------------------------------------------------
def fig_dinamica_iteraciones(X_train, y_train, X_val, y_val):
    print("  calculando curva de convergencia (entrena paso a paso)...")
    iteraciones, f1_tr, f1_va = [], [], []

    modelo = LogisticRegression(max_iter=1, warm_start=True, random_state=SEMILLA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for paso in range(1, 61):
            modelo.max_iter = paso
            modelo.fit(X_train, y_train)
            iteraciones.append(paso)
            f1_tr.append(f1_score(y_train, modelo.predict(X_train),
                                  average="macro", zero_division=0))
            f1_va.append(f1_score(y_val, modelo.predict(X_val),
                                  average="macro", zero_division=0))

    _plot_dinamica(iteraciones, f1_tr, f1_va,
                   titulo="Como aprende el modelo, iteracion a iteracion\n"
                          "El equivalente a las 'epocas' de una red neuronal.",
                   xlabel="Iteracion del optimizador (L-BFGS)")


# ---------------------------------------------------------------------------
# FIG 4b - Dinamica de aprendizaje: RANDOM FOREST (arbol a arbol)
# ---------------------------------------------------------------------------
def fig_dinamica_arboles(X_train, y_train, X_val, y_val, modelo_final):
    print("  calculando curva por cantidad de arboles...")
    # Se reusan los hiperparametros del modelo ya entrenado, cambiando solo
    # n_estimators, para que la curva describa a ESE modelo y no a otro.
    params = modelo_final.get_params()
    cantidades = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 180, 250, 300]
    cantidades = [n for n in cantidades if n <= params["n_estimators"]]
    if params["n_estimators"] not in cantidades:
        cantidades.append(params["n_estimators"])

    f1_tr, f1_va = [], []
    m = RandomForestClassifier(**{**params, "n_estimators": 1, "warm_start": True})
    for n in cantidades:
        m.set_params(n_estimators=n)
        m.fit(X_train, y_train)
        f1_tr.append(f1_score(y_train, m.predict(X_train), average="macro", zero_division=0))
        f1_va.append(f1_score(y_val, m.predict(X_val), average="macro", zero_division=0))

    _plot_dinamica(cantidades, f1_tr, f1_va,
                   titulo="Como aprende el bosque, arbol a arbol\n"
                          "Random Forest NO usa epocas: crece agregando arboles.",
                   xlabel="Cantidad de arboles en el bosque (n_estimators)",
                   log_x=True)


def _plot_dinamica(x, f1_tr, f1_va, titulo, xlabel, log_x=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, f1_tr, color=COLOR_TRAIN, linewidth=2, label="Entrenamiento", zorder=3)
    ax.plot(x, f1_va, color=COLOR_VAL, linewidth=2, label="Validacion", zorder=3)
    ax.scatter([x[-1]], [f1_tr[-1]], color=COLOR_TRAIN, s=45, zorder=4,
               edgecolor=SUPERFICIE, linewidth=2)
    ax.scatter([x[-1]], [f1_va[-1]], color=COLOR_VAL, s=45, zorder=4,
               edgecolor=SUPERFICIE, linewidth=2)
    ax.text(x[-1], f1_tr[-1] + 0.015, f"{f1_tr[-1]:.3f}", ha="right",
            color=TINTA, fontsize=9, weight="bold")
    ax.text(x[-1], f1_va[-1] - 0.035, f"{f1_va[-1]:.3f}", ha="right",
            color=TINTA, fontsize=9, weight="bold")

    estilo(ax, titulo, xlabel=xlabel, ylabel="F1 macro")
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x], fontsize=8)
        ax.minorticks_off()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar(fig, "fig4_dinamica.png")


# ---------------------------------------------------------------------------
# FIG 5 - Curva de aprendizaje: ¿serviria tener MAS datos?
# ---------------------------------------------------------------------------
def fig_curva_aprendizaje(X_train, y_train, X_val, y_val, modelo_final):
    print("  calculando curva de aprendizaje (varios tamanios de train)...")
    rng = np.random.default_rng(SEMILLA)
    fracciones = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]
    tamanios, f1_tr, f1_va = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for frac in fracciones:
            n = int(len(X_train) * frac)
            idx = rng.choice(len(X_train), size=n, replace=False)
            Xs, ys = X_train[idx], y_train[idx]
            m = _clonar(modelo_final)
            m.fit(Xs, ys)
            tamanios.append(n)
            f1_tr.append(f1_score(ys, m.predict(Xs), average="macro", zero_division=0))
            f1_va.append(f1_score(y_val, m.predict(X_val), average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tamanios, f1_tr, color=COLOR_TRAIN, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Entrenamiento", zorder=3)
    ax.plot(tamanios, f1_va, color=COLOR_VAL, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Validacion", zorder=3)
    estilo(ax, "Curva de aprendizaje: ¿mas datos ayudarian?\n"
               "Si la naranja sigue subiendo al final -> si. Si se aplana -> no.",
           xlabel="Cantidad de filas de entrenamiento", ylabel="F1 macro")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.text(tamanios[-1], f1_va[-1] - 0.035, f"{f1_va[-1]:.3f}", ha="right",
            color=TINTA, fontsize=9, weight="bold")
    leg = ax.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar(fig, "fig5_curva_aprendizaje.png")


def _clonar(modelo):
    """Copia sin entrenar del modelo, con los mismos hiperparametros."""
    from sklearn.base import clone
    m = clone(modelo)
    if hasattr(m, "warm_start"):
        m.set_params(warm_start=False)
    return m


# ---------------------------------------------------------------------------
# FIG 6 - Que variables pesan mas (interpretabilidad)
# ---------------------------------------------------------------------------
def fig_importancia_coeficientes(modelo, nombres_features):
    """Para modelos lineales: un peso por clase y por variable."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    orden_clases = list(modelo.classes_)

    for ax, clase in zip(axes.ravel(), CLASES):
        k = orden_clases.index(clase)
        pesos = modelo.coef_[k]
        top_idx = np.argsort(np.abs(pesos))[-10:]
        vals = pesos[top_idx]
        etiquetas = [_limpiar_nombre(nombres_features[i]) for i in top_idx]

        colores = [COLOR_CLASE[clase] if v > 0 else "#c3c2b7" for v in vals]
        barras = ax.barh(range(len(vals)), vals, color=colores, zorder=3, height=0.68)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(1.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(etiquetas, fontsize=8)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        estilo(ax, clase.replace("_", " "))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)

    fig.suptitle("Que variables empujan hacia cada clase (coeficientes del modelo)\n"
                 "Color = empuja HACIA la clase.  Gris = empuja EN CONTRA.",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    guardar(fig, "fig6_importancia.png")


def fig_importancia_arboles(modelo, nombres_features):
    """Para modelos de arboles: una sola importancia global por variable.

    OJO en la interpretacion: feature_importances_ NO dice hacia que clase
    empuja la variable, solo cuanto ayudo a separar las clases en general.
    Es menos informativo que los coeficientes de la logistica, pero sigue
    siendo interpretable (no es una caja negra).
    """
    imp = modelo.feature_importances_
    top_idx = np.argsort(imp)[-20:]
    vals = imp[top_idx]
    etiquetas = [_limpiar_nombre(nombres_features[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    barras = ax.barh(range(len(vals)), vals, color="#2a78d6", zorder=3, height=0.7)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(1.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(etiquetas, fontsize=9)
    estilo(ax, "Las 20 variables mas importantes del bosque\n"
               "Cuanto ayudo cada variable a separar las clases (suma = 1).",
           xlabel="Importancia (reduccion media de impureza)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, f"{v:.3f}", va="center",
                fontsize=8, color=TINTA_SEC)
    guardar(fig, "fig6_importancia.png")


def fig_importancia_permutacion(modelo, nombres_features, X_val, y_val):
    """Para modelos sin coeficientes ni feature_importances_ (ej. el MLP).

    Importancia por PERMUTACION: se baraja al azar UNA columna y se mide
    cuanto empeora el F1. Si empeora mucho, esa columna importaba.
    Es agnostica al modelo y facil de explicar: no depende de la estructura
    interna, solo de cuanto se degrada la prediccion sin esa variable.
    """
    from sklearn.inspection import permutation_importance
    print("  calculando importancia por permutacion (puede tardar)...")
    r = permutation_importance(
        modelo, X_val, y_val, n_repeats=5, random_state=SEMILLA,
        scoring="f1_macro", n_jobs=-1)

    top_idx = np.argsort(r.importances_mean)[-20:]
    vals = r.importances_mean[top_idx]
    errs = r.importances_std[top_idx]
    etiquetas = [_limpiar_nombre(nombres_features[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    barras = ax.barh(range(len(vals)), vals, xerr=errs, color="#2a78d6",
                     zorder=3, height=0.7,
                     error_kw={"ecolor": "#898781", "elinewidth": 1.2})
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(1.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(etiquetas, fontsize=9)
    estilo(ax, "Las 20 variables mas importantes (por permutacion)\n"
               "Cuanto cae el F1 al barajar esa columna al azar.",
           xlabel="Caida de F1 macro al permutar")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    guardar(fig, "fig6_importancia.png")


def _limpiar_nombre(nombre):
    return (nombre.replace("cat__", "").replace("num__", "")
                  .replace("bin__", "")[:44])


# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELOS:
        print(f"Uso: python 05_graficos.py [{' | '.join(MODELOS)}]")
        sys.exit(1)

    global PREFIJO, TITULO
    cfg = MODELOS[sys.argv[1]]
    PREFIJO, TITULO = cfg["prefijo"], cfg["titulo"]

    print(f"Python {sys.version.split()[0]} | graficos de: {TITULO}\n")

    X_train = np.load(DIR_PROC / "train_X.npy")
    y_train = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_val = np.load(DIR_PROC / "val_X.npy")
    y_val = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)

    modelo = joblib.load(DIR_MOD / cfg["archivo"])
    preproc = joblib.load(DIR_MOD / "preprocesador.joblib")
    nombres_features = list(preproc.get_feature_names_out())

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    fig_matriz(y_val, pred_val)
    fig_metricas_clase(y_val, pred_val)
    fig_train_vs_val(y_train, pred_train, y_val, pred_val)

    if cfg["dinamica"] == "iteraciones":
        fig_dinamica_iteraciones(X_train, y_train, X_val, y_val)
    elif cfg["dinamica"] == "arboles":
        fig_dinamica_arboles(X_train, y_train, X_val, y_val, modelo)
    else:
        print("  (la curva de aprendizaje por epoca la genera su propio script)")

    fig_curva_aprendizaje(X_train, y_train, X_val, y_val, modelo)

    if hasattr(modelo, "coef_"):
        fig_importancia_coeficientes(modelo, nombres_features)
    elif hasattr(modelo, "feature_importances_"):
        fig_importancia_arboles(modelo, nombres_features)
    else:
        fig_importancia_permutacion(modelo, nombres_features, X_val, y_val)

    print(f"\nTodas las figuras en: Graficos/")


if __name__ == "__main__":
    main()
