"""
ENTRENAMIENTO V2 - CATBOOST (Variante A vs Variante B)
=======================================================

QUE HACE EL MODELO (la formula, sin caja negra)
------------------------------------------------
CatBoost es BOOSTING: los arboles se construyen EN SECUENCIA y cada uno se
entrena para corregir los errores que dejo el conjunto anterior.

    F_m(x) = F_{m-1}(x) + eta * h_m(x)

h_m es un arbol nuevo ajustado al GRADIENTE NEGATIVO de la perdida respecto
a las predicciones actuales; eta es el learning_rate. Cada arbol mira
"donde me estoy equivocando mas ahora" y se especializa ahi.

Perdida (multiclase): MultiClass = LogLoss multinomial
    L = - SUM_i log P(y_i = clase_verdadera | x_i)
Es LA MISMA entropia cruzada de la regresion logistica. La diferencia no
esta en QUE minimiza, sino en COMO: sumando arboles en vez de ajustando
pesos lineales.

POR QUE ESTE MODELO NO USA LOS .npy
------------------------------------
Logistica y Random Forest usan variante_*_X.npy (one-hot + escalado) porque
no aceptan texto. CatBoost SI, y esa es su ventaja: en vez de convertir
'canton' en 211 columnas binarias (escondiendole al modelo que esas 211
columnas son la misma variable), usa ORDERED TARGET STATISTICS: reemplaza
cada categoria por una estadistica de la tasa de cada clase en esa
categoria, calculada SOLO con las filas anteriores en un orden aleatorio.
Ese "solo las anteriores" es lo que evita el leakage que tendria un
target-encoding ingenuo.

Por eso parte de variante_*_raw.csv, que 01_preparacion_v2.py guardo ANTES
del ColumnTransformer: mismo split, mismas filas, misma semilla, pero con
las categorias todavia en texto.

Dos ajustes propios:
  1. 'edad' tiene NaN -> se imputa con la MEDIANA DEL TRAIN, el mismo
     criterio y el mismo valor que usaron logistica y Random Forest, para
     que la comparacion entre algoritmos sea justa.
  2. NO se escala: a un arbol le da igual la escala. El escalado existia
     solo por la regresion logistica.

DESBALANCE: auto_class_weights, NO class_weight
------------------------------------------------
CatBoost no tiene el parametro class_weight de sklearn. Su equivalente es
auto_class_weights='Balanced', que calcula el mismo peso
n / (n_clases * n_clase) y lo aplica dentro de la funcion de perdida.

  VARIANTE A: auto_class_weights='Balanced'  (30.782 filas, 79% ARMA_FUEGO)
  VARIANTE B: sin pesos                      (10.409 filas, ARMA_FUEGO
                                              recortado a 5.800)

EXPERIMENTOS
  Experimento 1: barrido manual de learning_rate, con el early stopping
                 propio de CatBoost decidiendo cuantos arboles usar.
  Experimento 2: categoricas NATIVAS vs ONE-HOT (mismo algoritmo, mismo
                 split, misma configuracion) -> cuanto aporta realmente el
                 manejo nativo. En V1 dio +0.0203 a favor de nativo.

Entrada : data/variante_{a,b}_{train,val}_raw.csv  (texto)
          data/variante_{a,b}_{train,val}_X.npy    (solo para el experimento 2)
Salida  : modelos/catboost_variante_{a,b}.joblib y .cbm
          resultados/informe_modelo_catboost_v2.txt
          Graficos/catboost_v2_*.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

# 16 columnas categoricas (V2 no incluye 'mes', a diferencia de V1).
# CatBoost EXIGE declararlas: si no, las trataria como numericas.
COLS_CATEGORICAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "dia_semana",
    "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad", "es_fin_semana"]

LEARNING_RATES = [0.03, 0.06, 0.10, 0.20, 0.30]
DEPTH = 6

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
COLOR_NATIVO, COLOR_ONEHOT = "#1baf7a", "#eda100"
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


rep = Reporte(DIR_REP / "informe_modelo_catboost_v2.txt")


def estilo(ax, titulo="", xlabel="", ylabel=""):
    ax.set_facecolor(SUPERFICIE)
    if titulo:
        ax.set_title(titulo, color=TINTA, fontsize=11, pad=12, loc="left", weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=TINTA_SEC, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=TINTA_SEC, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def guardar_fig(fig, nombre):
    ruta = DIR_FIG / f"catboost_v2_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Graficos/{ruta.name}")
    return ruta


def cargar_variante(nombre):
    """Splits en TEXTO (antes del one-hot), con los mismos criterios que
    aplicaron logistica y Random Forest sobre las mismas filas."""
    tr = pd.read_csv(DIR_DATA / f"{nombre}_train_raw.csv")
    va = pd.read_csv(DIR_DATA / f"{nombre}_val_raw.csv")

    mediana = tr["edad"].median()
    n_tr, n_va = int(tr["edad"].isna().sum()), int(va["edad"].isna().sum())
    tr["edad"] = tr["edad"].fillna(mediana)
    va["edad"] = va["edad"].fillna(mediana)

    for c in COLS_CATEGORICAS:
        tr[c] = tr[c].astype(str)
        va[c] = va[c].astype(str)

    X_tr, y_tr = tr[COLS_CATEGORICAS + COLS_NUMERICAS], tr["y"]
    X_va, y_va = va[COLS_CATEGORICAS + COLS_NUMERICAS], va["y"]
    return X_tr, y_tr, X_va, y_va, mediana, n_tr, n_va


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


def experimento_learning_rate(pool_tr, pool_va, y_tr, y_va, pesos, etiqueta):
    """Barrido MANUAL de learning_rate.

    El early stopping propio de CatBoost (od_type='Iter', od_wait=50) decide
    cuantos arboles usar para cada lr. Eso NO es busqueda automatica de
    modelo: es el criterio de parada estandar del boosting, y queda
    documentado cuantas iteraciones eligio cada uno.
    """
    rep.p(f"\n  EXPERIMENTO 1: learning_rate ({etiqueta})")
    rep.p("  eta controla cuanto corrige CADA arbol nuevo.")
    rep.p("  Bajo -> aprende despacio, necesita muchos arboles, generaliza mejor.")
    rep.p("  Alto -> aprende rapido, pocos arboles, riesgo de sobreajustar.")
    rep.p("")
    rep.p(f"  {'lr':>6} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for lr in LEARNING_RATES:
        m = CatBoostClassifier(
            iterations=1500, learning_rate=lr, depth=DEPTH,
            loss_function="MultiClass", auto_class_weights=pesos,
            random_seed=SEMILLA, od_type="Iter", od_wait=50, verbose=False,
        )
        m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        f1_tr = f1_score(y_tr, m.predict(pool_tr).ravel(), average="macro", zero_division=0)
        f1_va = f1_score(y_va, m.predict(pool_va).ravel(), average="macro", zero_division=0)
        n_arb = m.tree_count_
        rep.p(f"  {lr:>6.2f} | {n_arb:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr-f1_va:7.4f}")
        filas.append({"lr": lr, "arboles": n_arb, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    # Misma regla de parsimonia que en Random Forest, logistica y MLP:
    # el lr MAS BAJO cuyo F1 val este dentro del 1% del mejor.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    i = df.index[df["f1_val"] >= umbral].tolist()[0]
    lr_elegido = LEARNING_RATES[i]
    n_arboles = int(df.loc[i, "arboles"])

    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {f1_mejor:.4f} (lr={df.loc[df['f1_val'].idxmax(),'lr']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : lr={lr_elegido} con {n_arboles} arboles "
          f"-> F1 val {df.loc[i,'f1_val']:.4f}")
    return lr_elegido, n_arboles, df, i


def entrenar_variante(nombre_variante, etiqueta, pesos):
    rep.titulo(f"{etiqueta} (auto_class_weights={pesos})")
    X_tr, y_tr, X_va, y_va, mediana, n_tr, n_va = cargar_variante(nombre_variante)
    rep.p(f"  Fuente: {nombre_variante}_train_raw.csv / _val_raw.csv (ANTES del one-hot)")
    rep.p(f"  Train: {X_tr.shape}  |  Val: {X_va.shape}")
    rep.p(f"  Categoricas declaradas: {len(COLS_CATEGORICAS)}  |  Numericas: {COLS_NUMERICAS}")
    rep.p(f"  'edad': {n_tr} NaN en train y {n_va} en val -> mediana del train ({mediana})")

    dist = y_tr.value_counts()
    rep.p("\n  Distribucion real en train:")
    for k in CLASES:
        v = int(dist.get(k, 0))
        rep.p(f"    {k:<18} {v:6,}  ({100*v/len(y_tr):5.2f}%)")
    if pesos == "Balanced":
        n, n_cl = len(y_tr), len(CLASES)
        rep.p("\n  Pesos que aplica auto_class_weights='Balanced':")
        for k in CLASES:
            rep.p(f"    {k:<18} peso = {n/(n_cl*int(dist.get(k,0))):6.3f}")

    idx_cat = [X_tr.columns.get_loc(c) for c in COLS_CATEGORICAS]
    pool_tr = Pool(X_tr, y_tr, cat_features=idx_cat)
    pool_va = Pool(X_va, y_va, cat_features=idx_cat)

    lr, n_arboles, df_lr, i_lr = experimento_learning_rate(
        pool_tr, pool_va, y_tr, y_va, pesos, etiqueta)

    rep.p(f"\n  HIPERPARAMETROS FINALES ({etiqueta}):")
    params = dict(
        iterations=n_arboles, learning_rate=lr, depth=DEPTH,
        loss_function="MultiClass", auto_class_weights=pesos,
        random_seed=SEMILLA, verbose=False,
    )
    for k, v in params.items():
        rep.p(f"    {k} = {v}")
    rep.p("    l2_leaf_reg = 3.0     (DEFAULT de CatBoost)")
    rep.p("    border_count = 254    (DEFAULT)")
    rep.p("    grow_policy = SymmetricTree (DEFAULT: arboles simetricos)")
    rep.p("")
    rep.p("    iterations    : lo fijo el early stopping del experimento 1.")
    rep.p("    learning_rate : elegido en el experimento 1 (regla de parsimonia).")
    rep.p("    depth=6       : arboles SIMETRICOS (todas las divisiones de un nivel")
    rep.p("                    usan la misma variable y el mismo corte). Eso ya es")
    rep.p("                    regularizacion: por eso 6 alcanza donde Random Forest")
    rep.p("                    necesitaba 25.")

    modelo = CatBoostClassifier(**params)
    modelo.fit(pool_tr)

    pred_tr = modelo.predict(pool_tr).ravel()
    pred_va = modelo.predict(pool_va).ravel()

    m_tr = metricas(y_tr, pred_tr, "TRAIN")
    m_va = metricas(y_va, pred_va, "VALIDACION")

    brecha = m_tr["f1_macro"] - m_va["f1_macro"]
    rep.p(f"\n  OVERFITTING / UNDERFITTING: brecha F1 train-val = {brecha:.4f}")
    if brecha > 0.08:
        rep.p("    -> OVERFITTING: el boosting memoriza parte del train.")
    elif m_tr["f1_macro"] < 0.5:
        rep.p("    -> Posible UNDERFITTING.")
    else:
        rep.p("    -> Sin senales fuertes de over/underfitting.")

    rep.p("\n  Reporte por clase (validacion):")
    rep.p(classification_report(y_va, pred_va, labels=CLASES, zero_division=0))

    cm = confusion_matrix(y_va, pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    imp = pd.Series(modelo.get_feature_importance(pool_tr),
                    index=X_tr.columns).sort_values(ascending=False)
    rep.p("\n  IMPORTANCIA DE VARIABLES (cada variable aparece UNA vez, no")
    rep.p("  dispersa en decenas de columnas one-hot):")
    rep.p(imp.round(4).to_string())

    modelo.save_model(str(DIR_MOD / f"catboost_{nombre_variante}.cbm"))
    joblib.dump(modelo, DIR_MOD / f"catboost_{nombre_variante}.joblib")

    return {
        "etiqueta": etiqueta, "variante": nombre_variante, "modelo": modelo,
        "lr": lr, "arboles": n_arboles, "df_lr": df_lr, "i_lr": i_lr,
        "cm": cm, "imp": imp, "y_val": y_va, "pred_val": pred_va,
        "f1_train": m_tr["f1_macro"], "f1_val": m_va["f1_macro"],
        "acc_val": m_va["accuracy"], "brecha": brecha,
        "pool_tr": pool_tr, "pool_va": pool_va, "pesos": pesos,
    }


def experimento_nativo_vs_onehot(r):
    """Mismo algoritmo, mismo split, misma configuracion. Solo cambia como se
    le entregan las categoricas. Responde: cuanto aporta el manejo nativo."""
    nombre = r["variante"]
    X_tr = np.load(DIR_DATA / f"{nombre}_train_X.npy")
    y_tr = np.load(DIR_DATA / f"{nombre}_train_y.npy", allow_pickle=True)
    X_va = np.load(DIR_DATA / f"{nombre}_val_X.npy")
    y_va = np.load(DIR_DATA / f"{nombre}_val_y.npy", allow_pickle=True)

    m = CatBoostClassifier(
        iterations=r["arboles"], learning_rate=r["lr"], depth=DEPTH,
        loss_function="MultiClass", auto_class_weights=r["pesos"],
        random_seed=SEMILLA, verbose=False,
    )
    m.fit(X_tr, y_tr)
    pred = m.predict(X_va).ravel()
    f1_oh = f1_score(y_va, pred, average="macro", zero_division=0)

    rep.p(f"\n  {r['etiqueta']}  ({X_tr.shape[1]} columnas tras one-hot)")
    rep.p(f"    NATIVAS (ordered target statistics): F1 macro val = {r['f1_val']:.4f}")
    rep.p(f"    ONE-HOT                            : F1 macro val = {f1_oh:.4f}")
    rep.p(f"    Diferencia a favor de nativo       : {r['f1_val']-f1_oh:+.4f}")
    return f1_oh, X_tr.shape[1]


# ---------------------------------------------------------------- graficos
def fig_learning_rate(ra, rb):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, r in enumerate((ra, rb)):
        df, i = r["df_lr"], r["i_lr"]
        x = list(range(len(df)))

        ax = axes[0, col]
        ax.fill_between(x, df["f1_val"], df["f1_train"], color=COLOR_VAL,
                        alpha=0.10, zorder=1, label="Brecha = overfitting")
        ax.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
                markersize=6, markeredgecolor=SUPERFICIE, markeredgewidth=2,
                label="Train", zorder=3)
        ax.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
                markersize=6, markeredgecolor=SUPERFICIE, markeredgewidth=2,
                label="Validacion", zorder=3)
        ax.axvline(i, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
        ax.text(i, df["f1_train"].max() + 0.006, f"elegido lr={r['lr']}",
                ha="center", fontsize=8.5, color=TINTA, weight="bold")
        estilo(ax, f"{r['etiqueta']}: learning_rate vs generalizacion",
               ylabel="F1 macro")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in df["lr"]], fontsize=9)
        if col == 0:
            leg = ax.legend(frameon=False, fontsize=8.5, loc="center right")
            for t in leg.get_texts():
                t.set_color(TINTA_SEC)

        ax = axes[1, col]
        barras = ax.bar(x, df["arboles"], color=COLOR_TRAIN, width=0.6, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        estilo(ax, "Arboles elegidos por early stopping (od_wait=50)",
               xlabel="learning_rate", ylabel="Cantidad de arboles")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in df["lr"]], fontsize=9)
        for xi, v in zip(x, df["arboles"]):
            ax.text(xi, v + max(df["arboles"]) * 0.02, str(v), ha="center",
                    fontsize=8.5, color=TINTA, weight="bold")
        ax.grid(axis="x", visible=False)

    fig.suptitle("Experimento 1: que learning_rate usar (barrido manual)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "experimento_learning_rate.png")


def fig_matrices(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, r in zip(axes, (ra, rb)):
        cm = r["cm"]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
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
                ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%", ha="center",
                        va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else TINTA)
        ax.tick_params(colors=MUTED)
        ax.grid(False)
    fig.suptitle("CatBoost - matriz de confusion en validacion "
                 "(celda: casos y % de la fila real)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "matrices_confusion.png")


def fig_comparacion(ra, rb):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ancho = 0.32

    x = np.arange(2)
    tr = [ra["f1_train"], rb["f1_train"]]
    va = [ra["f1_val"], rb["f1_val"]]
    for datos, off, color, lab in [(tr, -ancho/2, COLOR_TRAIN, "Train"),
                                    (va, ancho/2, COLOR_VAL, "Validacion")]:
        barras = ax1.bar(x + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax1.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=TINTA)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Variante A\nauto_class_weights", "Variante B\nrecorte ARMA_FUEGO"],
                        fontsize=9)
    estilo(ax1, "Rendimiento global", ylabel="F1 macro")
    leg = ax1.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    xc = np.arange(len(CLASES))
    f1a = f1_score(ra["y_val"], ra["pred_val"], labels=CLASES, average=None, zero_division=0)
    f1b = f1_score(rb["y_val"], rb["pred_val"], labels=CLASES, average=None, zero_division=0)
    for datos, off, color, lab in [(f1a, -ancho/2, COLOR_A, "Variante A"),
                                    (f1b, ancho/2, COLOR_B, "Variante B")]:
        barras = ax2.bar(xc + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax2.text(xi + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=TINTA)
    ax2.set_xticks(xc)
    ax2.set_xticklabels([c.replace("_", "\n") for c in CLASES], fontsize=8.5)
    estilo(ax2, "Rendimiento por clase", ylabel="F1 de la clase (validacion)")
    leg = ax2.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    fig.suptitle("CatBoost V2: auto_class_weights (A) vs recorte (B)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "comparacion.png")


def fig_nativo_vs_onehot(ra, rb, oh_a, oh_b, ncols_a, ncols_b):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(2)
    ancho = 0.32
    nat = [ra["f1_val"], rb["f1_val"]]
    ohs = [oh_a, oh_b]
    for datos, off, color, lab in [(nat, -ancho/2, COLOR_NATIVO, "Categoricas nativas"),
                                    (ohs, ancho/2, COLOR_ONEHOT, "One-Hot")]:
        barras = ax.bar(x + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax.text(xi + off, v + 0.006, f"{v:.4f}", ha="center", fontsize=9,
                    color=TINTA, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Variante A\n(16 cat. -> {ncols_a} cols one-hot)",
                        f"Variante B\n(16 cat. -> {ncols_b} cols one-hot)"], fontsize=9)
    estilo(ax, "Experimento 2: cuanto aporta el manejo nativo de categoricas\n"
               "Mismo algoritmo, mismo split, misma configuracion. Solo cambia la codificacion.",
           ylabel="F1 macro (validacion)")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    return guardar_fig(fig, "experimento_nativo_vs_onehot.png")


def fig_importancia(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, r in zip(axes, (ra, rb)):
        top = r["imp"].iloc[::-1]
        barras = ax.barh(range(len(top)), top.values, color=COLOR_A, zorder=3, height=0.7)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(1.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8.5)
        estilo(ax, r["etiqueta"], xlabel="Importancia (%)")
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(top.values):
            ax.text(v + top.max() * 0.012, i, f"{v:.1f}", va="center",
                    fontsize=7.5, color=TINTA_SEC)
    fig.suptitle("Importancia de variables - CatBoost\n"
                 "Cada variable aparece UNA vez, no dispersa en columnas one-hot",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "importancia.png")


def main():
    import catboost
    rep.titulo("ENTRENAMIENTO V2 - CATBOOST (categoricas nativas)", "#")
    rep.p(f"Python {sys.version.split()[0]} | catboost {catboost.__version__} | "
          f"semilla {SEMILLA}")

    ra = entrenar_variante("variante_a", "VARIANTE A", "Balanced")
    rb = entrenar_variante("variante_b", "VARIANTE B", None)

    rep.titulo("EXPERIMENTO 2: categoricas NATIVAS vs ONE-HOT")
    rep.p("  NATIVO  -> texto crudo, CatBoost usa ordered target statistics")
    rep.p("  ONE-HOT -> las mismas columnas 0/1 que usan logistica y Random Forest")
    oh_a, nc_a = experimento_nativo_vs_onehot(ra)
    oh_b, nc_b = experimento_nativo_vs_onehot(rb)
    rep.p("")
    if ra["f1_val"] > oh_a and rb["f1_val"] > oh_b:
        rep.p("  -> El manejo nativo aporta en AMBAS variantes. Justifica haber")
        rep.p("     partido de los *_raw.csv en vez de los .npy compartidos.")
    elif ra["f1_val"] > oh_a or rb["f1_val"] > oh_b:
        rep.p("  -> El manejo nativo aporta solo en una variante. Hallazgo valido:")
        rep.p("     el beneficio depende de cuantos datos hay por categoria.")
    else:
        rep.p("  -> El manejo nativo NO aporta aqui. Hallazgo valido: significa que")
        rep.p("     One-Hot ya capturaba lo necesario en estos datos.")
    rep.p("  (En V1, con dataset.xlsx, la diferencia fue +0.0203 a favor de nativo.)")

    rep.titulo("COMPARACION DIRECTA A vs B")
    comp = pd.DataFrame({
        "Variante A (auto_class_weights)": [ra["lr"], ra["arboles"], ra["acc_val"],
                                            ra["f1_val"], ra["brecha"]],
        "Variante B (recorte)": [rb["lr"], rb["arboles"], rb["acc_val"],
                                 rb["f1_val"], rb["brecha"]],
    }, index=["learning_rate", "arboles", "Accuracy (val)", "F1 macro (val)",
              "Brecha train-val"]).round(4)
    rep.p(comp.to_string())

    rep.titulo("COMPARACION CONTRA EL CATBOOST DE V1 (dataset.xlsx)")
    rep.p("  V1 (con filas fabricadas, lr=0.06, 1500 arboles, sin pesos de clase):")
    rep.p("      Accuracy val 0.6402 | F1 macro val 0.6392 | brecha 0.142")
    rep.p(f"  V2 Variante A: Accuracy val {ra['acc_val']:.4f} | "
          f"F1 macro val {ra['f1_val']:.4f} | brecha {ra['brecha']:.4f}")
    rep.p(f"  V2 Variante B: Accuracy val {rb['acc_val']:.4f} | "
          f"F1 macro val {rb['f1_val']:.4f} | brecha {rb['brecha']:.4f}")
    rep.p("")
    rep.p("  Igual que con la logistica: NO son comparables como 'quien es mejor'.")
    rep.p("  El val de V1 tambien contiene filas fabricadas y clases artificialmente")
    rep.p("  equilibradas (25% c/u), asi que su F1 macro esta inflado.")

    rep.titulo("COMPARATIVA DE MODELOS V2 (validacion)")
    filas = {}
    for et, r in [("CatBoost", ra), ("CatBoost", rb)]:
        filas[f"{et} {r['etiqueta'][-1]}"] = [r["acc_val"], r["f1_val"], r["brecha"]]
    for arch, et in [("randomforest_variante_a.joblib", "RandomForest A"),
                      ("randomforest_variante_b.joblib", "RandomForest B"),
                      ("logistica_variante_a.joblib", "Logistica A"),
                      ("logistica_variante_b.joblib", "Logistica B")]:
        ruta = DIR_MOD / arch
        if not ruta.exists():
            continue
        obj = joblib.load(ruta)
        m = obj["modelo"] if isinstance(obj, dict) else obj
        nv = "variante_a" if arch.endswith("_a.joblib") else "variante_b"
        Xv = np.load(DIR_DATA / f"{nv}_val_X.npy")
        yv = np.load(DIR_DATA / f"{nv}_val_y.npy", allow_pickle=True)
        Xt = np.load(DIR_DATA / f"{nv}_train_X.npy")
        yt = np.load(DIR_DATA / f"{nv}_train_y.npy", allow_pickle=True)
        pv, pt = m.predict(Xv), m.predict(Xt)
        f1v = f1_score(yv, pv, average="macro", zero_division=0)
        f1t = f1_score(yt, pt, average="macro", zero_division=0)
        filas[et] = [accuracy_score(yv, pv), f1v, f1t - f1v]
    comp2 = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)",
                                       "Brecha train-val"]).round(4)
    rep.p(comp2.T.to_string())

    rep.titulo("GRAFICOS")
    fig_learning_rate(ra, rb)
    fig_matrices(ra, rb)
    fig_comparacion(ra, rb)
    fig_nativo_vs_onehot(ra, rb, oh_a, oh_b, nc_a, nc_b)
    fig_importancia(ra, rb)

    rep.titulo("FIN", "#")
    rep.p("Guardado: modelos/catboost_variante_{a,b}.joblib (y .cbm nativo)")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
