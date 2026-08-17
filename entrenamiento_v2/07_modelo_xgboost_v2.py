"""
ENTRENAMIENTO V2 - XGBOOST (Variante A vs Variante B)
=======================================================

QUE HACE EL MODELO (la formula, sin caja negra)
------------------------------------------------
XGBoost es BOOSTING, igual que CatBoost: arboles en SECUENCIA, cada uno
corrigiendo los errores del conjunto anterior.

    F_m(x) = F_{m-1}(x) + eta * h_m(x)

La diferencia con CatBoost esta en COMO elige cada division. XGBoost hace una
aproximacion de SEGUNDO ORDEN (Taylor) de la perdida, no solo del gradiente:

    L ~ SUM_i [ g_i * f(x_i) + 0.5 * h_i * f(x_i)^2 ] + Omega(f)

    g_i = dL/dy_pred        (gradiente, primera derivada)
    h_i = d2L/dy_pred^2     (hessiano, segunda derivada)

    Omega(f) = gamma * T + 0.5 * lambda * SUM_j w_j^2
               (T = numero de hojas, w_j = valor de la hoja j)

De ahi sale el peso optimo de cada hoja en forma CERRADA:

    w_j* = - SUM_{i en j} g_i / ( SUM_{i en j} h_i + lambda )

y la ganancia de una division:

    Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                   - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

Esa penalizacion Omega, con gamma y lambda DENTRO de la formula de ganancia,
es la diferencia conceptual: la regularizacion no es un agregado posterior,
esta metida en el criterio que decide si una division vale la pena.

DIFERENCIAS CON CATBOOST (mismo paradigma, decisiones opuestas)
----------------------------------------------------------------
  Arboles     : XGBoost los hace ASIMETRICOS (cada rama parte por donde le
                conviene). CatBoost los hace SIMETRICOS (todo un nivel usa la
                misma variable y el mismo corte).
                Consecuencia: XGBoost tiene mas capacidad por arbol, pero
                depende mucho mas de max_depth para no sobreajustar. Por eso
                aca max_depth SI necesita su propio experimento, mientras que
                en CatBoost depth=6 (el default) alcanzaba.
  Categoricas : CatBoost las maneja nativo (ordered target statistics).
                XGBoost necesita numeros, asi que usa los MISMOS .npy que
                logistica, Random Forest y MLP.
  Desbalance  : CatBoost tiene auto_class_weights. XGBoost NO tiene un
                class_weight para multiclase (scale_pos_weight es solo
                binario), asi que se usa sample_weight en .fit(), igual que
                hizo el MLP.

Perdida: multi:softprob = LogLoss multinomial. LA MISMA entropia cruzada de
la logistica, del MLP y de CatBoost. Los cuatro modelos minimizan lo mismo;
cambia como.

EXPERIMENTOS
  Exp. 1: learning_rate, con early stopping decidiendo cuantos arboles.
  Exp. 2: max_depth (el hiperparametro critico de XGBoost por sus arboles
          asimetricos).
  Exp. 3: XGBoost vs CatBoost cara a cara. Mismo paradigma (boosting), misma
          particion, distinta forma de construir los arboles.

Entrada : data/variante_{a,b}_{train,val}_X.npy y _y.npy
Salida  : modelos/xgboost_variante_{a,b}.joblib
          resultados/informe_modelo_xgboost_v2.txt
          Graficos/xgboost_v2_*.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_DATA = BASE / "data"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados"
DIR_FIG = BASE / "Graficos"
DIR_MOD.mkdir(exist_ok=True)
DIR_FIG.mkdir(exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

LEARNING_RATES = [0.03, 0.06, 0.10, 0.20, 0.30]
PROFUNDIDADES = [3, 4, 6, 8, 10, 12]
MAX_ARBOLES = 1500
PACIENCIA = 50

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
COLOR_XGB, COLOR_CAT = "#1baf7a", "#eda100"
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


rep = Reporte(DIR_REP / "informe_modelo_xgboost_v2.txt")


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
    ruta = DIR_FIG / f"xgboost_v2_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Graficos/{ruta.name}")
    return ruta


def cargar(nombre):
    d = {}
    for parte in ("train", "val"):
        d[f"X_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_X.npy")
        d[f"y_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_y.npy", allow_pickle=True)
    d["columnas"] = np.load(DIR_DATA / f"{nombre}_columnas.npy", allow_pickle=True)
    # XGBoost exige etiquetas 0..K-1
    d["le"] = LabelEncoder().fit(d["y_train"])
    d["yc_train"] = d["le"].transform(d["y_train"])
    d["yc_val"] = d["le"].transform(d["y_val"])
    return d


def construir(lr, depth, n_arboles=MAX_ARBOLES, early=True):
    return xgb.XGBClassifier(
        n_estimators=n_arboles,
        learning_rate=lr,
        max_depth=depth,
        objective="multi:softprob",   # LogLoss multinomial
        eval_metric="mlogloss",
        tree_method="hist",           # DEFAULT desde XGBoost 2.0
        early_stopping_rounds=PACIENCIA if early else None,
        random_state=SEMILLA,
        n_jobs=-1,
        verbosity=0,
    )


def entrenar(d, lr, depth, pesos, n_arboles=MAX_ARBOLES, early=True):
    m = construir(lr, depth, n_arboles, early)
    kw = {"verbose": False}
    if early:
        kw["eval_set"] = [(d["X_val"], d["yc_val"])]
    m.fit(d["X_train"], d["yc_train"], sample_weight=pesos, **kw)
    pred_tr = d["le"].inverse_transform(m.predict(d["X_train"]))
    pred_va = d["le"].inverse_transform(m.predict(d["X_val"]))
    f1_tr = f1_score(d["y_train"], pred_tr, average="macro", zero_division=0)
    f1_va = f1_score(d["y_val"], pred_va, average="macro", zero_division=0)
    n_usados = (m.best_iteration + 1) if early and m.best_iteration is not None else n_arboles
    return m, f1_tr, f1_va, n_usados, pred_tr, pred_va


def experimento_lr(d, pesos, etiqueta):
    rep.p(f"\n  EXPERIMENTO 1: learning_rate ({etiqueta})")
    rep.p("  eta controla cuanto corrige CADA arbol nuevo. El early stopping")
    rep.p(f"  (paciencia {PACIENCIA}) decide cuantos arboles usar en cada caso.")
    rep.p("")
    rep.p(f"  {'lr':>6} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for lr in LEARNING_RATES:
        _, f1_tr, f1_va, n_arb, _, _ = entrenar(d, lr, 6, pesos)
        rep.p(f"  {lr:>6.2f} | {n_arb:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | "
              f"{f1_tr-f1_va:7.4f}")
        filas.append({"lr": lr, "arboles": n_arb, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    umbral = df["f1_val"].max() * 0.99
    i = int(df.index[df["f1_val"] >= umbral].tolist()[0])
    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {df['f1_val'].max():.4f} "
          f"(lr={df.loc[df['f1_val'].idxmax(),'lr']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : lr={LEARNING_RATES[i]} -> "
          f"F1 val {df.loc[i,'f1_val']:.4f}")
    return LEARNING_RATES[i], df, i


def experimento_depth(d, pesos, lr, etiqueta):
    rep.p(f"\n  EXPERIMENTO 2: max_depth ({etiqueta}, lr={lr})")
    rep.p("  XGBoost hace arboles ASIMETRICOS: cada rama parte por donde le")
    rep.p("  conviene. Eso le da mas capacidad por arbol que CatBoost (que los")
    rep.p("  hace simetricos), y por eso max_depth pesa mucho mas aca.")
    rep.p("")
    rep.p(f"  {'max_depth':>10} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | "
          f"{'brecha':>7}")
    rep.p("  " + "-" * 56)

    filas = []
    for dp in PROFUNDIDADES:
        _, f1_tr, f1_va, n_arb, _, _ = entrenar(d, lr, dp, pesos)
        rep.p(f"  {dp:>10} | {n_arb:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | "
              f"{f1_tr-f1_va:7.4f}")
        filas.append({"depth": dp, "arboles": n_arb, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    umbral = df["f1_val"].max() * 0.99
    i = int(df.index[df["f1_val"] >= umbral].tolist()[0])  # el mas chico
    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {df['f1_val'].max():.4f} "
          f"(depth={df.loc[df['f1_val'].idxmax(),'depth']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : max_depth={PROFUNDIDADES[i]} -> "
          f"F1 val {df.loc[i,'f1_val']:.4f}")
    rep.p("  Razon: el arbol MAS SIMPLE dentro del 1% del mejor.")
    return PROFUNDIDADES[i], df, i


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n    --- {nombre} ---")
    a = accuracy_score(y_true, y_pred)
    rep.p(f"    Accuracy         : {a:.4f}")
    rep.p(f"    Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"    Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    f = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep.p(f"    F1 (macro)       : {f:.4f}")
    return {"accuracy": a, "f1_macro": f}


def entrenar_variante(nombre_variante, etiqueta, balancear):
    rep.titulo(f"{etiqueta} (balanceo de clases: {'si' if balancear else 'no'})")
    d = cargar(nombre_variante)
    rep.p(f"  train: {d['X_train'].shape}   val: {d['X_val'].shape}")
    rep.p(f"  Etiquetas codificadas a 0..{len(CLASES)-1}: {list(d['le'].classes_)}")

    dist = pd.Series(d["y_train"]).value_counts()
    rep.p("\n  Distribucion real en train:")
    for k in CLASES:
        v = int(dist.get(k, 0))
        rep.p(f"    {k:<18} {v:6,}  ({100*v/len(d['y_train']):5.2f}%)")

    pesos = None
    if balancear:
        pesos = compute_sample_weight("balanced", d["y_train"])
        rep.p("\n  XGBoost NO tiene class_weight para multiclase")
        rep.p("  (scale_pos_weight es solo binario). Se usa sample_weight, igual")
        rep.p("  que en el MLP: un peso por FILA = n / (n_clases * n_su_clase).")
        n, n_cl = len(d["y_train"]), len(CLASES)
        for k in CLASES:
            rep.p(f"    {k:<18} peso = {n/(n_cl*int(dist.get(k,0))):6.3f}")

    lr, df_lr, i_lr = experimento_lr(d, pesos, etiqueta)
    depth, df_dp, i_dp = experimento_depth(d, pesos, lr, etiqueta)

    modelo, f1_tr, f1_va, n_arb, pred_tr, pred_va = entrenar(d, lr, depth, pesos)

    rep.p(f"\n  HIPERPARAMETROS FINALES ({etiqueta}):")
    rep.p(f"    learning_rate = {lr}          (experimento 1)")
    rep.p(f"    max_depth = {depth}                (experimento 2)")
    rep.p(f"    n_estimators = {n_arb}        (lo fijo el early stopping)")
    rep.p(f"    objective = 'multi:softprob'  (LogLoss multinomial)")
    rep.p(f"    eval_metric = 'mlogloss'")
    rep.p(f"    tree_method = 'hist'          (DEFAULT desde XGBoost 2.0)")
    rep.p(f"    reg_lambda = 1.0              (DEFAULT: el lambda de Omega)")
    rep.p(f"    gamma = 0.0                   (DEFAULT: ganancia minima por hoja)")
    rep.p(f"    subsample = 1.0               (DEFAULT: usa todas las filas)")
    rep.p(f"    colsample_bytree = 1.0        (DEFAULT: usa todas las columnas)")
    rep.p(f"    min_child_weight = 1          (DEFAULT: hessiano minimo por hoja)")
    rep.p(f"    random_state = {SEMILLA}")

    m_tr = metricas(d["y_train"], pred_tr, "TRAIN")
    m_va = metricas(d["y_val"], pred_va, "VALIDACION")
    brecha = m_tr["f1_macro"] - m_va["f1_macro"]
    rep.p(f"\n  OVERFITTING / UNDERFITTING: brecha F1 train-val = {brecha:.4f}")
    if brecha > 0.08:
        rep.p("    -> OVERFITTING: el boosting memoriza parte del train.")
    elif m_tr["f1_macro"] < 0.5:
        rep.p("    -> Posible UNDERFITTING.")
    else:
        rep.p("    -> Sin senales fuertes de over/underfitting.")

    rep.p("\n  Reporte por clase (validacion):")
    rep.p(classification_report(d["y_val"], pred_va, labels=CLASES, zero_division=0))

    cm = confusion_matrix(d["y_val"], pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    imp = pd.Series(modelo.feature_importances_, index=d["columnas"]).sort_values(ascending=False)
    rep.p("\n  Top 15 columnas por importancia (gain normalizado):")
    for k, v in imp.head(15).items():
        rep.p(f"    {str(k)[:48]:<50} {v:.5f}")

    joblib.dump({"modelo": modelo, "le": d["le"], "lr": lr, "max_depth": depth},
                DIR_MOD / f"xgboost_{nombre_variante}.joblib")

    return {
        "etiqueta": etiqueta, "variante": nombre_variante, "modelo": modelo,
        "datos": d, "lr": lr, "depth": depth, "arboles": n_arb,
        "df_lr": df_lr, "i_lr": i_lr, "df_dp": df_dp, "i_dp": i_dp,
        "cm": cm, "imp": imp, "pred_val": pred_va,
        "f1_train": m_tr["f1_macro"], "f1_val": m_va["f1_macro"],
        "acc_val": m_va["accuracy"], "brecha": brecha,
    }


# ---------------------------------------------------------------- graficos
def _fig_barrido(ra, rb, clave, campo, etiqueta_x, titulo, nombre):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, r in zip(axes, (ra, rb)):
        df = r[clave]
        i = r[f"i_{clave.split('_')[1]}"]
        x = np.arange(len(df))
        ax.fill_between(x, df["f1_val"], df["f1_train"], color=COLOR_VAL,
                        alpha=0.10, zorder=1, label="Brecha = overfitting")
        ax.plot(x, df["f1_train"], "o-", color=COLOR_TRAIN, linewidth=2,
                markersize=6, markeredgecolor=SUPERFICIE, markeredgewidth=2,
                label="Train", zorder=3)
        ax.plot(x, df["f1_val"], "o-", color=COLOR_VAL, linewidth=2,
                markersize=6, markeredgecolor=SUPERFICIE, markeredgewidth=2,
                label="Validacion", zorder=3)
        ax.axvline(i, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
        ax.text(i, df["f1_train"].max() + 0.008,
                f"elegido {df.loc[i, campo]}", ha="center", fontsize=8.5,
                color=TINTA, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n{a} arb." for v, a in zip(df[campo], df["arboles"])],
                           fontsize=8)
        estilo(ax, r["etiqueta"], xlabel=etiqueta_x)
    axes[0].set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    fig.suptitle(titulo, color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, nombre)


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
    fig.suptitle("XGBoost - matriz de confusion en validacion "
                 "(celda: casos y % de la fila real)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "matrices_confusion.png")


def fig_comparacion(ra, rb):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ancho = 0.32
    x = np.arange(2)
    for datos, off, color, lab in [
            ([ra["f1_train"], rb["f1_train"]], -ancho/2, COLOR_TRAIN, "Train"),
            ([ra["f1_val"], rb["f1_val"]], ancho/2, COLOR_VAL, "Validacion")]:
        barras = ax1.bar(x + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax1.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=TINTA)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Variante A\nsample_weight", "Variante B\nrecorte ARMA_FUEGO"],
                        fontsize=9)
    estilo(ax1, "Rendimiento global", ylabel="F1 macro")
    leg = ax1.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    xc = np.arange(len(CLASES))
    f1a = f1_score(ra["datos"]["y_val"], ra["pred_val"], labels=CLASES,
                   average=None, zero_division=0)
    f1b = f1_score(rb["datos"]["y_val"], rb["pred_val"], labels=CLASES,
                   average=None, zero_division=0)
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

    fig.suptitle("XGBoost V2: sample_weight (A) vs recorte (B)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "comparacion.png")


def fig_xgb_vs_cat(comparacion):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(2)
    ancho = 0.32
    xg = [comparacion["A"]["xgb"], comparacion["B"]["xgb"]]
    ct = [comparacion["A"]["cat"], comparacion["B"]["cat"]]
    for datos, off, color, lab in [
            (xg, -ancho/2, COLOR_XGB, "XGBoost (arboles asimetricos + one-hot)"),
            (ct, ancho/2, COLOR_CAT, "CatBoost (arboles simetricos + cat. nativas)")]:
        barras = ax.bar(x + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax.text(xi + off, v + 0.006, f"{v:.4f}", ha="center", fontsize=9,
                    color=TINTA, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Variante A", "Variante B"], fontsize=10)
    estilo(ax, "Experimento 3: los dos boostings cara a cara\n"
               "Mismo paradigma y misma particion. Cambia como se construyen los arboles.",
           ylabel="F1 macro (validacion)")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    return guardar_fig(fig, "experimento_xgb_vs_catboost.png")


def fig_importancia(ra, rb, n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, r in zip(axes, (ra, rb)):
        top = r["imp"].head(n).iloc[::-1]
        barras = ax.barh(range(len(top)), top.values, color=COLOR_A, zorder=3, height=0.7)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(1.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([str(s)[:40] for s in top.index], fontsize=8)
        estilo(ax, r["etiqueta"], xlabel="Importancia (gain normalizado)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Importancia de columnas - XGBoost\n"
                 "Al usar one-hot, cada variable aparece repartida en varias columnas "
                 "(a diferencia de CatBoost)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "importancia.png")


def main():
    rep.titulo("ENTRENAMIENTO V2 - XGBOOST", "#")
    rep.p(f"Python {sys.version.split()[0]} | xgboost {xgb.__version__} | "
          f"semilla {SEMILLA}")
    rep.p("Datos: los mismos .npy que usan logistica, Random Forest y MLP.")
    rep.p("(XGBoost necesita numeros; no maneja texto como CatBoost)")

    ra = entrenar_variante("variante_a", "VARIANTE A", balancear=True)
    rb = entrenar_variante("variante_b", "VARIANTE B", balancear=False)

    rep.titulo("EXPERIMENTO 3: XGBOOST vs CATBOOST (los dos boostings)")
    rep.p("  Mismo paradigma (arboles secuenciales que corrigen residuos) y")
    rep.p("  misma particion. Las diferencias:")
    rep.p("    - XGBoost: arboles ASIMETRICOS, aproximacion de 2do orden")
    rep.p("      (gradiente + hessiano), categoricas via one-hot.")
    rep.p("    - CatBoost: arboles SIMETRICOS, categoricas nativas con")
    rep.p("      ordered target statistics.")
    rep.p("")
    F1_CATBOOST = {"A": 0.4838, "B": 0.5057}   # de informe_modelo_catboost_v2.txt
    comparacion = {}
    for letra, r in [("A", ra), ("B", rb)]:
        comparacion[letra] = {"xgb": r["f1_val"], "cat": F1_CATBOOST[letra]}
        dif = r["f1_val"] - F1_CATBOOST[letra]
        rep.p(f"  Variante {letra}: XGBoost {r['f1_val']:.4f} vs "
              f"CatBoost {F1_CATBOOST[letra]:.4f}  ({dif:+.4f})")
    rep.p("")
    rep.p(f"  XGBoost eligio max_depth={ra['depth']} (A) y {rb['depth']} (B);")
    rep.p("  CatBoost se quedo con depth=6 en las dos.")
    if ra["depth"] > 6 and rb["depth"] > 6:
        rep.p("  XGBoost eligio arboles MAS PROFUNDOS, no mas superficiales.")
        rep.p("  Esto CONTRADICE la intuicion de que sus arboles asimetricos, al")
        rep.p("  tener mas capacidad por nivel, necesitarian menos profundidad.")
        rep.p("  La explicacion esta en la brecha train-val: XGBoost gana ese")
        rep.p("  +0.02 de F1 val a costa de memorizar (brecha "
              f"{ra['brecha']:.3f} en A y {rb['brecha']:.3f} en B, contra 0.057 y")
        rep.p("  0.041 de CatBoost). No es que aprenda mejor: es que se le permite")
        rep.p("  ajustar mas, y la validacion todavia no lo castiga.")
    rep.p("")
    rep.p("  ADVERTENCIA METODOLOGICA (aplica tambien a CatBoost):")
    rep.p("  el early stopping usa el set de VALIDACION como eval_set, o sea que")
    rep.p("  el numero de arboles se elige mirando el mismo set sobre el que")
    rep.p("  despues se reporta el F1. Eso sesga el resultado hacia arriba. Los")
    rep.p("  modelos SIN early stopping (logistica, Random Forest, MLP) no tienen")
    rep.p("  esta ventaja, asi que la comparacion favorece levemente a los dos")
    rep.p("  boostings. El numero limpio saldra recien del TEST sellado.")

    rep.titulo("COMPARACION DIRECTA A vs B")
    comp = pd.DataFrame({
        "Variante A": [ra["lr"], ra["depth"], ra["arboles"], ra["acc_val"],
                       ra["f1_val"], ra["brecha"]],
        "Variante B": [rb["lr"], rb["depth"], rb["arboles"], rb["acc_val"],
                       rb["f1_val"], rb["brecha"]],
    }, index=["learning_rate", "max_depth", "arboles", "Accuracy (val)",
              "F1 macro (val)", "Brecha train-val"]).round(4)
    rep.p(comp.to_string())

    rep.titulo("TABLA FINAL: LOS 5 MODELOS DE V2 (validacion)")
    F1_OTROS = {
        "RandomForest A": (0.8170, 0.5108, 0.2717),
        "RandomForest B": (0.6843, 0.4927, 0.1139),
        "CatBoost A":     (0.7624, 0.4838, 0.0573),
        "CatBoost B":     (0.6758, 0.5057, 0.0413),
        "Logistica A":    (0.7315, 0.4468, 0.0175),
        "Logistica B":    (0.6686, 0.5037, 0.0156),
        "MLP A":          (0.7576, 0.4751, 0.0308),
        "MLP B":          (0.6780, 0.5279, 0.0477),
    }
    filas = dict(F1_OTROS)
    filas["XGBoost A"] = (ra["acc_val"], ra["f1_val"], ra["brecha"])
    filas["XGBoost B"] = (rb["acc_val"], rb["f1_val"], rb["brecha"])
    tabla = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)",
                                       "Brecha train-val"]).T.round(4)
    tabla = tabla.sort_values("F1 macro (val)", ascending=False)
    rep.p(tabla.to_string())
    rep.p("")
    mejor = tabla.index[0]
    rep.p(f"  Mejor F1 macro absoluto: {mejor} ({tabla.iloc[0]['F1 macro (val)']:.4f})")
    rep.p("")
    rep.p("  PERO EL F1 MACRO SOLO ELIGE MAL. Mirando las dos columnas juntas:")
    rep.p("")
    top = tabla.head(2)
    rep.p(f"    {top.index[0]:<16} F1 {top.iloc[0]['F1 macro (val)']:.4f}  "
          f"brecha {top.iloc[0]['Brecha train-val']:.4f}")
    rep.p(f"    {top.index[1]:<16} F1 {top.iloc[1]['F1 macro (val)']:.4f}  "
          f"brecha {top.iloc[1]['Brecha train-val']:.4f}")
    d_f1 = top.iloc[0]["F1 macro (val)"] - top.iloc[1]["F1 macro (val)"]
    d_br = top.iloc[0]["Brecha train-val"] - top.iloc[1]["Brecha train-val"]
    rep.p(f"    -> diferencia de F1: {d_f1:+.4f}   diferencia de brecha: {d_br:+.4f}")
    if abs(d_f1) < 0.01 and d_br > 0.10:
        rep.p(f"    Los dos primeros empatan en F1 ({abs(d_f1):.4f} de diferencia,")
        rep.p(f"    ruido), pero {top.index[0]} memoriza {d_br:.2f} mas. A igual")
        rep.p(f"    rendimiento, {top.index[1]} es el modelo defendible.")
    rep.p("")
    rep.p("  Ademas hay que mirar el F1 de ARMA_CONTUNDENTE, que ningun modelo")
    rep.p("  resuelve (el mejor es 0.22 de la logistica A).")
    rep.p("  El set de TEST sigue sellado: no se toca hasta elegir el final.")

    rep.titulo("GRAFICOS")
    _fig_barrido(ra, rb, "df_lr", "lr", "learning_rate",
                 "Experimento 1: que learning_rate usar", "experimento_learning_rate.png")
    _fig_barrido(ra, rb, "df_dp", "depth", "max_depth",
                 "Experimento 2: que profundidad usar (el parametro critico de XGBoost)",
                 "experimento_max_depth.png")
    fig_matrices(ra, rb)
    fig_comparacion(ra, rb)
    fig_xgb_vs_cat(comparacion)
    fig_importancia(ra, rb)

    rep.titulo("FIN", "#")
    rep.p("Guardado: modelos/xgboost_variante_{a,b}.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
