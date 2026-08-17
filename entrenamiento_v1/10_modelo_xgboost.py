"""
MODELO 5: XGBOOST (gradient boosting con arboles asimetricos)
===============================================================

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

De ahi sale el peso optimo de cada hoja en forma CERRADA (no por busqueda):

    w_j* = - SUM_{i en j} g_i / ( SUM_{i en j} h_i + lambda )

y la ganancia de una division:

    Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                   - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

Lo conceptualmente distinto: gamma y lambda estan DENTRO de la formula de
ganancia. La regularizacion no es un castigo posterior, es parte del criterio
que decide si una division vale la pena.

DIFERENCIAS CON LOS MODELOS 1-4
---------------------------------
  vs Regresion Logistica : la logistica traza UNA frontera lineal sobre las
        463 columnas. XGBoost suma cientos de arboles: fronteras no lineales
        con interacciones. Misma perdida (entropia cruzada), distinta forma.
  vs Random Forest : RF es BAGGING (arboles independientes en paralelo que se
        promedian, sin funcion de perdida global). XGBoost es BOOSTING
        (secuencial, cada arbol mira el error del anterior).
  vs CatBoost : mismo paradigma, decisiones opuestas. XGBoost hace arboles
        ASIMETRICOS (cada rama parte por donde le conviene); CatBoost los hace
        SIMETRICOS (todo un nivel usa la misma variable y el mismo corte).
        Consecuencia: XGBoost tiene mas capacidad por arbol y depende mucho
        mas de max_depth, por eso aca ese parametro necesita su experimento.
  vs MLP : el MLP tambien minimiza la misma entropia cruzada, pero ajustando
        pesos por backpropagation en vez de sumando arboles.

POR QUE USA train_X.npy Y NO train_raw.csv
-------------------------------------------
XGBoost necesita numeros: no maneja texto como CatBoost. Asi que usa el mismo
train_X.npy (one-hot + escalado) que la logistica, Random Forest y el MLP.
El escalado le es indiferente (a un arbol no le importa la escala), pero no
molesta, y usar la MISMA matriz que los otros tres hace la comparacion justa.

DESBALANCE
-----------
No se aplica ningun peso de clase: dataset.xlsx ya viene balanceado
artificialmente (~25% cada clase), igual criterio que en la logistica y el
Random Forest de esta misma linea. (En entrenamiento_v2, donde los datos son
reales y desbalanceados, si se usa sample_weight.)

EXPERIMENTOS
  Exp. 1: learning_rate, con el early stopping decidiendo cuantos arboles.
  Exp. 2: max_depth (el hiperparametro critico por los arboles asimetricos).
  Exp. 3: XGBoost vs CatBoost cara a cara.

Salida:
  modelos/modelo5_xgboost.joblib
  resultados/modelos/informe_modelo5_xgboost.txt
  Graficos/modelo5_xgboost_*.png
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

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"
DIR_FIG = BASE / "Graficos"
DIR_REP.mkdir(parents=True, exist_ok=True)
DIR_FIG.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

LEARNING_RATES = [0.03, 0.06, 0.10, 0.20, 0.30]
PROFUNDIDADES = [3, 4, 6, 8, 10, 12]
MAX_ARBOLES = 1500
PACIENCIA = 50

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


rep = Reporte(DIR_REP / "informe_modelo5_xgboost.txt")


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
    ruta = DIR_FIG / f"modelo5_xgboost_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Grafico: Graficos/{ruta.name}")
    return ruta


def entrenar(X_tr, yc_tr, X_va, yc_va, le, y_tr, y_va, lr, depth):
    m = xgb.XGBClassifier(
        n_estimators=MAX_ARBOLES,
        learning_rate=lr,
        max_depth=depth,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        early_stopping_rounds=PACIENCIA,
        random_state=SEMILLA,
        n_jobs=-1,
        verbosity=0,
    )
    m.fit(X_tr, yc_tr, eval_set=[(X_va, yc_va)], verbose=False)
    p_tr = le.inverse_transform(m.predict(X_tr))
    p_va = le.inverse_transform(m.predict(X_va))
    f1_tr = f1_score(y_tr, p_tr, average="macro", zero_division=0)
    f1_va = f1_score(y_va, p_va, average="macro", zero_division=0)
    n_arb = (m.best_iteration + 1) if m.best_iteration is not None else MAX_ARBOLES
    return m, f1_tr, f1_va, n_arb, p_tr, p_va


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n  --- {nombre} ---")
    a = accuracy_score(y_true, y_pred)
    rep.p(f"  Accuracy         : {a:.4f}")
    rep.p(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    f = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rep.p(f"  F1 (macro)       : {f:.4f}")
    return {"accuracy": a, "f1_macro": f}


def experimento_lr(datos):
    rep.titulo("EXPERIMENTO 1: ¿que learning_rate usar? (barrido manual)")
    rep.p("  eta controla cuanto corrige CADA arbol nuevo.")
    rep.p("  Bajo -> aprende despacio, necesita muchos arboles, generaliza mejor.")
    rep.p("  Alto -> aprende rapido, pocos arboles, riesgo de sobreajustar.")
    rep.p(f"  El early stopping (paciencia {PACIENCIA}) decide cuantos arboles usar")
    rep.p("  en cada caso; queda documentado cuantos eligio.")
    rep.p("")
    rep.p(f"  {'lr':>6} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for lr in LEARNING_RATES:
        _, f1_tr, f1_va, n_arb, _, _ = entrenar(*datos, lr, 6)
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
    rep.p("  Misma regla de parsimonia que en Random Forest, CatBoost y MLP.")
    return LEARNING_RATES[i], df, i


def experimento_depth(datos, lr):
    rep.titulo("EXPERIMENTO 2: ¿que max_depth usar? (el parametro critico)")
    rep.p("  XGBoost hace arboles ASIMETRICOS: cada rama parte por donde le")
    rep.p("  conviene. Eso le da mas capacidad por arbol que CatBoost, que los")
    rep.p("  hace simetricos, y por eso max_depth pesa mucho mas aca.")
    rep.p(f"  (CatBoost se quedo con depth=6, su default, sin necesitar barrido.)")
    rep.p("")
    rep.p(f"  {'max_depth':>10} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | "
          f"{'brecha':>7}")
    rep.p("  " + "-" * 56)

    filas = []
    for dp in PROFUNDIDADES:
        _, f1_tr, f1_va, n_arb, _, _ = entrenar(*datos, lr, dp)
        rep.p(f"  {dp:>10} | {n_arb:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | "
              f"{f1_tr-f1_va:7.4f}")
        filas.append({"depth": dp, "arboles": n_arb, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    umbral = df["f1_val"].max() * 0.99
    i = int(df.index[df["f1_val"] >= umbral].tolist()[0])
    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {df['f1_val'].max():.4f} "
          f"(depth={df.loc[df['f1_val'].idxmax(),'depth']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : max_depth={PROFUNDIDADES[i]} -> "
          f"F1 val {df.loc[i,'f1_val']:.4f}")
    rep.p("  Razon: el arbol MAS SIMPLE dentro del 1% del mejor.")
    rep.p("")
    rep.p("  LIMITACION DE ESTA REGLA (hay que decirla): mira solo el F1 val, no")
    rep.p("  la brecha. Si las profundidades chicas rinden genuinamente peor en")
    rep.p("  validacion, la regla NO protege contra el sobreajuste. Ver la")
    rep.p("  columna 'brecha' de la tabla de arriba antes de confiar en el valor")
    rep.p("  elegido.")
    return PROFUNDIDADES[i], df, i


def fig_barrido(df, i, campo, etiqueta_x, titulo, nombre):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(df))
    ax.fill_between(x, df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax.plot(x, df["f1_train"], "o-", color=COLOR_TRAIN, linewidth=2, markersize=7,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, label="Train", zorder=3)
    ax.plot(x, df["f1_val"], "o-", color=COLOR_VAL, linewidth=2, markersize=7,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, label="Validacion", zorder=3)
    ax.axvline(i, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(i, df["f1_train"].max() + 0.008, f"elegido: {df.loc[i, campo]}",
            ha="center", fontsize=9, color=TINTA, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n{a} arb." for v, a in zip(df[campo], df["arboles"])],
                       fontsize=8.5)
    estilo(ax, titulo, xlabel=etiqueta_x, ylabel="F1 macro")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    return guardar_fig(fig, nombre)


def fig_matriz(y_va, pred_va, f1):
    cm = confusion_matrix(y_va, pred_va, labels=CLASES)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASES, fontsize=9)
    ax.set_xlabel("Prediccion", color=TINTA_SEC, fontsize=10)
    ax.set_ylabel("Real", color=TINTA_SEC, fontsize=10)
    ax.set_title(f"Matriz de confusion - XGBoost - Validacion (F1 macro {f1:.3f})\n"
                 "celda: casos y % de la fila real",
                 color=TINTA, fontsize=11, loc="left", weight="bold")
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if cmn[i, j] > 0.5 else TINTA)
    ax.tick_params(colors=MUTED)
    ax.grid(False)
    return guardar_fig(fig, "fig1_matriz_confusion.png")


def fig_xgb_vs_cat(f1_xgb, f1_cat, depth_xgb):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    barras = ax.bar([0, 1], [f1_xgb, f1_cat], color=[COLOR_XGB, COLOR_CAT],
                    width=0.5, zorder=3)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(2)
    for xi, v in zip([0, 1], [f1_xgb, f1_cat]):
        ax.text(xi, v + max(f1_xgb, f1_cat) * 0.02, f"{v:.4f}", ha="center",
                fontsize=11, color=TINTA, weight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"XGBoost\narboles asimetricos (depth={depth_xgb})\n+ one-hot",
                        "CatBoost\narboles simetricos (depth=6)\n+ categoricas nativas"],
                       fontsize=9)
    ax.set_ylim(0, max(f1_xgb, f1_cat) * 1.3)
    estilo(ax, "Experimento 3: los dos boostings cara a cara\n"
               "Mismo paradigma y mismo split. Cambia como se construyen los arboles.",
           ylabel="F1 macro (validacion)")
    ax.grid(axis="x", visible=False)
    return guardar_fig(fig, "fig2_xgb_vs_catboost.png")


def fig_importancia(imp, n=19):
    top = imp.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    barras = ax.barh(range(len(top)), top.values, color="#2a78d6", zorder=3, height=0.7)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(1.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(s)[:44] for s in top.index], fontsize=8.5)
    estilo(ax, "Importancia de columnas - XGBoost\n"
               "Al usar one-hot, cada variable aparece repartida en varias columnas "
               "(a diferencia de CatBoost)",
           xlabel="Importancia (gain normalizado)")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.012, i, f"{v:.4f}", va="center",
                fontsize=7.5, color=TINTA_SEC)
    return guardar_fig(fig, "fig3_importancia.png")


def f1_catboost_del_informe():
    """Lee el F1 de CatBoost de su propio informe en vez de hardcodearlo."""
    ruta = DIR_REP / "informe_modelo3_catboost.txt"
    if not ruta.exists():
        return None
    partes = ruta.read_text(encoding="utf-8").split("--- VALIDACION ---")
    if len(partes) < 2:
        return None
    for linea in partes[1].splitlines()[:6]:
        if "F1 (macro)" in linea:
            return float(linea.split(":")[-1])
    return None


def main():
    rep.titulo("MODELO 5: XGBOOST (boosting con arboles asimetricos)", "#")
    rep.p(f"Python {sys.version.split()[0]} | xgboost {xgb.__version__} | "
          f"semilla {SEMILLA}")

    X_tr = np.load(DIR_PROC / "train_X.npy")
    y_tr = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_va = np.load(DIR_PROC / "val_X.npy")
    y_va = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)
    columnas = joblib.load(DIR_MOD / "preprocesador.joblib").get_feature_names_out()

    rep.p(f"\nTrain: {X_tr.shape}  |  Val: {X_va.shape}")
    rep.p("Fuente: train_X.npy (el mismo que usan logistica, Random Forest y MLP).")
    rep.p("XGBoost necesita numeros: no maneja texto como CatBoost.")

    le = LabelEncoder().fit(y_tr)
    yc_tr, yc_va = le.transform(y_tr), le.transform(y_va)
    rep.p(f"Etiquetas codificadas a 0..{len(CLASES)-1}: {list(le.classes_)}")

    dist = pd.Series(y_tr).value_counts()
    rep.p("\nDistribucion en train (dataset.xlsx ya viene balanceado):")
    for k in CLASES:
        v = int(dist.get(k, 0))
        rep.p(f"  {k:<18} {v:6,}  ({100*v/len(y_tr):5.2f}%)")
    rep.p("Por eso NO se usa sample_weight, igual criterio que en la logistica")
    rep.p("y el Random Forest de esta misma linea.")

    datos = (X_tr, yc_tr, X_va, yc_va, le, y_tr, y_va)
    lr, df_lr, i_lr = experimento_lr(datos)
    depth, df_dp, i_dp = experimento_depth(datos, lr)

    modelo, f1_tr, f1_va, n_arb, pred_tr, pred_va = entrenar(*datos, lr, depth)

    rep.titulo("MODELO FINAL (hiperparametros explicitos)")
    rep.p(f"  learning_rate = {lr}            (experimento 1)")
    rep.p(f"  max_depth = {depth}                  (experimento 2)")
    rep.p(f"  n_estimators = {n_arb}          (lo fijo el early stopping)")
    rep.p(f"  objective = 'multi:softprob'    (LogLoss multinomial)")
    rep.p(f"  eval_metric = 'mlogloss'")
    rep.p(f"  tree_method = 'hist'            (DEFAULT desde XGBoost 2.0)")
    rep.p(f"  reg_lambda = 1.0                (DEFAULT: el lambda de Omega)")
    rep.p(f"  gamma = 0.0                     (DEFAULT: ganancia minima por hoja)")
    rep.p(f"  subsample = 1.0                 (DEFAULT: usa todas las filas)")
    rep.p(f"  colsample_bytree = 1.0          (DEFAULT: usa todas las columnas)")
    rep.p(f"  min_child_weight = 1            (DEFAULT: hessiano minimo por hoja)")
    rep.p(f"  random_state = {SEMILLA}")
    rep.p("")
    rep.p("  reg_lambda y gamma son los que aparecen en Omega(f) del docstring:")
    rep.p("  lambda suaviza el valor de cada hoja, gamma exige una ganancia")
    rep.p("  minima para que una division exista. Se dejan en su default porque")
    rep.p("  el control de capacidad ya se hizo con max_depth (experimento 2).")

    rep.titulo("METRICAS")
    m_train = metricas(y_tr, pred_tr, "TRAIN")
    m_val = metricas(y_va, pred_va, "VALIDACION")

    rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING")
    diff = m_train["f1_macro"] - m_val["f1_macro"]
    rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia     : {diff:.4f}")
    if diff > 0.08:
        rep.p("  -> OVERFITTING: el boosting memoriza parte del train.")
    elif m_train["f1_macro"] < 0.5:
        rep.p("  -> Posible UNDERFITTING.")
    else:
        rep.p("  -> Sin senales fuertes de over/underfitting.")

    rep.titulo("REPORTE POR CLASE (VALIDACION)")
    rep.p(classification_report(y_va, pred_va, labels=CLASES, zero_division=0))

    rep.titulo("MATRIZ DE CONFUSION (VALIDACION)")
    cm = confusion_matrix(y_va, pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    rep.titulo("EXPERIMENTO 3: XGBOOST vs CATBOOST (los dos boostings)")
    f1_cat = f1_catboost_del_informe()
    rep.p("  Mismo paradigma (arboles secuenciales que corrigen residuos) y")
    rep.p("  mismo split. Las diferencias:")
    rep.p("    - XGBoost : arboles ASIMETRICOS, aproximacion de 2do orden")
    rep.p("                (gradiente + hessiano), categoricas via one-hot.")
    rep.p("    - CatBoost: arboles SIMETRICOS, categoricas nativas con")
    rep.p("                ordered target statistics.")
    rep.p("")
    if f1_cat is None:
        rep.p("  CatBoost todavia no se entreno con esta preparacion: se omite.")
    else:
        rep.p(f"  XGBoost  (max_depth={depth}) : F1 macro val = {m_val['f1_macro']:.4f}")
        rep.p(f"  CatBoost (depth=6)      : F1 macro val = {f1_cat:.4f}")
        rep.p(f"  Diferencia a favor de XGBoost: {m_val['f1_macro']-f1_cat:+.4f}")
        rep.p("")
        if depth > 6:
            rep.p("  XGBoost eligio arboles MAS PROFUNDOS que CatBoost. Vale la pena")
            rep.p("  mirarlo junto con la brecha: si gana F1 val a costa de")
            rep.p(f"  memorizar ({diff:.3f} de brecha), no es que aprenda mejor,")
            rep.p("  es que se le permite ajustar mas.")
        fig_xgb_vs_cat(m_val["f1_macro"], f1_cat, depth)

    rep.p("")
    rep.p("  ADVERTENCIA METODOLOGICA (aplica tambien a CatBoost):")
    rep.p("  el early stopping usa el set de VALIDACION como eval_set, o sea que")
    rep.p("  el numero de arboles se elige mirando el mismo set sobre el que")
    rep.p("  despues se reporta el F1. Eso sesga el resultado hacia arriba. Los")
    rep.p("  modelos SIN early stopping (logistica, Random Forest) no tienen esa")
    rep.p("  ventaja, asi que la comparacion favorece levemente a los boostings.")
    rep.p("  El numero limpio saldra recien del TEST sellado.")

    rep.titulo("IMPORTANCIA DE COLUMNAS")
    imp = pd.Series(modelo.feature_importances_, index=columnas).sort_values(ascending=False)
    rep.p("  (con one-hot, cada variable aparece repartida en varias columnas;")
    rep.p("   CatBoost, al usar categoricas nativas, la muestra una sola vez)")
    rep.p("")
    for k, v in imp.head(20).items():
        rep.p(f"    {str(k)[:52]:<54} {v:.5f}")

    rep.titulo("COMPARATIVA DE LOS 5 MODELOS DE V1 (validacion)")
    filas = {"XGBoost": [m_val["accuracy"], m_val["f1_macro"], diff]}
    for nombre, arch in [("Regresion Logistica", "modelo1_logistica.joblib"),
                          ("Random Forest", "modelo2_randomforest.joblib"),
                          ("MLP (red neuronal)", "modelo4_mlp.joblib")]:
        ruta = DIR_MOD / arch
        if not ruta.exists():
            continue
        m = joblib.load(ruta)
        p_v, p_t = m.predict(X_va), m.predict(X_tr)
        f1v = f1_score(y_va, p_v, average="macro", zero_division=0)
        filas[nombre] = [accuracy_score(y_va, p_v), f1v,
                          f1_score(y_tr, p_t, average="macro", zero_division=0) - f1v]
    tabla = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)",
                                       "Brecha train-val"]).T
    if f1_cat is not None:
        rep.p(f"  (CatBoost usa categoricas nativas, no train_X.npy, asi que no se")
        rep.p(f"   recalcula aca. Segun su informe: F1 macro val {f1_cat:.4f})")
    tabla = tabla.sort_values("F1 macro (val)", ascending=False).round(4)
    rep.p("")
    rep.p(tabla.to_string())
    rep.p("")
    rep.p("  RECORDATORIO: el F1 macro solo no alcanza para elegir. La columna de")
    rep.p("  brecha dice cuanto memoriza cada uno. El set de TEST sigue sellado.")

    rep.titulo("GRAFICOS")
    fig_barrido(df_lr, i_lr, "lr", "learning_rate",
                "Experimento 1: learning_rate vs generalizacion",
                "fig0_learningrate.png")
    fig_barrido(df_dp, i_dp, "depth", "max_depth",
                "Experimento 2: max_depth (el parametro critico de XGBoost)",
                "fig0b_maxdepth.png")
    fig_matriz(y_va, pred_va, m_val["f1_macro"])
    fig_importancia(imp)

    joblib.dump(modelo, DIR_MOD / "modelo5_xgboost.joblib")

    rep.titulo("FIN MODELO 5", "#")
    rep.p("Guardado: modelos/modelo5_xgboost.joblib")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
