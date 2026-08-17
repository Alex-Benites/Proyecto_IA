"""
MODELO 3: CATBOOST (gradient boosting con categoricas nativas)
================================================================

DIFERENCIA CONCEPTUAL CON LOS MODELOS 1 Y 2
--------------------------------------------
Modelos 1 y 2 son BAGGING / lineal:
  - Regresion Logistica: ajusta pesos, todos a la vez.
  - Random Forest: 300 arboles INDEPENDIENTES en paralelo, se promedian.

CatBoost es BOOSTING: los arboles se construyen EN SECUENCIA y cada uno
se entrena para corregir los errores que dejo el conjunto anterior.

    F_m(x) = F_{m-1}(x) + eta * h_m(x)

donde h_m es un arbol nuevo ajustado al GRADIENTE NEGATIVO de la funcion
de perdida respecto a las predicciones actuales, y eta es learning_rate.
Es decir: cada arbol nuevo mira "donde me estoy equivocando mas ahora" y
se especializa en eso. Por eso boosting suele ganarle a bagging, y por eso
tambien es mas propenso a sobreajustar si no se regula.

Funcion de perdida (multiclase): MultiClass = LogLoss multinomial
    L = - suma_i log( p(y_i = clase_verdadera | x_i) )
la misma entropia cruzada de la regresion logistica, pero minimizada
agregando arboles en vez de ajustando pesos lineales.

POR QUE ESTE MODELO NO USA train_X.npy
---------------------------------------
Los modelos 1 y 2 comparten train_X.npy: 475 columnas, todo numerico,
generado con One-Hot + StandardScaler. Necesitan eso porque:
  - LogisticRegression calcula w*x: no puede multiplicar un peso por "GUAYAS".
  - RandomForestClassifier de sklearn tampoco acepta texto.

CatBoost SI acepta las categoricas como texto, y esa es justamente su
ventaja: en vez de One-Hot (que convierte 'canton' en 211 columnas binarias
y le esconde al modelo que esas 211 columnas son la misma variable), usa
ORDERED TARGET STATISTICS: reemplaza cada categoria por una estadistica de
la tasa de cada clase en esa categoria, calculada SOLO con las filas
anteriores en un orden aleatorio. Ese "solo las anteriores" es lo que
evita el data leakage que tendria un target-encoding ingenuo (donde la
fila usaria su propia etiqueta para construir su propia feature).

Por eso este script parte de train_raw.csv / val_raw.csv, que 03_preparacion.py
guardo ANTES de aplicar el ColumnTransformer: mismo split, mismas 14.630 /
3.135 filas, misma semilla, pero con las categorias todavia en texto.

Dos ajustes propios de este modelo:
  1. 'edad' tiene NaN. CatBoost los soporta nativamente, pero se imputan con
     la MEDIANA DEL TRAIN (mismo criterio y mismo valor que usaron los
     modelos 1 y 2) para que la comparacion sea justa.
  2. NO se escala: a un arbol le da igual la escala de una variable. El
     escalado en los modelos 1 y 2 existia solo por la regresion logistica.

NO ES CAJA NEGRA:
  - Algoritmo publicado y documentado (Prokhorenkova et al., 2018).
  - Funcion de perdida explicita (MultiClass / LogLoss).
  - Hiperparametros elegidos con experimentos manuales documentados abajo,
    NO con GridSearchCV ni busqueda automatica.
  - Expone feature_importances_.

EXPERIMENTOS INCLUIDOS
  Experimento 1: learning_rate vs profundidad del bosque (curva de validacion
                 usando el propio early stopping de CatBoost, documentado).
  Experimento 2: categoricas NATIVAS vs ONE-HOT (mismo algoritmo, mismo
                 split) -> responde "¿cuanto aporta realmente el manejo
                 nativo de categoricas de CatBoost?".

Salida:
  modelos/modelo3_catboost.joblib
  resultados/modelos/informe_modelo3_catboost.txt
  Graficos/modelo3_catboost_fig0_learningrate.png
  Graficos/modelo3_catboost_fig7_nativo_vs_onehot.png
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

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"
DIR_FIG = BASE / "Graficos"
DIR_REP.mkdir(parents=True, exist_ok=True)
DIR_FIG.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Las 16 columnas categoricas. CatBoost EXIGE que se declaren explicitamente:
# si no, las trataria como numericas y se perderia todo el proposito.
# ('mes' se excluye por fuga; ver el comentario en 03_preparacion.py)
COLS_CATEGORICAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "dia_semana",
    "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad", "es_fin_semana"]

COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
COLOR_NATIVO, COLOR_ONEHOT = "#1baf7a", "#eda100"
TINTA, TINTA_SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SUPERFICIE = "#e1e0d9", "#fcfcfb"

LEARNING_RATES = [0.03, 0.06, 0.10, 0.20, 0.30]


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


rep = Reporte(DIR_REP / "informe_modelo3_catboost.txt")


def cargar_datos():
    """Carga los splits en TEXTO (antes del One-Hot) y prepara los tipos."""
    tr = pd.read_csv(DIR_PROC / "train_raw.csv")
    va = pd.read_csv(DIR_PROC / "val_raw.csv")

    # 'edad': imputar con la MEDIANA DEL TRAIN (mismo criterio que modelos 1 y 2)
    mediana = tr["edad"].median()
    n_tr = int(tr["edad"].isna().sum())
    n_va = int(va["edad"].isna().sum())
    tr["edad"] = tr["edad"].fillna(mediana)
    va["edad"] = va["edad"].fillna(mediana)

    # CatBoost exige que las categoricas no tengan NaN: se marcan como texto.
    for c in COLS_CATEGORICAS:
        tr[c] = tr[c].astype(str)
        va[c] = va[c].astype(str)

    X_tr, y_tr = tr[COLS_CATEGORICAS + COLS_NUMERICAS], tr["y"]
    X_va, y_va = va[COLS_CATEGORICAS + COLS_NUMERICAS], va["y"]
    return X_tr, y_tr, X_va, y_va, mediana, n_tr, n_va


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    rep.p(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  F1 (macro)       : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def base_estilo(ax, titulo, xlabel="", ylabel=""):
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
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def guardar_fig(fig, nombre):
    ruta = DIR_FIG / f"modelo3_catboost_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Grafico: Graficos/{ruta.name}")


# ---------------------------------------------------------------------------
# EXPERIMENTO 1 - learning_rate
# ---------------------------------------------------------------------------
def experimento_learning_rate(pool_tr, pool_va, y_tr, y_va):
    """Barrido MANUAL de learning_rate.

    Se deja que el propio early stopping de CatBoost elija cuantos arboles
    usar para cada learning_rate (od_type='Iter', od_wait=50): esto NO es
    busqueda automatica de modelo, es el criterio de parada estandar del
    boosting, y queda documentado cuantas iteraciones eligio cada uno.
    """
    rep.titulo("EXPERIMENTO 1: ¿que learning_rate usar? (barrido manual)")
    rep.p("  learning_rate (eta) controla cuanto corrige CADA arbol nuevo.")
    rep.p("  Bajo -> aprende despacio, necesita muchos arboles, generaliza mejor.")
    rep.p("  Alto -> aprende rapido, pocos arboles, riesgo de sobreajustar.")
    rep.p("")
    rep.p(f"  {'lr':>6} | {'arboles':>8} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for lr in LEARNING_RATES:
        m = CatBoostClassifier(
            iterations=1500, learning_rate=lr, depth=6,
            loss_function="MultiClass", random_seed=SEMILLA,
            od_type="Iter", od_wait=50, verbose=False,
        )
        m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        f1_tr = f1_score(y_tr, m.predict(pool_tr).ravel(), average="macro", zero_division=0)
        f1_va = f1_score(y_va, m.predict(pool_va).ravel(), average="macro", zero_division=0)
        n_arb = m.tree_count_
        rep.p(f"  {lr:>6.2f} | {n_arb:>8} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
        filas.append({"lr": lr, "arboles": n_arb, "f1_train": f1_tr,
                      "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas)

    # Mismo criterio de parsimonia que se uso en Random Forest: el
    # learning_rate MAS BAJO cuyo F1 este dentro del 1% del mejor.
    # Un lr mas bajo = correcciones mas suaves = modelo mas estable.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    i_elegido = df.index[df["f1_val"] >= umbral].tolist()[0]
    lr_elegido = float(df.loc[i_elegido, "lr"])
    n_arboles = int(df.loc[i_elegido, "arboles"])

    rep.p("")
    rep.p(f"  Mejor F1 absoluto : {f1_mejor:.4f} (lr={df.loc[df['f1_val'].idxmax(), 'lr']})")
    rep.p(f"  Umbral del 1%     : {umbral:.4f}")
    rep.p(f"  ELEGIDO (lr mas bajo dentro del umbral): lr={lr_elegido} con "
          f"{n_arboles} arboles -> F1 val {df.loc[i_elegido, 'f1_val']:.4f}")
    rep.p("  Mismo criterio de parsimonia que se aplico en Random Forest.")

    graficar_learning_rate(df, i_elegido)
    return lr_elegido, n_arboles


def graficar_learning_rate(df, i_elegido):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(df))

    ax1.fill_between(list(x), df["f1_val"], df["f1_train"], color=COLOR_VAL,
                     alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax1.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
             markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
             label="Entrenamiento", zorder=3)
    ax1.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
             markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
             label="Validacion", zorder=3)
    ax1.axvline(i_elegido, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax1.text(i_elegido, df["f1_train"].max() + 0.008,
             f"elegido: lr={df.loc[i_elegido, 'lr']}", ha="center",
             fontsize=9, color=TINTA, weight="bold")
    base_estilo(ax1, "learning_rate vs generalizacion",
                xlabel="learning_rate", ylabel="F1 macro")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(v) for v in df["lr"]], fontsize=9)
    leg = ax1.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    barras = ax2.bar(list(x), df["arboles"], color=COLOR_TRAIN, width=0.6, zorder=3)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(2)
    base_estilo(ax2, "Arboles que necesito cada learning_rate\n"
                     "(elegidos por early stopping, od_wait=50)",
                xlabel="learning_rate", ylabel="Cantidad de arboles")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(v) for v in df["lr"]], fontsize=9)
    for xi, v in zip(x, df["arboles"]):
        ax2.text(xi, v + max(df["arboles"]) * 0.02, str(v), ha="center",
                 fontsize=9, color=TINTA, weight="bold")

    guardar_fig(fig, "fig0_learningrate.png")


# ---------------------------------------------------------------------------
# EXPERIMENTO 2 - categoricas nativas vs One-Hot
# ---------------------------------------------------------------------------
def experimento_nativo_vs_onehot(lr, n_arboles, pool_tr, pool_va, y_tr, y_va,
                                  f1_nativo):
    """Mismo algoritmo, mismo split, misma configuracion: solo cambia como
    se le entregan las categoricas.

    Responde directamente: ¿cuanto aporta el manejo nativo de CatBoost?
    """
    rep.titulo("EXPERIMENTO 2: categoricas NATIVAS vs ONE-HOT")
    rep.p("  Mismo CatBoost, mismos hiperparametros, mismo split.")
    rep.p("  Unica diferencia: como recibe las 17 columnas categoricas.")
    rep.p("    NATIVO  -> texto crudo, CatBoost usa ordered target statistics")
    rep.p("    ONE-HOT -> las 475 columnas 0/1 que usan los modelos 1 y 2")
    rep.p("")

    X_tr_oh = np.load(DIR_PROC / "train_X.npy")
    y_tr_oh = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_va_oh = np.load(DIR_PROC / "val_X.npy")
    y_va_oh = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)

    m_oh = CatBoostClassifier(
        iterations=n_arboles, learning_rate=lr, depth=6,
        loss_function="MultiClass", random_seed=SEMILLA, verbose=False,
    )
    m_oh.fit(X_tr_oh, y_tr_oh)
    pred_oh = m_oh.predict(X_va_oh).ravel()
    f1_oh = f1_score(y_va_oh, pred_oh, average="macro", zero_division=0)
    acc_oh = accuracy_score(y_va_oh, pred_oh)

    rep.p(f"  CatBoost + categoricas NATIVAS : F1 macro val = {f1_nativo:.4f}")
    rep.p(f"  CatBoost + ONE-HOT (475 cols)  : F1 macro val = {f1_oh:.4f}")
    rep.p(f"  Diferencia a favor de nativo   : {f1_nativo - f1_oh:+.4f}")
    rep.p("")
    if f1_nativo > f1_oh:
        rep.p("  -> El manejo nativo SI aporta. Justifica haber usado train_raw.csv")
        rep.p("     en vez del train_X.npy compartido con los modelos 1 y 2.")
    else:
        rep.p("  -> El manejo nativo NO aporta en este dataset. Hallazgo valido:")
        rep.p("     significa que One-Hot ya capturaba lo necesario aqui.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    barras = ax.bar([0, 1], [f1_nativo, f1_oh],
                    color=[COLOR_NATIVO, COLOR_ONEHOT], width=0.5, zorder=3)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(2)
    base_estilo(ax, "CatBoost: categoricas nativas vs One-Hot\n"
                    "Mismo algoritmo y mismo split. Solo cambia la codificacion.",
                ylabel="F1 macro (validacion)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nativas\n(ordered target statistics)",
                        "One-Hot\n(475 columnas 0/1)"], fontsize=9)
    ax.set_ylim(0, max(f1_nativo, f1_oh) * 1.25)
    for xi, v in zip([0, 1], [f1_nativo, f1_oh]):
        ax.text(xi, v + max(f1_nativo, f1_oh) * 0.03, f"{v:.4f}", ha="center",
                fontsize=11, color=TINTA, weight="bold")
    ax.grid(axis="x", visible=False)
    guardar_fig(fig, "fig7_nativo_vs_onehot.png")
    return f1_oh, acc_oh


# ---------------------------------------------------------------------------
def main():
    rep.titulo("MODELO 3: CATBOOST (categoricas nativas)", "#")
    import catboost
    rep.p(f"Python {sys.version.split()[0]} | catboost {catboost.__version__} | "
          f"semilla {SEMILLA}")

    X_tr, y_tr, X_va, y_va, mediana, n_tr, n_va = cargar_datos()
    rep.p(f"\nFuente: train_raw.csv / val_raw.csv (ANTES del One-Hot)")
    rep.p(f"Train: {X_tr.shape}  |  Val: {X_va.shape}")
    rep.p(f"Columnas categoricas declaradas: {len(COLS_CATEGORICAS)}")
    rep.p(f"Columnas numericas: {COLS_NUMERICAS}")
    rep.p(f"'edad': {n_tr} NaN en train y {n_va} en val -> imputados con la "
          f"mediana del train ({mediana})")
    rep.p("NO se escala: a un arbol le da igual la escala de una variable.")

    idx_cat = [X_tr.columns.get_loc(c) for c in COLS_CATEGORICAS]
    pool_tr = Pool(X_tr, y_tr, cat_features=idx_cat)
    pool_va = Pool(X_va, y_va, cat_features=idx_cat)

    lr, n_arboles = experimento_learning_rate(pool_tr, pool_va, y_tr, y_va)

    # --- Modelo final ---
    rep.titulo("MODELO FINAL (hiperparametros explicitos)")
    params = dict(
        iterations=n_arboles,       # fijado por el early stopping del experimento 1
        learning_rate=lr,           # elegido en el experimento 1
        depth=6,                    # profundidad de los arboles simetricos
        loss_function="MultiClass",  # LogLoss multinomial (entropia cruzada)
        random_seed=SEMILLA,
        verbose=False,
    )
    for k, v in params.items():
        rep.p(f"  {k} = {v}")
    rep.p("")
    rep.p("  Justificacion:")
    rep.p("    iterations    : cantidad de arboles del boosting. La fijo el early")
    rep.p("                    stopping del experimento 1, no una eleccion a dedo.")
    rep.p("    learning_rate : cuanto corrige cada arbol nuevo. Ver experimento 1.")
    rep.p("    depth=6       : CatBoost usa arboles SIMETRICOS (todas las divisiones")
    rep.p("                    de un nivel usan la misma variable y el mismo corte).")
    rep.p("                    Eso ya actua como regularizacion, por eso 6 alcanza")
    rep.p("                    donde Random Forest necesitaba 25.")
    rep.p("    loss_function : MultiClass = LogLoss multinomial, la MISMA entropia")
    rep.p("                    cruzada de la regresion logistica. La diferencia no")
    rep.p("                    esta en QUE minimiza, sino en COMO: sumando arboles")
    rep.p("                    en vez de ajustando pesos lineales.")

    modelo = CatBoostClassifier(**params)
    modelo.fit(pool_tr)

    pred_tr = modelo.predict(pool_tr).ravel()
    pred_va = modelo.predict(pool_va).ravel()

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

    f1_oh, acc_oh = experimento_nativo_vs_onehot(
        lr, n_arboles, pool_tr, pool_va, y_tr, y_va, m_val["f1_macro"])

    rep.titulo("IMPORTANCIA DE VARIABLES")
    imp = pd.Series(modelo.get_feature_importance(pool_tr),
                    index=X_tr.columns).sort_values(ascending=False)
    rep.p("  (a diferencia de los modelos 1 y 2, aqui cada variable aparece UNA")
    rep.p("   vez, no dispersa en decenas de columnas one-hot)")
    rep.p("")
    rep.p(imp.round(4).to_string())
    graficar_importancia(imp)

    rep.titulo("COMPARATIVA DE LOS 3 MODELOS (validacion)")
    filas = {"CatBoost (nativo)": [m_val["accuracy"], m_val["f1_macro"]]}
    for nombre, arch in [("Regresion Logistica", "modelo1_logistica.joblib"),
                          ("Random Forest", "modelo2_randomforest.joblib")]:
        ruta = DIR_MOD / arch
        if ruta.exists():
            m = joblib.load(ruta)
            Xv = np.load(DIR_PROC / "val_X.npy")
            yv = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)
            p = m.predict(Xv)
            filas[nombre] = [accuracy_score(yv, p),
                             f1_score(yv, p, average="macro", zero_division=0)]
    comp = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)"]).round(4)
    rep.p(comp.to_string())

    modelo.save_model(str(DIR_MOD / "modelo3_catboost.cbm"))
    joblib.dump(modelo, DIR_MOD / "modelo3_catboost.joblib")

    rep.titulo("FIN MODELO 3", "#")
    rep.p("Guardado: modelos/modelo3_catboost.joblib (y .cbm nativo)")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


def graficar_importancia(imp):
    top = imp.head(19).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    barras = ax.barh(range(len(top)), top.values, color="#2a78d6",
                     zorder=3, height=0.7)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(1.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    base_estilo(ax, "Importancia de variables - CatBoost\n"
                    "Cada variable aparece UNA vez, no dispersa en columnas one-hot.",
                xlabel="Importancia (%)")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.012, i, f"{v:.2f}", va="center",
                fontsize=8, color=TINTA_SEC)
    guardar_fig(fig, "fig6_importancia.png")


if __name__ == "__main__":
    main()
