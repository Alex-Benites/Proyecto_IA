"""
ENTRENAMIENTO V2 - RED NEURONAL (MLP) : sklearn vs Keras, Variante A vs B
==========================================================================

COMO APRENDE (la formula, sin caja negra)
------------------------------------------
Forward pass (495 entradas -> 4 salidas en la Variante A):

    a1 = ReLU(W1 . x  + b1)      capa oculta 1
    a2 = ReLU(W2 . a1 + b2)      capa oculta 2 (si existe)
    z  = W_out . a_ultima + b_out
    p  = softmax(z)              4 probabilidades

ReLU(v) = max(0, v). Se usa porque su derivada es 0 o 1: no se "desvanece"
al propagar el gradiente hacia atras, problema que si tenia la sigmoide.

Perdida: entropia cruzada multinomial + regularizacion L2

    L = - SUM_i log P(y_i = clase_verdadera | x_i)  +  alpha * ||W||^2

Es LA MISMA perdida de la regresion logistica y de CatBoost. Las tres
diferencias estan en COMO se minimiza y con que capacidad:
    logistica -> pesos lineales directos, frontera lineal
    CatBoost  -> suma de arboles
    MLP       -> capas ocultas con ReLU, frontera NO lineal, backpropagation

Backpropagation: se calcula dL/dW en cada capa con la regla de la cadena,
desde la salida hacia la entrada, y Adam actualiza los pesos.

UNA EPOCA = una pasada completa por las filas de entrenamiento. Aqui el
concepto SI aplica (en Random Forest y CatBoost no existia).

DESBALANCE: el problema particular del MLP
-------------------------------------------
MLPClassifier NO tiene el parametro class_weight (a diferencia de
LogisticRegression y RandomForestClassifier). Desde sklearn 1.8 si acepta
sample_weight en .fit(), que es matematicamente equivalente: en vez de un
peso por clase, un peso por FILA. Se calcula con
compute_sample_weight('balanced', y), que asigna a cada fila el peso de su
clase, n / (n_clases * n_clase).

Keras, en cambio, si tiene class_weight nativo en model.fit(). Esa es una de
las cosas que compara el Experimento 3.

EXPERIMENTOS
  Exp. 1: ARQUITECTURA (hidden_layer_sizes). El default (100,) no es un
          detalle tecnico: es la red ENTERA decidida sin escribirla. En los
          otros modelos los defaults ajustaban COMO aprende; aqui define QUE
          ES el modelo. Por eso es el primer experimento.
  Exp. 2: ALPHA (fuerza de la regularizacion L2).
  Exp. 3: sklearn vs KERAS con la MISMA arquitectura, mismo alpha, mismo
          optimizador, mismo batch, misma semilla. Responde: ¿el resultado
          depende de la libreria o de los hiperparametros?

Entrada : data/variante_{a,b}_{train,val}_X.npy y _y.npy
Salida  : modelos/mlp_variante_{a,b}.joblib
          modelos/mlp_keras_variante_{a,b}.keras
          resultados/informe_modelo_mlp_v2.txt
          Graficos/mlp_v2_*.png
"""

import os
import sys
import warnings
from pathlib import Path

# Silenciar los logs de arranque de TensorFlow (no afectan el resultado).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from utilidades import ModeloEtiquetado

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_DATA = BASE / "data"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados"
DIR_FIG = BASE / "Graficos"
DIR_MOD.mkdir(exist_ok=True)
DIR_FIG.mkdir(exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# EXPERIMENTO 1: de menos a mas capacidad, para ver donde empieza a sobreajustar.
ARQUITECTURAS = [
    (50,),            # 1 capa chica
    (100,),           # 1 capa - EL DEFAULT de sklearn
    (200,),           # 1 capa grande
    (100, 50),        # 2 capas
    (200, 100),       # 2 capas grandes
    (100, 50, 25),    # 3 capas (embudo)
]
# EXPERIMENTO 2: fuerza de la regularizacion L2.
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

MAX_EPOCAS = 300
BATCH = 200          # sklearn 'auto' = min(200, n). Se fija para igualar Keras.
LR_INICIAL = 0.001   # default de Adam en ambas librerias
PACIENCIA = 15
FRACCION_VAL_INTERNA = 0.1

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
COLOR_SK, COLOR_KERAS = "#1baf7a", "#eda100"
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


rep = Reporte(DIR_REP / "informe_modelo_mlp_v2.txt")


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
    ruta = DIR_FIG / f"mlp_v2_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Graficos/{ruta.name}")
    return ruta


def n_parametros(arq, n_entradas, n_salidas=4):
    """Pesos + sesgos de la red. Sirve para la regla de parsimonia."""
    capas = [n_entradas] + list(arq) + [n_salidas]
    return sum(capas[i] * capas[i + 1] + capas[i + 1] for i in range(len(capas) - 1))


def cargar(nombre):
    d = {}
    for parte in ("train", "val"):
        d[f"X_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_X.npy")
        d[f"y_{parte}"] = np.load(DIR_DATA / f"{nombre}_{parte}_y.npy", allow_pickle=True)
    d["columnas"] = np.load(DIR_DATA / f"{nombre}_columnas.npy", allow_pickle=True)
    return d


def construir_sklearn(arq, alpha):
    return ModeloEtiquetado(MLPClassifier(
        hidden_layer_sizes=arq,
        alpha=alpha,
        activation="relu",
        solver="adam",
        learning_rate_init=LR_INICIAL,
        batch_size=BATCH,
        max_iter=MAX_EPOCAS,
        early_stopping=True,
        validation_fraction=FRACCION_VAL_INTERNA,
        n_iter_no_change=PACIENCIA,
        random_state=SEMILLA,
    ))


def entrenar_sk(arq, alpha, d, pesos):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = construir_sklearn(arq, alpha)
        m.fit(d["X_train"], d["y_train"], sample_weight=pesos)
    f1_tr = f1_score(d["y_train"], m.predict(d["X_train"]), average="macro", zero_division=0)
    f1_va = f1_score(d["y_val"], m.predict(d["X_val"]), average="macro", zero_division=0)
    return m, f1_tr, f1_va


# ---------------------------------------------------------------- experimentos
def experimento_arquitectura(d, pesos, etiqueta):
    n_ent = d["X_train"].shape[1]
    rep.p(f"\n  EXPERIMENTO 1: arquitectura ({etiqueta})")
    rep.p("  El default de sklearn es (100,). No es un detalle: es la red entera.")
    rep.p("")
    rep.p(f"  {'arquitectura':>16} | {'parametros':>11} | {'epocas':>6} | "
          f"{'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 76)

    filas = []
    for arq in ARQUITECTURAS:
        m, f1_tr, f1_va = entrenar_sk(arq, ALPHAS[0], d, pesos)
        npar = n_parametros(arq, n_ent)
        ep = len(m.loss_curve_)
        et = str(arq) + (" <- default" if arq == (100,) else "")
        rep.p(f"  {et:>16} | {npar:>11,} | {ep:>6} | {f1_tr:9.4f} | "
              f"{f1_va:9.4f} | {f1_tr-f1_va:7.4f}")
        filas.append({"arq": arq, "etiqueta": str(arq), "parametros": npar,
                      "epocas": ep, "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    # Parsimonia: entre las que estan dentro del 1% del mejor F1 val, la de
    # MENOS parametros (red mas chica = menos capacidad de memorizar).
    umbral = df["f1_val"].max() * 0.99
    cand = df[df["f1_val"] >= umbral]
    i = int(cand["parametros"].idxmin())
    arq_elegida = ARQUITECTURAS[i]
    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {df['f1_val'].max():.4f} "
          f"({df.loc[df['f1_val'].idxmax(),'etiqueta']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDA               : {arq_elegida} con {df.loc[i,'parametros']:,} "
          f"parametros -> F1 val {df.loc[i,'f1_val']:.4f}")
    rep.p("  Razon: es la red con MENOS parametros dentro del 1% del mejor.")
    return arq_elegida, df, i


def experimento_alpha(d, pesos, arq, etiqueta):
    rep.p(f"\n  EXPERIMENTO 2: alpha (regularizacion L2) ({etiqueta})")
    rep.p(f"  Arquitectura fija en {arq}. El default de sklearn es alpha=0.0001.")
    rep.p("")
    rep.p(f"  {'alpha':>8} | {'epocas':>6} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for a in ALPHAS:
        m, f1_tr, f1_va = entrenar_sk(arq, a, d, pesos)
        et = f"{a}" + (" <-def" if a == 0.0001 else "")
        rep.p(f"  {et:>8} | {len(m.loss_curve_):>6} | {f1_tr:9.4f} | "
              f"{f1_va:9.4f} | {f1_tr-f1_va:7.4f}")
        filas.append({"alpha": a, "epocas": len(m.loss_curve_),
                      "f1_train": f1_tr, "f1_val": f1_va})

    df = pd.DataFrame(filas)
    # Parsimonia: el alpha MAS ALTO dentro del 1% del mejor (a igual
    # rendimiento, se prefiere mas regularizacion).
    umbral = df["f1_val"].max() * 0.99
    i = int(df.index[df["f1_val"] >= umbral].tolist()[-1])
    alpha_elegido = ALPHAS[i]
    rep.p("")
    rep.p(f"  Mejor F1 val absoluto : {df['f1_val'].max():.4f} "
          f"(alpha={df.loc[df['f1_val'].idxmax(),'alpha']})")
    rep.p(f"  Umbral de parsimonia  : {umbral:.4f} (99% del mejor)")
    rep.p(f"  ELEGIDO               : alpha={alpha_elegido} -> "
          f"F1 val {df.loc[i,'f1_val']:.4f}")
    rep.p("  Razon: a igual rendimiento se prefiere MAS regularizacion.")
    return alpha_elegido, df, i


def alpha_keras_equivalente(alpha_sk, n_train):
    """Convierte el alpha de sklearn al l2() de Keras.

    Verificado leyendo el codigo fuente de sklearn
    (_multilayer_perceptron.py, dentro de _backprop):

        sw_sum = n_samples if sample_weight is None else sample_weight.sum()
        loss += (0.5 * self.alpha) * values / sw_sum

    LA CLAVE: _backprop se llama UNA VEZ POR BATCH y recibe el batch, no el
    dataset completo. Asi que 'sw_sum' vale ~batch_size (200), NO el total
    de filas. Keras, en cambio:

        l2(a):  loss += a * SUM(W^2)      <- no divide por nada

    Por lo tanto, para que penalicen igual:
        a_keras = 0.5 * alpha_sklearn / batch_size

    Sin esta correccion, pasar el mismo numero regulariza 2*batch = 400 veces
    mas fuerte en Keras.

    (n_train se recibe solo para documentarlo en el informe. Dividir por n en
     vez de por batch sobre-corrige por un factor n/batch: 154x en la
     Variante A y 52x en la B, y la red termina sin regularizacion efectiva.)
    """
    return 0.5 * alpha_sk / BATCH


def entrenar_keras(d, arq, alpha, pesos_clase, etiqueta, sufijo="",
                   alpha_l2=None, alinear=False):
    """Misma arquitectura, mismo alpha, mismo optimizador, misma semilla.

    Equivalencias explicitas sklearn -> Keras:
      hidden_layer_sizes=(200,)  -> Dense(200, activation='relu')
      (capa de salida implicita) -> Dense(4, activation='softmax')
      alpha                      -> kernel_regularizer=l2(alpha)
      solver='adam', lr=0.001    -> Adam(learning_rate=0.001)
      batch_size=200             -> batch_size=200
      early_stopping=True        -> EarlyStopping(restore_best_weights=True)
      n_iter_no_change=15        -> patience=15
      validation_fraction=0.1    -> validation_split=0.1
      sample_weight('balanced')  -> class_weight={...}  (Keras SI lo tiene)

    Diferencias que NO se pueden igualar (y hay que declararlas):
      - epsilon de Adam: 1e-8 en sklearn, 1e-7 en Keras.
      - sklearn divide la penalizacion L2 por el numero de muestras; Keras no.
      - El orden de los batches y la inicializacion de pesos usan generadores
        distintos, aunque ambos parten de Glorot uniform.
    """
    import keras
    from keras import layers, regularizers

    keras.utils.set_random_seed(SEMILLA)
    if alpha_l2 is None:
        alpha_l2 = alpha

    clases_ord = sorted(set(d["y_train"]))
    idx = {c: i for i, c in enumerate(clases_ord)}
    y_tr = np.array([idx[v] for v in d["y_train"]])
    y_va = np.array([idx[v] for v in d["y_val"]])

    modelo = keras.Sequential(
        [layers.Input(shape=(d["X_train"].shape[1],))]
        + [layers.Dense(n, activation="relu",
                        kernel_regularizer=regularizers.l2(alpha_l2)) for n in arq]
        + [layers.Dense(len(clases_ord), activation="softmax")]
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INICIAL),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if alinear:
        # sklearn detiene por ACCURACY (_score), no por perdida; y separa su
        # 10% interno de forma ESTRATIFICADA. Se replican las dos cosas.
        from sklearn.model_selection import train_test_split
        Xa, Xb, ya, yb = train_test_split(
            d["X_train"], y_tr, test_size=FRACCION_VAL_INTERNA,
            stratify=y_tr, random_state=SEMILLA)
        parada = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=PACIENCIA,
            restore_best_weights=True)
        hist = modelo.fit(
            Xa, ya, epochs=MAX_EPOCAS, batch_size=BATCH,
            validation_data=(Xb, yb),
            class_weight=pesos_clase, callbacks=[parada], verbose=0,
        )
    else:
        parada = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
        hist = modelo.fit(
            d["X_train"], y_tr,
            epochs=MAX_EPOCAS, batch_size=BATCH,
            validation_split=FRACCION_VAL_INTERNA,
            class_weight=pesos_clase, callbacks=[parada], verbose=0,
        )

    pred_tr = np.array(clases_ord)[modelo.predict(d["X_train"], verbose=0).argmax(axis=1)]
    pred_va = np.array(clases_ord)[modelo.predict(d["X_val"], verbose=0).argmax(axis=1)]
    f1_tr = f1_score(d["y_train"], pred_tr, average="macro", zero_division=0)
    f1_va = f1_score(d["y_val"], pred_va, average="macro", zero_division=0)

    rep.p(f"\n  {etiqueta} - Keras{sufijo}")
    rep.p(f"    Arquitectura: {arq} | batch={BATCH} | Adam lr={LR_INICIAL}")
    rep.p(f"    l2 aplicado en Keras: {alpha_l2:.3e}  (alpha de sklearn: {alpha})")
    rep.p(f"    Criterio de parada: {'val_accuracy + split estratificado (como sklearn)' if alinear else 'val_loss + ultimo 10% (default Keras)'}")
    rep.p(f"    Parametros entrenables: {modelo.count_params():,}")
    rep.p(f"    Epocas corridas: {len(hist.history['loss'])} (tope {MAX_EPOCAS}, "
          f"paciencia {PACIENCIA})")
    rep.p(f"    class_weight nativo: {'si' if pesos_clase else 'no aplica'}")
    rep.p(f"    F1 macro train: {f1_tr:.4f}")
    rep.p(f"    F1 macro val  : {f1_va:.4f}")
    rep.p(f"    Brecha        : {f1_tr-f1_va:.4f}")

    nombre = f"mlp_keras_{etiqueta.lower().replace(' ', '_')}{sufijo.replace(' ', '_')}.keras"
    modelo.save(DIR_MOD / nombre)
    return {"f1_train": f1_tr, "f1_val": f1_va, "pred_val": pred_va,
            "hist": hist.history, "params": modelo.count_params(),
            "epocas": len(hist.history["loss"])}


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

    dist = pd.Series(d["y_train"]).value_counts()
    rep.p("\n  Distribucion real en train:")
    for k in CLASES:
        v = int(dist.get(k, 0))
        rep.p(f"    {k:<18} {v:6,}  ({100*v/len(d['y_train']):5.2f}%)")

    pesos = None
    pesos_clase = None
    if balancear:
        pesos = compute_sample_weight("balanced", d["y_train"])
        rep.p("\n  MLPClassifier NO tiene class_weight. Se usa sample_weight")
        rep.p("  (un peso por FILA en vez de por clase, equivalente):")
        n, n_cl = len(d["y_train"]), len(CLASES)
        clases_ord = sorted(set(d["y_train"]))
        pesos_clase = {}
        for k in CLASES:
            w = n / (n_cl * int(dist.get(k, 0)))
            rep.p(f"    {k:<18} peso = {w:6.3f}")
        for i, c in enumerate(clases_ord):
            pesos_clase[i] = n / (n_cl * int(dist.get(c, 0)))

    arq, df_arq, i_arq = experimento_arquitectura(d, pesos, etiqueta)
    alpha, df_alpha, i_alpha = experimento_alpha(d, pesos, arq, etiqueta)

    rep.p(f"\n  HIPERPARAMETROS FINALES ({etiqueta}):")
    rep.p(f"    hidden_layer_sizes = {arq}          (experimento 1)")
    rep.p(f"    alpha = {alpha}                     (experimento 2)")
    rep.p(f"    activation = 'relu'                 (DEFAULT)")
    rep.p(f"    solver = 'adam'                     (DEFAULT)")
    rep.p(f"    learning_rate_init = {LR_INICIAL}   (DEFAULT)")
    rep.p(f"    beta_1=0.9, beta_2=0.999            (DEFAULT: momentos de Adam)")
    rep.p(f"    batch_size = {BATCH}                (DEFAULT 'auto' = min(200, n))")
    rep.p(f"    max_iter = {MAX_EPOCAS}             (DEFAULT 200; se sube para no")
    rep.p(f"                                         cortar antes que early stopping)")
    rep.p(f"    early_stopping = True               (DEFAULT False; se activa)")
    rep.p(f"    n_iter_no_change = {PACIENCIA}      (DEFAULT 10)")
    rep.p(f"    validation_fraction = {FRACCION_VAL_INTERNA}  (DEFAULT)")
    rep.p(f"    random_state = {SEMILLA}")

    modelo, f1_tr, f1_va = entrenar_sk(arq, alpha, d, pesos)
    pred_tr, pred_va = modelo.predict(d["X_train"]), modelo.predict(d["X_val"])

    m_tr = metricas(d["y_train"], pred_tr, "TRAIN")
    m_va = metricas(d["y_val"], pred_va, "VALIDACION")
    brecha = m_tr["f1_macro"] - m_va["f1_macro"]
    rep.p(f"\n  Epocas corridas: {len(modelo.loss_curve_)} (tope {MAX_EPOCAS})")
    rep.p(f"  Perdida final  : {modelo.loss_curve_[-1]:.4f}")
    rep.p(f"  OVERFITTING / UNDERFITTING: brecha F1 train-val = {brecha:.4f}")
    if brecha > 0.08:
        rep.p("    -> OVERFITTING.")
    elif m_tr["f1_macro"] < 0.5:
        rep.p("    -> Posible UNDERFITTING.")
    else:
        rep.p("    -> Sin senales fuertes de over/underfitting.")

    rep.p("\n  Reporte por clase (validacion):")
    rep.p(classification_report(d["y_val"], pred_va, labels=CLASES, zero_division=0))

    cm = confusion_matrix(d["y_val"], pred_va, labels=CLASES)
    rep.p(pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                       columns=[f"pred_{c}" for c in CLASES]).to_string())

    joblib.dump({"modelo": modelo, "arquitectura": arq, "alpha": alpha},
                DIR_MOD / f"mlp_{nombre_variante}.joblib")

    return {
        "etiqueta": etiqueta, "variante": nombre_variante, "modelo": modelo,
        "datos": d, "arq": arq, "alpha": alpha, "pesos_clase": pesos_clase,
        "df_arq": df_arq, "i_arq": i_arq, "df_alpha": df_alpha, "i_alpha": i_alpha,
        "cm": cm, "pred_val": pred_va, "f1_train": m_tr["f1_macro"],
        "f1_val": m_va["f1_macro"], "acc_val": m_va["accuracy"], "brecha": brecha,
    }


def importancia_permutacion(r, n=15):
    """El MLP no tiene coef_ interpretable ni feature_importances_. La
    importancia por permutacion mide cuanto cae el F1 al barajar UNA columna:
    si barajarla no cambia nada, el modelo no la estaba usando."""
    d = r["datos"]
    res = permutation_importance(
        r["modelo"], d["X_val"], d["y_val"], n_repeats=3,
        random_state=SEMILLA, scoring="f1_macro", n_jobs=-1)
    imp = pd.Series(res.importances_mean, index=d["columnas"]).sort_values(ascending=False)
    rep.p(f"\n  {r['etiqueta']} - top {n} por importancia de permutacion:")
    for k, v in imp.head(n).items():
        rep.p(f"    {str(k)[:48]:<50} {v:+.5f}")
    return imp


# ---------------------------------------------------------------- graficos
def fig_experimento_arq(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, r in zip(axes, (ra, rb)):
        df, i = r["df_arq"], r["i_arq"]
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
        ax.text(i, df["f1_train"].max() + 0.006, f"elegida {r['arq']}",
                ha="center", fontsize=8.5, color=TINTA, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e}\n{p:,}p" for e, p in zip(df["etiqueta"], df["parametros"])],
                           fontsize=7.5)
        estilo(ax, r["etiqueta"], xlabel="arquitectura (y cantidad de parametros)")
    axes[0].set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    fig.suptitle("Experimento 1: que arquitectura usar\n"
                 "El default de sklearn es (100,) - no es un detalle, es la red entera",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "experimento_arquitectura.png")


def fig_experimento_alpha(ra, rb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, r in zip(axes, (ra, rb)):
        df, i = r["df_alpha"], r["i_alpha"]
        ax.plot(df["alpha"], df["f1_train"], "o-", color=COLOR_TRAIN,
                label="Train", zorder=3)
        ax.plot(df["alpha"], df["f1_val"], "o-", color=COLOR_VAL,
                label="Validacion", zorder=3)
        ax.axvline(r["alpha"], color=MUTED, linestyle="--", linewidth=1, zorder=2)
        ax.set_xscale("log")
        estilo(ax, f"{r['etiqueta']}  (arquitectura fija {r['arq']})",
               xlabel="alpha  (mas a la derecha = mas regularizado)")
        ax.annotate(f"elegido alpha={r['alpha']}", xy=(r["alpha"], df["f1_val"].min()),
                    xytext=(-6, 10), textcoords="offset points",
                    fontsize=8.5, color=TINTA_SEC, ha="right")
    axes[0].set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    fig.suptitle("Experimento 2: cuanta regularizacion L2 (alpha)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "experimento_alpha.png")


def fig_curvas_aprendizaje(ra, rb, ka, kb):
    """Dos paneles APILADOS por variante (nunca dos ejes Y en un mismo grafico:
    invitaria a leer un 'cruce' entre perdida y accuracy que no significa nada)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for col, (r, k) in enumerate([(ra, ka), (rb, kb)]):
        ax = axes[0, col]
        ax.plot(range(1, len(r["modelo"].loss_curve_) + 1), r["modelo"].loss_curve_,
                color=COLOR_SK, linewidth=2, label="sklearn", zorder=3)
        if k:
            ax.plot(range(1, len(k["hist"]["loss"]) + 1), k["hist"]["loss"],
                    color=COLOR_KERAS, linewidth=2, label="Keras", zorder=3)
        estilo(ax, f"{r['etiqueta']}: perdida por epoca", ylabel="Perdida (entropia cruzada)")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(TINTA_SEC)

        ax = axes[1, col]
        vs = getattr(r["modelo"], "validation_scores_", None)
        if vs is not None:
            ax.plot(range(1, len(vs) + 1), vs, color=COLOR_SK, linewidth=2,
                    label="sklearn", zorder=3)
        if k and "val_accuracy" in k["hist"]:
            ax.plot(range(1, len(k["hist"]["val_accuracy"]) + 1),
                    k["hist"]["val_accuracy"], color=COLOR_KERAS, linewidth=2,
                    label="Keras", zorder=3)
        estilo(ax, "Accuracy en la validacion interna (10% del train)",
               xlabel="Epoca", ylabel="Accuracy")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(TINTA_SEC)

    fig.suptitle("Como aprende la red, epoca a epoca\n"
                 "El eje X es el mismo en ambos paneles; nunca se superponen dos escalas",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "curvas_aprendizaje.png")


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
    fig.suptitle("MLP - matriz de confusion en validacion "
                 "(celda: casos y % de la fila real)",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "matrices_confusion.png")


def fig_sklearn_vs_keras(ra, rb, ka, kb, ka_c, kb_c):
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(2)
    ancho = 0.26
    series = [
        ([ra["f1_val"], rb["f1_val"]], -ancho, COLOR_SK, "sklearn MLPClassifier"),
        ([ka["f1_val"], kb["f1_val"]], 0, COLOR_KERAS,
         "Keras, mismo numero de alpha"),
        ([ka_c["f1_val"], kb_c["f1_val"]], ancho, COLOR_A,
         "Keras alineado (L2 + criterio de parada + split)"),
    ]
    for datos, off, color, lab in series:
        barras = ax.bar(x + off, datos, ancho, label=lab, color=color, zorder=3)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(2)
        for xi, v in enumerate(datos):
            ax.text(xi + off, v + 0.006, f"{v:.4f}", ha="center", fontsize=8.5,
                    color=TINTA, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Variante A\n{ra['arq']}, alpha={ra['alpha']}",
                        f"Variante B\n{rb['arq']}, alpha={rb['alpha']}"],
                       fontsize=9)
    estilo(ax, "Experimento 3: ¿el resultado depende de la libreria?\n"
               "sklearn divide la penalizacion L2 por el batch; Keras no. Pasar el "
               f"mismo numero regulariza {2*BATCH}x mas fuerte.",
           ylabel="F1 macro (validacion)")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    return guardar_fig(fig, "experimento_sklearn_vs_keras.png")


def fig_importancia(imp_a, imp_b, n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (imp, et) in zip(axes, [(imp_a, "VARIANTE A"), (imp_b, "VARIANTE B")]):
        top = imp.head(n).iloc[::-1]
        barras = ax.barh(range(len(top)), top.values, color=COLOR_A, zorder=3, height=0.7)
        for b in barras:
            b.set_edgecolor(SUPERFICIE)
            b.set_linewidth(1.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([str(s)[:40] for s in top.index], fontsize=8)
        estilo(ax, et, xlabel="Caida de F1 macro al barajar la columna")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Importancia por permutacion - MLP\n"
                 "El MLP no tiene coef_ ni feature_importances_: se mide cuanto "
                 "cae el F1 al barajar cada columna",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    return guardar_fig(fig, "importancia_permutacion.png")


def main():
    rep.titulo("ENTRENAMIENTO V2 - RED NEURONAL (MLP)", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")

    ra = entrenar_variante("variante_a", "VARIANTE A", balancear=True)
    rb = entrenar_variante("variante_b", "VARIANTE B", balancear=False)

    rep.titulo("EXPERIMENTO 3: sklearn vs KERAS (misma configuracion)")
    rep.p("  Equivalencias explicitas:")
    rep.p("    hidden_layer_sizes  -> Dense(n, activation='relu')")
    rep.p("    (salida implicita)  -> Dense(4, activation='softmax')")
    rep.p("    alpha               -> kernel_regularizer=l2(alpha)")
    rep.p("    solver='adam'       -> optimizers.Adam(learning_rate=0.001)")
    rep.p("    batch_size=200      -> batch_size=200")
    rep.p("    n_iter_no_change=15 -> EarlyStopping(patience=15)")
    rep.p("    sample_weight       -> class_weight  (Keras SI lo tiene nativo)")
    rep.p("")
    rep.p("  Diferencias que NO se pueden igualar pasando el mismo numero:")
    rep.p("    - epsilon de Adam: 1e-8 en sklearn, 1e-7 en Keras.")
    rep.p("    - El orden de los batches y la inicializacion usan generadores")
    rep.p("      distintos, aunque ambos parten de Glorot uniform.")
    rep.p("    - LA PENALIZACION L2 NO SIGNIFICA LO MISMO:")
    rep.p("        sklearn: loss += 0.5 * alpha * SUM(W^2) / sw_sum")
    rep.p("                 y sw_sum ~ batch_size, porque _backprop recibe el")
    rep.p("                 BATCH, no el dataset entero.")
    rep.p("        Keras  : loss += alpha * SUM(W^2)   (no divide por nada)")
    rep.p(f"      Pasar el mismo numero regulariza 2*batch = {2*BATCH}x mas fuerte")
    rep.p("      en Keras. Por eso se corren DOS versiones: la ingenua (mismo")
    rep.p("      numero) y la corregida (a_keras = 0.5*alpha/batch).")

    ka = kb = ka_c = kb_c = None
    try:
        import keras
        rep.p(f"\n  keras {keras.__version__}")

        rep.p("\n  --- 3a. Keras INGENUO: se le pasa el mismo alpha que a sklearn ---")
        ka = entrenar_keras(ra["datos"], ra["arq"], ra["alpha"],
                            ra["pesos_clase"], "VARIANTE A", " ingenuo")
        kb = entrenar_keras(rb["datos"], rb["arq"], rb["alpha"],
                            rb["pesos_clase"], "VARIANTE B", " ingenuo")
        rep.p("")
        for r, k in [(ra, ka), (rb, kb)]:
            rep.p(f"  {r['etiqueta']}: sklearn {r['f1_val']:.4f} vs "
                  f"Keras {k['f1_val']:.4f}  ({k['f1_val']-r['f1_val']:+.4f})")

        rep.p("\n  --- 3b. Keras ALINEADO: se igualan las 3 diferencias controlables ---")
        rep.p("    (1) l2 reescalado: a_keras = 0.5 * alpha_sklearn / batch")
        rep.p("        El divisor es el BATCH, no n: _backprop de sklearn recibe el")
        rep.p("        batch, asi que su sw_sum vale ~200, no 30.782.")
        rep.p("    (2) criterio de parada: sklearn detiene por ACCURACY (_score),")
        rep.p("        no por perdida. Se pasa Keras a monitor='val_accuracy'.")
        rep.p("    (3) split interno: sklearn separa su 10% de forma ESTRATIFICADA;")
        rep.p("        el validation_split de Keras toma el ultimo 10% sin mirar")
        rep.p("        la clase. Se replica el split estratificado a mano.")
        rep.p("")
        for r in (ra, rb):
            a_eq = alpha_keras_equivalente(r["alpha"], len(r["datos"]["y_train"]))
            rep.p(f"    {r['etiqueta']}: alpha={r['alpha']}, batch={BATCH} -> "
                  f"l2={a_eq:.3e}  (factor {r['alpha']/a_eq:,.0f}x mas suave)")
        ka_c = entrenar_keras(
            ra["datos"], ra["arq"], ra["alpha"], ra["pesos_clase"], "VARIANTE A",
            " alineado",
            alpha_keras_equivalente(ra["alpha"], len(ra["datos"]["y_train"])),
            alinear=True)
        kb_c = entrenar_keras(
            rb["datos"], rb["arq"], rb["alpha"], rb["pesos_clase"], "VARIANTE B",
            " alineado",
            alpha_keras_equivalente(rb["alpha"], len(rb["datos"]["y_train"])),
            alinear=True)

        rep.p("\n  RESUMEN DEL EXPERIMENTO 3")
        rep.p(f"  {'':>12} | {'sklearn':>8} | {'Keras ingenuo':>14} | "
              f"{'Keras alineado':>15}")
        rep.p("  " + "-" * 60)
        for r, k, kc in [(ra, ka, ka_c), (rb, kb, kb_c)]:
            rep.p(f"  {r['etiqueta']:>12} | {r['f1_val']:8.4f} | {k['f1_val']:14.4f} | "
                  f"{kc['f1_val']:15.4f}")

        d_ing = np.mean([abs(ka["f1_val"] - ra["f1_val"]),
                         abs(kb["f1_val"] - rb["f1_val"])])
        d_cor = np.mean([abs(ka_c["f1_val"] - ra["f1_val"]),
                         abs(kb_c["f1_val"] - rb["f1_val"])])
        rep.p("")
        rep.p(f"  Diferencia media de F1 macro contra sklearn:")
        rep.p(f"    Keras ingenuo  : {d_ing:.4f}")
        rep.p(f"    Keras alineado : {d_cor:.4f}")
        rep.p(f"    Explicado por las 3 correcciones: {100*(d_ing-d_cor)/d_ing:.0f}%")
        rep.p("")
        rep.p("  COMO LEER ESTO (sin sobreafirmar):")
        if d_cor < 0.015:
            rep.p("    La diferencia que queda es del orden del ruido de")
            rep.p("    inicializacion. Las dos librerias implementan el MISMO modelo:")
            rep.p("    lo que parecia una brecha de libreria era en realidad que los")
            rep.p("    hiperparametros NO SIGNIFICAN LO MISMO en cada una.")
        else:
            rep.p(f"    Queda {d_cor:.4f} de diferencia sin explicar. Las causas")
            rep.p("    restantes NO son configurables desde la API:")
            rep.p("      - epsilon de Adam: 1e-8 en sklearn vs 1e-7 en Keras.")
            rep.p("      - la inicializacion de pesos y el orden de los batches")
            rep.p("        salen de generadores distintos (ambos Glorot uniform,")
            rep.p("        pero con RNG y secuencia distintos).")
            rep.p("    Conclusion honesta: las correcciones explican la mayor parte")
            rep.p("    de la brecha, pero NO se puede afirmar que las dos librerias")
            rep.p("    den el mismo numero. Lo defendible es que son EQUIVALENTES EN")
            rep.p("    METODO, no identicas en resultado.")
    except ImportError:
        rep.p("\n  keras/tensorflow no esta instalado: se omite el experimento 3.")
        rep.p("  Instalar con: pip install tensorflow==2.20.0")

    rep.titulo("IMPORTANCIA POR PERMUTACION")
    imp_a = importancia_permutacion(ra)
    imp_b = importancia_permutacion(rb)

    rep.titulo("COMPARACION DIRECTA A vs B")
    comp = pd.DataFrame({
        "Variante A": [str(ra["arq"]), ra["alpha"], ra["acc_val"], ra["f1_val"], ra["brecha"]],
        "Variante B": [str(rb["arq"]), rb["alpha"], rb["acc_val"], rb["f1_val"], rb["brecha"]],
    }, index=["arquitectura", "alpha", "Accuracy (val)", "F1 macro (val)",
              "Brecha train-val"])
    rep.p(comp.to_string())

    rep.titulo("COMPARACION CONTRA EL MLP DE V1 (dataset.xlsx)")
    rep.p("  V1 (con filas fabricadas, (200,), alpha=0.01, 35 epocas):")
    rep.p("      Accuracy val 0.6376 | F1 macro val 0.6369 | brecha 0.140")
    rep.p(f"  V2 Variante A: Accuracy val {ra['acc_val']:.4f} | "
          f"F1 macro val {ra['f1_val']:.4f} | brecha {ra['brecha']:.4f}")
    rep.p(f"  V2 Variante B: Accuracy val {rb['acc_val']:.4f} | "
          f"F1 macro val {rb['f1_val']:.4f} | brecha {rb['brecha']:.4f}")

    rep.titulo("COMPARATIVA DE MODELOS V2 (validacion)")
    filas = {"MLP A": [ra["acc_val"], ra["f1_val"], ra["brecha"]],
             "MLP B": [rb["acc_val"], rb["f1_val"], rb["brecha"]]}
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
        filas[et] = [accuracy_score(yv, pv), f1v,
                     f1_score(yt, pt, average="macro", zero_division=0) - f1v]
    for et, r in [("CatBoost A", ra), ("CatBoost B", rb)]:
        pass  # los de CatBoost se leen del informe propio (usan otra matriz)
    comp2 = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)",
                                       "Brecha train-val"]).round(4)
    rep.p(comp2.T.to_string())
    rep.p("\n  (CatBoost no aparece aca porque no usa los .npy: sus numeros estan")
    rep.p("   en informe_modelo_catboost_v2.txt -> A: 0.4838 | B: 0.5057)")

    rep.titulo("GRAFICOS")
    fig_experimento_arq(ra, rb)
    fig_experimento_alpha(ra, rb)
    fig_curvas_aprendizaje(ra, rb, ka_c or ka, kb_c or kb)
    fig_matrices(ra, rb)
    fig_importancia(imp_a, imp_b)
    if ka and kb and ka_c and kb_c:
        fig_sklearn_vs_keras(ra, rb, ka, kb, ka_c, kb_c)

    rep.titulo("FIN", "#")
    rep.p("Guardado: modelos/mlp_variante_{a,b}.joblib")
    if ka:
        rep.p("Guardado: modelos/mlp_keras_variante_{a,b}.keras")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
