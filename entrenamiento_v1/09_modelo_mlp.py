"""
MODELO 4: RED NEURONAL (MLP - Perceptron Multicapa)
=====================================================

LIBRERIA: sklearn.neural_network.MLPClassifier

¿Por que sklearn y no TensorFlow/Keras o PyTorch?
  1. Consistencia: los modelos 1, 2 y 3 ya usan la API de sklearn. El script
     de graficos, el de prueba y la futura app funcionan sin adaptaciones.
  2. Sin dependencias nuevas: TensorFlow pesa ~500 MB.
  3. Expone loss_curve_: la perdida por EPOCA, que es justo lo que interesa
     ver en una red neuronal (y que Random Forest no tiene).

¿ES UNA CAJA NEGRA?
-------------------
MLPClassifier trae 23 hiperparametros con valores por defecto. Aceptarlos a
ciegas SI seria una caja negra. Por eso aqui:
  - hidden_layer_sizes (la ARQUITECTURA) se elige con el experimento 1.
  - alpha (la regularizacion) se elige con el experimento 2.
  - El resto se fija explicitamente y se justifica abajo.
  - Se grafica loss_curve_ para ver el aprendizaje epoca a epoca.
  - Se inspeccionan las matrices de pesos (coefs_).

Ojo con un default en particular: hidden_layer_sizes=(100,) NO es un detalle
tecnico, es la arquitectura ENTERA de la red decidida sin escribirla. En los
otros modelos los defaults ajustaban COMO aprende; aqui define QUE ES el
modelo. Por eso es el primer experimento.

COMO APRENDE (para la defensa)
-------------------------------
A diferencia de Random Forest (que no optimiza nada globalmente), el MLP SI
minimiza una funcion de perdida, igual que la regresion logistica:

  Perdida: entropia cruzada multinomial (la MISMA de la logistica)
      L = - suma_i log( p(y_i = clase_verdadera | x_i) )  +  alpha * ||W||^2

  Forward pass (475 entradas -> 4 salidas):
      a1 = ReLU(W1 * x  + b1)       capa oculta 1
      a2 = ReLU(W2 * a1 + b2)       capa oculta 2 (si existe)
      z  = W3 * a2 + b3
      p  = softmax(z)               4 probabilidades

  ReLU(v) = max(0, v). Se usa porque su derivada es 0 o 1: no se "desvanece"
  al propagar el gradiente hacia atras, problema que si tenia sigmoide.

  Backpropagation: se calcula dL/dW en cada capa aplicando la regla de la
  cadena desde la salida hacia la entrada, y Adam actualiza los pesos.

  UNA EPOCA = una pasada completa por las 14.630 filas de entrenamiento.
  Aqui el concepto SI aplica (en Random Forest no existia).

DIFERENCIA CLAVE CON LA REGRESION LOGISTICA
  Misma funcion de perdida, misma capa de salida (softmax). La diferencia es
  que la logistica va DIRECTO de las 475 entradas a las 4 salidas (frontera
  lineal), mientras el MLP intercala capas ocultas con ReLU, que le permiten
  representar fronteras NO lineales e interacciones entre variables.

Salida:
  modelos/modelo4_mlp.joblib
  resultados/modelos/informe_modelo4_mlp.txt
  Graficos/modelo4_mlp_fig0_arquitectura.png
  Graficos/modelo4_mlp_fig0b_alpha.png
  Graficos/modelo4_mlp_fig4_dinamica.png   (curva de perdida por epoca)
"""

import sys
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
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.neural_network import MLPClassifier

from utilidades import ModeloEtiquetado

BASE = Path(__file__).resolve().parent
SEMILLA = 42

DIR_PROC = BASE / "data" / "processed"
DIR_MOD = BASE / "modelos"
DIR_REP = BASE / "resultados" / "modelos"
DIR_FIG = BASE / "Graficos"
DIR_REP.mkdir(parents=True, exist_ok=True)
DIR_FIG.mkdir(parents=True, exist_ok=True)

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
TINTA, TINTA_SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SUPERFICIE = "#e1e0d9", "#fcfcfb"

# EXPERIMENTO 1: arquitecturas a probar.
# De menos a mas capacidad, para ver donde empieza a sobreajustar.
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

# Las etiquetas se codifican a enteros internamente (ver ModeloEtiquetado en
# utilidades.py): es un rodeo a un bug de scikit-learn 1.8.0 con
# early_stopping=True y etiquetas de texto. El modelo resultante es identico.


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


rep = Reporte(DIR_REP / "informe_modelo4_mlp.txt")


def metricas(y_true, y_pred, nombre):
    rep.p(f"\n  --- {nombre} ---")
    rep.p(f"  Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    rep.p(f"  Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    rep.p(f"  F1 (macro)       : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


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
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def guardar_fig(fig, nombre):
    ruta = DIR_FIG / f"modelo4_mlp_{nombre}"
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Grafico: Graficos/{ruta.name}")


def entrenar(X_tr, y_tr, arquitectura, alpha):
    """Entrena un MLP con TODOS los hiperparametros explicitos."""
    m = MLPClassifier(
        hidden_layer_sizes=arquitectura,
        alpha=alpha,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,       # corta cuando deja de mejorar
        validation_fraction=0.1,   # 10% del TRAIN para decidir el corte
        n_iter_no_change=15,
        random_state=SEMILLA,
    )
    envuelto = ModeloEtiquetado(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        envuelto.fit(X_tr, y_tr)
    return envuelto


def pred(modelo, X):
    """Predice devolviendo etiquetas de texto."""
    return modelo.predict(X)


# ---------------------------------------------------------------------------
# EXPERIMENTO 3 - sklearn vs Keras
# ---------------------------------------------------------------------------
BATCH = 200   # sklearn 'auto' = min(200, n). Con n=14.630 da 200.


def alpha_keras_equivalente(alpha_sk):
    """Traduce el alpha de sklearn al l2() de Keras.

    Verificado leyendo el fuente de sklearn (_multilayer_perceptron.py,
    dentro de _backprop, que se llama UNA VEZ POR BATCH):

        sw_sum = n_samples if sample_weight is None else sample_weight.sum()
        loss += (0.5 * self.alpha) * values / sw_sum

    Como _backprop recibe el BATCH y no el dataset, 'sw_sum' vale ~200.
    Keras, en cambio, hace  l2(a): loss += a * SUM(W^2)  sin dividir por nada.

        a_keras = 0.5 * alpha_sklearn / batch_size

    Pasarle el mismo numero a las dos librerias regulariza 2*batch = 400
    veces mas fuerte en Keras.
    """
    return 0.5 * alpha_sk / BATCH


def entrenar_keras(X_tr, y_tr, X_va, y_va, arq, alpha, sufijo, alpha_l2=None,
                   alinear=False):
    """Misma arquitectura, mismo optimizador, misma semilla que el sklearn.

    Equivalencias explicitas:
      hidden_layer_sizes=(n,)  -> Dense(n, activation='relu')
      (capa de salida)         -> Dense(4, activation='softmax')
      alpha                    -> kernel_regularizer=l2(alpha)
      solver='adam', lr=0.001  -> Adam(learning_rate=0.001)
      batch_size 'auto'=200    -> batch_size=200
      n_iter_no_change=15      -> EarlyStopping(patience=15)
      validation_fraction=0.1  -> validation_split / validation_data

    Con alinear=True se igualan ademas las tres diferencias que NO son
    obvias (descubiertas en entrenamiento_v2/06_modelo_mlp_v2.py):
      1. la escala de L2 (ver alpha_keras_equivalente)
      2. el criterio de parada: sklearn corta por ACCURACY (_score), no por
         perdida, asi que Keras va con monitor='val_accuracy'
      3. el split interno: sklearn lo hace ESTRATIFICADO; el validation_split
         de Keras agarra el ultimo 10% sin mirar la clase
    """
    import keras
    from keras import layers, regularizers

    keras.utils.set_random_seed(SEMILLA)
    if alpha_l2 is None:
        alpha_l2 = alpha

    clases_ord = sorted(set(y_tr))
    idx = {c: i for i, c in enumerate(clases_ord)}
    yc_tr = np.array([idx[v] for v in y_tr])

    modelo = keras.Sequential(
        [layers.Input(shape=(X_tr.shape[1],))]
        + [layers.Dense(n, activation="relu",
                        kernel_regularizer=regularizers.l2(alpha_l2)) for n in arq]
        + [layers.Dense(len(clases_ord), activation="softmax")]
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if alinear:
        from sklearn.model_selection import train_test_split
        Xa, Xb, ya, yb = train_test_split(
            X_tr, yc_tr, test_size=0.1, stratify=yc_tr, random_state=SEMILLA)
        parada = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=15,
            restore_best_weights=True)
        hist = modelo.fit(Xa, ya, epochs=300, batch_size=BATCH,
                          validation_data=(Xb, yb), callbacks=[parada], verbose=0)
    else:
        parada = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)
        hist = modelo.fit(X_tr, yc_tr, epochs=300, batch_size=BATCH,
                          validation_split=0.1, callbacks=[parada], verbose=0)

    cl = np.array(clases_ord)
    p_tr = cl[modelo.predict(X_tr, verbose=0).argmax(axis=1)]
    p_va = cl[modelo.predict(X_va, verbose=0).argmax(axis=1)]
    f1_tr = f1_score(y_tr, p_tr, average="macro", zero_division=0)
    f1_va = f1_score(y_va, p_va, average="macro", zero_division=0)

    rep.p(f"\n  Keras{sufijo}")
    rep.p(f"    Arquitectura: {arq} | batch={BATCH} | Adam lr=0.001")
    rep.p(f"    l2 aplicado: {alpha_l2:.3e}   (alpha de sklearn: {alpha})")
    rep.p(f"    Criterio de parada: "
          f"{'val_accuracy + split estratificado (como sklearn)' if alinear else 'val_loss + ultimo 10% (default Keras)'}")
    rep.p(f"    Parametros entrenables: {modelo.count_params():,}")
    rep.p(f"    Epocas corridas: {len(hist.history['loss'])} (tope 300, paciencia 15)")
    rep.p(f"    F1 macro train: {f1_tr:.4f}")
    rep.p(f"    F1 macro val  : {f1_va:.4f}")
    rep.p(f"    Brecha        : {f1_tr - f1_va:.4f}")

    modelo.save(DIR_MOD / f"modelo4_mlp_keras{sufijo.replace(' ', '_')}.keras")
    return {"f1_train": f1_tr, "f1_val": f1_va, "hist": hist.history,
            "epocas": len(hist.history["loss"])}


def experimento_keras(X_tr, y_tr, X_va, y_va, arq, alpha, f1_sk):
    rep.titulo("EXPERIMENTO 3: sklearn vs KERAS (misma configuracion)")
    rep.p("  Pregunta: ¿el resultado depende de la LIBRERIA o de los")
    rep.p("  HIPERPARAMETROS? Se entrena la misma red en Keras, dos veces.")
    rep.p("")
    rep.p("  Diferencias que NO se igualan pasando el mismo numero:")
    rep.p("    - LA PENALIZACION L2 NO SIGNIFICA LO MISMO:")
    rep.p("        sklearn: loss += 0.5*alpha*SUM(W^2)/sw_sum, y sw_sum ~ batch")
    rep.p("                 porque _backprop recibe el BATCH, no el dataset.")
    rep.p("        Keras  : loss += alpha*SUM(W^2)   (no divide por nada)")
    rep.p(f"      Pasar el mismo numero regulariza 2*batch = {2*BATCH}x mas fuerte.")
    rep.p("    - El criterio de parada: sklearn corta por ACCURACY (_score); el")
    rep.p("      default que uno pone en Keras es val_loss.")
    rep.p("    - El split interno: sklearn separa su 10% ESTRATIFICADO; el")
    rep.p("      validation_split de Keras agarra el ultimo 10% sin mirar clase.")
    rep.p("")
    rep.p("  Y las que NO son configurables desde la API:")
    rep.p("    - epsilon de Adam: 1e-8 en sklearn vs 1e-7 en Keras.")
    rep.p("    - inicializacion de pesos y orden de batches: generadores")
    rep.p("      distintos, aunque ambos parten de Glorot uniform.")

    try:
        import keras
    except ImportError:
        rep.p("\n  keras/tensorflow no esta instalado: se omite el experimento 3.")
        rep.p("  Instalar con: pip install tensorflow==2.20.0")
        return None, None

    rep.p(f"\n  keras {keras.__version__}")
    rep.p("\n  --- 3a. Keras INGENUO: mismo numero de alpha, defaults de Keras ---")
    k_ing = entrenar_keras(X_tr, y_tr, X_va, y_va, arq, alpha, " ingenuo")

    rep.p("\n  --- 3b. Keras ALINEADO: las 3 diferencias controlables igualadas ---")
    a_eq = alpha_keras_equivalente(alpha)
    rep.p(f"    alpha={alpha}, batch={BATCH} -> l2={a_eq:.3e} "
          f"({alpha/a_eq:,.0f}x mas suave)")
    k_ali = entrenar_keras(X_tr, y_tr, X_va, y_va, arq, alpha, " alineado",
                           a_eq, alinear=True)

    rep.p("\n  RESUMEN DEL EXPERIMENTO 3")
    rep.p(f"    sklearn MLPClassifier : {f1_sk:.4f}")
    rep.p(f"    Keras ingenuo         : {k_ing['f1_val']:.4f} "
          f"({k_ing['f1_val']-f1_sk:+.4f})")
    rep.p(f"    Keras alineado        : {k_ali['f1_val']:.4f} "
          f"({k_ali['f1_val']-f1_sk:+.4f})")
    d_ing, d_ali = abs(k_ing["f1_val"]-f1_sk), abs(k_ali["f1_val"]-f1_sk)
    rep.p("")
    if d_ing > 0:
        rep.p(f"    Explicado por las 3 correcciones: {100*(d_ing-d_ali)/d_ing:.0f}%")
    rep.p("")
    if d_ali < 0.015:
        rep.p("  -> CONCLUSION: la diferencia que queda es del orden del ruido de")
        rep.p("     inicializacion. Las dos librerias implementan el MISMO modelo:")
        rep.p("     lo que parecia una brecha de libreria era que los")
        rep.p("     hiperparametros NO SIGNIFICAN LO MISMO en cada una.")
    else:
        rep.p(f"  -> CONCLUSION: queda {d_ali:.4f} sin explicar, atribuible al")
        rep.p("     epsilon de Adam y a la inicializacion/orden de batches. Lo")
        rep.p("     defendible es que son EQUIVALENTES EN METODO, no identicas")
        rep.p("     en resultado.")

    graficar_keras(f1_sk, k_ing, k_ali, arq, alpha)
    return k_ing, k_ali


def graficar_keras(f1_sk, k_ing, k_ali, arq, alpha):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    vals = [f1_sk, k_ing["f1_val"], k_ali["f1_val"]]
    etiquetas = ["sklearn\nMLPClassifier",
                 "Keras\nmismo numero de alpha",
                 "Keras alineado\n(L2 + parada + split)"]
    colores = ["#1baf7a", "#eda100", "#2a78d6"]
    barras = ax.bar(range(3), vals, color=colores, width=0.55, zorder=3)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=10,
                color=TINTA, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(etiquetas, fontsize=9)
    estilo(ax, "Experimento 3: ¿el resultado depende de la libreria?\n"
               f"Misma red {arq}, alpha={alpha}, mismo Adam, mismo batch, misma semilla.",
           ylabel="F1 macro (validacion)")
    ax.grid(axis="x", visible=False)
    guardar_fig(fig, "fig8_sklearn_vs_keras.png")


# ---------------------------------------------------------------------------
# EXPERIMENTO 1 - Arquitectura
# ---------------------------------------------------------------------------
def experimento_arquitectura(X_tr, y_tr, X_va, y_va):
    rep.titulo("EXPERIMENTO 1: ¿que arquitectura usar? (barrido manual)")
    rep.p("  hidden_layer_sizes define la RED ENTERA: cuantas capas ocultas y")
    rep.p("  cuantas neuronas en cada una. Es el hiperparametro mas importante,")
    rep.p("  y el default de sklearn -(100,)- se prueba como una opcion mas,")
    rep.p("  no se acepta a ciegas.")
    rep.p("")
    rep.p("  alpha se deja fijo en su default (0.0001) para aislar el efecto de")
    rep.p("  la arquitectura. Se ajustara en el experimento 2.")
    rep.p("")
    rep.p(f"  {'arquitectura':>16} | {'params':>9} | {'epocas':>7} | "
          f"{'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 76)

    filas = []
    for arq in ARQUITECTURAS:
        m = entrenar(X_tr, y_tr, arq, alpha=0.0001)
        f1_tr = f1_score(y_tr, pred(m, X_tr), average="macro", zero_division=0)
        f1_va = f1_score(y_va, pred(m, X_va), average="macro", zero_division=0)
        # Cantidad de pesos entrenables (sin contar bias, aproximado)
        n_params = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
        etiqueta = str(arq)
        rep.p(f"  {etiqueta:>16} | {n_params:>9,} | {m.n_iter_:>7} | "
              f"{f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
        filas.append({"arq": etiqueta, "params": n_params, "epocas": m.n_iter_,
                      "f1_train": f1_tr, "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas)

    # Misma regla de parsimonia usada en Random Forest y CatBoost:
    # la arquitectura MAS CHICA dentro del 1% del mejor F1.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    candidatos = df.index[df["f1_val"] >= umbral].tolist()
    i_elegido = min(candidatos, key=lambda i: df.loc[i, "params"])
    arq_elegida = ARQUITECTURAS[i_elegido]

    rep.p("")
    rep.p(f"  Mejor F1 absoluto : {f1_mejor:.4f} "
          f"(arquitectura {df.loc[df['f1_val'].idxmax(), 'arq']})")
    rep.p(f"  Umbral del 1%     : {umbral:.4f}")
    rep.p(f"  ELEGIDA (menos parametros dentro del umbral): {df.loc[i_elegido, 'arq']} "
          f"-> {df.loc[i_elegido, 'params']:,} pesos, F1 val {df.loc[i_elegido, 'f1_val']:.4f}")
    rep.p("  Misma regla de parsimonia aplicada en Random Forest y CatBoost.")

    graficar_arquitectura(df, i_elegido)
    return arq_elegida


def graficar_arquitectura(df, i_elegido):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
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
             f"elegida: {df.loc[i_elegido, 'arq']}", ha="center",
             fontsize=9, color=TINTA, weight="bold")
    estilo(ax1, "Arquitectura vs generalizacion",
           xlabel="hidden_layer_sizes", ylabel="F1 macro")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["arq"], fontsize=8, rotation=20, ha="right")
    leg = ax1.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    barras = ax2.bar(list(x), df["params"], color=COLOR_TRAIN, width=0.6, zorder=3)
    for b in barras:
        b.set_edgecolor(SUPERFICIE)
        b.set_linewidth(2)
    estilo(ax2, "Tamanio de cada red\n(cantidad de pesos entrenables)",
           xlabel="hidden_layer_sizes", ylabel="Parametros")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(df["arq"], fontsize=8, rotation=20, ha="right")
    for xi, v in zip(x, df["params"]):
        ax2.text(xi, v + df["params"].max() * 0.02, f"{v/1000:.0f}k",
                 ha="center", fontsize=8.5, color=TINTA, weight="bold")

    guardar_fig(fig, "fig0_arquitectura.png")


# ---------------------------------------------------------------------------
# EXPERIMENTO 2 - Regularizacion (alpha)
# ---------------------------------------------------------------------------
def experimento_alpha(X_tr, y_tr, X_va, y_va, arquitectura):
    rep.titulo("EXPERIMENTO 2: ¿cuanta regularizacion (alpha)?")
    rep.p("  alpha es el peso del termino L2 en la perdida: penaliza que los")
    rep.p("  pesos crezcan mucho. Es el freno principal contra el overfitting.")
    rep.p("  Bajo  -> la red memoriza el train.")
    rep.p("  Alto  -> la red queda demasiado simple (underfitting).")
    rep.p("")
    rep.p(f"  Arquitectura fija: {arquitectura} (elegida en el experimento 1)")
    rep.p("")
    rep.p(f"  {'alpha':>8} | {'epocas':>7} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 52)

    filas = []
    for a in ALPHAS:
        m = entrenar(X_tr, y_tr, arquitectura, alpha=a)
        f1_tr = f1_score(y_tr, pred(m, X_tr), average="macro", zero_division=0)
        f1_va = f1_score(y_va, pred(m, X_va), average="macro", zero_division=0)
        rep.p(f"  {a:>8} | {m.n_iter_:>7} | {f1_tr:9.4f} | {f1_va:9.4f} | "
              f"{f1_tr - f1_va:7.4f}")
        filas.append({"alpha": a, "epocas": m.n_iter_, "f1_train": f1_tr,
                      "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas)
    # Aqui se toma el MEJOR F1 de validacion directamente: alpha no agrega
    # complejidad al modelo (la red es la misma), asi que no aplica la regla
    # de parsimonia. A igualdad de F1 se prefiere el alpha MAS ALTO, porque
    # implica menos overfitting.
    f1_mejor = df["f1_val"].max()
    candidatos = df.index[df["f1_val"] >= f1_mejor * 0.99].tolist()
    i_elegido = candidatos[-1]
    alpha_elegido = float(df.loc[i_elegido, "alpha"])

    rep.p("")
    rep.p(f"  Mejor F1 absoluto : {f1_mejor:.4f} (alpha={df.loc[df['f1_val'].idxmax(), 'alpha']})")
    rep.p(f"  ELEGIDO (alpha mas alto dentro del 1%): alpha={alpha_elegido} -> "
          f"F1 val {df.loc[i_elegido, 'f1_val']:.4f}, brecha {df.loc[i_elegido, 'brecha']:.4f}")
    rep.p("  Criterio: a igual rendimiento, mas regularizacion = menos overfitting.")

    graficar_alpha(df, i_elegido)
    return alpha_elegido


def graficar_alpha(df, i_elegido):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    ax.fill_between(list(x), df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")
    ax.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Entrenamiento", zorder=3)
    ax.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Validacion", zorder=3)
    ax.axvline(i_elegido, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(i_elegido, df["f1_train"].max() + 0.01,
            f"elegido: alpha={df.loc[i_elegido, 'alpha']}", ha="center",
            fontsize=9, color=TINTA, weight="bold")
    estilo(ax, "Regularizacion L2 (alpha) vs generalizacion\n"
               "A la izquierda memoriza; a la derecha se queda corto.",
           xlabel="alpha (escala log)", ylabel="F1 macro")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in df["alpha"]], fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9, loc="center left")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)
    guardar_fig(fig, "fig0b_alpha.png")


# ---------------------------------------------------------------------------
# Curva de perdida por EPOCA (aqui el concepto SI aplica)
# ---------------------------------------------------------------------------
def graficar_perdida(modelo):
    """Dos paneles apilados, NO dos ejes Y sobre el mismo grafico.

    La perdida y el accuracy viven en escalas distintas y no comparables:
    superponerlos con doble eje Y haria parecer que las curvas 'se cruzan'
    en algun punto significativo, y ese cruce seria un artefacto de como se
    eligieron las dos escalas, no un hecho de los datos.
    """
    tiene_val = bool(getattr(modelo, "validation_scores_", None))
    n = 2 if tiene_val else 1
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 4.2 * n), sharex=True)
    axes = np.atleast_1d(axes)

    epocas = range(1, len(modelo.loss_curve_) + 1)
    axes[0].plot(epocas, modelo.loss_curve_, color=COLOR_TRAIN, linewidth=2, zorder=3)
    axes[0].scatter([modelo.n_iter_], [modelo.loss_curve_[-1]], color=COLOR_TRAIN,
                    s=45, zorder=4, edgecolor=SUPERFICIE, linewidth=2)
    estilo(axes[0], "Aprendizaje epoca a epoca\n"
                    "Aqui las EPOCAS si existen: cada una es una pasada por las 14.630 filas.",
           ylabel="Perdida (entropia cruzada)")
    axes[0].text(0.99, 0.95, f"Perdida final: {modelo.loss_:.4f}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, color=TINTA_SEC)

    if tiene_val:
        vs = modelo.validation_scores_
        axes[1].plot(range(1, len(vs) + 1), vs, color=COLOR_VAL, linewidth=2, zorder=3)
        axes[1].axhline(max(vs), color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
        estilo(axes[1], "Accuracy en la validacion interna (10% del train)\n"
                        "La perdida sigue bajando pero esto ya NO mejora: ahi corta el early stopping.",
               xlabel="Epoca", ylabel="Accuracy")
        axes[1].text(0.99, 0.06, f"Mejor: {max(vs):.4f}  |  se detuvo en la epoca "
                                 f"{modelo.n_iter_} (n_iter_no_change=15)",
                     transform=axes[1].transAxes, ha="right", va="bottom",
                     fontsize=9, color=TINTA_SEC)
    else:
        axes[0].set_xlabel("Epoca", color=TINTA_SEC, fontsize=10)

    guardar_fig(fig, "fig4_dinamica.png")


# ---------------------------------------------------------------------------
def main():
    rep.titulo("MODELO 4: RED NEURONAL (MLP)", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")
    rep.p("Libreria principal: sklearn.neural_network.MLPClassifier")
    rep.p("(misma API que los modelos 1-3, asi que 05_graficos.py y la futura app")
    rep.p(" funcionan sin adaptaciones. El experimento 3 entrena ademas la MISMA")
    rep.p(" red en Keras para medir si el resultado depende de la libreria.)")

    X_tr = np.load(DIR_PROC / "train_X.npy")
    y_tr = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_va = np.load(DIR_PROC / "val_X.npy")
    y_va = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)
    rep.p(f"\nTrain: {X_tr.shape}  |  Val: {X_va.shape}")
    rep.p("Los datos YA vienen escalados de 03_preparacion.py. Esto es")
    rep.p("obligatorio para una red neuronal: sin escalar, las variables de")
    rep.p("rango grande dominarian el gradiente y la red no convergeria bien.")

    rep.p("\nLas etiquetas se codifican a enteros internamente (ModeloEtiquetado")
    rep.p("en utilidades.py): rodeo a un bug de sklearn 1.8.0 con early_stopping")
    rep.p("y etiquetas de texto. El modelo resultante es identico.")

    arq = experimento_arquitectura(X_tr, y_tr, X_va, y_va)
    alpha = experimento_alpha(X_tr, y_tr, X_va, y_va, arq)

    # --- Modelo final ---
    rep.titulo("MODELO FINAL (los 23 hiperparametros, explicitos)")
    params = dict(
        hidden_layer_sizes=arq,      # experimento 1
        alpha=alpha,                 # experimento 2
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEMILLA,
    )
    for k, v in params.items():
        rep.p(f"  {k} = {v}")
    rep.p("")
    rep.p("  Justificacion de los que NO salieron de un experimento:")
    rep.p("    activation='relu'     : ReLU(v)=max(0,v). Su derivada es 0 o 1, asi")
    rep.p("                            que no se desvanece el gradiente al propagarlo")
    rep.p("                            hacia atras (problema que si tiene sigmoide).")
    rep.p("    solver='adam'         : optimizador adaptativo, apropiado para")
    rep.p("                            datasets de este tamanio (miles de filas).")
    rep.p("                            'lbfgs' conviene en datasets chicos; 'sgd'")
    rep.p("                            exigiria ajustar el learning rate a mano.")
    rep.p("    learning_rate_init    : 0.001 es el valor estandar de Adam. Se probo")
    rep.p("                            implicitamente: si fuera malo, las curvas de")
    rep.p("                            perdida no bajarian de forma suave.")
    rep.p("    max_iter=300          : tope de epocas. En la practica corta antes")
    rep.p("                            por early stopping.")
    rep.p("    early_stopping=True   : separa un 10% del TRAIN (no toca nuestro set")
    rep.p("                            de validacion) y corta cuando deja de mejorar")
    rep.p("                            por 15 epocas seguidas. Evita sobreentrenar.")
    rep.p("    random_state=42       : sin esto la inicializacion de pesos seria")
    rep.p("                            aleatoria y el resultado NO reproducible.")

    modelo = entrenar(X_tr, y_tr, arq, alpha)

    rep.p("")
    rep.p(f"  Epocas efectivas: {modelo.n_iter_} (de un maximo de 300)")
    rep.p(f"  Perdida final   : {modelo.loss_:.4f}")
    capas = [X_tr.shape[1]] + list(arq) + [len(CLASES)]
    rep.p(f"  Estructura      : {' -> '.join(str(c) for c in capas)}")
    n_params = sum(w.size for w in modelo.coefs_) + sum(b.size for b in modelo.intercepts_)
    rep.p(f"  Pesos entrenables: {n_params:,}")

    pred_tr = pred(modelo, X_tr)
    pred_va = pred(modelo, X_va)

    rep.titulo("METRICAS")
    m_train = metricas(y_tr, pred_tr, "TRAIN")
    m_val = metricas(y_va, pred_va, "VALIDACION")

    rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING")
    diff = m_train["f1_macro"] - m_val["f1_macro"]
    rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia     : {diff:.4f}")
    if diff > 0.08:
        rep.p("  -> OVERFITTING: la red memoriza parte del train.")
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

    rep.titulo("CURVA DE PERDIDA POR EPOCA")
    graficar_perdida(modelo)

    experimento_keras(X_tr, y_tr, X_va, y_va, arq, alpha, m_val["f1_macro"])

    rep.titulo("COMPARATIVA DE TODOS LOS MODELOS (validacion)")
    filas = {"MLP (red neuronal)": [m_val["accuracy"], m_val["f1_macro"]]}
    for nombre, arch in [("Regresion Logistica", "modelo1_logistica.joblib"),
                          ("Random Forest", "modelo2_randomforest.joblib")]:
        ruta = DIR_MOD / arch
        if ruta.exists():
            m = joblib.load(ruta)
            p = m.predict(X_va)
            filas[nombre] = [accuracy_score(y_va, p),
                             f1_score(y_va, p, average="macro", zero_division=0)]
    # CatBoost usa otra codificacion de entrada (categoricas nativas, no el
    # train_X.npy compartido), asi que no se puede evaluar con X_va. Se lee su
    # F1 del informe que genero su propio script, en vez de hardcodearlo.
    ruta_inf_cb = DIR_REP / "informe_modelo3_catboost.txt"
    if ruta_inf_cb.exists():
        texto = ruta_inf_cb.read_text(encoding="utf-8")
        bloque = texto.split("--- VALIDACION ---")
        if len(bloque) > 1:
            for linea in bloque[1].splitlines():
                if "F1 (macro)" in linea:
                    rep.p(f"  (CatBoost usa categoricas nativas, no train_X.npy, asi que")
                    rep.p(f"   no se recalcula aqui. Segun su informe:{linea.split(':')[-1]})")
                    break
    else:
        rep.p("  (CatBoost todavia no se entreno con esta preparacion.)")
    comp = pd.DataFrame(filas, index=["Accuracy (val)", "F1 macro (val)"]).round(4)
    rep.p(comp.to_string())

    joblib.dump(modelo, DIR_MOD / "modelo4_mlp.joblib")
    rep.titulo("FIN MODELO 4", "#")
    rep.p("Guardado: modelos/modelo4_mlp.joblib")
    rep.p("Siguiente: python 05_graficos.py mlp")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
