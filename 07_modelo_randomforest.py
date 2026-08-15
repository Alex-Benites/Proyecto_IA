"""
MODELO 2: RANDOM FOREST
=========================

Por que este modelo:
  - Es la propuesta original del proyecto.
  - A diferencia de la regresion logistica (que solo puede trazar fronteras
    LINEALES), un bosque captura INTERACCIONES: por ejemplo "si es violencia
    intrafamiliar Y ocurre en casa Y la victima es mujer" puede llevar a una
    prediccion distinta que cada condicion por separado. La logistica no
    puede representar eso salvo que se creen las interacciones a mano.

QUE ES RANDOM FOREST (para la defensa, sin caja negra):
  Es un conjunto ("ensemble") de N arboles de decision. Cada arbol se
  entrena con DOS fuentes de azar:
    1. Bootstrap: una muestra aleatoria CON REEMPLAZO de las filas de train.
       Cada arbol ve un subconjunto distinto de los datos.
    2. max_features: en cada division, el arbol solo considera un subconjunto
       aleatorio de las columnas, no las 476.
  La prediccion final es el VOTO de los N arboles (para probabilidades, el
  promedio de las probabilidades de cada arbol).
  La idea de fondo: arboles individuales se equivocan, pero se equivocan de
  forma DISTINTA gracias al azar, y al promediar los errores se cancelan.

COMO DIVIDE CADA ARBOL (la formula):
  En cada nodo prueba divisiones y elige la que mas reduce la IMPUREZA DE GINI:
      Gini(nodo) = 1 - suma_k (p_k)^2
  donde p_k es la proporcion de la clase k en ese nodo. Gini = 0 significa
  nodo puro (una sola clase). La ganancia de una division es:
      ganancia = Gini(padre) - [ n_izq/n * Gini(izq) + n_der/n * Gini(der) ]
  El arbol se queda con la division de mayor ganancia.

NO ES CAJA NEGRA:
  - No hay seleccion automatica de modelo ni de hiperparametros.
  - La profundidad se elige con un EXPERIMENTO EXPLICITO documentado abajo
    (barrido manual de max_depth, con su grafico), no con GridSearchCV.
  - El modelo expone feature_importances_, que se grafica.

SOBRE EL DESBALANCE:
  No se usa class_weight porque dataset.xlsx ya viene balanceado (~25% por
  clase). Aplicarlo encima corregiria un desbalance que ya no existe.

Salida:
  modelos/modelo2_randomforest.joblib
  resultados/modelos/informe_modelo2_randomforest.txt
  Graficos/modelo2_randomforest_fig0_profundidad.png
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
from sklearn.ensemble import RandomForestClassifier
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

# Estilo de graficos (mismo sistema que 05_graficos.py)
COLOR_TRAIN, COLOR_VAL = "#2a78d6", "#eb6834"
TINTA, TINTA_SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SUPERFICIE = "#e1e0d9", "#fcfcfb"

# Profundidades a probar en el experimento explicito. None = sin limite.
PROFUNDIDADES = [4, 6, 8, 10, 12, 15, 20, 25, 30, None]


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


rep = Reporte(DIR_REP / "informe_modelo2_randomforest.txt")


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


def experimento_profundidad(X_train, y_train, X_val, y_val):
    """Barrido MANUAL y documentado de max_depth.

    Se hace con pocos arboles (60) porque aqui solo interesa la FORMA de la
    curva, no el rendimiento final. El modelo definitivo usa mas arboles.
    """
    rep.titulo("EXPERIMENTO 1: ¿que profundidad usar? (barrido manual)")
    rep.p("  Se prueba cada profundidad y se mide F1 en train y en validacion.")
    rep.p("  Objetivo: encontrar donde el modelo deja de generalizar y empieza")
    rep.p("  a memorizar. NO se usa GridSearchCV: el barrido es explicito.")
    rep.p("")
    rep.p(f"  {'max_depth':>10} | {'F1 train':>9} | {'F1 val':>9} | {'brecha':>7}")
    rep.p("  " + "-" * 45)

    filas = []
    for d in PROFUNDIDADES:
        m = RandomForestClassifier(
            n_estimators=60, max_depth=d, random_state=SEMILLA, n_jobs=-1)
        m.fit(X_train, y_train)
        f1_tr = f1_score(y_train, m.predict(X_train), average="macro", zero_division=0)
        f1_va = f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)
        etiqueta = "sin limite" if d is None else str(d)
        rep.p(f"  {etiqueta:>10} | {f1_tr:9.4f} | {f1_va:9.4f} | {f1_tr - f1_va:7.4f}")
        filas.append({"etiqueta": etiqueta, "f1_train": f1_tr,
                      "f1_val": f1_va, "brecha": f1_tr - f1_va})

    df = pd.DataFrame(filas)

    # CRITERIO DE SELECCION (regla de parsimonia, documentada):
    # No se toma el maximo F1 a secas. Se toma la profundidad MAS PEQUENIA
    # cuyo F1 de validacion este dentro del 1% del mejor.
    # Motivo: entre max_depth=20 y sin limite la ganancia de F1 es minima,
    # pero la brecha train-val (el overfitting) crece mucho. Un modelo mas
    # simple con rendimiento equivalente es preferible: generaliza mejor
    # ante datos nuevos y es mas defendible academicamente.
    f1_mejor = df["f1_val"].max()
    umbral = f1_mejor * 0.99
    candidatos = df.index[df["f1_val"] >= umbral].tolist()
    i_elegido = candidatos[0]          # el primero = la profundidad mas chica
    depth_elegida = PROFUNDIDADES[i_elegido]

    rep.p("")
    rep.p(f"  Mejor F1 absoluto      : {f1_mejor:.4f} "
          f"(max_depth={df.loc[df['f1_val'].idxmax(), 'etiqueta']}, "
          f"brecha {df['brecha'].max():.4f})")
    rep.p(f"  Umbral del 1%          : {umbral:.4f}")
    rep.p(f"  ELEGIDA (mas simple dentro del umbral): max_depth="
          f"{df.loc[i_elegido, 'etiqueta']} -> F1 val {df.loc[i_elegido, 'f1_val']:.4f}, "
          f"brecha {df.loc[i_elegido, 'brecha']:.4f}")
    rep.p("  Regla: entre modelos de rendimiento equivalente se prefiere el mas")
    rep.p("  simple (navaja de Occam). Gana poco F1 y ahorra mucho overfitting.")

    graficar_profundidad(df, i_elegido)
    return depth_elegida, df


def graficar_profundidad(df, i_elegido):
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, df["f1_train"], color=COLOR_TRAIN, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Entrenamiento", zorder=3)
    ax.plot(x, df["f1_val"], color=COLOR_VAL, linewidth=2, marker="o",
            markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
            label="Validacion", zorder=3)

    # Sombrea la brecha train-val: visualmente ES el overfitting.
    ax.fill_between(list(x), df["f1_val"], df["f1_train"], color=COLOR_VAL,
                    alpha=0.10, zorder=1, label="Brecha = overfitting")

    ax.axvline(i_elegido, color=MUTED, linestyle=":", linewidth=1.4, zorder=2)
    ax.text(i_elegido, df["f1_train"].max() + 0.03,
            f"elegido: max_depth={df.loc[i_elegido, 'etiqueta']}", ha="center",
            fontsize=9, color=TINTA, weight="bold")

    ax.set_facecolor(SUPERFICIE)
    ax.set_title("Experimento: profundidad del bosque vs generalizacion\n"
                 "La brecha entre las curvas ES el overfitting.",
                 color=TINTA, fontsize=12, pad=14, loc="left", weight="bold")
    ax.set_xlabel("max_depth (profundidad maxima de cada arbol)", color=TINTA_SEC, fontsize=10)
    ax.set_ylabel("F1 macro", color=TINTA_SEC, fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["etiqueta"], fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9, loc="center right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "modelo2_randomforest_fig0_profundidad.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"  Grafico: Graficos/{ruta.name}")


def main():
    rep.titulo("MODELO 2: RANDOM FOREST", "#")
    rep.p(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
          f"semilla {SEMILLA}")

    X_train = np.load(DIR_PROC / "train_X.npy")
    y_train = np.load(DIR_PROC / "train_y.npy", allow_pickle=True)
    X_val = np.load(DIR_PROC / "val_X.npy")
    y_val = np.load(DIR_PROC / "val_y.npy", allow_pickle=True)
    rep.p(f"\nTrain: {X_train.shape}  |  Val: {X_val.shape}")

    # --- Experimento explicito de profundidad ---
    mejor_depth, tabla = experimento_profundidad(X_train, y_train, X_val, y_val)

    # --- Modelo final ---
    rep.titulo("MODELO FINAL (hiperparametros explicitos)")
    params = dict(
        n_estimators=300,          # mas arboles = prediccion mas estable
        max_depth=mejor_depth,     # elegido por el experimento de arriba
        min_samples_leaf=2,        # una hoja con 1 sola fila memoriza esa fila
        max_features="sqrt",       # sqrt(476) ~ 22 columnas por division
        random_state=SEMILLA,      # reproducibilidad del bootstrap
        n_jobs=-1,
    )
    for k, v in params.items():
        rep.p(f"  {k} = {v}")
    rep.p("")
    rep.p("  Justificacion de cada uno:")
    rep.p("    n_estimators=300  : con la curva de arboles se vera que a partir")
    rep.p("                        de ~100 la mejora es marginal; 300 da margen.")
    rep.p("    max_depth         : lo fijo el experimento 1 (ver tabla).")
    rep.p("    min_samples_leaf=2: impide hojas de 1 sola fila (memorizacion pura).")
    rep.p("    max_features=sqrt : el estandar para clasificacion. Es la fuente de")
    rep.p("                        decorrelacion entre arboles; sin esto todos los")
    rep.p("                        arboles se pareceran y el ensemble no sirve.")
    rep.p("    class_weight      : None. El dataset ya viene balanceado.")

    modelo = RandomForestClassifier(**params)
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    rep.titulo("METRICAS")
    m_train = metricas(y_train, pred_train, "TRAIN")
    m_val = metricas(y_val, pred_val, "VALIDACION")

    rep.titulo("CHEQUEO DE OVERFITTING / UNDERFITTING")
    diff = m_train["f1_macro"] - m_val["f1_macro"]
    rep.p(f"  F1 macro train : {m_train['f1_macro']:.4f}")
    rep.p(f"  F1 macro val   : {m_val['f1_macro']:.4f}")
    rep.p(f"  Diferencia     : {diff:.4f}")
    if diff > 0.08:
        rep.p("  -> OVERFITTING: el bosque memoriza el train. Es el comportamiento")
        rep.p("     esperado de un Random Forest profundo; lo relevante es si aun")
        rep.p("     asi generaliza MEJOR que la logistica en validacion.")
    elif m_train["f1_macro"] < 0.5:
        rep.p("  -> Posible UNDERFITTING.")
    else:
        rep.p("  -> Sin senales fuertes de over/underfitting.")

    rep.titulo("REPORTE POR CLASE (VALIDACION)")
    rep.p(classification_report(y_val, pred_val, labels=CLASES, zero_division=0))

    rep.titulo("MATRIZ DE CONFUSION (VALIDACION)")
    cm = confusion_matrix(y_val, pred_val, labels=CLASES)
    cm_df = pd.DataFrame(cm, index=[f"real_{c}" for c in CLASES],
                         columns=[f"pred_{c}" for c in CLASES])
    rep.p(cm_df.to_string())

    rep.titulo("COMPARATIVA CONTRA EL MODELO 1")
    ruta_m1 = DIR_MOD / "modelo1_logistica.joblib"
    if ruta_m1.exists():
        m1 = joblib.load(ruta_m1)
        p1 = m1.predict(X_val)
        f1_m1 = f1_score(y_val, p1, average="macro", zero_division=0)
        acc_m1 = accuracy_score(y_val, p1)
        comp = pd.DataFrame({
            "Regresion Logistica": [acc_m1, f1_m1],
            "Random Forest": [m_val["accuracy"], m_val["f1_macro"]],
        }, index=["Accuracy (val)", "F1 macro (val)"]).round(4)
        comp["Diferencia"] = (comp["Random Forest"] - comp["Regresion Logistica"]).round(4)
        rep.p(comp.to_string())

    joblib.dump(modelo, DIR_MOD / "modelo2_randomforest.joblib")
    rep.titulo("FIN MODELO 2", "#")
    rep.p("Guardado: modelos/modelo2_randomforest.joblib")
    rep.p("Siguiente: python 05_graficos.py randomforest")
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
