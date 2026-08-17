"""
ENTRENAMIENTO V2 - EXPERIMENTO: ¿cuanta granularidad geografica conviene?
==========================================================================

MOTIVACION (medida, no supuesta)
---------------------------------
Con datos 100% reales, ARMA_CONTUNDENTE tiene solo 1.180 casos repartidos
en 162 cantones distintos, y 37 de esos cantones aparecen UNA SOLA VEZ.
Al hacer one-hot, 'canton' genera 211 columnas que para esa clase estan
casi siempre en cero: el modelo no tiene con que aprender.

La hipotesis es que bajar la granularidad (usar provincia o zona en vez de
canton) da menos columnas pero MEJOR pobladas, y eso deberia ayudar
especialmente a las clases chicas.

NIVELES QUE SE PRUEBAN
  completo   : zona + provincia + canton   (16 cat. -> ~250 columnas)
  sin_canton : zona + provincia            (menos granular)
  solo_prov  : provincia                   (24 categorias)
  solo_zona  : zona                        (9 categorias, el mas grueso)

Se prueba sobre las DOS variantes (A: class_weight / B: recorte) con Random
Forest y los mismos hiperparametros, para que la unica diferencia sea la
geografia.

Se mira F1 macro, pero sobre todo el F1 DE ARMA_CONTUNDENTE, que es la clase
que motivo el experimento.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
SEMILLA = 42
DIR_DATA = BASE / "data"
DIR_REP = BASE / "resultados"
DIR_FIG = BASE / "Graficos"

CLASES = ["ARMA_FUEGO", "ARMA_BLANCA", "ARMA_CONTUNDENTE", "OTRAS"]

# Columnas NO geograficas (constantes en todos los niveles)
COLS_BASE = [
    "area_hecho", "lugar", "tipo_lugar", "presunta_motivacion",
    "presun_motiva_observada", "sexo", "etnia", "estado_civil",
    "nacionalidad", "discapacidad", "dia_semana", "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad"]
COL_BINARIA = ["es_fin_semana"]

NIVELES = {
    "completo":   ["zona", "provincia", "canton"],
    "sin_canton": ["zona", "provincia"],
    "solo_prov":  ["provincia"],
    "solo_zona":  ["zona"],
}

COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
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


rep = Reporte(DIR_REP / "informe_experimento_geografia.txt")


def cargar(nombre):
    tr = pd.read_csv(DIR_DATA / f"{nombre}_train_raw.csv")
    va = pd.read_csv(DIR_DATA / f"{nombre}_val_raw.csv")
    return tr, va


def evaluar(tr, va, cols_geo, class_weight):
    cols_cat = cols_geo + COLS_BASE
    for df in (tr, va):
        df[cols_cat] = df[cols_cat].fillna("DESCONOCIDO").astype(str)

    pre = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), COLS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat),
        ("bin", "passthrough", COL_BINARIA),
    ])
    cols = cols_cat + COLS_NUMERICAS + COL_BINARIA
    X_tr = pre.fit_transform(tr[cols])
    X_va = pre.transform(va[cols])

    m = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_leaf=2,
                               max_features="sqrt", class_weight=class_weight,
                               random_state=SEMILLA, n_jobs=-1)
    m.fit(X_tr, tr["y"])
    pred = m.predict(X_va)

    f1_macro = f1_score(va["y"], pred, average="macro", zero_division=0)
    f1_por_clase = f1_score(va["y"], pred, labels=CLASES, average=None, zero_division=0)
    return X_tr.shape[1], f1_macro, dict(zip(CLASES, f1_por_clase))


def correr_variante(nombre_variante, etiqueta, class_weight):
    rep.titulo(f"{etiqueta} (class_weight={class_weight})")
    tr, va = cargar(nombre_variante)
    rep.p(f"  train={len(tr):,}  val={len(va):,}")
    rep.p("")
    rep.p(f"  {'nivel':>12} | {'columnas':>8} | {'F1 macro':>9} | "
          f"{'F1 CONTUND.':>11} | {'F1 OTRAS':>9} | {'F1 BLANCA':>9} | {'F1 FUEGO':>9}")
    rep.p("  " + "-" * 92)

    filas = []
    for nivel, cols_geo in NIVELES.items():
        n_cols, f1m, f1c = evaluar(tr.copy(), va.copy(), cols_geo, class_weight)
        rep.p(f"  {nivel:>12} | {n_cols:>8} | {f1m:9.4f} | "
              f"{f1c['ARMA_CONTUNDENTE']:11.4f} | {f1c['OTRAS']:9.4f} | "
              f"{f1c['ARMA_BLANCA']:9.4f} | {f1c['ARMA_FUEGO']:9.4f}")
        filas.append({"nivel": nivel, "columnas": n_cols, "f1_macro": f1m, **f1c})

    df = pd.DataFrame(filas)
    mejor_macro = df.loc[df["f1_macro"].idxmax()]
    mejor_cont = df.loc[df["ARMA_CONTUNDENTE"].idxmax()]
    rep.p("")
    rep.p(f"  Mejor F1 macro           : {mejor_macro['nivel']} ({mejor_macro['f1_macro']:.4f})")
    rep.p(f"  Mejor F1 ARMA_CONTUNDENTE: {mejor_cont['nivel']} ({mejor_cont['ARMA_CONTUNDENTE']:.4f})")
    return df


def graficar(df_a, df_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(NIVELES))
    ancho = 0.36

    for ax, col, titulo in [
        (ax1, "f1_macro", "F1 macro (todas las clases)"),
        (ax2, "ARMA_CONTUNDENTE", "F1 de ARMA_CONTUNDENTE\n(la clase que motivo el experimento)"),
    ]:
        b1 = ax.bar(x - ancho/2, df_a[col], ancho, label="Variante A (class_weight)",
                    color=COLOR_A, zorder=3)
        b2 = ax.bar(x + ancho/2, df_b[col], ancho, label="Variante B (recorte)",
                    color=COLOR_B, zorder=3)
        for barras in (b1, b2):
            for b in barras:
                b.set_edgecolor(SUPERFICIE)
                b.set_linewidth(2)
        for xi, (va_, vb_) in enumerate(zip(df_a[col], df_b[col])):
            ax.text(xi - ancho/2, va_ + 0.008, f"{va_:.3f}", ha="center", fontsize=8, color=TINTA)
            ax.text(xi + ancho/2, vb_ + 0.008, f"{vb_:.3f}", ha="center", fontsize=8, color=TINTA)

        ax.set_facecolor(SUPERFICIE)
        ax.set_title(titulo, color=TINTA, fontsize=12, pad=14, loc="left", weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n({c} cols)" for n, c in zip(df_a["nivel"], df_a["columnas"])],
                           fontsize=8.5)
        ax.set_ylabel("F1", color=TINTA_SEC, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    leg = ax1.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TINTA_SEC)

    fig.suptitle("¿Menos granularidad geografica ayuda a las clases chicas?",
                 color=TINTA, fontsize=13, weight="bold", x=0.01, ha="left")
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    ruta = DIR_FIG / "experimento_geografia.png"
    fig.savefig(ruta, dpi=160, facecolor=SUPERFICIE)
    plt.close(fig)
    rep.p(f"\nGrafico: Graficos/{ruta.name}")


def main():
    rep.titulo("EXPERIMENTO: GRANULARIDAD GEOGRAFICA", "#")
    rep.p(f"Python {sys.version.split()[0]} | semilla {SEMILLA}")
    rep.p("Random Forest con hiperparametros FIJOS (n_estimators=300,")
    rep.p("max_depth=25, min_samples_leaf=2). Lo unico que cambia es que")
    rep.p("columnas geograficas entran.")

    df_a = correr_variante("variante_a", "VARIANTE A", "balanced")
    df_b = correr_variante("variante_b", "VARIANTE B", None)

    rep.titulo("CONCLUSION")
    base_a = df_a[df_a["nivel"] == "completo"].iloc[0]
    base_b = df_b[df_b["nivel"] == "completo"].iloc[0]
    for nombre, df, base in [("A", df_a, base_a), ("B", df_b, base_b)]:
        mejor = df.loc[df["ARMA_CONTUNDENTE"].idxmax()]
        delta = mejor["ARMA_CONTUNDENTE"] - base["ARMA_CONTUNDENTE"]
        rep.p(f"  Variante {nombre}: bajar de 'completo' a '{mejor['nivel']}' cambia el F1")
        rep.p(f"              de ARMA_CONTUNDENTE en {delta:+.4f} "
              f"({base['ARMA_CONTUNDENTE']:.4f} -> {mejor['ARMA_CONTUNDENTE']:.4f})")

    graficar(df_a, df_b)
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
