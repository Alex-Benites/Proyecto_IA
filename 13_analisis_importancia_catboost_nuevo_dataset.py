"""
ANALISIS DE IMPORTANCIA Y ABLACION DE VARIABLES - CATBOOST
===========================================================

Objetivo:
1. Cargar el nuevo dataset equilibrado.
2. Dividir 80% TRAIN / 10% VALIDATION / 10% TEST de forma estratificada.
3. Mantener TEST completamente sellado.
4. Entrenar un CatBoost de referencia SOLO con TRAIN.
5. Medir importancia de variables mediante:
      a) CatBoost PredictionValuesChange.
      b) Permutation Importance sobre VALIDATION usando F1 macro.
6. Tomar las variables menos importantes como CANDIDATAS.
7. Hacer ablacion: eliminar una candidata, reentrenar con TRAIN
   y comprobar si el F1 macro de VALIDATION mejora o se mantiene.

IMPORTANTE:
- Este script NO decide de forma ciega qué variables eliminar.
- Una variable solo se considera candidata si aparece con baja importancia.
- La evidencia final para eliminarla es la ablacion.
- TEST se guarda, pero NO se usa para seleccionar variables.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# CONFIGURACION
# =============================================================================

BASE = Path(__file__).resolve().parent

RUTA_DATASET = (
    BASE
    / "data"
    / "processed"
    / "A_2014_2025_muestraFinalV2.xls"
)

DIR_PROC = BASE / "data" / "processed"
DIR_RESULTADOS = BASE / "resultados" / "seleccion_variables"

DIR_PROC.mkdir(parents=True, exist_ok=True)
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

TARGET = "y"
SEMILLA = 42

CLASES = [
    "ARMA_FUEGO",
    "ARMA_BLANCA",
    "ARMA_CONTUNDENTE",
    "OTRAS",
]

# Se mantiene la misma lógica utilizada anteriormente:
# edad y es_fin_semana son numéricas.
# Las demás variables se tratan como categóricas.
NUMERICAS_ESPERADAS = [
    "edad",
    "es_fin_semana",
]

# Número de variables de menor importancia que serán sometidas
# a prueba de ablación individual.
N_CANDIDATOS_ABLACION = 5

# Tolerancia para considerar que quitar una variable no perjudica
# materialmente al modelo.
# 0.005 = medio punto porcentual de F1 macro.
TOLERANCIA_F1 = 0.005


# =============================================================================
# UTILIDADES
# =============================================================================

def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def preparar_datos_catboost(X_train, X_val, X_test):
    """
    Imputación aprendida SOLO en TRAIN.

    Aunque el nuevo dataset indique que no tiene vacíos, dejamos esta
    protección para evitar fallos inesperados.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    numericas = [
        c for c in NUMERICAS_ESPERADAS
        if c in X_train.columns
    ]

    categoricas = [
        c for c in X_train.columns
        if c not in numericas
    ]

    medianas = {}

    for c in numericas:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

        mediana = X_train[c].median()

        if pd.isna(mediana):
            mediana = 0.0

        medianas[c] = float(mediana)

        X_train[c] = X_train[c].fillna(mediana)
        X_val[c] = X_val[c].fillna(mediana)
        X_test[c] = X_test[c].fillna(mediana)

    for c in categoricas:
        X_train[c] = X_train[c].fillna("DESCONOCIDO").astype(str)
        X_val[c] = X_val[c].fillna("DESCONOCIDO").astype(str)
        X_test[c] = X_test[c].fillna("DESCONOCIDO").astype(str)

    return X_train, X_val, X_test, categoricas, numericas, medianas


def crear_modelo(cat_features):
    """
    Modelo de referencia para comparar variables.

    NO se presenta todavía como modelo final.
    Se usa la MISMA configuración en todas las ablaciones,
    de modo que la única diferencia sea la variable eliminada.
    """
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=7.0,
        loss_function="MultiClass",
        cat_features=cat_features,
        random_seed=SEMILLA,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluar(modelo, X, y):
    pred = np.asarray(modelo.predict(X)).reshape(-1).astype(str)

    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(
            y,
            pred,
            labels=CLASES,
            average="macro",
            zero_division=0,
        ),
        "pred": pred,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    titulo("1. CARGA DEL NUEVO DATASET")

    if not RUTA_DATASET.exists():
        raise FileNotFoundError(
            f"No existe el dataset:\n{RUTA_DATASET}"
        )

    df = pd.read_excel(RUTA_DATASET)

    print(f"Dataset: {RUTA_DATASET}")
    print(f"Dimensiones: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(
            f"No existe la columna objetivo '{TARGET}'."
        )

    df[TARGET] = df[TARGET].astype(str).str.strip()

    clases_desconocidas = sorted(
        set(df[TARGET].unique()) - set(CLASES)
    )

    if clases_desconocidas:
        raise ValueError(
            f"Hay clases inesperadas: {clases_desconocidas}"
        )

    print("\nDistribución:")
    print(df[TARGET].value_counts())

    # =========================================================================
    # SPLIT 80 / 10 / 10
    # =========================================================================

    titulo("2. DIVISION ESTRATIFICADA 80 / 10 / 10")

    COLUMNAS_NO_PREDICTORAS = [
        "arma_categoria",  # leakage directo del target

        "fecha",           # sesgo temporal / cobertura
        "anio",            # sesgo temporal / cobertura
        "mes",             # proxy del mismo sesgo temporal
    ]

    X = df.drop(
        columns=[TARGET] + COLUMNAS_NO_PREDICTORAS,
        errors="ignore"
    ).copy()

    y = df[TARGET].copy()

    # 80% train, 20% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEMILLA,
        stratify=y,
    )

    # 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEMILLA,
        stratify=y_temp,
    )

    (
        X_train,
        X_val,
        X_test,
        cat_cols,
        num_cols,
        medianas,
    ) = preparar_datos_catboost(
        X_train,
        X_val,
        X_test,
    )

    print(f"TRAIN: {len(X_train)} ({len(X_train)/len(df):.2%})")
    print(f"VAL  : {len(X_val)} ({len(X_val)/len(df):.2%})")
    print(f"TEST : {len(X_test)} ({len(X_test)/len(df):.2%})")

    print(f"\nCategoricas: {len(cat_cols)}")
    print(cat_cols)
    print(f"\nNumericas: {num_cols}")

    # Guardamos los splits para reutilizarlos después.
    train_raw = X_train.copy()
    train_raw[TARGET] = y_train.values

    val_raw = X_val.copy()
    val_raw[TARGET] = y_val.values

    test_raw = X_test.copy()
    test_raw[TARGET] = y_test.values

    train_raw.to_csv(
        DIR_PROC / "nuevo_train_raw.csv",
        index=False,
    )
    val_raw.to_csv(
        DIR_PROC / "nuevo_val_raw.csv",
        index=False,
    )
    test_raw.to_csv(
        DIR_PROC / "nuevo_test_raw.csv",
        index=False,
    )

    print("\nSplits guardados:")
    print("  data/processed/nuevo_train_raw.csv")
    print("  data/processed/nuevo_val_raw.csv")
    print("  data/processed/nuevo_test_raw.csv")
    print("\nTEST queda SELLADO desde este punto.")

    # =========================================================================
    # MODELO BASE DE REFERENCIA
    # =========================================================================

    titulo("3. CATBOOST DE REFERENCIA")

    modelo = crear_modelo(cat_cols)

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo = time.perf_counter() - inicio

    train_m = evaluar(
        modelo,
        X_train,
        y_train,
    )

    val_m = evaluar(
        modelo,
        X_val,
        y_val,
    )

    print(f"F1 macro TRAIN: {train_m['f1_macro']:.4f}")
    print(f"F1 macro VAL  : {val_m['f1_macro']:.4f}")
    print(
        f"Gap           : "
        f"{train_m['f1_macro'] - val_m['f1_macro']:.4f}"
    )
    print(f"Tiempo        : {tiempo:.2f} s")

    print("\nReporte VALIDATION:")
    print(
        classification_report(
            y_val,
            val_m["pred"],
            labels=CLASES,
            zero_division=0,
        )
    )

    # =========================================================================
    # IMPORTANCIA NATIVA DE CATBOOST
    # =========================================================================

    titulo("4. IMPORTANCIA NATIVA CATBOOST")

    importancia_cat = np.asarray(
        modelo.get_feature_importance(
            type="PredictionValuesChange"
        )
    )

    df_cat = pd.DataFrame({
        "feature": X_train.columns,
        "catboost_importance": importancia_cat,
    }).sort_values(
        "catboost_importance",
        ascending=False,
    )

    print(
        df_cat.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================

    titulo("5. PERMUTATION IMPORTANCE SOBRE VALIDATION")

    print(
        "Se permuta una variable a la vez y se mide cuánto cae F1 macro."
    )
    print(
        "Mayor caída = la variable era más útil para el modelo."
    )

    perm = permutation_importance(
        modelo,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=10,
        random_state=SEMILLA,
        n_jobs=1,
    )

    df_perm = pd.DataFrame({
        "feature": X_val.columns,
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    })

    # =========================================================================
    # TABLA COMBINADA
    # =========================================================================

    titulo("6. RANKING COMBINADO")

    ranking = df_cat.merge(
        df_perm,
        on="feature",
        how="inner",
    )

    ranking = ranking.sort_values(
        ["permutation_mean", "catboost_importance"],
        ascending=False,
    ).reset_index(drop=True)

    ranking["ranking_perm"] = (
        ranking["permutation_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking["ranking_catboost"] = (
        ranking["catboost_importance"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking.to_csv(
        DIR_RESULTADOS / "ranking_importancia_variables.csv",
        index=False,
    )

    print(
        ranking.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # =========================================================================
    # GRAFICO
    # =========================================================================

    orden = ranking.sort_values(
        "permutation_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        orden["feature"],
        orden["permutation_mean"],
    )
    ax.set_xlabel("Caída media de F1 macro al permutar")
    ax.set_title(
        "Permutation Importance - Validation"
    )
    fig.tight_layout()
    fig.savefig(
        DIR_RESULTADOS / "permutation_importance.png",
        dpi=160,
    )
    plt.close(fig)

    # =========================================================================
    # CANDIDATAS PARA ABLACION
    # =========================================================================

    titulo("7. VARIABLES CANDIDATAS A ELIMINAR")

    # Seleccionamos las N con menor permutation importance.
    candidatas = (
        ranking
        .sort_values(
            ["permutation_mean", "catboost_importance"],
            ascending=True,
        )
        .head(N_CANDIDATOS_ABLACION)["feature"]
        .tolist()
    )

    print(
        "Estas variables NO se eliminan todavía."
    )
    print(
        "Solamente pasan a una prueba de ablación:"
    )

    for c in candidatas:
        fila = ranking.loc[
            ranking["feature"] == c
        ].iloc[0]

        print(
            f"  {c:<30} "
            f"perm={fila['permutation_mean']:.6f} | "
            f"catboost={fila['catboost_importance']:.6f}"
        )

    # =========================================================================
    # ABLACION INDIVIDUAL
    # =========================================================================

    titulo("8. PRUEBA DE ABLACION INDIVIDUAL")

    print(
        f"F1 macro base VAL = {val_m['f1_macro']:.4f}"
    )
    print(
        f"Tolerancia permitida = {TOLERANCIA_F1:.4f}"
    )

    resultados_ablacion = []

    for feature in candidatas:

        print(f"\nProbando SIN: {feature}")

        columnas_reducidas = [
            c for c in X_train.columns
            if c != feature
        ]

        Xtr = X_train[columnas_reducidas].copy()
        Xv = X_val[columnas_reducidas].copy()

        cats_reducidas = [
            c for c in cat_cols
            if c != feature
        ]

        modelo_reducido = crear_modelo(
            cats_reducidas
        )

        inicio = time.perf_counter()

        modelo_reducido.fit(
            Xtr,
            y_train,
        )

        t = time.perf_counter() - inicio

        met = evaluar(
            modelo_reducido,
            Xv,
            y_val,
        )

        delta = (
            met["f1_macro"]
            - val_m["f1_macro"]
        )

        # Si la caída no supera la tolerancia,
        # la variable es provisionalmente eliminable.
        eliminable = (
            delta >= -TOLERANCIA_F1
        )

        resultados_ablacion.append({
            "feature_eliminada": feature,
            "f1_base": val_m["f1_macro"],
            "f1_sin_feature": met["f1_macro"],
            "delta_f1": delta,
            "eliminable_provisional": eliminable,
            "tiempo_segundos": t,
        })

        print(
            f"  F1 sin variable : {met['f1_macro']:.4f}"
        )
        print(
            f"  Cambio F1       : {delta:+.4f}"
        )
        print(
            "  Resultado        : "
            + (
                "CANDIDATA A ELIMINAR"
                if eliminable
                else "CONSERVAR"
            )
        )

    df_ab = pd.DataFrame(
        resultados_ablacion
    ).sort_values(
        "delta_f1",
        ascending=False,
    )

    df_ab.to_csv(
        DIR_RESULTADOS / "resultado_ablacion_individual.csv",
        index=False,
    )

    titulo("9. RESUMEN FINAL")

    print(
        df_ab.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nInterpretación:")
    print(
        "- delta_f1 > 0: quitar la variable mejoró VALIDATION."
    )
    print(
        "- delta_f1 cerca de 0: aporta muy poco al modelo."
    )
    print(
        "- delta_f1 muy negativo: quitarla perjudicó; debe conservarse."
    )

    print("\nIMPORTANTE:")
    print(
        "No se utilizó TEST para medir importancia ni para decidir variables."
    )
    print(
        "Después de seleccionar features se deben volver a entrenar "
        "los 3 escenarios de CatBoost y recién al final evaluar TEST."
    )

    print("\nArchivos:")
    print(
        "  resultados/seleccion_variables/ranking_importancia_variables.csv"
    )
    print(
        "  resultados/seleccion_variables/resultado_ablacion_individual.csv"
    )
    print(
        "  resultados/seleccion_variables/permutation_importance.png"
    )


if __name__ == "__main__":
    main()
