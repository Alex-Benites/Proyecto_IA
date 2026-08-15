# =============================================================
# 03_evaluacion.py — Evaluación y comparación de los 5 modelos
# Ejecutar después de 02_modelos.py
# =============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # para guardar gráficos sin pantalla
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

os.makedirs("resultados/", exist_ok=True)

# ── Carga de datos de prueba ──────────────────────────────────
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

CLASES = ["ARMA DE FUEGO", "ARMA BLANCA", "ARMA CONTUNDENTE", "CONSTRICTORA", "OTROS"]
N_CLASES = 5

# ── Función auxiliar: evaluar un modelo ──────────────────────
def evaluar_modelo(nombre, y_pred, y_proba=None):
    """Calcula todas las métricas y genera la matriz de confusión."""
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(N_CLASES)))
    if y_proba is not None:
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    else:
        auc = float("nan")

    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"{'='*55}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision (macro):  {prec:.4f}")
    print(f"  Recall (macro):     {rec:.4f}")
    print(f"  F1-score (macro):   {f1:.4f}")
    print(f"  AUC-ROC (macro):    {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASES, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASES))
    ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASES, fontsize=9)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {nombre}")
    for i in range(N_CLASES):
        for j in range(N_CLASES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    plt.tight_layout()
    nombre_archivo = nombre.lower().replace(" ", "_").replace("á","a").replace("é","e")
    plt.savefig(f"resultados/confusion_{nombre_archivo}.png", dpi=120)
    plt.close()
    print(f"  Matriz guardada en resultados/confusion_{nombre_archivo}.png")

    return {"modelo": nombre, "accuracy": acc, "precision": prec,
            "recall": rec, "f1_macro": f1, "auc_roc": auc}

# =============================================================
# Evaluación de cada modelo
# =============================================================
resultados = []

# ── Modelo 1: Regresión Logística ─────────────────────────────
lr = joblib.load("modelos/01_regresion_logistica.pkl")
y_pred_lr   = lr.predict(X_test)
y_proba_lr  = lr.predict_proba(X_test)
resultados.append(evaluar_modelo("Regresion Logistica", y_pred_lr, y_proba_lr))

# ── Modelo 2: Árbol de Decisión ───────────────────────────────
dt = joblib.load("modelos/02_arbol_decision.pkl")
y_pred_dt   = dt.predict(X_test)
y_proba_dt  = dt.predict_proba(X_test)
resultados.append(evaluar_modelo("Arbol de Decision", y_pred_dt, y_proba_dt))

# ── Modelo 3: Random Forest ───────────────────────────────────
rf = joblib.load("modelos/03_random_forest.pkl")
y_pred_rf   = rf.predict(X_test)
y_proba_rf  = rf.predict_proba(X_test)
resultados.append(evaluar_modelo("Random Forest", y_pred_rf, y_proba_rf))

# ── Modelo 4: XGBoost ─────────────────────────────────────────
xgb = joblib.load("modelos/04_xgboost.pkl")
y_pred_xgb  = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)
resultados.append(evaluar_modelo("XGBoost", y_pred_xgb, y_proba_xgb))

# ── Modelo 5: MLP ─────────────────────────────────────────────
mlp = tf.keras.models.load_model("modelos/05_mlp.keras")
y_proba_mlp = mlp.predict(X_test)
y_pred_mlp  = np.argmax(y_proba_mlp, axis=1)
resultados.append(evaluar_modelo("MLP Red Neuronal", y_pred_mlp, y_proba_mlp))

# =============================================================
# Tabla comparativa final
# =============================================================
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values("f1_macro", ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("TABLA COMPARATIVA FINAL (ordenada por F1-score macro)")
print("="*70)
print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

df_res.to_csv("resultados/comparativa_modelos.csv", index=False)
print("\n  Tabla guardada en resultados/comparativa_modelos.csv")

# Gráfico comparativo de F1-score macro
fig, ax = plt.subplots(figsize=(9, 5))
colores = ["#185FA5" if i == 0 else "#9FE1CB" for i in range(len(df_res))]
bars = ax.barh(df_res["modelo"], df_res["f1_macro"], color=colores, edgecolor="#333", height=0.5)
ax.set_xlabel("F1-score macro")
ax.set_title("Comparativa de modelos — F1-score macro sobre conjunto de prueba")
ax.set_xlim(0, 1)
for bar, val in zip(bars, df_res["f1_macro"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("resultados/comparativa_f1.png", dpi=120)
plt.close()
print("  Gráfico comparativo guardado en resultados/comparativa_f1.png")

mejor = df_res.iloc[0]["modelo"]
mejor_f1 = df_res.iloc[0]["f1_macro"]
print(f"\n✓ MEJOR MODELO: {mejor} (F1-macro = {mejor_f1:.4f})")
print("  Usa este modelo para construir la aplicación Streamlit.")

# Curva de entrenamiento del MLP
historia = joblib.load("modelos/05_mlp_historia.pkl")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(historia["loss"],     label="Train loss")
axes[0].plot(historia["val_loss"], label="Val loss")
axes[0].set_title("MLP — Pérdida por época")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
axes[0].legend()
axes[1].plot(historia["accuracy"],     label="Train acc")
axes[1].plot(historia["val_accuracy"], label="Val acc")
axes[1].set_title("MLP — Accuracy por época")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
axes[1].legend()
plt.tight_layout()
plt.savefig("resultados/mlp_curvas_entrenamiento.png", dpi=120)
plt.close()
print("  Curvas de entrenamiento MLP guardadas en resultados/mlp_curvas_entrenamiento.png")
