# =============================================================
# 02_modelos.py — Entrenamiento de los 5 modelos
# Ejecutar después de 01_preprocesamiento.py
# =============================================================

import numpy as np
import joblib
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Carga de datos preprocesados ──────────────────────────────
print("Cargando datos preprocesados...")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_val   = np.load("data/processed/X_val.npy")
y_val   = np.load("data/processed/y_val.npy")

N_CLASES    = 5
N_FEATURES  = X_train.shape[1]
RANDOM_STATE = 42

print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}")

# ── Pesos de clase para MLP (class_weight) ────────────────────
# Calculamos el peso inverso a la frecuencia de cada clase
conteos = np.bincount(y_train)
pesos   = {i: len(y_train) / (N_CLASES * c) for i, c in enumerate(conteos)}
print("\nClass weights para MLP:")
clases = ["ARMA DE FUEGO", "ARMA BLANCA", "ARMA CONTUNDENTE", "CONSTRICTORA", "OTROS"]
for i, c in enumerate(clases):
    print(f"  {c}: {pesos[i]:.3f}")

# =============================================================
# MODELO 1: Regresión Logística Multinomial
# =============================================================
print("\n" + "="*50)
print("Entrenando Regresión Logística...")
lr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    C=1.0,               # regularización inversa
    class_weight="balanced",
    random_state=RANDOM_STATE
)
lr.fit(X_train, y_train)
joblib.dump(lr, "modelos/01_regresion_logistica.pkl")
print("  ✓ Guardado en modelos/01_regresion_logistica.pkl")

# =============================================================
# MODELO 2: Árbol de Decisión
# =============================================================
print("\n" + "="*50)
print("Entrenando Árbol de Decisión...")
dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
dt.fit(X_train, y_train)
joblib.dump(dt, "modelos/02_arbol_decision.pkl")
print("  ✓ Guardado en modelos/02_arbol_decision.pkl")

# =============================================================
# MODELO 3: Random Forest
# =============================================================
print("\n" + "="*50)
print("Entrenando Random Forest (esto puede tardar 1-2 min)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1           # usa todos los núcleos del CPU
)
rf.fit(X_train, y_train)
joblib.dump(rf, "modelos/03_random_forest.pkl")
print("  ✓ Guardado en modelos/03_random_forest.pkl")

# =============================================================
# MODELO 4: XGBoost
# =============================================================
print("\n" + "="*50)
print("Entrenando XGBoost...")

# XGBoost maneja desbalance con scale_pos_weight solo para binario
# Para multiclase usamos sample_weight
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight("balanced", y_train)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=N_CLASES,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False
)
joblib.dump(xgb, "modelos/04_xgboost.pkl")
print("  ✓ Guardado en modelos/04_xgboost.pkl")

# =============================================================
# MODELO 5: Perceptrón Multicapa (MLP) con Keras
# =============================================================
print("\n" + "="*50)
print("Entrenando MLP (red neuronal)...")

# Convertir y a one-hot para Keras
y_train_oh = tf.keras.utils.to_categorical(y_train, N_CLASES)
y_val_oh   = tf.keras.utils.to_categorical(y_val,   N_CLASES)

# Arquitectura: definida manualmente capa por capa
mlp = Sequential([
    Dense(128, activation="relu",  input_shape=(N_FEATURES,),
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.4),
    Dense(64,  activation="relu",
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(32,  activation="relu",
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(N_CLASES, activation="softmax")   # 5 neuronas, una por clase
])

mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

mlp.summary()

# Callbacks: detención temprana y reducción de learning rate
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

historia = mlp.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=100,
    batch_size=32,
    class_weight=pesos,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

mlp.save("modelos/05_mlp.keras")
joblib.dump(historia.history, "modelos/05_mlp_historia.pkl")
print("  ✓ Guardado en modelos/05_mlp.keras")

print("\n✓ Todos los modelos entrenados correctamente.")
print("  Ejecuta 03_evaluacion.py para ver las métricas comparativas.")
