# Carpeta `backup/` — archivos que YA NO SE USAN

**No ejecutes nada de esta carpeta.** Estos scripts son la versión original del
proyecto, anterior a la reconstrucción. Se conservan únicamente como registro
histórico: sirven para explicar en la defensa *qué* se descartó y *por qué*.

## Por qué se descartaron

| Script | Problema |
|---|---|
| `01_preprocesamiento.py` | Escala y codifica **antes** del split → los parámetros del `StandardScaler` y del target-encoding se calculan con datos de validación y test. Es **data leakage**. Además usa **SMOTE**, que fabrica víctimas sintéticas interpolando registros reales. |
| `02_modelos.py` | Entrena 5 modelos sin justificar hiperparámetros, incluyendo **XGBoost** y una red **Keras/TensorFlow** montada como caja negra. |
| `03_evaluacion.py` | Evalúa sobre **5 clases** (`ARMA DE FUEGO`, `ARMA BLANCA`, `ARMA CONTUNDENTE`, `CONSTRICTORA`, `OTROS`), esquema que se reemplazó por 4 clases. |

Otros motivos generales:
- Leen los datasets desde `data/raw/`, ruta que ya no se usa (están en la raíz).
- No documentan las decisiones por columna ni el análisis de leakage.
- No separan transformaciones deterministas de las que aprenden parámetros.

### Aclaración: XGBoost y Keras SÍ se usan en el proyecto actual

Puede parecer contradictorio que se descartara `02_modelos.py` por usar
XGBoost y Keras, y que el pipeline actual los use igual. La diferencia no es
la librería, es **cómo**:

| | `backup/02_modelos.py` | Pipeline actual |
|---|---|---|
| Hiperparámetros | los defaults, sin justificar | elegidos con experimentos manuales documentados (barridos de `learning_rate`, `max_depth`, arquitectura, `alpha`, `C`) |
| Fórmula del algoritmo | no se explica | escrita en el docstring de cada script |
| Keras | red montada sin explicar la arquitectura | `06_modelo_mlp_v2.py` la usa como **experimento controlado** contra sklearn, documentando las 3 diferencias de traducción entre librerías |
| Interpretabilidad | no se reporta | `feature_importances_` en XGBoost, importancia por permutación en el MLP |

"Sin cajas negras" nunca significó "sin estas librerías": significa que no se
acepta ningún hiperparámetro ni ningún resultado sin poder explicar de dónde
sale.

## ¿Interfieren con el pipeline actual?

No. Usan nombres de archivo completamente distintos, así que ni siquiera
sobrescribirían nada si se ejecutaran por error:

| | Estos scripts (viejos) | Pipeline actual |
|---|---|---|
| Matrices | `X_train.npy`, `y_train.npy` | `train_X.npy`, `train_y.npy` |
| Modelos | `01_regresion_logistica.pkl` | `modelo1_logistica.joblib` |
| Preprocesador | `scaler.pkl`, `target_encoder.pkl` | `preprocesador.joblib` |

Ningún script del pipeline actual los importa ni los invoca.

## ¿Qué usar en su lugar?

Ver el `README.md` de la raíz del proyecto.
