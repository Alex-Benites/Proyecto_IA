# Clasificación del Tipo de Arma en Homicidios Intencionales — Ecuador

Proyecto de Inteligencia Artificial. Predice **qué tipo de arma** se usó en un
homicidio intencional a partir del **contexto del hecho** (ubicación, momento,
perfil de la víctima, motivación presunta).

Fuente original: Ministerio del Interior / DINASED, 2014–2026.

---

## ⚠️ Lee esto antes de tocar nada

**1. No modifiques `03_preparacion.py` sin avisar al grupo.**
Ahí viven las decisiones compartidas: qué columnas entran, la semilla (42), el
split 70/15/15. Todos los modelos leen los `.npy` que genera ese script. Si
alguien lo cambia por su cuenta, los `.npy` dejan de coincidir y **los modelos
del equipo ya no son comparables entre sí**.

**2. La carpeta `backup/` contiene archivos que YA NO SE USAN.**
Son la versión original del proyecto, descartada por data leakage y uso de
cajas negras. Están ahí solo como registro histórico. No los ejecutes.
Ver `backup/LEEME.md`.

**3. Usa las versiones exactas de `requirements.txt`.**
Están fijadas con `==` a propósito: el pipeline es determinista, pero esa
garantía solo se sostiene si todos usan las mismas versiones de librería.

---

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.13.3.

## Cómo correrlo

```bash
# 1. Genera los splits (train/val/test). Obligatorio antes de cualquier modelo.
python 03_preparacion.py

# 2. Entrena el modelo que te toque (son independientes entre sí)
python 04_modelo_logistica.py
python 07_modelo_randomforest.py
python 08_modelo_catboost.py

# 3. Genera los gráficos de diagnóstico
python 05_graficos.py logistica
python 05_graficos.py randomforest
# (CatBoost genera los suyos automáticamente)

# 4. Prueba un modelo con casos nuevos (prototipo de la app)
python 06_probar_modelo.py
```

**Cada modelo es independiente.** Ninguno necesita el `.joblib` de otro, así que
varias personas pueden entrenar en paralelo sin esperarse. Solo hace falta que
todos hayan corrido `03_preparacion.py` primero.

## Estructura

```
clasificacion-armas/
├── mdi_homicidiosintencionales_pm_2014_2025.xlsx   ← datos originales
├── mdi_homicidiosintencionalse_pm_2026_enero_junio.xlsx
│
├── 01_inspeccion.py          Etapa 1: perfilado, sin modificar nada
├── 02_limpieza.py            Etapa 2: limpieza determinista de los datos oficiales
├── 03_preparacion.py         Etapa 3: split + imputación + one-hot + escalado  ⚠️ COMPARTIDO
├── 04_modelo_logistica.py    Modelo 1: Regresión Logística multinomial
├── 07_modelo_randomforest.py Modelo 2: Random Forest
├── 08_modelo_catboost.py     Modelo 3: CatBoost
├── 05_graficos.py            Gráficos de diagnóstico (parametrizado por modelo)
├── 06_probar_modelo.py       Predice sobre casos nuevos
│
├── data/processed/dataset.xlsx    ← dataset de entrenamiento (ver nota abajo)
├── Graficos/                      ← figuras de cada modelo
├── resultados/                    ← informes .txt de cada etapa
└── backup/                        ← ⚠️ archivos obsoletos, NO USAR
```

Los `.npy`, los `.joblib` y los CSV intermedios **no están en el repo**: son
regenerables y pesan mucho (`train_X.npy` 54 MB, `modelo2_randomforest.joblib`
70 MB). Se recrean corriendo los scripts.

## El problema

**Variable objetivo:** `y`, con 4 clases.

| Clase | Categorías originales que agrupa |
|---|---|
| `ARMA_FUEGO` | ARMA DE FUEGO |
| `ARMA_BLANCA` | ARMA BLANCA |
| `ARMA_CONTUNDENTE` | ARMA CONTUNDENTE |
| `OTRAS` | CONSTRICTORA, OTROS, SUSTANCIAS |

**Variables eliminadas por data leakage** (revelan la respuesta):
- `tipo_arma` — es la subcategoría directa (PISTOLA → ARMA DE FUEGO). Determina el target al 100%.
- `probable_causa_motivada` — resultado forense posterior. Lo determina al 99.8%.
- `arma` — es el target mismo.
- `arma_categoria` — el target sin colapsar.

**`anio` también se eliminó**, y conviene saber por qué: en `dataset.xlsx`,
`ARMA_FUEGO` tiene 0% de presencia en 2014–2023 y 2026, concentrándose casi toda
en 2024–2025. Cualquier modelo aprendía el atajo *"si el año no es 2024/2025, no
es arma de fuego"*. Medido: Random Forest con `anio` daba F1 macro 0.7833; sin
`anio`, 0.6283. Se excluye para **todos** los modelos, no solo Random Forest —
el problema está en el dato, no en el algoritmo.

**Métrica principal:** F1-score macro (no accuracy), para que las 4 clases pesen
igual.

## Resultados actuales (validación)

| Modelo | Accuracy | F1 macro | Brecha train-val |
|---|---|---|---|
| Regresión Logística | 0.5939 | 0.5923 | 0.027 |
| Random Forest | 0.6300 | 0.6283 | 0.307 |
| **CatBoost** | **0.6402** | **0.6392** | 0.142 |
| XGBoost | — | pendiente | |
| MLP | — | pendiente | |

El set de **test permanece sellado**: no se toca hasta elegir el modelo final.

### Hallazgo adicional (CatBoost)

Se comparó el mismo CatBoost con dos codificaciones distintas:

| Codificación de categóricas | F1 macro val |
|---|---|
| Nativas (ordered target statistics) | 0.6392 |
| One-Hot (475 columnas) | 0.6189 |

El manejo nativo aporta **+0.020**. Por eso `08_modelo_catboost.py` parte de
`train_raw.csv` (antes del One-Hot) en vez de `train_X.npy`.

## Nota sobre `dataset.xlsx`

Este archivo **no son los datos oficiales sin modificar**. Se armó a partir de
los datos limpios para balancear las clases (~25% cada una). Al verificarlo
contra los datasets del Ministerio:

| Clase | Filas que corresponden a casos reales |
|---|---|
| ARMA_FUEGO | 99.2% |
| ARMA_BLANCA | 99.0% |
| ARMA_CONTUNDENTE | 22.7% |
| OTRAS | 31.1% |

En `ARMA_FUEGO` y `ARMA_BLANCA` es submuestreo de datos reales (sin problema
metodológico). En `ARMA_CONTUNDENTE` y `OTRAS` hay filas que no corresponden a
registros oficiales. Queda documentado para que las métricas se interpreten con
ese contexto.

Los datos oficiales sin alterar se obtienen con `python 02_limpieza.py`, que
genera `A_2014_2025_limpio.csv` y `B_2026_limpio.csv`.

## Restricción del proyecto: sin cajas negras

- Sin modelos preentrenados ni descargados.
- Sin AutoML ni selección automática de modelo.
- Sin `GridSearchCV`: los hiperparámetros se eligen con **experimentos manuales
  documentados** (ver `resultados/modelos/*.txt`).
- Todos los modelos exponen su interpretabilidad: `coef_` en la logística,
  `feature_importances_` en Random Forest y CatBoost.
- Reproducibilidad: `SEMILLA = 42` en todos los scripts, versiones fijadas.
