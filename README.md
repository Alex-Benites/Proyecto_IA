# Clasificación del Tipo de Arma en Homicidios Intencionales — Ecuador

Proyecto de Inteligencia Artificial. Predice **qué tipo de arma** se usó en un
homicidio intencional a partir del **contexto del hecho** (ubicación, momento,
perfil de la víctima, motivación presunta).

Fuente original: Ministerio del Interior / DINASED, 2014–2026.

---

## ⚠️ Lee esto antes de tocar nada

**1. El proyecto tiene DOS líneas de entrenamiento en paralelo — no las mezcles.**

| | `entrenamiento_v1/` | `entrenamiento_v2/` |
|---|---|---|
| Fuente de datos | `dataset.xlsx` (rebalanceado a mano) | Datos oficiales reales (A+B), sin alterar |
| Contiene filas fabricadas | Sí, en `ARMA_CONTUNDENTE` (77%) y `OTRAS` (69%) | No, 100% real |
| Manejo del desbalance | Ya viene ~equilibrado por diseño | `class_weight` (Variante A) vs recorte real de `ARMA_FUEGO` (Variante B) |
| Estado | **Los 5 modelos listos.** | **Los 5 modelos listos** en ambas variantes. |

V1 existe primero por orden histórico; V2 se abrió después al confirmar que
`dataset.xlsx` no es defendible como única fuente (ver la nota más abajo).

**El modelo elegido para la app sale de V2** (ver la sección "Conclusión").
V1 se conserva como **contraejemplo documentado**: muestra que entrenar con
filas fabricadas no mejoró nada una vez quitadas las fugas de `anio` y `mes`.

**2. No modifiques `entrenamiento_v1/03_preparacion.py` ni
`entrenamiento_v2/01_preparacion_v2.py` sin avisar al grupo.**
Ahí viven las decisiones compartidas de cada línea: qué columnas entran, la
semilla (42), los splits. Todos los modelos de esa línea leen lo que generan
esos scripts. Si alguien los cambia por su cuenta, dejan de ser comparables
entre sí.

**3. La carpeta `backup/` contiene archivos que YA NO SE USAN.**
Son la versión original del proyecto, descartada por data leakage y uso de
cajas negras. No los ejecutes. Ver `backup/LEEME.md`.

**4. Usa las versiones exactas de `requirements.txt`.**
Fijadas con `==` a propósito: el pipeline es determinista, pero esa garantía
solo se sostiene si todos usan las mismas versiones de librería.

---

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.13.3.

## Estructura

```
clasificacion-armas/
├── mdi_homicidiosintencionales_pm_2014_2025.xlsx   ← datos oficiales originales
├── mdi_homicidiosintencionalse_pm_2026_enero_junio.xlsx
│
├── 01_inspeccion.py       Etapa 1: perfilado, sin modificar nada (compartido)
├── 02_limpieza.py         Etapa 2: limpieza determinista (compartido, genera A/B limpios)
├── data/processed/A_2014_2025_limpio.csv   ← usado por ambas lineas
├── data/processed/B_2026_limpio.csv
├── resultados/inspeccion/, resultados/limpieza/
│
├── entrenamiento_v1/      Linea 1: dataset.xlsx (ver tabla de arriba)
│   ├── 03_preparacion.py
│   ├── 04_modelo_logistica.py
│   ├── 05_graficos.py
│   ├── 06_probar_modelo.py
│   ├── 07_modelo_randomforest.py
│   ├── 08_modelo_catboost.py
│   ├── 09_modelo_mlp.py                (sklearn + comparacion con Keras)
│   ├── 10_modelo_xgboost.py
│   ├── utilidades.py
│   ├── data/processed/dataset.xlsx
│   ├── modelos/, resultados/, Graficos/
│
├── entrenamiento_v2/      Linea 2: datos reales, class_weight vs recorte
│   ├── 01_preparacion_v2.py           (genera .npy + preprocesador por variante)
│   ├── 02_modelo_randomforest_v2.py
│   ├── 03_experimento_geografia.py
│   ├── 04_modelo_logistica_v2.py
│   ├── 05_modelo_catboost_v2.py       (unico que NO usa .npy: categoricas nativas)
│   ├── 06_modelo_mlp_v2.py            (sklearn + Keras; unico que usa TensorFlow)
│   ├── 07_modelo_xgboost_v2.py
│   ├── utilidades.py                  (ModeloEtiquetado, rodeo a un bug de sklearn 1.8)
│   ├── data/, modelos/, resultados/, Graficos/
│
└── backup/                ⚠️ archivos obsoletos, NO USAR
```

Los `.npy`, `.joblib`, `.cbm`, `.keras` y los CSV intermedios **no están en el
repo**: son regenerables y pesan mucho. Se recrean corriendo los scripts de
cada carpeta. Sí están versionados los informes `.txt` y los `.png`, que son
la evidencia de los experimentos.

## Cómo correrlo

### Paso previo (una sola vez, alimenta a ambas líneas)
```bash
python 02_limpieza.py    # genera data/processed/A_2014_2025_limpio.csv y B_2026_limpio.csv
```

### Línea V1 (`entrenamiento_v1/`)
```bash
cd entrenamiento_v1
python 03_preparacion.py              # obligatorio antes de cualquier modelo
python 04_modelo_logistica.py
python 07_modelo_randomforest.py
python 08_modelo_catboost.py
python 09_modelo_mlp.py
python 10_modelo_xgboost.py
python 05_graficos.py logistica       # gráficos por modelo
python 05_graficos.py randomforest
python 05_graficos.py mlp
python 06_probar_modelo.py            # prueba con casos nuevos
```

`05_graficos.py` solo cubre los modelos que leen `train_X.npy`. CatBoost y
XGBoost generan sus propios gráficos dentro de su script, porque sus
experimentos son específicos (categóricas nativas, `max_depth`).

### Línea V2 (`entrenamiento_v2/`)
```bash
cd entrenamiento_v2
python 01_preparacion_v2.py           # obligatorio: genera Variante A y B (.npy + preprocesador)
python 02_modelo_randomforest_v2.py   # entrena ambas variantes y las compara
python 04_modelo_logistica_v2.py      # linea base + experimento de regularizacion C
python 05_modelo_catboost_v2.py       # categoricas nativas + experimento learning_rate
python 06_modelo_mlp_v2.py            # red neuronal: sklearn + comparacion con Keras
python 07_modelo_xgboost_v2.py        # boosting asimetrico + comparacion con CatBoost
python 03_experimento_geografia.py    # experimento: cantón vs provincia vs zona
```

`06_modelo_mlp_v2.py` es el único que necesita TensorFlow, y solo para su
Experimento 3. Si no lo tenés instalado, el script corre igual y salta ese
experimento (está envuelto en `try/ImportError`).

`01_preparacion_v2.py` ajusta el preprocesador (imputar edad + escalar +
one-hot) **una sola vez por variante**, solo con su train, y guarda las
matrices en `.npy`. Ningún script de modelo lo reconstruye por su cuenta:
así todos los algoritmos se comparan sobre exactamente la misma matriz.
La única excepción es `03_experimento_geografia.py`, que por definición
cambia qué columnas entran y necesita reajustarlo.

**Dentro de cada línea, los modelos son independientes entre sí** — no
necesitan el `.joblib` de otro modelo, así que varias personas pueden
entrenar en paralelo. Sí necesitan haber corrido el script de preparación de
su línea primero.

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

**`anio` y `mes` también se eliminaron en V1**, y conviene saber por qué: en
`dataset.xlsx`, `ARMA_FUEGO` tiene 0% de presencia en 2014–2023/2026 y en
marzo-abril, concentrándose de forma artificial en 2024–2025. Cualquier
modelo aprendía atajos sin sentido causal. Medido en Random Forest:

| Variable eliminada | F1 macro CON ella | F1 macro SIN ella |
|---|---|---|
| `anio` | 0.7833 | 0.6283 |
| `mes` | 0.6283 | 0.4825 |

Las dos exclusiones ya están aplicadas en `entrenamiento_v1/03_preparacion.py`
(`COLS_ELIMINAR`), y todos los modelos de V1 se reentrenaron con ellas
—ver la nota de corrección en la sección de resultados.

**En V2 (datos reales) se verificó que ninguna de las dos tiene esa fuga**
(ej. `ARMA_FUEGO` se mantiene 76–81% en los 12 meses, sin excepción) — se
excluyen igual por consistencia entre líneas, no porque hagan falta.

**Métrica principal:** F1-score macro (no accuracy), para que las 4 clases pesen
igual.

## Resultados V1 (validación, con `dataset.xlsx`)

| Modelo | Accuracy | F1 macro | Brecha train-val |
|---|---|---|---|
| **CatBoost** | **0.5352** | **0.5188** | 0.188 |
| MLP (red neuronal) | 0.5024 | 0.4969 | 0.153 |
| Random Forest | 0.5113 | 0.4825 | 0.178 |
| XGBoost | 0.5011 | 0.4799 | 0.127 |
| Regresión Logística | 0.4746 | 0.4570 | **0.026** |

### ⚠️ Estos números son la CORRECCIÓN de los que se reportaron antes

Los cuatro modelos se habían entrenado **con `mes` incluido**, pese a que ya
se había detectado que era una fuga. Al reentrenarlos sin esa columna:

| Modelo | CON `mes` (inflado) | SIN `mes` (real) | Caída |
|---|---|---|---|
| Regresión Logística | 0.5923 | 0.4570 | −0.135 |
| Random Forest | 0.6283 | 0.4825 | −0.146 |
| CatBoost | 0.6392 | 0.5188 | −0.120 |
| MLP | 0.6369 | 0.4969 | −0.140 |

**La conclusión importante:** una vez sacadas las dos fugas (`anio` y `mes`),
V1 queda al mismo nivel que V2 (0.457–0.519 contra 0.447–0.528). Es decir,
la ventaja aparente de entrenar con filas fabricadas **no era señal real, era
atajo**. Ese es el argumento central para justificar por qué se abrió V2.

En V1 el manejo nativo de categóricas de CatBoost aporta **+0.0354** sobre
one-hot — bastante más que en V2 (+0.008 en A, −0.009 en B).

## Resultados V2 (validación, datos 100% reales)

Ordenados por F1 macro. La columna de brecha es tan importante como la de F1.

| Modelo | Variante | Accuracy | F1 macro | Brecha |
|---|---|---|---|---|
| XGBoost | B: recorte | 0.6807 | 0.5282 | 0.233 ⚠️ |
| **MLP (red neuronal)** | B: recorte | 0.6780 | **0.5279** | **0.048** |
| Random Forest | A: class_weight | 0.8170 | 0.5108 | 0.272 ⚠️ |
| CatBoost | B: recorte | 0.6758 | 0.5057 | 0.041 |
| Logística (base) | B: recorte | 0.6686 | 0.5037 | 0.016 |
| XGBoost | A: sample_weight | 0.8129 | 0.5034 | 0.425 ⚠️ |
| Random Forest | B: recorte | 0.6843 | 0.4927 | 0.114 |
| CatBoost | A: auto_class_weights | 0.7624 | 0.4838 | 0.057 |
| MLP (red neuronal) | A: sample_weight | 0.7576 | 0.4751 | 0.031 |
| Logística (base) | A: class_weight | 0.7315 | 0.4468 | 0.018 |

**XGBoost B y MLP B empatan en F1 (0.0003 de diferencia = ruido), pero
XGBoost memoriza 5 veces más** (brecha 0.233 vs 0.048). A igual rendimiento,
el MLP es el candidato defendible.

⚠️ **Sesgo a favor de los boostings**: CatBoost y XGBoost usan el set de
validación como `eval_set` del early stopping, así que eligen su número de
árboles mirando el mismo set sobre el que se reporta el F1. Logística, Random
Forest y MLP no tienen esa ventaja. El número limpio saldrá del test sellado.

### Cada modelo maneja el desbalance con un mecanismo distinto

| Modelo | Parámetro | Nota |
|---|---|---|
| Logística, Random Forest | `class_weight='balanced'` | nativo |
| CatBoost | `auto_class_weights='Balanced'` | no tiene `class_weight` |
| MLP | `sample_weight` en `.fit()` | no tiene `class_weight` |
| XGBoost | `sample_weight` en `.fit()` | `scale_pos_weight` es **solo binario** |

Los cuatro calculan el mismo peso `n/(K·n_clase)`; cambia dónde se pasa.

**Ojo con el accuracy en la Variante A**: tiene 79% de `ARMA_FUEGO` en
validación, así que predecir siempre esa clase ya daría 0.79 de accuracy sin
aprender nada. Por eso la métrica que decide es el F1 macro.

La logística sirve de **línea base**: Random Forest la supera por +0.064 de
F1 macro en la Variante A, lo que justifica su complejidad extra. A cambio,
la logística tiene una brecha train-val 15x menor (0.018 vs 0.272): no
memoriza, pero tampoco tiene capacidad para capturar interacciones.

En la logística, `class_weight='balanced'` sí rescata a `ARMA_CONTUNDENTE`
(F1 0.22 en A vs 0.11 en B): sin peso, el modelo solo predice esa clase 16
veces en 2.230 casos de validación, cuando hay 177 reales.

**El F1 macro por sí solo elige mal.** Random Forest A tiene el F1 más alto
(0.5108) pero una brecha train-val de 0.272: memoriza. CatBoost A saca 0.484
con una brecha de 0.057 — 4,7x menor por 0.027 de F1. Para la defensa, el
segundo es el modelo más confiable de la Variante A.

En la Variante B los cinco modelos caben en 0.035 de F1 (0.4927 a 0.5282):
con 10.409 filas de entrenamiento, la complejidad extra deja de rendir. En
términos prácticos, están empatados.

### Hallazgo: `canton` es inútil en la Variante B

La importancia que CatBoost le asigna a `canton` en la Variante B es
**exactamente 0.0000** — nunca lo usó para dividir. Coincide con el
experimento de granularidad geográfica (`03_experimento_geografia.py`), que
había concluido lo mismo por otra vía. Con 214 cantones y 10.409 filas, la
variable no aporta señal: `provincia` (8.3%) y `zona` (8.3%) sí.

Consecuencia práctica para la app: **no hace falta pedir el cantón**.

### Hallazgo: sklearn vs Keras dan lo mismo, si sabés traducir

El MLP se entrenó con las dos librerías (`06_modelo_mlp_v2.py`, Experimento 3).
Pasándoles literalmente los mismos números, Keras salía **0.056 peor**. Ese
gap NO era de implementación: eran tres cosas que significan distinto en
cada librería.

| | Variante A | Variante B |
|---|---|---|
| sklearn `MLPClassifier` | 0.4751 | 0.5279 |
| Keras, mismos números | 0.4192 | 0.4710 |
| **Keras alineado** | **0.4659** | **0.5264** |

Las tres traducciones, verificadas leyendo el código fuente de sklearn:

1. **`alpha` no es `l2()`.** sklearn hace `loss += 0.5*alpha*Σ(W²)/sw_sum`, y
   como `_backprop` recibe el *batch*, `sw_sum` ≈ 200. Keras hace
   `loss += alpha*Σ(W²)` sin dividir. Equivalencia: `l2 = 0.5*alpha/batch`
   (400x más suave).
2. **El criterio de parada.** sklearn detiene por *accuracy* (`_score`); el
   default que uno pone en Keras es `val_loss`.
3. **El split interno.** sklearn separa su 10% **estratificado**; el
   `validation_split` de Keras agarra el último 10% sin mirar la clase — con
   `ARMA_CONTUNDENTE` al 2,7%, eso importa.

Corregidas las tres, la diferencia media cae de 0.0564 a **0.0054** (90%
explicado), que es ruido de inicialización.

El mismo experimento corrió en V1 (`09_modelo_mlp.py`) y **ahí las
correcciones explicaron solo el 33%** (0.024 → 0.016):

| | sklearn | Keras ingenuo | Keras alineado | Explicado |
|---|---|---|---|---|
| V1 | 0.4969 | 0.4731 | 0.4809 | 33% |
| V2 (Variante A) | 0.4751 | 0.4192 | 0.4659 | 90% |
| V2 (Variante B) | 0.5279 | 0.4710 | 0.5264 | 90% |

Conclusión defendible: **las dos librerías implementan el mismo modelo, y las
diferencias grandes vienen de que los hiperparámetros no significan lo mismo
en cada una.** Lo que queda después de traducir (0.005 en V2, 0.016 en V1) es
ruido de inicialización, no de implementación: son *equivalentes en método,
no idénticas en resultado*. Una hipótesis para el residuo mayor de V1 es que
tiene la mitad de filas de train (14.630 vs 30.782), así que la inicialización
pesa más — pero no está medido.

### La clase que ningún modelo resuelve

`ARMA_CONTUNDENTE` es la más difícil en todas las variantes (mejor F1: 0.22 en
V2 Variante A) — con solo 1.180 casos reales dispersos en 162 cantones, el
contexto del hecho no alcanza a distinguirla de forma confiable.

Se probó si reducir la granularidad geográfica (`canton` → `provincia` →
`zona`) ayudaba: el efecto fue nulo (+0.005 de F1), porque Random Forest ya
ignora por su cuenta las columnas one-hot casi vacías.

⚠️ En V1 esta clase muestra F1 0.30 (XGBoost), el valor más alto del proyecto.
**Es un artefacto**: ahí tiene 777 casos de validación en vez de 177, y el 77%
son fabricados. El modelo aprendió el patrón sintético, no la realidad.

El set de **test permanece sellado** en ambas líneas: no se toca hasta elegir
el modelo final.

---

## Conclusión: qué modelo va a la app

**`entrenamiento_v2`, MLP Variante B** (F1 macro 0.5279, brecha 0.048).

Por qué V2 y no V1: una vez corregidas las dos fugas, **las dos líneas dan
prácticamente lo mismo** (V1: 0.457–0.519, V2: 0.447–0.528). Si entrenar con
filas fabricadas no mejora nada, no hay razón para preferirlo — y V2 se
defiende sin asteriscos.

| | Rango F1 macro | Mejor modelo | Brecha del mejor |
|---|---|---|---|
| V1 | 0.457 – 0.519 | CatBoost 0.5188 | 0.188 |
| **V2** | 0.447 – 0.528 | **MLP B 0.5279** | **0.048** |

Por qué el MLP y no XGBoost B (que empata en F1): XGBoost memoriza 5 veces
más. A igual rendimiento medido, gana el que generaliza.

### ⚠️ Las probabilidades del modelo NO son probabilidades reales

Aplica a las dos variantes. La Variante B se entrenó con `ARMA_FUEGO`
recortada de 34.905 a 5.800, así que el modelo "cree" que las armas de fuego
son el 39% de los casos cuando en la realidad son el 79%. La Variante A no
recorta, pero `class_weight='balanced'` produce el mismo efecto por otra vía.

Si la app muestra *"65% probabilidad de ARMA_BLANCA"*, ese número está
inflado respecto de la realidad ecuatoriana. Dos salidas posibles:

1. **Mostrar solo la clase predicha y el ranking**, sin porcentajes. Es lo más
   honesto y no requiere nada extra.
2. **Recalibrar** multiplicando por el ratio de prevalencias reales/entrenadas.
   Defendible, pero agrega un paso que hay que justificar.

Y un límite que la app debe comunicar: con F1 macro 0.53, acierta bien
`ARMA_FUEGO` (0.80) y `ARMA_BLANCA` (0.68), pero `ARMA_CONTUNDENTE` está en
0.12. **Es confiable en 2 de 4 clases**, no en las cuatro.

## Nota sobre `dataset.xlsx` (solo aplica a V1)

Este archivo **no son los datos oficiales sin modificar**. Se armó a partir
de los datos limpios para balancear las clases (~25% cada una). Al
verificarlo contra los datasets del Ministerio:

| Clase | Filas que corresponden a casos reales |
|---|---|
| ARMA_FUEGO | 99.2% |
| ARMA_BLANCA | 99.0% |
| ARMA_CONTUNDENTE | 22.7% |
| OTRAS | 31.1% |

En `ARMA_FUEGO` y `ARMA_BLANCA` es submuestreo de datos reales (sin problema
metodológico). En `ARMA_CONTUNDENTE` y `OTRAS` hay filas que no corresponden
a registros oficiales. Por eso se abrió `entrenamiento_v2/`, que entrena
exclusivamente con datos reales.

## 📱 Para quien haga la app: leer esto primero

### Los 3 archivos que necesitás

**No hace falta reentrenar nada.** Estos dos vienen en el repo (0,55 MB en
total, son la excepción del `.gitignore`):

| Archivo | Qué es |
|---|---|
| `entrenamiento_v2/modelos/preprocesador_variante_b.joblib` | Imputa `edad`, escala y hace el one-hot. **Ya viene ajustado** con el train — no lo re-ajustes. |
| `entrenamiento_v2/modelos/mlp_variante_b.joblib` | El modelo. Es un `dict`: `{"modelo": ..., "arquitectura": (50,), "alpha": 0.01}` |
| `entrenamiento_v2/utilidades.py` | **Obligatorio.** Ver la trampa de abajo. |

### ⚠️ La trampa: `ModuleNotFoundError: No module named 'utilidades'`

El `.joblib` del modelo guarda un objeto `ModeloEtiquetado`, que está definido
en `utilidades.py`. Si cargás el modelo desde otra carpeta sin que ese módulo
sea importable, joblib **falla**. Verificado, no supuesto.

La solución es agregar la carpeta al path (o copiar `utilidades.py` junto a
la app):

```python
import sys, joblib
from pathlib import Path

V2 = Path("entrenamiento_v2")
sys.path.insert(0, str(V2))          # <-- sin esto, joblib.load revienta

pre    = joblib.load(V2 / "modelos" / "preprocesador_variante_b.joblib")
modelo = joblib.load(V2 / "modelos" / "mlp_variante_b.joblib")["modelo"]
```

### Cómo predecir

El preprocesador espera un `DataFrame` con **exactamente estas 18 columnas, en
este orden**:

```python
COLS_CAT = ["zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
            "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
            "estado_civil", "nacionalidad", "discapacidad", "dia_semana",
            "hora", "franja_horaria"]        # texto (str)
COLS_NUM = ["edad"]                          # float, puede ser NaN
COL_BIN  = ["es_fin_semana"]                 # 0 o 1

X = pre.transform(caso[COLS_CAT + COLS_NUM + COL_BIN])
proba = modelo.predict_proba(X)[0]
clases = modelo.classes_    # ['ARMA_BLANCA','ARMA_CONTUNDENTE','ARMA_FUEGO','OTRAS']
```

Ejemplo real de salida (riña nocturna, vía pública, Guayaquil, hombre de 28):

```
ARMA_FUEGO          60.5%
ARMA_BLANCA         21.8%
OTRAS                9.7%
ARMA_CONTUNDENTE     8.0%
```

`categorias_desconocidas`: el `OneHotEncoder` se creó con
`handle_unknown="ignore"`, así que si el usuario manda un cantón o un lugar
que no estaba en el train, **no falla** — esa columna queda en cero.

### ⚠️ Dos límites que la app DEBE comunicar

**1. Los porcentajes no son probabilidades reales.** El modelo se entrenó con
`ARMA_FUEGO` recortada de 34.905 a 5.800 casos, así que "cree" que las armas
de fuego son el 39% cuando en la realidad ecuatoriana son el 79%. Ese 60,5%
del ejemplo está *subestimado*. Opciones: mostrar solo el ranking sin
porcentajes (lo más simple y honesto), o recalibrar por el ratio de
prevalencias y explicarlo.

**2. Es confiable en 2 de 4 clases.** F1 por clase en validación:

| Clase | F1 | ¿Confiable? |
|---|---|---|
| `ARMA_FUEGO` | 0.80 | sí |
| `ARMA_BLANCA` | 0.68 | aceptable |
| `OTRAS` | 0.51 | dudoso |
| `ARMA_CONTUNDENTE` | 0.12 | **no** |

No presentarla como un clasificador confiable de las cuatro. Es una
herramienta de apoyo que sugiere una hipótesis, no un dictamen.

### Si querés regenerar los modelos igual

```bash
python 02_limpieza.py                      # desde la raiz
cd entrenamiento_v2
python 01_preparacion_v2.py
python 06_modelo_mlp_v2.py
```

Con `SEMILLA = 42` y las versiones de `requirements.txt`, da idéntico.

## Inputs que necesita la app (no son todas las columnas del dataset)

Varias columnas se pueden derivar automáticamente en vez de pedírselas al
usuario — verificado, no supuesto:

| El usuario ingresa... | Se deriva automáticamente |
|---|---|
| Fecha y hora del hecho | `dia_semana`, `es_fin_semana`, `hora`, `franja_horaria` |
| Cantón | `provincia`, `zona` (212 de 214 cantones mapean a una única provincia/zona) |
| Motivación observada (36 opciones) | `presunta_motivacion` (100% determinista desde la observada) |

`lugar → tipo_lugar` **no** es determinista (9 de 102 lugares tienen ambos
valores), así que ese sí se pide aparte. El resto del perfil de la víctima
(sexo, edad, etnia, estado civil, nacionalidad, discapacidad) y el contexto
del hecho (área, lugar, tipo de lugar, motivación observada) son campos
independientes.

## Restricción del proyecto: sin cajas negras

- Sin modelos preentrenados ni descargados.
- Sin AutoML ni selección automática de modelo.
- Sin `GridSearchCV`: los hiperparámetros se eligen con **experimentos manuales
  documentados** (ver `resultados/` de cada línea).
- Todos los modelos exponen su interpretabilidad: `coef_` en la logística,
  `feature_importances_` en Random Forest, CatBoost y XGBoost, importancia por
  permutación en el MLP.
- Las fórmulas están escritas en el docstring de cada script (softmax,
  entropía cruzada, ganancia con hessiano en XGBoost, ordered target
  statistics en CatBoost, backpropagation en el MLP).
- Reproducibilidad: `SEMILLA = 42` en todos los scripts, versiones fijadas.

### Limitaciones que asumimos y declaramos

No todo salió limpio. Estas son las que hay que poder defender:

1. **La regla de parsimonia mira solo el F1 val, no la brecha.** Por eso en
   V2 eligió `max_depth=10` para XGBoost, que sobreajusta fuerte. Está
   anotado dentro del experimento de `max_depth` en ambas líneas.
2. **El early stopping de CatBoost y XGBoost usa el set de validación** como
   `eval_set`, así que su F1 reportado está levemente inflado respecto de los
   modelos que no lo usan.
3. **Los splits son aleatorios estratificados, no temporales**, pese a que los
   datos tienen fecha. Se decidió así porque `dataset.xlsx` desordenó el
   significado temporal; en V2 se mantuvo por comparabilidad entre líneas.
4. **En algunos barridos las diferencias fueron menores al ruido** (ej. el
   `learning_rate` de XGBoost en V1 varió solo 0.004 entre el mejor y el
   peor): ahí el valor elegido no está realmente fundamentado por los datos.
