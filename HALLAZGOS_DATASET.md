# Hallazgos sobre `dataset.xlsx` — investigación durante el desarrollo del MLP

Documento de registro de una investigación que surgió al entrenar el Modelo 5
(MLP, `09_modelo_mlp.py`) y que afecta a los 4 modelos del proyecto, no solo
al MLP. Se deja acá como respaldo escrito del mensaje enviado al equipo,
para que quede trazable y cualquiera pueda reproducir los pasos.

**Estado: propuesta pendiente de decisión del equipo. No se modificó
`03_preparacion.py` ni ningún archivo de otro modelo.**

---

## Resumen ejecutivo

`dataset.xlsx` tiene tres problemas de fondo que inflan el F1 de los 4
modelos de forma similar:

1. La columna `mes` es *data leakage* (mismo mecanismo que ya llevó a
   excluir `anio`).
2. La columna `hora` es redundante con `franja_horaria`.
3. Las filas fabricadas para balancear `ARMA_CONTUNDENTE`/`OTRAS` no son
   sintéticas generadas al azar: son filas reales de `ARMA_FUEGO` con la
   etiqueta cambiada, lo que les da un contenido que no corresponde a la
   clase que dicen representar.

Corrigiendo los tres problemas (solo filas reales, sin `mes`/`hora`,
clases equiparadas por submuestreo), los 4 modelos caen de forma
consistente entre -0.115 y -0.127 de F1 macro. No es un problema de un
modelo puntual: los 4 medían con la misma ventaja artificial incluida.

---

## 1. `mes` es leakage

**Verificación:** se cruzó `dataset.xlsx` contra los datos oficiales sin
alterar (`A_2014_2025_limpio.csv` + `B_2026_limpio.csv` generados por
`02_limpieza.py`, 43.975 filas reales).

- En los datos oficiales completos, `ARMA_FUEGO` representa entre 76% y
  81% de los casos en **todos** los meses del año, sin excepción — no hay
  estacionalidad real.
- En `dataset.xlsx`, marzo y abril muestran **0.0%** de `ARMA_FUEGO`.
- Las filas fabricadas de `ARMA_CONTUNDENTE`/`OTRAS` quedaron concentradas
  por bloques de mes casi perfectos: enero/febrero/octubre/noviembre/
  diciembre ≈ 100% `OTRAS`; marzo-agosto ≈ 90% `ARMA_CONTUNDENTE`.

**Impacto medido (MLP):** F1 macro val con `mes` = 0.6133, sin `mes` =
0.4715.

**Mismo mecanismo que `anio`** (ya documentado en el README del proyecto):
una variable temporal que no tiene relación causal con el arma, pero que
el modelo puede usar como atajo porque el dataset fue construido sin
controlar por esa variable.

## 2. `hora` es redundante con `franja_horaria`

`02_limpieza.py` construye `franja_horaria` directamente a partir de
`hora` (`pd.cut` en 4 bloques: MADRUGADA/MAÑANA/TARDE/NOCHE) — es la misma
información en dos formatos, no dos variables independientes.

**Impacto medido:** sacar `hora` además de `mes` no cambió el F1
(0.4715 → 0.4705, diferencia de 0.001, dentro del ruido normal entre
corridas). Confirma que no aportaba señal adicional ni dañaba — solo
duplicaba.

## 3. Origen y calidad de las filas fabricadas

**Confirmado por el equipo:** las filas de `ARMA_CONTUNDENTE`/`OTRAS` que
no corresponden a registros oficiales (documentadas ya en el README:
77.3% y 68.9% respectivamente) no se generaron sintéticamente — se tomaron
filas reales de `ARMA_FUEGO` y se les cambió la etiqueta de arma.

Esto se verificó de forma independiente antes de tener esa confirmación,
comparando las filas fabricadas contra las reales de la misma clase:

- **Redundancia de "plantilla":** 21.8% de las filas fabricadas de
  `ARMA_CONTUNDENTE` son duplicados casi exactos entre sí (mismo
  lugar/motivo/perfil, solo cambia fecha/edad), contra 1.4% en las filas
  reales de esa clase.
- **Concentración de perfil:** la data fabricada está mucho menos
  diversificada que la real. Ejemplo: `presunta_motivacion` = DELINCUENCIA
  COMUN es 44.9% en los casos reales de `ARMA_CONTUNDENTE` pero 97.8% en
  los fabricados — el perfil típico de violencia con arma de fuego, no de
  arma contundente.
- **El modelo aprende el "molde", no la clase real:** entrenado con la
  mezcla, el modelo predice mejor en las filas fabricadas de
  `ARMA_CONTUNDENTE` (F1=0.35) que en las reales (F1=0.09).

**Conclusión:** no se puede "arreglar" generando mejor data sintética — el
proyecto ya descarta esa vía (SMOTE/oversampling, ver README: "fabrica
víctimas sintéticas interpolando registros reales"). La alternativa sin
fabricar nada es usar solo filas reales.

## 4. Alternativa evaluada: submuestreo balanceado (sin fabricar nada)

Se probaron 5 configuraciones sobre el MLP (mismos hiperparámetros salvo
que se indique lo contrario):

| Configuración | F1 macro val | `ARMA_CONTUNDENTE` recall |
|---|---|---|
| Mezclado, con `mes`/`hora` (como está hoy) | 0.6133 | 0.66 (inflado) |
| Mezclado, sin `mes`/`hora` | 0.4705 | 0.19 |
| Solo real, sin equiparar (desbalanceado) | 0.4917 | **0.03** (casi invisible) |
| Solo real, equiparado por submuestreo (1 corrida) | 0.4987 | 0.28 |
| Solo real, equiparado, **promedio de 10 semillas de submuestreo** | **0.4964 ± 0.0073** | 0.28-0.44 según la semilla |

El número recomendado para reportar es el promedio de 10 semillas
(0.4964 ± 0.0073): una sola corrida de submuestreo depende de qué filas de
las clases mayoritarias sobrevivieron al recorte al azar; promediar varias
semillas da un resultado más estable y defendible (desvío de solo 0.007,
confirma que no es un número con suerte).

Sin equiparar las clases, el modelo prácticamente abandona
`ARMA_CONTUNDENTE` (recall 0.03 = detecta 6 de 190 casos reales).
Equiparando, se vuelve una clase que el modelo sí intenta predecir
(recall 0.28-0.44), a costa de algo de precisión general.

## 5. Los 4 modelos, no solo el MLP

Se replicó el mismo proceso de selección de hiperparámetros de cada script
(mismo barrido de `max_depth` para Random Forest, mismo barrido de
`learning_rate` para CatBoost) sobre los datos corregidos (sin `mes`/
`hora`, solo filas reales, submuestreado balanceado):

| Modelo | F1 macro original | F1 macro corregido | Diferencia |
|---|---|---|---|
| Regresión Logística | 0.5923 | 0.4707 | -0.1216 |
| Random Forest | 0.6283 | 0.5009 | -0.1274 |
| CatBoost | 0.6392 | 0.5241 | -0.1151 |
| MLP | 0.6133 | 0.4964 (± 0.0073) | -0.1169 |

Los 4 caen en un rango muy estrecho (-0.115 a -0.127). El orden entre
modelos se mantiene parecido al original (CatBoost > Random Forest >
MLP ≈ Random Forest > Regresión Logística), solo que ahora sin la ventaja
artificial.

### `ARMA_CONTUNDENTE` en detalle, los 4 modelos (antes vs después)

| Modelo | Precisión antes | Recall antes | Precisión después | Recall después |
|---|---|---|---|---|
| Regresión Logística | 0.63 | 0.66 | 0.17 | 0.33 |
| Random Forest | 0.67 | 0.67 | 0.22 | 0.40 |
| CatBoost | 0.68 | 0.68 | 0.23 | 0.40 |
| MLP | 0.66 | 0.66 | 0.19 | 0.28 |

Los 4 modelos "antes" rondan 0.63-0.68 en ambas métricas para esta clase
(números inflados por el mecanismo del punto 3), y los 4 "después" caen a
un rango similar entre sí (precisión 0.17-0.23, recall 0.28-0.40). Ningún
modelo se salva ni es "peor" que los demás frente a este problema.

---

## Propuesta para el equipo

1. Sacar `mes` y `hora` de `COLS_CATEGORICAS` en `03_preparacion.py`.
2. Filtrar `dataset.xlsx` a solo filas verificadas como reales (cruzando
   contra `A_2014_2025_limpio.csv`/`B_2026_limpio.csv`).
3. Submuestrear el train para equiparar las 4 clases, en vez de depender
   del balanceo artificial de las filas fabricadas.
4. Reentrenar los 4 modelos con esto y documentar la caída en el README,
   con el mismo nivel de transparencia que ya tiene la sección de `anio`.

## Qué queda pendiente

- Confirmación del equipo antes de tocar `03_preparacion.py` (archivo
  compartido).
- Si se aprueba, reentrenar Regresión Logística, Random Forest y CatBoost
  con el dataset corregido y actualizar sus informes/gráficos.
- Actualizar la tabla de resultados del README con los números finales.

## Reproducibilidad

Toda la verificación de este documento se hizo con scripts de diagnóstico
que NO se agregaron al repositorio (para no tocar archivos existentes sin
necesidad) — viven fuera del proyecto, en el entorno donde se hizo la
investigación. La metodología está descrita paso a paso en cada sección de
este documento y en el docstring de `09_modelo_mlp.py`, que sí forma parte
del repo y es completamente autónomo (no depende de que el equipo apruebe
esta propuesta para poder ejecutarse). Si el equipo quiere, esos scripts
de diagnóstico se pueden formalizar y agregar al repo como parte del
proceso de aprobación.
