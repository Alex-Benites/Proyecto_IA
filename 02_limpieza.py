"""
ETAPA 2 - VALIDACION Y LIMPIEZA
================================

REGLA DE ORO DE ESTA ETAPA
--------------------------
Aqui SOLO se aplican transformaciones DETERMINISTICAS: reglas fijas que no
"aprenden" nada de los datos (quitar acentos, convertir coma a punto,
marcar (0,0) como faltante).

NO se imputan valores faltantes, NO se escala, NO se codifica.
Motivo: imputar con la media/moda es APRENDER un parametro de los datos.
Si lo hacemos ahora, ese parametro se calcularia con datos de 2026 (test)
y eso seria DATA LEAKAGE. La imputacion va DESPUES del split (Etapa 5).

Los dos datasets se limpian POR SEPARADO con las MISMAS reglas y se guardan
por separado. Todavia NO se combinan.

Salida:
  data/processed/A_2014_2025_limpio.csv
  data/processed/B_2026_limpio.csv
  resultados/limpieza/informe_limpieza.txt
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
HOJA = "1. Homicidios Intencionales"
SEMILLA = 42  # reproducibilidad (aqui no hay azar, pero queda documentado)

ARCHIVOS = {
    "A_2014_2025": BASE / "mdi_homicidiosintencionales_pm_2014_2025.xlsx",
    "B_2026": BASE / "mdi_homicidiosintencionalse_pm_2026_enero_junio.xlsx",
}

DIR_PROC = BASE / "data" / "processed"
DIR_REP = BASE / "resultados" / "limpieza"
DIR_PROC.mkdir(parents=True, exist_ok=True)
DIR_REP.mkdir(parents=True, exist_ok=True)

# --- Rango geografico valido de Ecuador (incluye Galapagos) ---
LAT_MIN, LAT_MAX = -5.5, 2.0
LON_MIN, LON_MAX = -92.5, -74.5

# --- Rango de edad humanamente posible ---
EDAD_MIN, EDAD_MAX = 0, 110

# ---------------------------------------------------------------------------
# DECISIONES POR COLUMNA (documentadas para la presentacion)
# ---------------------------------------------------------------------------

# Columnas que se ELIMINAN por LEAKAGE: revelan la respuesta.
COLS_LEAKAGE = {
    "tipo_arma": "Es la subcategoria directa de 'arma' (PISTOLA->ARMA DE FUEGO). "
                 "Determina el target al 100%.",
    "probable_causa_motivada": "Resultado forense posterior. Determina 'arma' al "
                               "99.8% (HERIDA POR ARMA DE FUEGO -> ARMA DE FUEGO).",
    "arma": "Es el target. Se transforma en 'y' y se elimina el original.",
}

# Columnas que se ELIMINAN por decision de diseno.
COLS_DESCARTADAS = {
    "tipo_muerte": "Calificacion JURIDICA posterior al hecho, no un dato de la "
                   "escena. Ademas tiene sesgo administrativo severo: ZONA 9 "
                   "clasifica 71.6% como HOMICIDIO vs ZONA 8 con 2.5% (29x). "
                   "El modelo aprenderia la practica de la fiscalia, no el crimen.",
    "medida_edad": "Constante util: solo vale 'A' (anios) o SIN_DATO. El SIN_DATO "
                   "es redundante con edad nula. No aporta informacion.",
    "instruccion": "75.8% SIN_DATO en A y 90.1% en B. Ademas la tasa de vacio "
                   "CAMBIA entre datasets -> variable inestable en el tiempo.",
    "genero": "Redundante con 'sexo': 30.339 HOMBRE->MASCULINO y 2.662 "
              "MUJER->FEMENINO (>99% de coincidencia). Ademas 16.6% de "
              "faltantes en A contra 0.6% en B: cambio la practica de "
              "registro, seria inestable al predecir en 2026.",
    "codigo_subcircuito": "1745 categorias. Redundante con 'subcircuito' y "
                          "demasiado granular: permite memorizar registros.",
    "subcircuito": "1609 categorias. Mismo problema. La geografia ya esta "
                   "representada por zona/provincia/canton/area_hecho.",
    "circuito": "933 categorias. Mismo problema de granularidad.",
    "codigo_provincia": "Redundante: es el codigo numerico de 'provincia'.",
    "codigo_canton": "Redundante: es el codigo numerico de 'canton'.",
    "profesion_registro_civil": "64 categorias muy dispersas, 17-19% SIN_DATO, y "
                                "listas de categorias distintas entre A y B "
                                "(24 solo en A, 18 solo en B).",
}

# Columnas "dudosas": se CONSERVAN pero marcadas. En la etapa de modelado
# se mide el impacto entrenando CON y SIN ellas (experimento documentado).
COLS_DUDOSAS = {
    "presunta_motivacion": "Resultado de investigacion posterior al hecho. "
                           "Podria no estar disponible al momento de predecir.",
    "presun_motiva_observada": "Igual, y mas granular (35 categorias).",
}

# Valores que significan "no hay dato" -> se convierten a NaN.
CENTINELAS_FALTANTE = {"SIN_DATO", "SIN DATO", "NO DETERMINADO", "NO APLICA", ""}

# OJO: 'NINGUNA' y 'OTROS' NO entran arriba porque en varias columnas son
# categorias VALIDAS, no ausencia de dato:
#   discapacidad='NINGUNA' -> la victima no tenia discapacidad (informacion real)
#   etnia='OTROS'          -> etnia registrada pero fuera del catalogo
# Convertirlos a NaN destruiria informacion legitima.

# Mapeo del target: 6 categorias originales -> 4 clases
MAPA_TARGET = {
    "ARMA DE FUEGO": "ARMA_FUEGO",
    "ARMA BLANCA": "ARMA_BLANCA",
    "ARMA CONTUNDENTE": "ARMA_CONTUNDENTE",
    "CONSTRICTORA": "OTRAS",
    "OTROS": "OTRAS",
    "SUSTANCIAS": "OTRAS",   # 40 casos en 12 anios: inaprendible como clase propia
}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
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


rep = Reporte(DIR_REP / "informe_limpieza.txt")


# ---------------------------------------------------------------------------
# Funciones de limpieza (todas DETERMINISTICAS)
# ---------------------------------------------------------------------------
def quitar_acentos(texto):
    """'PÚBLICO' -> 'PUBLICO'. Resuelve que A escriba con tilde y B sin tilde."""
    if pd.isna(texto):
        return texto
    s = unicodedata.normalize("NFKD", str(texto))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalizar_texto(serie):
    """Mayusculas, sin acentos, sin espacios sobrantes, espacios internos colapsados.

    Esto hace que 'TERRENO BALDÍO' (A) y 'TERRENO BALDIO' (B) sean el MISMO
    valor. Sin esto, al combinar los datasets el codificador crearia dos
    categorias distintas para el mismo concepto y B quedaria lleno de
    categorias 'desconocidas'.
    """
    return (serie.astype(str)
                 .str.strip()
                 .str.upper()
                 .map(quitar_acentos)
                 .str.replace(r"\s+", " ", regex=True)
                 .replace({"NAN": np.nan, "NONE": np.nan}))


def marcar_faltantes(df, cols, rep_dict):
    """Convierte los centinelas textuales en NaN reales."""
    for c in cols:
        if c not in df.columns:
            continue
        antes = df[c].isna().sum()
        df[c] = df[c].replace(list(CENTINELAS_FALTANTE), np.nan)
        rep_dict[c] = int(df[c].isna().sum() - antes)
    return df


def limpiar_coordenadas(df, log):
    """Coma decimal -> punto, (0,0) -> NaN, fuera de Ecuador -> NaN."""
    for c in ["coordenada_x", "coordenada_y"]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # (0,0) esta en el Golfo de Guinea, Africa: es un nulo disfrazado de numero.
    cero = (df["coordenada_x"] == 0) & (df["coordenada_y"] == 0)
    log["coord_cero_a_nan"] = int(cero.sum())
    df.loc[cero, ["coordenada_x", "coordenada_y"]] = np.nan

    # Fuera del territorio nacional -> error de digitacion
    fuera = (~df["coordenada_x"].between(LON_MIN, LON_MAX)
             | ~df["coordenada_y"].between(LAT_MIN, LAT_MAX)) \
            & df["coordenada_x"].notna()
    log["coord_fuera_ecuador_a_nan"] = int(fuera.sum())
    df.loc[fuera, ["coordenada_x", "coordenada_y"]] = np.nan
    return df


def limpiar_edad(df, log):
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    fuera = ~df["edad"].between(EDAD_MIN, EDAD_MAX) & df["edad"].notna()
    log["edad_fuera_rango_a_nan"] = int(fuera.sum())
    df.loc[fuera, "edad"] = np.nan
    log["edad_nula_total"] = int(df["edad"].isna().sum())
    log["edad_cero"] = int((df["edad"] == 0).sum())
    return df


def derivar_fecha_hora(df, log):
    """Extrae variables temporales USABLES al momento del hecho.

    Todas estas se conocen en el instante del crimen -> no hay leakage temporal.
    """
    f = pd.to_datetime(df["fecha_infraccion"], errors="coerce")
    log["fecha_no_parseable"] = int(f.isna().sum())

    df["anio"] = f.dt.year
    df["mes"] = f.dt.month
    df["dia_semana"] = f.dt.dayofweek          # 0=lunes ... 6=domingo
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["fecha"] = f                            # se conserva para el split temporal

    # Hora: 'HH:MM:SS' -> entero 0..23
    h = pd.to_numeric(df["hora_infraccion"].astype(str).str[:2], errors="coerce")
    h = h.where(h.between(0, 23))
    log["hora_no_parseable"] = int(h.isna().sum())
    df["hora"] = h

    # Franja horaria: agrupa 24 horas en 4 bloques con sentido criminologico.
    # Reduce ruido y es facil de explicar en la presentacion.
    df["franja_horaria"] = pd.cut(
        h, bins=[-0.1, 5.99, 11.99, 17.99, 23.99],
        labels=["MADRUGADA", "MANANA", "TARDE", "NOCHE"]).astype(object)

    df = df.drop(columns=["fecha_infraccion", "hora_infraccion"])
    return df


def construir_target(df, log):
    """arma (6 categorias) -> y (4 clases)."""
    arma_norm = normalizar_texto(df["arma"])
    df["arma_categoria"] = arma_norm  # 6 categorias originales, conservadas para analisis
    df["y"] = arma_norm.map(MAPA_TARGET)

    sin_mapear = df.loc[df["y"].isna() & arma_norm.notna(), "arma"].unique()
    if len(sin_mapear):
        rep.p(f"  !! ATENCION: categorias de 'arma' sin mapear: {list(sin_mapear)}")
    log["target_nulo"] = int(df["y"].isna().sum())
    return df


def analizar_duplicados(df, nombre):
    """Los analiza y DOCUMENTA, pero NO los elimina. Ver justificacion abajo."""
    rep.p("")
    rep.p(f"  --- Duplicados en {nombre} ---")
    n_exactos = df.duplicated().sum()
    rep.p(f"  Filas identicas en todas las columnas: {n_exactos:,} "
          f"({100*n_exactos/len(df):.2f}%)")
    rep.p("  DECISION: NO se eliminan.")
    rep.p("  Justificacion: una fila = una VICTIMA, no un hecho. Se verifico en")
    rep.p("  los datos que estos duplicados corresponden a hechos con multiples")
    rep.p("  victimas (mismo canton, misma hora, mismo perfil demografico).")
    rep.p("  Ejemplo real: 3 filas identicas en QUEVEDO 2016-11-02 10:20 con")
    rep.p("  sexo NO DETERMINADO = 3 cuerpos sin identificar del mismo evento.")
    rep.p("  Eliminarlos borraria personas reales y subestimaria la violencia")
    rep.p("  de alto impacto (masacres), que es justamente donde el arma de")
    rep.p("  fuego domina. Seria un sesgo contra la clase mayoritaria.")
    return n_exactos


# ---------------------------------------------------------------------------
# Pipeline de limpieza para UN dataset
# ---------------------------------------------------------------------------
def limpiar(ruta, nombre):
    rep.titulo(f"LIMPIEZA DE {nombre}", "#")
    log = {}

    df = pd.read_excel(ruta, sheet_name=HOJA, dtype=str)
    filas_ini, cols_ini = df.shape
    rep.p(f"  Entrada: {filas_ini:,} filas x {cols_ini} columnas")

    # --- 1. Normalizacion de texto ---
    # Se aplica a TODAS las columnas de texto, antes que nada, para que las
    # comparaciones y mapeos posteriores funcionen igual en A y en B.
    cols_texto = [c for c in df.columns
                  if c not in ("coordenada_x", "coordenada_y", "edad",
                               "fecha_infraccion", "hora_infraccion")]
    for c in cols_texto:
        df[c] = normalizar_texto(df[c])
    rep.p(f"  1. Texto normalizado (mayusculas, sin acentos) en {len(cols_texto)} columnas")

    # --- 2. Target (ANTES de borrar 'arma') ---
    df = construir_target(df, log)
    rep.p("  2. Target construido: arma (6 cat.) -> y (4 clases)")
    dist = df["y"].value_counts()
    for k, v in dist.items():
        rep.p(f"       {k:<14} {v:6,}  {100*v/len(df):5.2f}%")

    # --- 3. Centinelas -> NaN ---
    log_cent = {}
    df = marcar_faltantes(df, cols_texto, log_cent)
    rep.p("  3. Centinelas (SIN_DATO / NO DETERMINADO / NO APLICA) -> NaN:")
    for c, n in sorted(log_cent.items(), key=lambda x: -x[1]):
        if n > 0:
            rep.p(f"       {c:<28} {n:6,} valores marcados como faltantes")

    # --- 4. Numericas ---
    df = limpiar_coordenadas(df, log)
    df = limpiar_edad(df, log)
    rep.p("  4. Numericas validadas:")
    rep.p(f"       coordenadas (0,0) -> NaN      : {log['coord_cero_a_nan']:,}")
    rep.p(f"       coordenadas fuera Ecuador     : {log['coord_fuera_ecuador_a_nan']:,}")
    rep.p(f"       edad fuera de [0,110] -> NaN  : {log['edad_fuera_rango_a_nan']:,}")
    rep.p(f"       edad nula (total)             : {log['edad_nula_total']:,}")
    rep.p(f"       edad == 0 (se CONSERVAN)      : {log['edad_cero']:,}")
    rep.p("       Nota: edad 0 puede ser real (infanticidio). No se elimina.")

    # --- 5. Fecha y hora ---
    df = derivar_fecha_hora(df, log)
    rep.p("  5. Variables temporales derivadas: anio, mes, dia_semana,")
    rep.p("     es_fin_semana, hora, franja_horaria")
    rep.p(f"       fechas no parseables: {log['fecha_no_parseable']:,}")
    rep.p(f"       horas no parseables : {log['hora_no_parseable']:,}")

    # --- 6. Duplicados (solo analisis) ---
    analizar_duplicados(df, nombre)

    # --- 7. Eliminacion de columnas ---
    rep.p("")
    rep.p("  7. Columnas eliminadas por LEAKAGE:")
    for c, motivo in COLS_LEAKAGE.items():
        if c in df.columns:
            df = df.drop(columns=[c])
            rep.p(f"       [{c}] {motivo}")

    rep.p("")
    rep.p("  8. Columnas eliminadas por decision de diseno:")
    for c, motivo in COLS_DESCARTADAS.items():
        if c in df.columns:
            df = df.drop(columns=[c])
            rep.p(f"       [{c}] {motivo}")

    rep.p("")
    rep.p("  9. Columnas DUDOSAS conservadas (se medira su impacto con/sin):")
    for c, motivo in COLS_DUDOSAS.items():
        if c in df.columns:
            rep.p(f"       [{c}] {motivo}")

    # --- 8. Filas sin target ---
    n_sin_y = df["y"].isna().sum()
    if n_sin_y:
        df = df[df["y"].notna()].copy()
        rep.p(f"\n  10. Filas sin target eliminadas: {n_sin_y:,}")

    rep.p("")
    rep.p(f"  SALIDA: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    rep.p(f"  (se conservo el {100*len(df)/filas_ini:.2f}% de las filas)")

    return df


# ---------------------------------------------------------------------------
# Verificacion de compatibilidad tras la limpieza
# ---------------------------------------------------------------------------
def verificar_compatibilidad(a, b):
    rep.titulo("VERIFICACION: ¿son compatibles A y B DESPUES de limpiar?", "#")

    rep.p("Columnas identicas:", list(a.columns) == list(b.columns))

    rep.p("")
    rep.p("Categorias de B que NO existen en A (serian 'desconocidas' para el modelo):")
    cats = [c for c in a.columns
            if a[c].dtype == object and c not in ("y", "fecha")]
    total_problema = 0
    for c in cats:
        va, vb = set(a[c].dropna()), set(b[c].dropna())
        nuevas = vb - va
        if nuevas:
            n_filas = b[c].isin(nuevas).sum()
            total_problema += n_filas
            rep.p(f"  [{c}] {len(nuevas)} categorias nuevas, afectan {n_filas} filas: "
                  f"{sorted(nuevas)[:6]}")
    if total_problema == 0:
        rep.p("  NINGUNA. La normalizacion de texto resolvio las diferencias.")
    else:
        rep.p(f"  -> Se trataran en la Etapa 5 (categoria 'DESCONOCIDA').")

    rep.p("")
    rep.p("Distribucion del target en cada dataset:")
    da = a["y"].value_counts(normalize=True).mul(100).round(2)
    db = b["y"].value_counts(normalize=True).mul(100).round(2)
    comp = pd.DataFrame({"%A_2014_2025": da, "%B_2026": db})
    comp["diff"] = (comp["%B_2026"] - comp["%A_2014_2025"]).round(2)
    rep.p(comp.to_string())

    rep.p("")
    rep.p("Faltantes (%) por columna en cada dataset:")
    fa = (100 * a.isna().mean()).round(2)
    fb = (100 * b.isna().mean()).round(2)
    tab = pd.DataFrame({"%NaN_A": fa, "%NaN_B": fb})
    tab["diff"] = (tab["%NaN_B"] - tab["%NaN_A"]).round(2)
    tab = tab.sort_values("diff", key=abs, ascending=False)
    rep.p(tab.to_string())
    rep.p("")
    rep.p("ALERTA si |diff| es grande: la practica de registro cambio entre")
    rep.p("periodos y esa variable sera inestable al predecir en 2026.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    rep.titulo("ETAPA 2 - VALIDACION Y LIMPIEZA", "#")
    rep.p(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | semilla {SEMILLA}")
    rep.p("")
    rep.p("Solo transformaciones DETERMINISTICAS. Sin imputar, sin escalar,")
    rep.p("sin codificar: eso requiere aprender parametros y va DESPUES del split.")

    limpios = {}
    for nombre, ruta in ARCHIVOS.items():
        limpios[nombre] = limpiar(ruta, nombre)

    a, b = limpios["A_2014_2025"], limpios["B_2026"]
    verificar_compatibilidad(a, b)

    # --- Guardado (siguen SEPARADOS) ---
    pa = DIR_PROC / "A_2014_2025_limpio.csv"
    pb = DIR_PROC / "B_2026_limpio.csv"
    a.to_csv(pa, index=False, encoding="utf-8")
    b.to_csv(pb, index=False, encoding="utf-8")

    rep.titulo("FIN DE LA ETAPA 2", "#")
    rep.p(f"Guardado: {pa.relative_to(BASE)}  ({a.shape[0]:,} x {a.shape[1]})")
    rep.p(f"Guardado: {pb.relative_to(BASE)}  ({b.shape[0]:,} x {b.shape[1]})")
    rep.p("")
    rep.p("Los datasets siguen SEPARADOS. B_2026 es el test final sellado.")
    rep.p("Columnas finales: " + ", ".join(a.columns))
    rep.guardar()
    print(f"\nInforme: {rep.ruta}")


if __name__ == "__main__":
    main()
