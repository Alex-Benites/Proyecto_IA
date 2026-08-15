"""
ETAPA 1 - INSPECCION DE LOS DATASETS
=====================================
Objetivo: entender los datos ANTES de limpiar, transformar o modelar.

Este script NO modifica los datos, NO combina los datasets y NO entrena nada.
Solo lee, describe y compara.

Salida: consola + resultados/inspeccion/informe_inspeccion.txt

Autor: proyecto homicidios intencionales (Ecuador, MDI-DINASED)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
HOJA_DATOS = "1. Homicidios Intencionales"

ARCHIVOS = {
    "A_2014_2025": BASE / "mdi_homicidiosintencionales_pm_2014_2025.xlsx",
    "B_2026_ene_jun": BASE / "mdi_homicidiosintencionalse_pm_2026_enero_junio.xlsx",
}

SALIDA = BASE / "resultados" / "inspeccion"
SALIDA.mkdir(parents=True, exist_ok=True)

# Valores centinela: no son NaN reales, pero significan "sin informacion".
# Los tratamos aparte porque pandas los cuenta como datos validos.
CENTINELAS = {"SIN_DATO", "NO DETERMINADO", "SIN DATO", "NO ESPECIFICA",
              "NO ESPECIFICADO", "DESCONOCIDO", "N/A", "NA", "-", ""}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ---------------------------------------------------------------------------
# Utilidad: escribir a consola y a archivo al mismo tiempo
# ---------------------------------------------------------------------------
class Reporte:
    def __init__(self, ruta):
        self.buffer = []
        self.ruta = ruta

    def p(self, *args):
        texto = " ".join(str(a) for a in args)
        print(texto)
        self.buffer.append(texto)

    def titulo(self, texto, char="="):
        self.p("")
        self.p(char * 78)
        self.p(texto)
        self.p(char * 78)

    def guardar(self):
        self.ruta.write_text("\n".join(self.buffer), encoding="utf-8")


rep = Reporte(SALIDA / "informe_inspeccion.txt")


# ---------------------------------------------------------------------------
# Carga cruda: sin ninguna conversion, todo como texto
# ---------------------------------------------------------------------------
def cargar_crudo(ruta):
    """Lee la hoja de datos SIN inferir tipos (dtype=str).

    Motivo: queremos ver los valores tal como estan escritos en el Excel,
    incluyendo coordenadas con coma decimal, codigos con cero a la izquierda
    ('09' de provincia) y fechas. Si dejamos que pandas infiera, perdemos
    informacion de como esta escrito el dato original.
    """
    df = pd.read_excel(ruta, sheet_name=HOJA_DATOS, dtype=str)
    # Normalizamos solo espacios sobrantes (no cambia el contenido semantico)
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


# ---------------------------------------------------------------------------
# Bloques de analisis
# ---------------------------------------------------------------------------
def resumen_general(df, nombre):
    rep.titulo(f"[{nombre}] 1. DIMENSIONES Y COLUMNAS")
    rep.p(f"Filas    : {df.shape[0]:,}")
    rep.p(f"Columnas : {df.shape[1]}")
    rep.p(f"Memoria  : {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    rep.p("")
    rep.p("Listado de columnas (en orden):")
    for i, c in enumerate(df.columns, 1):
        rep.p(f"  {i:2d}. {c}")


def perfil_columnas(df, nombre):
    """Tabla por columna: nulos reales, centinelas, unicos, ejemplos."""
    rep.titulo(f"[{nombre}] 2. PERFIL POR COLUMNA")

    filas = []
    for c in df.columns:
        s = df[c]
        n_nulos = s.isna().sum()
        n_cent = s.isin(CENTINELAS).sum()
        n_unicos = s.nunique(dropna=True)
        # muestra los 2 valores mas frecuentes para dar contexto
        top = s.dropna().value_counts().head(2)
        ejemplo = " | ".join(f"{k}({v})" for k, v in top.items())
        filas.append({
            "columna": c,
            "nulos": n_nulos,
            "%nulos": round(100 * n_nulos / len(df), 2),
            "centinela": n_cent,
            "%centinela": round(100 * n_cent / len(df), 2),
            "unicos": n_unicos,
            "top2": ejemplo[:60],
        })

    perfil = pd.DataFrame(filas)
    rep.p(perfil.to_string(index=False))
    perfil.to_csv(SALIDA / f"perfil_columnas_{nombre}.csv", index=False, encoding="utf-8")
    rep.p("")
    rep.p(f"(guardado en resultados/inspeccion/perfil_columnas_{nombre}.csv)")
    return perfil


def analisis_duplicados(df, nombre):
    rep.titulo(f"[{nombre}] 3. DUPLICADOS")

    dup_total = df.duplicated().sum()
    rep.p(f"Filas identicas en TODAS las columnas : {dup_total:,} "
          f"({100*dup_total/len(df):.2f}%)")
    rep.p("")
    rep.p("OJO: en datos de homicidios, dos filas identicas pueden ser")
    rep.p("dos victimas distintas del mismo hecho (misma hora, mismo lugar,")
    rep.p("misma edad, mismo sexo). NO son necesariamente un error de carga.")
    rep.p("Esto hay que decidirlo en la Etapa 2, no ahora.")

    # Duplicados sobre un subconjunto que 'casi' identifica el hecho
    claves = ["fecha_infraccion", "hora_infraccion", "codigo_subcircuito",
              "coordenada_x", "coordenada_y"]
    claves = [c for c in claves if c in df.columns]
    dup_hecho = df.duplicated(subset=claves).sum()
    rep.p("")
    rep.p(f"Filas que repiten (fecha+hora+subcircuito+coordenadas): {dup_hecho:,}")
    rep.p("-> indicio de hechos con multiples victimas (masacres, ataques multiples)")


def analisis_identificadores(df, nombre):
    rep.titulo(f"[{nombre}] 4. POSIBLES IDENTIFICADORES")
    rep.p("Buscamos columnas con cardinalidad muy alta (casi un valor por fila).")
    rep.p("Esas columnas permiten 'memorizar' registros -> riesgo de leakage.")
    rep.p("")
    n = len(df)
    hay = False
    for c in df.columns:
        ratio = df[c].nunique(dropna=True) / n
        if ratio > 0.5:
            rep.p(f"  ALTA CARDINALIDAD: {c} -> {ratio:.1%} de valores unicos")
            hay = True
    if not hay:
        rep.p("  No existe ninguna columna con >50% de valores unicos.")
        rep.p("  => El dataset NO trae un ID de caso explicito.")


def analisis_fechas(df, nombre):
    rep.titulo(f"[{nombre}] 5. FECHAS Y COBERTURA TEMPORAL")

    if "fecha_infraccion" not in df.columns:
        rep.p("No existe columna fecha_infraccion.")
        return None

    f = pd.to_datetime(df["fecha_infraccion"], errors="coerce")
    n_malas = f.isna().sum()
    rep.p(f"Fechas no parseables : {n_malas:,}")
    rep.p(f"Fecha minima         : {f.min()}")
    rep.p(f"Fecha maxima         : {f.max()}")
    rep.p("")
    rep.p("Registros por anio:")
    por_anio = f.dt.year.value_counts().sort_index()
    for anio, cnt in por_anio.items():
        barra = "#" * int(50 * cnt / por_anio.max())
        rep.p(f"  {int(anio)} : {cnt:6,}  {barra}")

    # hora
    if "hora_infraccion" in df.columns:
        h = df["hora_infraccion"]
        rep.p("")
        rep.p(f"hora_infraccion -> nulos: {h.isna().sum():,}, "
              f"formatos distintos detectados: {h.dropna().str.len().value_counts().to_dict()}")
    return por_anio


def analisis_numericas(df, nombre):
    """Columnas que 'parecen' numericas aunque vengan como texto."""
    rep.titulo(f"[{nombre}] 6. VARIABLES POTENCIALMENTE NUMERICAS")

    candidatas = ["edad", "coordenada_x", "coordenada_y"]
    candidatas = [c for c in candidatas if c in df.columns]

    for c in candidatas:
        # coma decimal -> punto (solo para inspeccionar, no modificamos el df)
        s = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
        rep.p("")
        rep.p(f"--- {c} ---")
        rep.p(f"  no convertibles a numero : {s.isna().sum():,} "
              f"({100*s.isna().sum()/len(df):.2f}%)")
        if s.notna().any():
            rep.p(f"  min / max                : {s.min()} / {s.max()}")
            rep.p(f"  media / mediana          : {s.mean():.3f} / {s.median():.3f}")
            q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(3)
            rep.p(f"  percentiles 1/25/50/75/99: {list(q.values)}")
        # valores no convertibles mas comunes
        malos = df.loc[s.isna() & df[c].notna(), c].value_counts().head(5)
        if len(malos):
            rep.p(f"  valores no numericos frecuentes: {malos.to_dict()}")


def analisis_categoricas(df, nombre, max_cats=25):
    rep.titulo(f"[{nombre}] 7. VARIABLES CATEGORICAS (distribuciones)")

    # las de baja cardinalidad son las interesantes para mirar completas
    interes = ["tipo_muerte", "arma", "tipo_arma", "area_hecho", "tipo_lugar",
               "lugar", "presunta_motivacion", "probable_causa_motivad",
               "sexo", "genero", "etnia", "estado_civil", "nacionalidad",
               "discapacidad", "medida_edad", "instruccion", "zona"]
    interes = [c for c in interes if c in df.columns]

    for c in interes:
        vc = df[c].value_counts(dropna=False)
        rep.p("")
        rep.p(f"--- {c}  ({df[c].nunique(dropna=True)} categorias distintas) ---")
        for k, v in vc.head(max_cats).items():
            rep.p(f"   {str(k)[:45]:<47} {v:6,}  {100*v/len(df):5.2f}%")
        if len(vc) > max_cats:
            rep.p(f"   ... y {len(vc)-max_cats} categorias mas")


def relacion_arma_tipoarma(df, nombre):
    """Clave para detectar leakage entre columnas relacionadas."""
    rep.titulo(f"[{nombre}] 8. RELACION arma <-> tipo_arma <-> probable_causa")

    if not {"arma", "tipo_arma"}.issubset(df.columns):
        rep.p("Faltan columnas.")
        return

    tab = pd.crosstab(df["arma"], df["tipo_arma"])
    rep.p("Tabla cruzada arma x tipo_arma (conteos):")
    rep.p(tab.to_string())
    tab.to_csv(SALIDA / f"cruce_arma_tipoarma_{nombre}.csv", encoding="utf-8")

    rep.p("")
    rep.p("Lectura: si cada 'arma' mapea a un conjunto EXCLUSIVO de 'tipo_arma',")
    rep.p("entonces una columna determina a la otra -> LEAKAGE si predecimos una")
    rep.p("usando la otra como feature.")

    if "probable_causa_motivad" in df.columns:
        tab2 = pd.crosstab(df["arma"], df["probable_causa_motivad"])
        rep.p("")
        rep.p("Tabla cruzada arma x probable_causa_motivad:")
        rep.p(tab2.to_string())
        tab2.to_csv(SALIDA / f"cruce_arma_causa_{nombre}.csv", encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparacion entre los dos datasets (sin combinarlos)
# ---------------------------------------------------------------------------
def comparar(dfs):
    rep.titulo("9. COMPARACION ENTRE LOS DOS DATASETS (sin combinarlos)", "#")

    nombres = list(dfs.keys())
    a, b = dfs[nombres[0]], dfs[nombres[1]]

    rep.p(f"{nombres[0]}: {a.shape[0]:,} filas x {a.shape[1]} columnas")
    rep.p(f"{nombres[1]}: {b.shape[0]:,} filas x {b.shape[1]} columnas")

    # --- columnas ---
    rep.p("")
    rep.p("--- 9.1 Nombres de columnas ---")
    ca, cb = list(a.columns), list(b.columns)
    if ca == cb:
        rep.p("IDENTICAS en nombre y ORDEN.")
    else:
        rep.p(f"Solo en {nombres[0]}: {sorted(set(ca)-set(cb))}")
        rep.p(f"Solo en {nombres[1]}: {sorted(set(cb)-set(ca))}")
        if set(ca) == set(cb):
            rep.p("Mismos nombres pero DISTINTO ORDEN.")

    # --- categorias ---
    rep.p("")
    rep.p("--- 9.2 Categorias presentes en uno y no en el otro ---")
    rep.p("(esto rompe la codificacion si combinamos sin normalizar)")

    comunes = [c for c in ca if c in cb]
    problemas = []
    for c in comunes:
        va = set(a[c].dropna().unique())
        vb = set(b[c].dropna().unique())
        if len(va) > 200 or len(vb) > 200:
            continue  # alta cardinalidad (subcircuitos, coordenadas): se revisa aparte
        solo_a = va - vb
        solo_b = vb - va
        if solo_a or solo_b:
            problemas.append(c)
            rep.p("")
            rep.p(f"  [{c}]")
            if solo_a:
                rep.p(f"     solo en {nombres[0]} ({len(solo_a)}): {sorted(solo_a)[:12]}")
            if solo_b:
                rep.p(f"     solo en {nombres[1]} ({len(solo_b)}): {sorted(solo_b)[:12]}")

    if not problemas:
        rep.p("  Sin diferencias de categorias en columnas de baja cardinalidad.")

    # --- deteccion especifica de tildes/mayusculas ---
    rep.p("")
    rep.p("--- 9.3 Diferencias por acentos / mayusculas (mismo concepto, texto distinto) ---")

    def normaliza(s):
        import unicodedata
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s.upper().strip()

    encontrados = 0
    for c in comunes:
        va = set(a[c].dropna().unique())
        vb = set(b[c].dropna().unique())
        if len(va) > 200 or len(vb) > 200:
            continue
        mapa_a = {normaliza(v): v for v in va}
        mapa_b = {normaliza(v): v for v in vb}
        for k in set(mapa_a) & set(mapa_b):
            if mapa_a[k] != mapa_b[k]:
                rep.p(f"  [{c}] '{mapa_a[k]}' (A)  vs  '{mapa_b[k]}' (B)  -> mismo concepto")
                encontrados += 1
    if encontrados == 0:
        rep.p("  Ninguna.")

    # --- solapamiento temporal ---
    rep.p("")
    rep.p("--- 9.4 Solapamiento temporal ---")
    fa = pd.to_datetime(a["fecha_infraccion"], errors="coerce")
    fb = pd.to_datetime(b["fecha_infraccion"], errors="coerce")
    rep.p(f"  {nombres[0]}: {fa.min()}  ->  {fa.max()}")
    rep.p(f"  {nombres[1]}: {fb.min()}  ->  {fb.max()}")
    if fa.max() < fb.min():
        rep.p("  NO hay solapamiento: los periodos son consecutivos y disjuntos.")
    else:
        rep.p("  ATENCION: hay solapamiento de fechas -> revisar duplicacion de casos.")

    # --- distribucion de la variable candidata a objetivo ---
    rep.p("")
    rep.p("--- 9.5 Distribucion de 'arma' en cada dataset (comparacion de proporciones) ---")
    pa = a["arma"].value_counts(normalize=True) * 100
    pb = b["arma"].value_counts(normalize=True) * 100
    comp = pd.DataFrame({f"%{nombres[0]}": pa.round(2), f"%{nombres[1]}": pb.round(2)})
    comp["diff"] = (comp.iloc[:, 1] - comp.iloc[:, 0]).round(2)
    rep.p(comp.to_string())
    rep.p("")
    rep.p("Si las proporciones cambian mucho -> DRIFT: el fenomeno cambia en el tiempo.")
    rep.p("Eso es informacion clave para decidir la division train/test temporal.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    rep.titulo("ETAPA 1 - INSPECCION DE DATASETS (no se modifica ni combina nada)", "#")
    rep.p(f"Python  : {sys.version.split()[0]}")
    rep.p(f"pandas  : {pd.__version__}")
    rep.p(f"numpy   : {np.__version__}")

    dfs = {}
    for nombre, ruta in ARCHIVOS.items():
        if not ruta.exists():
            rep.p(f"ERROR: no existe {ruta}")
            continue
        rep.p("")
        rep.p(f"Cargando {nombre} desde {ruta.name} ...")
        dfs[nombre] = cargar_crudo(ruta)

    for nombre, df in dfs.items():
        rep.titulo(f"DATASET {nombre}", "#")
        resumen_general(df, nombre)
        perfil_columnas(df, nombre)
        analisis_duplicados(df, nombre)
        analisis_identificadores(df, nombre)
        analisis_fechas(df, nombre)
        analisis_numericas(df, nombre)
        analisis_categoricas(df, nombre)
        relacion_arma_tipoarma(df, nombre)

    if len(dfs) == 2:
        comparar(dfs)

    rep.titulo("FIN DE LA INSPECCION", "#")
    rep.p("No se combino nada. No se entreno nada. No se modifico ningun archivo.")
    rep.guardar()
    print(f"\nInforme guardado en: {rep.ruta}")


if __name__ == "__main__":
    main()
