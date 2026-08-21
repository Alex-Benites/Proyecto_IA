from pathlib import Path
from collections import Counter

import xlrd
import xlwt


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE = Path(__file__).resolve().parent

ENTRADA = (
    BASE
    / "data"
    / "processed"
    / "A_2014_2025_limpio-_1__1_sin_vacios.xls"
)

SALIDA = (
    BASE
    / "data"
    / "processed"
    / "A_2014_2025_muestraFinalV2.xls"
)

COLUMNA_CATEGORIA = "y"
COLUMNA_ANIO = "anio"


LIMITES = {
    "ARMA_FUEGO": 1000,
    "OTRAS": 1000,
    "ARMA_BLANCA": 1000,
    "ARMA_CONTUNDENTE": 875,
}


# =============================================================================
# NORMALIZAR AÑO
# =============================================================================

def normalizar_anio(valor):
    """
    Convierte valores como:
        2025
        2025.0
        "2025"

    a:
        2025
    """

    try:
        return int(float(valor))
    except (ValueError, TypeError):
        return None


# =============================================================================
# CALCULAR CUÁNTOS ARMA_FUEGO TOMAR DE CADA AÑO
# =============================================================================

def calcular_cupos_arma_fuego(
    hoja,
    indice_y,
    indice_anio,
    cantidad_total=1000,
):
    """
    Recorre primero el archivo para conocer cuántos registros
    ARMA_FUEGO existen en cada año.

    Luego reparte los 1000 registros de la forma más equilibrada
    posible entre todos los años disponibles.

    No selecciona aleatoriamente.
    """

    disponibles_por_anio = Counter()

    # -------------------------------------------------------------------------
    # Contar ARMA_FUEGO disponibles por año
    # -------------------------------------------------------------------------

    for fila in range(1, hoja.nrows):

        categoria = str(
            hoja.cell_value(
                fila,
                indice_y,
            )
        ).strip()

        if categoria != "ARMA_FUEGO":
            continue

        anio = normalizar_anio(
            hoja.cell_value(
                fila,
                indice_anio,
            )
        )

        if anio is not None:
            disponibles_por_anio[anio] += 1

    if not disponibles_por_anio:
        raise ValueError(
            "No se encontraron registros ARMA_FUEGO "
            "con un año válido."
        )

    total_disponible = sum(
        disponibles_por_anio.values()
    )

    if total_disponible < cantidad_total:
        raise ValueError(
            f"Solo existen {total_disponible} registros "
            f"ARMA_FUEGO y se necesitan {cantidad_total}."
        )

    # -------------------------------------------------------------------------
    # Orden cronológico
    # -------------------------------------------------------------------------

    anios = sorted(
        disponibles_por_anio.keys()
    )

    # -------------------------------------------------------------------------
    # Crear cupos inicialmente en cero
    # -------------------------------------------------------------------------

    cupos = {
        anio: 0
        for anio in anios
    }

    restantes = cantidad_total

    # -------------------------------------------------------------------------
    # Repartir uno por uno entre los años.
    #
    # Esto hace que los cupos queden lo más equilibrados posible.
    #
    # Si son 13 años y queremos 1000:
    #
    # 2014 -> 77
    # 2015 -> 77
    # ...
    # 2025 -> 77
    # 2026 -> 76
    #
    # siempre que cada año tenga suficientes registros.
    # -------------------------------------------------------------------------

    while restantes > 0:

        se_asigno = False

        for anio in anios:

            if restantes == 0:
                break

            # Solo aumentar si todavía existen registros
            # disponibles para ese año.
            if (
                cupos[anio]
                < disponibles_por_anio[anio]
            ):

                cupos[anio] += 1
                restantes -= 1
                se_asigno = True

        if not se_asigno:
            raise RuntimeError(
                "No fue posible distribuir los registros "
                "ARMA_FUEGO entre los años disponibles."
            )

    # -------------------------------------------------------------------------
    # Mostrar distribución calculada
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        "DISTRIBUCIÓN DE ARMA_FUEGO POR AÑO"
    )

    print(
        "=" * 65
    )

    print(
        f"{'Año':<10}"
        f"{'Disponibles':>15}"
        f"{'A seleccionar':>18}"
    )

    print("-" * 65)

    for anio in anios:

        print(
            f"{anio:<10}"
            f"{disponibles_por_anio[anio]:>15}"
            f"{cupos[anio]:>18}"
        )

    print("-" * 65)

    print(
        f"{'TOTAL':<10}"
        f"{total_disponible:>15}"
        f"{sum(cupos.values()):>18}"
    )

    return cupos


# =============================================================================
# EXTRAER MUESTRA
# =============================================================================

def extraer_muestra():

    # -------------------------------------------------------------------------
    # Verificar archivos
    # -------------------------------------------------------------------------

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n{ENTRADA}"
        )

    if SALIDA.exists():
        raise FileExistsError(
            f"El archivo de salida ya existe:\n"
            f"{SALIDA}\n"
            "Elimínalo antes de volver a ejecutar."
        )

    # -------------------------------------------------------------------------
    # Abrir Excel original
    # -------------------------------------------------------------------------

    libro_entrada = xlrd.open_workbook(
        str(ENTRADA)
    )

    hoja_entrada = (
        libro_entrada.sheet_by_index(0)
    )

    # -------------------------------------------------------------------------
    # Encabezados
    # -------------------------------------------------------------------------

    encabezados = [
        str(valor).strip()
        for valor in hoja_entrada.row_values(0)
    ]

    # -------------------------------------------------------------------------
    # Buscar columna y
    # -------------------------------------------------------------------------

    if COLUMNA_CATEGORIA not in encabezados:
        raise ValueError(
            f"No existe la columna "
            f"'{COLUMNA_CATEGORIA}'."
        )

    indice_y = encabezados.index(
        COLUMNA_CATEGORIA
    )

    # -------------------------------------------------------------------------
    # Buscar columna anio
    # -------------------------------------------------------------------------

    if COLUMNA_ANIO not in encabezados:
        raise ValueError(
            f"No existe la columna "
            f"'{COLUMNA_ANIO}'."
        )

    indice_anio = encabezados.index(
        COLUMNA_ANIO
    )

    print(
        f"Columna '{COLUMNA_CATEGORIA}' "
        f"en índice {indice_y}"
    )

    print(
        f"Columna '{COLUMNA_ANIO}' "
        f"en índice {indice_anio}"
    )

    # =========================================================================
    # CALCULAR DISTRIBUCIÓN DE ARMA_FUEGO
    # =========================================================================

    cupos_arma_fuego = (
        calcular_cupos_arma_fuego(
            hoja=hoja_entrada,
            indice_y=indice_y,
            indice_anio=indice_anio,
            cantidad_total=LIMITES[
                "ARMA_FUEGO"
            ],
        )
    )

    # Contadores específicos por año
    contadores_fuego_anio = {
        anio: 0
        for anio in cupos_arma_fuego
    }

    # -------------------------------------------------------------------------
    # Crear nuevo Excel
    # -------------------------------------------------------------------------

    libro_salida = xlwt.Workbook()

    hoja_salida = libro_salida.add_sheet(
        hoja_entrada.name
    )

    # -------------------------------------------------------------------------
    # Copiar encabezados
    # -------------------------------------------------------------------------

    for columna in range(
        hoja_entrada.ncols
    ):

        hoja_salida.write(
            0,
            columna,
            hoja_entrada.cell_value(
                0,
                columna,
            ),
        )

    # -------------------------------------------------------------------------
    # Contadores generales
    # -------------------------------------------------------------------------

    contadores = {
        "ARMA_FUEGO": 0,
        "OTRAS": 0,
        "ARMA_BLANCA": 0,
        "ARMA_CONTUNDENTE": 0,
    }

    fila_salida = 1

    # =========================================================================
    # RECORRER FILA POR FILA
    # =========================================================================

    for fila in range(
        1,
        hoja_entrada.nrows
    ):

        categoria = str(
            hoja_entrada.cell_value(
                fila,
                indice_y,
            )
        ).strip()

        # ---------------------------------------------------------------------
        # Ignorar categorías diferentes
        # ---------------------------------------------------------------------

        if categoria not in LIMITES:
            continue

        # =====================================================================
        # CASO ESPECIAL: ARMA_FUEGO
        # =====================================================================

        if categoria == "ARMA_FUEGO":

            anio = normalizar_anio(
                hoja_entrada.cell_value(
                    fila,
                    indice_anio,
                )
            )

            # Año no válido
            if anio not in cupos_arma_fuego:
                continue

            # Ya completamos el cupo de ese año
            if (
                contadores_fuego_anio[anio]
                >= cupos_arma_fuego[anio]
            ):
                continue

        # =====================================================================
        # RESTO DE CATEGORÍAS
        # =====================================================================

        else:

            if (
                contadores[categoria]
                >= LIMITES[categoria]
            ):
                continue

        # ---------------------------------------------------------------------
        # Copiar TODA la fila sin modificar nada
        # ---------------------------------------------------------------------

        for columna in range(
            hoja_entrada.ncols
        ):

            valor_original = (
                hoja_entrada.cell_value(
                    fila,
                    columna,
                )
            )

            hoja_salida.write(
                fila_salida,
                columna,
                valor_original,
            )

        # ---------------------------------------------------------------------
        # Actualizar contadores
        # ---------------------------------------------------------------------

        contadores[categoria] += 1

        if categoria == "ARMA_FUEGO":

            contadores_fuego_anio[
                anio
            ] += 1

        fila_salida += 1

        # ---------------------------------------------------------------------
        # ¿Se completaron las cuatro clases?
        # ---------------------------------------------------------------------

        completado = all(
            contadores[categoria]
            >= LIMITES[categoria]
            for categoria in LIMITES
        )

        if completado:

            print(
                "\nSe completaron todas "
                "las categorías."
            )

            break

    # =========================================================================
    # RESUMEN GENERAL
    # =========================================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "RESUMEN GENERAL"
    )

    print(
        "=" * 65
    )

    todo_correcto = True

    for categoria in LIMITES:

        obtenido = contadores[
            categoria
        ]

        esperado = LIMITES[
            categoria
        ]

        print(
            f"{categoria:<22}"
            f"{obtenido:>6} / "
            f"{esperado}"
        )

        if obtenido != esperado:
            todo_correcto = False

    total = sum(
        contadores.values()
    )

    total_esperado = sum(
        LIMITES.values()
    )

    print("-" * 65)

    print(
        f"{'TOTAL':<22}"
        f"{total:>6} / "
        f"{total_esperado}"
    )

    # =========================================================================
    # RESUMEN ARMA_FUEGO POR AÑO
    # =========================================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "ARMA_FUEGO SELECCIONADAS POR AÑO"
    )

    print(
        "=" * 65
    )

    for anio in sorted(
        cupos_arma_fuego
    ):

        print(
            f"{anio}: "
            f"{contadores_fuego_anio[anio]} "
            f"/ {cupos_arma_fuego[anio]}"
        )

        if (
            contadores_fuego_anio[anio]
            != cupos_arma_fuego[anio]
        ):
            todo_correcto = False

    # =========================================================================
    # VALIDACIÓN
    # =========================================================================

    if not todo_correcto:

        print(
            "\nERROR:"
        )

        print(
            "No fue posible completar "
            "correctamente la muestra."
        )

        print(
            "No se generó el archivo."
        )

        return

    # =========================================================================
    # GUARDAR
    # =========================================================================

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    libro_salida.save(
        str(SALIDA)
    )

    print(
        "\nMuestra creada correctamente."
    )

    print(
        f"Salida:\n{SALIDA}"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    extraer_muestra()