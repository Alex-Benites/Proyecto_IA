"""
ETAPA 6 - PROBAR EL MODELO CON UN CASO NUEVO
==============================================

Este script existe para responder: "¿donde pruebo el modelo?".

Es el PROTOTIPO de lo que hara la app Streamlit. Muestra la cadena completa
de prediccion sobre un caso que el modelo nunca vio:

    datos crudos (lo que el usuario escribe en la interfaz)
        -> preprocesador.joblib   (imputa + codifica + escala IGUAL que en train)
        -> modelo1_logistica.joblib
        -> probabilidad por clase

IMPORTANTE: el preprocesador debe ser el MISMO objeto guardado en la Etapa 3.
Si se recalculara aqui, las columnas one-hot quedarian en otro orden y el
modelo recibiria basura. Por eso se guarda y se reutiliza, no se re-entrena.

Uso:
    python 06_probar_modelo.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DIR_MOD = BASE / "modelos"

# Orden EXACTO de columnas que espera el preprocesador (definido en 03_preparacion.py)
COLS_CATEGORICAS = [
    "zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar",
    "presunta_motivacion", "presun_motiva_observada", "sexo", "etnia",
    "estado_civil", "nacionalidad", "discapacidad", "mes", "dia_semana",
    "hora", "franja_horaria",
]
COLS_NUMERICAS = ["edad", "anio"]
COL_BINARIA = ["es_fin_semana"]
ORDEN_COLUMNAS = COLS_CATEGORICAS + COLS_NUMERICAS + COL_BINARIA


# --- Casos de ejemplo (esto es lo que en la app vendra de los formularios) ---
CASOS = [
    {
        "descripcion": "Rina nocturna en via publica, Guayaquil, hombre joven",
        "zona": "ZONA 8", "provincia": "GUAYAS", "canton": "GUAYAQUIL",
        "area_hecho": "URBANO", "lugar": "VIA PUBLICA", "tipo_lugar": "PUBLICO",
        "presunta_motivacion": "VIOLENCIA COMUNITARIA",
        "presun_motiva_observada": "RINAS",
        "sexo": "HOMBRE", "etnia": "MESTIZO/A", "estado_civil": "SOLTERO",
        "nacionalidad": "ECUADOR", "discapacidad": "NINGUNA",
        "mes": "7", "dia_semana": "5", "hora": "23", "franja_horaria": "NOCHE",
        "edad": 24, "anio": 2025, "es_fin_semana": 1,
    },
    {
        "descripcion": "Violencia intrafamiliar en casa, zona rural, mujer adulta",
        "zona": "ZONA 6", "provincia": "CANAR", "canton": "CANAR",
        "area_hecho": "RURAL", "lugar": "CASA/VILLA", "tipo_lugar": "PRIVADO",
        "presunta_motivacion": "VIOLENCIA INTRAFAMILIAR",
        "presun_motiva_observada": "SENTIMENTAL",
        "sexo": "MUJER", "etnia": "INDIGENA", "estado_civil": "CASADO",
        "nacionalidad": "ECUADOR", "discapacidad": "NINGUNA",
        "mes": "4", "dia_semana": "2", "hora": "10", "franja_horaria": "MANANA",
        "edad": 47, "anio": 2025, "es_fin_semana": 0,
    },
    {
        "descripcion": "Microtrafico, madrugada, via publica en Machala",
        "zona": "ZONA 7", "provincia": "EL ORO", "canton": "MACHALA",
        "area_hecho": "URBANO", "lugar": "VIA PUBLICA", "tipo_lugar": "PUBLICO",
        "presunta_motivacion": "DELINCUENCIA COMUN",
        "presun_motiva_observada": "TRAFICO INTERNO DE DROGAS (MICROTRAFICO)",
        "sexo": "HOMBRE", "etnia": "MESTIZO/A", "estado_civil": "SOLTERO",
        "nacionalidad": "VENEZUELA", "discapacidad": "NINGUNA",
        "mes": "12", "dia_semana": "4", "hora": "3", "franja_horaria": "MADRUGADA",
        "edad": 31, "anio": 2025, "es_fin_semana": 0,
    },
]


def predecir(preproc, modelo, caso):
    """Convierte un dict de datos crudos en una prediccion con probabilidades."""
    datos = {k: v for k, v in caso.items() if k != "descripcion"}
    df = pd.DataFrame([datos])[ORDEN_COLUMNAS]

    # El preprocesador ya sabe imputar y codificar: se le pasa tal cual.
    X = preproc.transform(df)

    clase = modelo.predict(X)[0]
    probas = modelo.predict_proba(X)[0]
    return clase, dict(zip(modelo.classes_, probas))


def barra(p, ancho=32):
    """Barra de texto para ver la probabilidad de un vistazo."""
    lleno = int(round(p * ancho))
    return "#" * lleno + "." * (ancho - lleno)


def main():
    preproc = joblib.load(DIR_MOD / "preprocesador.joblib")
    modelo = joblib.load(DIR_MOD / "modelo1_logistica.joblib")

    print("=" * 74)
    print("PRUEBA DEL MODELO 1 (Regresion Logistica) CON CASOS NUEVOS")
    print("=" * 74)
    print("El modelo NO recibe ningun dato sobre el arma. Solo contexto del hecho.")

    for caso in CASOS:
        clase, probas = predecir(preproc, modelo, caso)
        print()
        print("-" * 74)
        print(f"CASO: {caso['descripcion']}")
        print(f"  {caso['canton']} / {caso['lugar']} / {caso['franja_horaria']} / "
              f"{caso['sexo']} {caso['edad']} anios / {caso['presunta_motivacion']}")
        print()
        for k, v in sorted(probas.items(), key=lambda x: -x[1]):
            marca = "  <-- PREDICCION" if k == clase else ""
            print(f"    {k:<18} {v*100:5.1f}%  {barra(v)}{marca}")

    print()
    print("-" * 74)
    print("Nota: la clase mas probable NO es una certeza. Con F1 macro de 0.61,")
    print("el modelo acierta aproximadamente 6 de cada 10 veces. En la app hay")
    print("que MOSTRAR estas probabilidades, no solo la clase ganadora.")


if __name__ == "__main__":
    main()
