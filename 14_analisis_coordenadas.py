import pandas as pd

df = pd.read_excel(
    "data/processed/A_2014_2025_muestraFinal.xls"
)

for columna in [
    "coordenada_x",
    "coordenada_y"
]:
    print("\n", columna)
    print("-" * 40)

    print("Tipo:")
    print(df[columna].dtype)

    print("Valores únicos:")
    print(df[columna].nunique(dropna=False))

    print("Primeros valores:")
    print(df[columna].head(20).tolist())

    print("Más frecuentes:")
    print(
        df[columna]
        .value_counts(dropna=False)
        .head(10)
    )