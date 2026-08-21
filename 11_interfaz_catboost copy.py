import pandas as pd

df = pd.read_excel(
    "data/processed/A_2014_2025_muestraFinalv2.xls"
)

print("\n==============================")
print("AÑO VS CLASE - CONTEOS")
print("==============================")

print(
    pd.crosstab(
        df["anio"],
        df["y"]
    )
)


print("\n==============================")
print("AÑO VS CLASE - PORCENTAJES")
print("==============================")

print(
    (
        pd.crosstab(
            df["anio"],
            df["y"],
            normalize="columns"
        ) * 100
    ).round(2)
)


print("\n==============================")
print("FECHA MIN/MAX POR CLASE")
print("==============================")

df["fecha"] = pd.to_datetime(
    df["fecha"],
    errors="coerce"
)

print(
    df.groupby("y")["fecha"]
      .agg(
          cantidad="count",
          fecha_min="min",
          fecha_max="max"
      )
)