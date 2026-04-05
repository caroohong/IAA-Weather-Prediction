import pandas as pd

archivo = "weather_curico_2025.csv"  # usa tu CSV final

df = pd.read_csv(archivo)


# convertir a datetime
df["date"] = pd.to_datetime(df["date"])

print("===================================")
print("RESUMEN GENERAL")
print("===================================")

print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\nColumnas:")
print(df.columns.tolist())

print("\n===================================")
print("PERÍODO DE TIEMPO")
print("===================================")

print(f"Fecha inicio: {df['date'].min()}")
print(f"Fecha fin: {df['date'].max()}")

duracion = df["date"].max() - df["date"].min()
print(f"Duración total: {duracion}")

print(f"Número de timestamps únicos: {df['date'].nunique()}")

# frecuencia estimada
print("\nFrecuencia estimada:")
print(df["date"].diff().value_counts().head())

df["month"] = df["date"].dt.month

print("\n===================================")
print("REGISTROS POR MES")
print("===================================")

print(df["month"].value_counts().sort_index())

print("\nDistribución por mes y clima:")
print(pd.crosstab(df["month"], df["weather_condition"]))

print("\n===================================")
print("VALORES NULOS")
print("===================================")

nulos = df.isnull().sum()
print(nulos)

print(f"\nTotal nulos: {nulos.sum()}")

print("\n===================================")
print("DISTRIBUCIÓN weather_condition")
print("===================================")

conteo = df["weather_condition"].value_counts()
print(conteo)

print("\nPorcentaje:")
porcentaje = (df["weather_condition"].value_counts(normalize=True) * 100).round(2)
print(porcentaje)

print("\n===================================")
print("UNKNOWN")
print("===================================")

unknown_count = (df["weather_condition"] == "Unknown").sum()
print(f"Unknown: {unknown_count}")
print(f"% Unknown: {(unknown_count / len(df) * 100):.2f}%")


print("\n===================================")
print("ESTADÍSTICAS NUMÉRICAS")
print("===================================")

print(df.describe())


print("\n===================================")
print("INFO DATASET")
print("===================================")

df.info()

print("\n===================================")
print("INSIGHTS CLAVE")
print("===================================")

print(f"- Total observaciones: {len(df)}")
print(f"- Variables: {len(df.columns)}")
print(f"- Clases distintas: {df['weather_condition'].nunique()}")

# clase dominante
top_class = df["weather_condition"].value_counts().idxmax()
top_pct = df["weather_condition"].value_counts(normalize=True).max() * 100

print(f"- Clase dominante: {top_class} ({top_pct:.2f}%)")

# clase menos frecuente
min_class = df["weather_condition"].value_counts().idxmin()
min_pct = df["weather_condition"].value_counts(normalize=True).min() * 100

print(f"- Clase menos frecuente: {min_class} ({min_pct:.2f}%)")