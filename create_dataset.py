import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import date

LUGARES = {
    "1":  {"nombre": "Curico",            "lat": -34.98, "lon": -71.24},
    "2":  {"nombre": "Santiago",           "lat": -33.45, "lon": -70.67},
    "3":  {"nombre": "Valparaíso",         "lat": -33.05, "lon": -71.62},
    "4":  {"nombre": "Concepción",         "lat": -36.83, "lon": -73.05},
    "5":  {"nombre": "La Serena",          "lat": -29.91, "lon": -71.25},
    "6":  {"nombre": "Antofagasta",        "lat": -23.65, "lon": -70.40},
    "7":  {"nombre": "Temuco",             "lat": -38.74, "lon": -72.59},
    "8":  {"nombre": "Puerto Montt",       "lat": -41.47, "lon": -72.94},
    "9":  {"nombre": "Punta Arenas",       "lat": -53.16, "lon": -70.91},
    "10": {"nombre": "Arica",              "lat": -18.48, "lon": -70.32},
    "11": {"nombre": "Valle Central",      "lat": -34.50, "lon": -71.00},
    "12": {"nombre": "Rancagua",           "lat": -34.17, "lon": -70.74},
    "13": {"nombre": "O'Higgins", "lat": -34.17, "lon": -70.74},
    "14": {"nombre": "Maule",     "lat": -35.43, "lon": -71.67},
}
print("===================================")
print("LUGARES DISPONIBLES")
print("===================================")
for key, lugar in LUGARES.items():
    print(f"  {key:>2}. {lugar['nombre']:<20} (lat: {lugar['lat']}, lon: {lugar['lon']})")
print("===================================")

while True:
    opcion = input("\nSelecciona un lugar (número): ").strip()
    if opcion in LUGARES:
        lugar_seleccionado = LUGARES[opcion]
        break
    print(f"Opción inválida. Elige entre 1 y {len(LUGARES)}.")
nombre_lugar = lugar_seleccionado["nombre"]
lat = lugar_seleccionado["lat"]
lon = lugar_seleccionado["lon"]
print(f"\nLugar seleccionado: {nombre_lugar} ({lat}, {lon})")

while True:
    year = input("Año a analizar (ej: 2024): ").strip()
    if year.isdigit() and 1940 <= int(year) <= 2026:
        break
    print("Año inválido. Debe ser entre 1940 y 2026.")

# Setup API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
year_int = int(year)
today = date.today()
if year_int == today.year:
    end_date = today.strftime("%Y-%m-%d")
else:
    end_date = f"{year}-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": f"{year}-01-01",
    "end_date": end_date,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
        "weather_code",
        "pressure_msl",
        "is_day",
        "vapour_pressure_deficit",
        "shortwave_radiation",
        "apparent_temperature",
        "et0_fao_evapotranspiration",
        "soil_temperature_0_to_7cm",
        "soil_moisture_0_to_7cm"
    ],
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]
print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

hourly = response.Hourly()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m":             hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m":       hourly.Variables(1).ValuesAsNumpy(),
    "precipitation":              hourly.Variables(2).ValuesAsNumpy(),
    "rain":                       hourly.Variables(3).ValuesAsNumpy(),
    "cloud_cover":                hourly.Variables(4).ValuesAsNumpy(),
    "wind_speed_10m":             hourly.Variables(5).ValuesAsNumpy(),
    "weather_code":               hourly.Variables(6).ValuesAsNumpy(),
    "pressure_msl":               hourly.Variables(7).ValuesAsNumpy(),
    "is_day":                     hourly.Variables(8).ValuesAsNumpy(),
    "vapour_pressure_deficit":    hourly.Variables(9).ValuesAsNumpy(),
    "shortwave_radiation":        hourly.Variables(10).ValuesAsNumpy(),
    "apparent_temperature":       hourly.Variables(11).ValuesAsNumpy(),
    "et0_fao_evapotranspiration": hourly.Variables(12).ValuesAsNumpy(),
    "soil_temperature_0_to_7cm":  hourly.Variables(13).ValuesAsNumpy(),
    "soil_moisture_0_to_7cm":     hourly.Variables(14).ValuesAsNumpy(),
}
df = pd.DataFrame(hourly_data)

def map_weather(code, is_day):
    if code in [0, 1]:
        return "Sunny"
    elif code in [2, 3]:
        return "Cloudy"
    elif code in [45, 48]:
        return "Fog"
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
        return "Rain"
    elif code in [71, 73, 75, 77, 85, 86]:
        return "Snow"
    elif code in [95, 96, 99]:
        return "Storm"
    else:
        return "Unknown"

df["weather_condition"] = df.apply(
    lambda row: map_weather(row["weather_code"], row["is_day"]),
    axis=1
)
print("\nShape:", df.shape)
print("\nColumnas:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nDistribución weather_condition:")
print(df["weather_condition"].value_counts())

nombre_archivo = f"weather_{nombre_lugar.lower().replace(' ', '_')}_{year}.csv"
df.to_csv(nombre_archivo, index=False)

print(f"\nCSV guardado como {nombre_archivo}")