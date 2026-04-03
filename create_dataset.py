import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": -34.98,
    "longitude": -71.24,
    "start_date": "2025-01-01",
    "end_date": "202",
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

print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

hourly = response.Hourly()

# Variables
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "precipitation": hourly.Variables(2).ValuesAsNumpy(),
    "rain": hourly.Variables(3).ValuesAsNumpy(),
    "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
    "wind_speed_10m": hourly.Variables(5).ValuesAsNumpy(),
    "weather_code": hourly.Variables(6).ValuesAsNumpy(),
    "pressure_msl": hourly.Variables(7).ValuesAsNumpy(),
    "is_day": hourly.Variables(8).ValuesAsNumpy(),
    "vapour_pressure_deficit": hourly.Variables(9).ValuesAsNumpy(),
    "shortwave_radiation": hourly.Variables(10).ValuesAsNumpy(),
    "apparent_temperature": hourly.Variables(11).ValuesAsNumpy(),
    "et0_fao_evapotranspiration": hourly.Variables(12).ValuesAsNumpy(),
    "soil_temperature_0_to_7cm": hourly.Variables(13).ValuesAsNumpy(),
    "soil_moisture_0_to_7cm": hourly.Variables(14).ValuesAsNumpy()
}

df = pd.DataFrame(hourly_data)

def map_weather(code, is_day):
    if code in[0, 1]:
        return "Sunny" if is_day == 1 else "Clear"
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

df.to_csv("weather_curico_2025.csv", index=False)

print("\nCSV guardado como weather_curico_2025.csv")