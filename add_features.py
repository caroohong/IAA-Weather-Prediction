import pandas as pd
import numpy as np
import glob
import os

def add_features(file_path):
    print(f"Procesando: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al leer {file_path}: {e}")
        return

    # Verificar que las columnas necesarias existan antes de calcular
    required_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'et0_fao_evapotranspiration']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ Saltando {file_path}: No contiene todas las columnas base necesarias.")
        return
    
    modified = False

    # VPD (Déficit de Presión de Vapor) en kPa
    if 'VPD' not in df.columns:
        # Presión de Vapor de Saturación (SVP)
        svp = 0.61078 * np.exp((17.27 * df['temperature_2m']) / (df['temperature_2m'] + 237.3))
        # Presión de Vapor Real (AVP)
        avp = svp * (df['relative_humidity_2m'] / 100)
        df['VPD'] = svp - avp
        modified = True

    # Water Balance (P - ET0) en mm
    if 'water_balance' not in df.columns:
        df['water_balance'] = df['precipitation'] - df['et0_fao_evapotranspiration']
        modified = True

    # Heat Stress Duration (Horas > 32°C en las últimas 24h)
    if 'heat_stress_duration' not in df.columns:
        # serie booleana (1 si > 32, 0 si no)
        is_hot = (df['temperature_2m'] > 32).astype(int)
        # suma móvil de las últimas 24 registros (asumiendo datos horarios)
        df['heat_stress_duration'] = is_hot.rolling(window=24, min_periods=1).sum()
        modified = True

    if modified:
        df.to_csv(file_path, index=False)
        print(f"Archivo actualizado: {file_path}")
    else:
        print(f"ℹEl archivo {file_path} ya contenía las variables. No se realizaron cambios.")

if __name__ == "__main__":
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No se encontraron archivos CSV en el directorio.")
    else:
        for file in csv_files:
            # filtrar
            if "weather_curico" in os.path.basename(file):
                add_features(file)
            else:
                pass
    print("\nProceso finalizado.")
