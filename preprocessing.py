import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    csv_files = sorted(glob.glob("weather_curico_*.csv"))
    if not csv_files:
        print("No se encontraron archivos CSV.")
        return
    print(f"Cargando {len(csv_files)} archivos...")
    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error al leer {f}: {e}")
    df = pd.concat(df_list, ignore_index=True)

    print("ESTADO INICIAL DEL DATASET")
    print("="*40)
    print(f"Total de registros: {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    print("\nValores nulos por columna:")
    nulos_iniciales = df.isnull().sum()
    print(nulos_iniciales[nulos_iniciales > 0] if nulos_iniciales.sum() > 0 else "No se detectaron nulos inicialmente.")
    
    print("\nDistribución de clases (weather_condition):")
    print(df['weather_condition'].value_counts() if 'weather_condition' in df.columns else "Variable objetivo no encontrada.")
    print("="*40 + "\n")

    # convertir fecha y ordenar cronológicamente
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print("Limpieza y manejo de nulos...")
    cols_to_drop = ['weather_code']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Tratamiento de nulos: Mediana para numéricos
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Tratamiento de nulos: Moda para la variable objetivo
    if 'weather_condition' in df.columns:
        df['weather_condition'] = df['weather_condition'].fillna(df['weather_condition'].mode()[0])

    print("Nulos imputados (Mediana para numéricos, Moda para categóricos).")
    print(f"Nulos restantes: {df.isnull().sum().sum()}")

    # Normalización de datos numéricos (StandardScaler)
    print("Normalizando variables numéricas...")
    # excluimos 'date' y la variable objetivo 'weather_condition'
    features_to_scale = [c for c in num_cols if c not in ['weather_condition', 'is_day']]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print(f"Variables normalizadas: {len(features_to_scale)}")

    # División de datos (70% Train, 30% Test) respetando temporalidad
    print("\nDividiendo dataset (70% Train / 30% Test)...")
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Split temporal realizado: {len(train_df)} registros para Train, {len(test_df)} para Test.")

    # Crear estructura de directorios y guardar por clase
    def save_by_class(dataset, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        classes = dataset['weather_condition'].unique()
        for cls in classes:
            # Reemplazar espacios por guiones bajos para nombres de carpetas seguros
            folder_name = cls.lower().replace(" ", "_")
            class_dir = os.path.join(base_dir, folder_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Guardar los registros de esa clase
            class_data = dataset[dataset['weather_condition'] == cls]
            class_data.to_csv(os.path.join(class_dir, "data.csv"), index=False)
            print(f"  - Guardada clase '{cls}' en {base_dir} ({len(class_data)} registros)")
    print("\nOrganizando carpeta 'train'...")
    save_by_class(train_df, "train")
    print("\nOrganizando carpeta 'test'...")
    save_by_class(test_df, "test")

    print("\nProceso de preprocesamiento y organización finalizado.")

if __name__ == "__main__":
    preprocess_data()
