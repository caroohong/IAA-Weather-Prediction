import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_recall_fscore_support,
    log_loss
)
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Configuración de consola para outputs estéticos
console = Console()

def load_test_data(file_path='weather_curico_preprocessed.csv', split_ratio=0.7):
    """Carga los datos y realiza el split temporal idéntico al entrenamiento."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    split_index = int(len(df) * split_ratio)
    test_df = df.iloc[split_index:]
    
    # Separar Features y Target (idéntico a notebooks de entrenamiento)
    X_test = test_df.drop(columns=['date', 'weather_condition', 'weather_code'])
    y_test = test_df['weather_condition']
    
    return X_test, y_test

def evaluate_model(model_name, model_path, encoder_path, X_test, y_test):
    """Carga un modelo y su encoder, predice y calcula métricas."""
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
    except Exception as e:
        return {"error": str(e)}

    # Codificar etiquetas reales
    y_test_encoded = le.transform(y_test)
    
    # Medir tiempo de inferencia
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Obtener probabilidades para Log Loss
    try:
        y_prob = model.predict_proba(X_test)
        loss = log_loss(y_test_encoded, y_prob)
    except:
        loss = np.nan

    # Métricas generales
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average='macro')
    
    # Métricas por clase
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_test_encoded, y_pred, average=None)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "F1-Score (Macro)": f1,
        "Log Loss": loss,
        "Avg Inference Time (ms)": inference_time * 1000,
        "Per Class": {}
    }
    
    for i, cls in enumerate(le.classes_):
        metrics["Per Class"][cls] = {
            "Precision": p_class[i],
            "Recall": r_class[i],
            "F1": f1_class[i]
        }
        
    return metrics

def run_comparison():
    console.print("[bold cyan]Iniciando Evaluación Comparativa de Modelos...[/bold cyan]\n")
    
    # 1. Cargar Datos
    X_test, y_test = load_test_data()
    console.print(f"Set de prueba cargado: [bold]{len(X_test)}[/bold] registros.\n")
    
    # 2. Definir modelos a evaluar
    models_config = {
        "XGBoost": {
            "model": "models/outputs/xgboost_model.joblib",
            "encoder": "models/outputs/label_encoder_xgb.joblib"
        },
        "Random Forest": {
            "model": "models/outputs/random_forest_model.joblib",
            "encoder": "models/outputs/label_encoder_rf.joblib"
        },
        "Logistic Regression": {
            "model": "models/outputs/logistic_regression_model.joblib",
            "encoder": "models/outputs/label_encoder_lg.joblib"
        }
    }
    
    results = {}
    
    for name, paths in track(models_config.items(), description="Evaluando modelos..."):
        results[name] = evaluate_model(name, paths["model"], paths["encoder"], X_test, y_test)

    # 3. Mostrar Tabla de Métricas Generales
    table = Table(title="Comparación General de Modelos")
    table.add_column("Modelo", style="magenta")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1-Score (Macro)", justify="right")
    table.add_column("Log Loss", justify="right")
    table.add_column("Inference (ms/obs)", justify="right")

    for name, m in results.items():
        if "error" in m:
            table.add_row(name, "[red]Error", "-", "-", "-")
            continue
        table.add_row(
            name, 
            f"{m['Accuracy']:.4f}", 
            f"{m['F1-Score (Macro)']:.4f}", 
            f"{m['Log Loss']:.4f}", 
            f"{m['Avg Inference Time (ms)']:.4f}"
        )
    
    console.print(table)

    # 4. Mostrar Desglose por Clase (F1-Score)
    class_table = Table(title="F1-Score por Clase")
    class_table.add_column("Modelo", style="magenta")
    
    # Obtener todas las clases únicas presentes
    all_classes = sorted(results["XGBoost"]["Per Class"].keys())
    for cls in all_classes:
        class_table.add_column(cls, justify="right")

    for name, m in results.items():
        if "error" in m: continue
        row = [name]
        for cls in all_classes:
            f1 = m["Per Class"].get(cls, {}).get("F1", 0)
            row.append(f"{f1:.4f}")
        class_table.add_row(*row)

    console.print(class_table)

    # 5. Determinar Ganador
    # Criterio: Mayor F1-Score Macro, desempate por Accuracy y luego Log Loss
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    winner = max(valid_results, key=lambda k: (valid_results[k]["F1-Score (Macro)"], valid_results[k]["Accuracy"]))
    
    console.print(f"\n[bold green]EL MEJOR MODELO ES: {winner}[/bold green]")
    console.print(f"Justificación: Lidera en F1-Score Macro ({valid_results[winner]['F1-Score (Macro)']:.4f}) y equilibrio entre clases.")

if __name__ == "__main__":
    run_comparison()
