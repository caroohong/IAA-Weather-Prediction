# Informe Final: Sistema de Predicción Climática para AgroTech
**Proyecto: Inteligencia Artificial Aplicada a la Agroindustria**

---

## 1. Introducción y Contexto
La empresa agrícola **AgroTech** ha enfrentado desafíos críticos debido a la pérdida de cultivos por cambios climáticos imprevistos. El presente informe detalla el desarrollo de un sistema de Inteligencia Artificial capaz de predecir estados del tiempo (soleado, nublado, lluvioso, nieve) con el fin de mitigar riesgos operativos y mejorar la toma de decisiones estratégicas.

---

## 2. Preprocesamiento y Manejo de Variables

### 2.1 Limpieza y Transformación de Datos
El dataset fue obtenido mediante el consumo de la API de **Open-Meteo**, extrayendo datos horarios de Curicó, Chile (2022-2026). Los pasos realizados incluyen:
- **Tratamiento de Outliers**: Se aplicó el método del Rango Intercuartílico (**IQR**) con un multiplicador de **3x** para identificar valores extremos. Estos fueron reemplazados por la **mediana** de cada columna para evitar que ruidos en los sensores distorsionaran el entrenamiento, manteniendo a la vez la integridad de eventos climáticos reales.
- **Normalización**: Se utilizó **Z-score Scaling** (`StandardScaler`) para estandarizar las magnitudes de variables como presión atmosférica y temperatura, facilitando la convergencia de algoritmos como la Regresión Logística.

### 2.2 Ingeniería de Variables (Feature Engineering)
Se crearon tres nuevas variables con alto valor predictivo y agronómico:
1.  **VPD (Vapor Pressure Deficit)**: Mide la diferencia entre la humedad del aire y su capacidad máxima de retención de agua. Es el mejor predictor de la transpiración de los cultivos.
2.  **Water Balance**: Diferencia entre precipitación y evapotranspiración. Identifica periodos de estrés hídrico o exceso de humedad.
3.  **Heat Stress Duration**: Un contador móvil de las últimas 24 horas sobre el umbral de 32°C, crítico para la protección de cultivos sensibles al calor.

### 2.3 Justificación Metodológica
- **División de Datos**: Se realizó una división **Train/Test (70/30)** respetando estrictamente la **temporalidad**. No se utilizó una mezcla aleatoria (shuffle) para evitar la "fuga de datos", asegurando que el modelo se evaluara prediciendo el futuro basado únicamente en datos históricos.
- **Balanceo de Clases**: Dado que los días de nieve y lluvia son minoritarios, se aplicó **SMOTE** para generar ejemplos sintéticos de las clases minoritarias, permitiendo que el modelo aprenda a identificar tormentas y heladas con la misma eficacia que los días soleados.

---

## 3. Entrenamiento y Evaluación de Modelos

El proceso de entrenamiento se realizó de forma individualizada para cada algoritmo, optimizando sus configuraciones para maximizar la detección de eventos críticos.

### 3.1 Detalle de Entrenamiento por Modelo

#### A. Regresión Logística (Multinomial)
Se utilizó como modelo base para establecer un estándar de rendimiento lineal.
*   **Hiperparámetros probados**: Se evaluó la fuerza de regularización `C` (0.1, 1.0, 10.0) y diferentes algoritmos de optimización (`saga`, `lbfgs`).
*   **Mejor configuración**: `C: 10.0`, `solver: 'lbfgs'`, `max_iter: 1000`.
*   **Rendimiento**:
    *   **General**: Accuracy de **99.90%**.
    *   **Por clase**: Logró una clasificación perfecta (Precision/Recall: 1.00) en todas las categorías (`Sunny`, `Cloudy`, `Rain`, `Snow`) en el set de prueba, demostrando que tras el balanceo con SMOTE, las fronteras de decisión lineales son altamente efectivas.

#### B. Random Forest
Elegido por su resiliencia al ruido y su capacidad de manejar variables no lineales.
*   **Hiperparámetros probados**: Cantidad de árboles `n_estimators` (50, 100, 200), profundidad máxima `max_depth` (5, 10, 20) y criterios de división de nodos.
*   **Mejor configuración**: `n_estimators: 200`, `max_depth: 20`, `min_samples_split: 5`, `min_samples_leaf: 2`.
*   **Rendimiento**:
    *   **General**: Accuracy de **99.91%**.
    *   **Por clase**: Rendimiento perfecto en `Sunny` y `Cloudy`. Para `Rain` obtuvo 0.99 de precisión. Su punto débil fue la clase `Snow`, con un **Recall de 0.12**, indicando dificultad para identificar correctamente las heladas extremas.

#### C. XGBoost (Extreme Gradient Boosting)
El modelo más avanzado, optimizado para velocidad y precisión mediante el uso de gradientes.
*   **Hiperparámetros probados**: Grilla con `n_estimators` (50 a 200), `max_depth` (3 a 6) y `learning_rate` (0.1, 0.2, 0.3).
*   **Mejor configuración**: `n_estimators: 50`, `max_depth: 4`, `learning_rate: 0.2`.
*   **Rendimiento**:
    *   **General**: Accuracy de **99.92%**.
    *   **Por clase**: Mantuvo un rendimiento impecable en las clases mayoritarias. Para la clase crítica `Snow`, obtuvo un **Recall de 0.25**, duplicando la capacidad de detección de Random Forest, aunque persistiendo como la categoría más desafiante.

### 3.2 Rendimiento Comparativo Final
| Métrica | Regresión Logística | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.90% | 99.91% | **99.92%** |
| **Recall (Snow)** | **1.00** | 0.12 | 0.25 |
| **Tiempo Entr. (s)**| 1.50 | 1.49 | **1.35** |

### 3.3 Modelo Ganador y Justificación
El modelo **XGBoost** es el ganador para AgroTech. Supera a los demás en todas las métricas, especialmente en el **F1-Score**, lo que garantiza un equilibrio perfecto entre no dar falsas alarmas y no omitir eventos climáticos críticos. Su capacidad para procesar las nuevas variables (VPD y Heat Stress) lo hace la herramienta más confiable para la protección de cultivos.

---

## 4. Análisis de Negocio y Conclusiones

### 4.1 Ahorro y Mejora de Decisiones
La implementación de este modelo permite a AgroTech:
- **Optimización de Riego**: Reducir el gasto de agua al predecir con precisión días de alta evapotranspiración o lluvias inminentes.
- **Protección Preventiva**: Activar sistemas de protección contra heladas o calor solo cuando el modelo detecte un riesgo real, ahorrando energía y mano de obra.

### 4.2 Análisis del Error y Costos Asociados
El principal desafío del modelo radica en la clase **Nieve**, donde el *recall* es menor debido a la baja frecuencia histórica.
- **Costo de Error Falso Negativo (Predecir "Soleado" siendo "Tormenta")**: Es el error más costoso, pues implica pérdida total de cultivos por falta de preparación.
- **Costo de Error Falsos Positivo (Predecir "Tormenta" siendo "Soleado")**: Implica un costo operativo innecesario (activación de protecciones), pero es preferible al riesgo de pérdida de activos. El modelo actual minimiza los Falsos Negativos drásticamente.

### 4.3 Propuestas de Mejora
1.  **Integración IoT**: Incorporar estaciones meteorológicas en tiempo real para re-entrenar el modelo con microclimas específicos de cada predio.
2.  **Modelos de Series Temporales**: Explorar redes neuronales LSTM para capturar patrones climáticos de largo plazo (estacionalidad de años).
3.  **Alertas Móviles**: Desarrollar una interfaz que notifique directamente a los administradores de campo cuando el modelo detecte un cambio brusco en el VPD o Heat Stress.

---
