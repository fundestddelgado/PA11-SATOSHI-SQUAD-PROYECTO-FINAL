# PA11-SATOSHI-SQUAD-PROYECTO-FINAL
---

# 🍌 Clasificador de Madurez del Plátano
Sistema de clasificación inteligente que utiliza Deep Learning para determinar el estado de madurez de plátanos mediante análisis de imágenes. El proyecto implementa una Red Neuronal Convolucional (CNN) entrenada con más de 11,000 imágenes para clasificar plátanos en cuatro estados de madurez.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Dataset](#-dataset)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Autores](#-autores)

## ✨ Características

- **Clasificación precisa** en 4 estados de madurez: Inmaduro, Maduro, Podrido y Sobremaduro
- **Interfaz web interactiva** desarrollada con Streamlit
- **Visualizaciones avanzadas** con gráficos de barras y pastel usando Plotly
- **Modelo CNN optimizado** con capas de convolución, pooling y dropout
- **Data augmentation** para mejorar la generalización del modelo
- **Predicción por consola** para análisis rápido de imágenes individuales
- **Confianza del modelo** mostrada en porcentaje para cada predicción



## 🛠 Tecnologías Utilizadas

### Backend y Machine Learning
- **TensorFlow 2.x** - Framework de Deep Learning
- **Keras** - API de alto nivel para redes neuronales
- **NumPy** - Procesamiento numérico y arrays
- **PIL (Pillow)** - Manipulación de imágenes

### Frontend y Visualización
- **Streamlit** - Framework para aplicaciones web interactivas
- **Plotly** - Gráficos interactivos avanzados
- **Matplotlib** - Visualización de métricas de entrenamiento

### Preprocesamiento de Datos
- **ImageDataGenerator** - Aumento de datos y normalización

## 📊 Dataset

El dataset contiene **11,793 imágenes** de plátanos organizadas en 4 categorías:

| Estado | Cantidad de Imágenes | Descripción |
|--------|---------------------|-------------|
| **Inmaduro** | 1,902 | Plátanos verdes, no aptos para consumo |
| **Maduro** | 3,522 | Plátanos amarillos, listos para consumir |
| **Podrido** | 4,020 | Plátanos en estado de descomposición |
| **Sobremaduro** | 2,349 | Plátanos muy maduros con manchas oscuras |

### Estructura del Dataset
```
dataset/
├── inmaduro/     # 1,902 imágenes
├── maduro/       # 3,522 imágenes
├── podrido/      # 4,020 imágenes
└── sobremaduro/  # 2,349 imágenes
```

### Preprocesamiento Aplicado
- **Normalización**: Valores de píxeles escalados de [0, 255] a [0, 1]
- **Redimensionamiento**: Todas las imágenes a 150x150 píxeles
- **Aumento de datos**:
  - Rotación aleatoria de ±15°
  - Desplazamiento horizontal y vertical (10%)
  - Volteo horizontal aleatorio
- **División de datos**: 80% entrenamiento, 20% validación

## 🚀 Instalación

### Instalar dependencias

```bash
pip install tensorflow numpy pillow matplotlib streamlit plotly
```

O usando el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Contenido de `requirements.txt`
```
tensorflow>=2.10.0
numpy>=1.23.0
pillow>=9.0.0
matplotlib>=3.5.0
streamlit>=1.20.0
plotly>=5.10.0
```

## 📖 Uso

### 1. Entrenamiento del Modelo

Para entrenar el modelo con el dataset:

```bash
python entrenar.py
```

Este proceso:
- Carga y procesa las 11,793 imágenes del dataset
- Entrena la red neuronal durante 15 épocas
- Guarda el modelo entrenado en `modelo/modelo_banana.h5`
- Genera gráficos de precisión y pérdida

**Tiempo estimado**: 15-30 minutos (depende del hardware)

### 2. Predicción por Consola

Para analizar una imagen específica:

```bash
# Opción 1: Proporcionar la ruta como argumento
python predecir.py ruta/a/imagen.jpg

# Opción 2: Ejecutar y luego ingresar la ruta
python predecir.py
# Se te pedirá ingresar la ruta de la imagen
```

**Salida esperada**:
```
==============================
RESULTADO DEL ANÁLISIS
==============================
Estado detectado: MADURO
Confianza de la IA: 94.32%
==============================

Detalle de probabilidades:
inmaduro: 2.15%
maduro: 94.32%
podrido: 0.08%
sobremaduro: 3.45%
```

### 3. Interfaz Web Interactiva

Para lanzar la aplicación web:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
Proyecto-Platanos/
│
├── app.py                 # Aplicación web con Streamlit
├── entrenar.py            # Script de entrenamiento del modelo
├── predecir.py            # Script de predicción por consola
├── README.md              # Documentación del proyecto
├── requirements.txt       # Dependencias del proyecto
│
├── dataset/              # Dataset de imágenes
│   ├── inmaduro/        # 1,902 imágenes
│   ├── maduro/          # 3,522 imágenes
│   ├── podrido/         # 4,020 imágenes
│   └── sobremaduro/     # 2,349 imágenes
│
└── modelo/              # Modelos entrenados
    └── modelo_banana.h5 # Modelo guardado (generado tras entrenamiento)
```

## 📈 Resultados

### Métricas del Modelo

El modelo alcanzó las siguientes métricas después de 15 épocas de entrenamiento:

| Métrica | Entrenamiento | Validación |
|---------|--------------|------------|
| **Accuracy** | 94.12% | 81.93% |
| **Loss** | 0.1669 | 0.5525 |

El modelo muestra un buen rendimiento en el conjunto de entrenamiento con una precisión del 94.12%. En validación alcanza un 81.93% de precisión, lo que indica capacidad de generalización a nuevas imágenes de plátanos.

Durante el entrenamiento, se generan automáticamente dos gráficos que muestran la evolución de la precisión y la pérdida a lo largo de las épocas.


## 👥 Autores
**Satoshi Squad** - Proyecto desarrollado en **SAMSUNG INNOVATION CAMPUS SIC 2025**

| Integrante | Rol |
|------------|-----|
| **Nayim Rodríguez** | Desarrollador |
| **Hassan El Rada** | Preparación del Dataset (Analista de Datos) y Desarrollador Backend |
| **Li Chao Wu** | Preparación del Dataset (Analista de Datos) y Desarrollador Backend |
| **Diego Corrales** | Documentación y Desarrollador |
| **Valentín Rodríguez** | Documentación |
