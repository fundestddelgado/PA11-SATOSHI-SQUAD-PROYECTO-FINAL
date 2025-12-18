# Sistema de Clasificación de Madurez de Frutas

Sistema de inteligencia artificial para clasificar el estado de madurez de 5 tipos de frutas usando Deep Learning y Transfer Learning con MobileNetV2.

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Dataset](#dataset)
- [Modelo](#modelo)
- [Resultados](#resultados)
- [Personalización](#personalización)
- [Autores](#autores)

---

## 🎯 Descripción

Este proyecto implementa un sistema de clasificación de madurez de frutas usando técnicas de **Deep Learning** y **Transfer Learning**. El modelo es capaz de clasificar 5 tipos de frutas (manzana, banana, mango, naranja y papaya) en 3 estados de madurez diferentes: verde (sin madurar), madura y podrida.

### ¿Por qué es útil?

- 🏪 **Supermercados y tiendas**: Automatizar la clasificación de frutas
- 🏭 **Industria alimentaria**: Control de calidad en líneas de producción
- 🏠 **Consumidores**: Determinar si una fruta está lista para consumir
- 📚 **Educación**: Aprender sobre clasificación de imágenes con IA

---

## ✨ Características

- ✅ **15 clases de clasificación** (5 frutas × 3 estados)
- ✅ **Transfer Learning** con MobileNetV2 pre-entrenado en ImageNet
- ✅ **85.28% de precisión** en el conjunto de validación
- ✅ **Interfaz web interactiva** con Streamlit
- ✅ **Visualizaciones dinámicas** con gráficas de pastel personalizadas
- ✅ **Interfaz en español** (código y mensajes)
- ✅ **Predicciones en tiempo real** con confianza del modelo
- ✅ **Gráficas de entrenamiento** para visualizar el aprendizaje
- ✅ **Scripts automatizados** para descarga de dataset
- ✅ **Fácil de usar** con scripts bien documentados

---

## 🔧 Requisitos

### Software

- **Python 3.8+** (recomendado 3.9 o 3.10)
- **pip** (gestor de paquetes de Python)

### Hardware

- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Almacenamiento**: ~3GB libres
- **GPU**: Opcional (acelera el entrenamiento, pero funciona con CPU)

---

## 📦 Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### Dependencias Principales:

- `tensorflow==2.15.0` - Framework de Deep Learning
- `numpy==1.24.3` - Operaciones numéricas
- `matplotlib==3.8.2` - Visualización de datos
- `Pillow==10.1.0` - Procesamiento de imágenes
- `opencv-python==4.8.1.78` - Visión por computadora
- `streamlit==1.52.2` - Interfaz web interactiva
- `plotly==6.5.0` - Gráficas interactivas
- `bing-image-downloader==1.1.2` - Descarga de imágenes

---

## 📁 Estructura del Proyecto

```
SatoshiSquad-Hackathon/
│
├── data/                          # Dataset de imágenes
│   ├── manzana_verde/            # 92 imágenes
│   ├── manzana_madura/           # 94 imágenes
│   ├── manzana_podrida/          # 73 imágenes
│   ├── banana_verde/             # 73 imágenes
│   ├── banana_madura/            # 89 imágenes
│   ├── banana_podrida/           # 72 imágenes
│   ├── mango_verde/              # 69 imágenes
│   ├── mango_maduro/             # 97 imágenes
│   ├── mango_podrido/            # 55 imágenes
│   ├── naranja_verde/            # 84 imágenes
│   ├── naranja_madura/           # 81 imágenes
│   ├── naranja_podrida/          # 81 imágenes
│   ├── papaya_verde/             # 90 imágenes
│   ├── papaya_madura/            # 88 imágenes
│   └── papaya_podrida/           # 50 imágenes
│
├── models/                        # Modelos entrenados
│   ├── modelo_frutas/            # Modelo en formato TensorFlow
│   ├── modelo_frutas.h5          # Modelo en formato H5
│   ├── etiquetas.txt             # Lista de clases
│   └── historial_entrenamiento.png  # Gráficas de entrenamiento
│
├── scripts/                       # Scripts principales
│   ├── descargar_imagenes_bing.py   # Descarga del dataset
│   ├── train_model.py               # Entrenamiento del modelo
│   └── predecir.py                  # Predicciones
│
├── test_images/                   # Imágenes de prueba
│
├── app.py                         # Aplicación web con Streamlit
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

---

## 🚀 Uso

### Paso 1: Obtener el Dataset

#### Opción A: Descargar Automáticamente con Bing

```bash
python scripts/descargar_imagenes_bing.py
```

Este script:
- Descarga ~120 imágenes por cada una de las 15 clases
- Organiza las imágenes en carpetas
- Tiempo estimado: 30-45 minutos

#### Opción B: Dataset Manual

Descarga imágenes manualmente y organízalas en la estructura de carpetas mostrada arriba.

**Importante**: Necesitas mínimo 50 imágenes por clase para un buen entrenamiento.

### Paso 2: Entrenar el Modelo

```bash
python scripts/train_model.py
```

**Parámetros configurables** (en `train_model.py`):

```python
CARPETA_DATOS = "data"
TAMAÑO_IMAGEN = (224, 224)
TAMAÑO_LOTE = 16           # Optimizado para mejor precisión
EPOCAS_ETAPA1 = 10         # Épocas de entrenamiento inicial
EPOCAS_ETAPA2 = 10         # Épocas de ajuste fino
RUTA_MODELO = "models/modelo_frutas"
```

**Salida esperada**:
- Modelo entrenado: `models/modelo_frutas/`
- Gráficas: `models/historial_entrenamiento.png`
- Tiempo estimado: 60-90 minutos (depende de tu CPU/GPU)

### Paso 3: Hacer Predicciones

#### Opción A: Usando la Aplicación Web (Recomendado) 🌐

```bash
streamlit run app.py
```

#### Opción B: Usando Terminal (Script de Línea de Comandos)

```bash
python scripts/predecir.py <ruta_a_imagen>
```

**Ejemplos**:

```bash
# Predecir una imagen específica
python scripts/predecir.py test_images/banana.jpg

# Con ruta completa
python scripts/predecir.py C:/Users/tu_usuario/Pictures/manzana.jpg
```

---

## 📊 Dataset

### Composición

- **Total de imágenes**: 1,188
- **Total de clases**: 15 (5 frutas × 3 estados)
- **Promedio por clase**: ~79 imágenes

### Distribución por Fruta

| Fruta    | Verde | Madura | Podrida | Total |
|----------|-------|--------|---------|-------|
| Manzana  | 92    | 94     | 73      | 259   |
| Banana   | 73    | 89     | 72      | 234   |
| Mango    | 69    | 97     | 55      | 221   |
| Naranja  | 84    | 81     | 81      | 246   |
| Papaya   | 90    | 88     | 50      | 228   |

### Preprocesamiento

- **Tamaño**: Todas las imágenes se redimensionan a 224×224 pixels
- **Normalización**: Valores de píxeles normalizados a [0, 1]
- **División**: 80% entrenamiento, 20% validación

---

## 🧠 Modelo

### Arquitectura

El modelo utiliza **Transfer Learning** con MobileNetV2:

```
MobileNetV2 (pre-entrenado en ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense (1024 neuronas, ReLU)
    ↓
Dense (15 neuronas, Softmax)
```

### Especificaciones Técnicas

- **Modelo base**: MobileNetV2 pre-entrenado
- **Parámetros entrenables**: 2,853,199
- **Optimizador Etapa 1**: RMSprop
- **Optimizador Etapa 2**: SGD (lr=0.0001, momentum=0.9)
- **Función de pérdida**: Categorical Crossentropy
- **Métrica**: Accuracy

### Proceso de Entrenamiento

**Etapa 1** (5 épocas):
- Congela todas las capas de MobileNetV2
- Entrena solo las capas superiores añadidas

**Etapa 2** (5 épocas):
- Descongela las últimas capas de MobileNetV2
- Ajuste fino con learning rate bajo

---

## 📈 Resultados

### Métricas del Modelo

| Métrica                      | Valor   |
|------------------------------|---------|
| **Precisión en Entrenamiento** | 98.75%  |
| **Precisión en Validación**    | 85.28%  |
| **Pérdida en Validación**      | 0.6496  |
| **Tiempo de Entrenamiento**    | ~6 min  |

---

## 🛠️ Personalización

### Cambiar Frutas o Estados

Para modificar las clases del modelo:

1. **Actualizar carpetas** en `data/`
2. **Modificar listas** en los scripts:
   - `CONSULTAS_BUSQUEDA` en `descargar_imagenes_bing.py`
   - `CLASES` en `predecir.py`
3. **Ajustar capas del modelo**:
   ```python
   predictions = Dense(NUM_CLASES, activation='softmax')(x)
   ```
4. **Re-entrenar el modelo**

### Ajustar Hiperparámetros

En `train_model.py`:

```python
# Configuración actual (optimizada para precisión)
TAMAÑO_LOTE = 16
EPOCAS_ETAPA1 = 10
EPOCAS_ETAPA2 = 10

# Para entrenamiento más rápido (menor precisión)
TAMAÑO_LOTE = 32
EPOCAS_ETAPA1 = 5
EPOCAS_ETAPA2 = 5

# Para máxima precisión (más lento)
TAMAÑO_LOTE = 8
EPOCAS_ETAPA1 = 15
EPOCAS_ETAPA2 = 15
```

---

## 👥 Autores

**Satoshi Squad** - Proyecto desarrollado en **SAMSUNG INNOVATION CAMPUS SIC 2025**

| Integrante | Rol |
|------------|-----|
| **Nayim Rodríguez** | Documentación |
| **Hassan El Rada** | Preparación del Dataset (Analista de Datos) y Desarrollador Backend |
| **Li Chao Wu** | Preparación del Dataset (Analista de Datos) y Desarrollador Backend |
| **Diego Corrales** | Desarrollador Front-end |
| **Valentín Rodríguez** | Documentación |

---

<div align="center">

Hecho en Panamá 🇵🇦

</div>

