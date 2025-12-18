"""
Script para entrenar modelo de clasificación de madurez de frutas
5 frutas (manzana, banana, mango, naranja, papaya) x 3 estados = 15 clases
Usa Transfer Learning con MobileNetV2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import os
from datetime import datetime

# CONFIGURACIÓN
CARPETA_DATOS = "data"
TAMAÑO_IMAGEN = (224, 224)
TAMAÑO_LOTE = 16
EPOCAS_ETAPA1 = 10  # Primera etapa
EPOCAS_ETAPA2 = 10  # Segunda etapa (ajuste fino)
RUTA_MODELO = "models\modelo_frutas"

# VERIFICAR CONFIGURACIÓN
print("=" * 80)
print("CONFIGURACIÓN DEL ENTRENAMIENTO")
print("=" * 80)
print(f"Carpeta de datos: {CARPETA_DATOS}")
print(f"Carpeta de modelos: {os.path.dirname(RUTA_MODELO)}")
print(f"Tamaño de imagen: {TAMAÑO_IMAGEN}")
print(f"Tamaño de lote: {TAMAÑO_LOTE}")
print(f"Épocas Etapa 1: {EPOCAS_ETAPA1}")
print(f"Épocas Etapa 2: {EPOCAS_ETAPA2}")

# Crear carpeta de modelos si no existe
os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)

# Verificar que existe la carpeta de datos
if not os.path.exists(CARPETA_DATOS):
    print(f"\nERROR: No se encuentra la carpeta de datos: {CARPETA_DATOS}")
    print("Verifica la ruta y vuelve a intentar.")
    input("Presiona ENTER para salir...")
    exit(1)

print("\n¿Todo correcto? El entrenamiento puede tomar 30-60 minutos.")
input("Presiona ENTER para comenzar el entrenamiento...")

# PREPARAR DATOS
print("\n" + "=" * 80)
print("CARGANDO DATOS")
print("=" * 80)

# Configuración para aumento de datos y validación
config_generador = dict(rescale=1./255, validation_split=0.20)

# Generador de datos de validación
generador_validacion = tf.keras.preprocessing.image.ImageDataGenerator(**config_generador)
datos_validacion = generador_validacion.flow_from_directory(
    CARPETA_DATOS, 
    subset="validation", 
    shuffle=True,
    batch_size=TAMAÑO_LOTE,
    target_size=TAMAÑO_IMAGEN
)

# Generador de datos de entrenamiento
generador_entrenamiento = tf.keras.preprocessing.image.ImageDataGenerator(**config_generador)
datos_entrenamiento = generador_entrenamiento.flow_from_directory(
    CARPETA_DATOS, 
    subset="training", 
    shuffle=True,
    batch_size=TAMAÑO_LOTE,
    target_size=TAMAÑO_IMAGEN
)

print(f"\n Clases encontradas: {len(datos_entrenamiento.class_indices)}")
print(f"   Imágenes de entrenamiento: {datos_entrenamiento.samples}")
print(f"   Imágenes de validación: {datos_validacion.samples}")
print(f"   Lotes por época: {datos_entrenamiento.samples // TAMAÑO_LOTE}")

# Guardar las clases en un archivo
print("\nGuardando etiquetas...")
etiquetas = '\n'.join(sorted(datos_entrenamiento.class_indices.keys()))
ruta_etiquetas = os.path.join(os.path.dirname(RUTA_MODELO), 'etiquetas.txt')
with open(ruta_etiquetas, 'w', encoding='utf-8') as f:
    f.write(etiquetas)
print(f"   Etiquetas guardadas en: {ruta_etiquetas}")

# Mostrar las clases
print("\n Clases (en orden):")
for nombre_clase, id_clase in sorted(datos_entrenamiento.class_indices.items(), key=lambda x: x[1]):
    print(f"   {id_clase}: {nombre_clase}")

# CONSTRUIR MODELO
print("\n" + "=" * 80)
print("CONSTRUYENDO MODELO")
print("=" * 80)

# Cargar modelo base pre-entrenado
print("Cargando MobileNetV2 pre-entrenado...")
modelo_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Añadir capas personalizadas
x = modelo_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predicciones = Dense(15, activation='softmax')(x)  # 15 clases

# Crear modelo completo
modelo = Model(inputs=modelo_base.input, outputs=predicciones)

# Congelar capas del modelo base
for capa in modelo_base.layers:
    capa.trainable = False

print(f"\n   Total de capas: {len(modelo.layers)}")
print(f"   Parámetros entrenables: {sum([tf.keras.backend.count_params(w) for w in modelo.trainable_weights]):,}")

# ============== ETAPA 1: ENTRENAR CAPAS SUPERIORES ==============
print("\n" + "=" * 80)
print("ETAPA 1: ENTRENANDO CAPAS SUPERIORES")
print("=" * 80)
print(f"Épocas: {EPOCAS_ETAPA1}")
print("Esto puede tomar 15-20 minutos...")
print("=" * 80)

modelo.compile(
    optimizer='rmsprop', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

pasos_por_epoca = np.ceil(datos_entrenamiento.samples / datos_entrenamiento.batch_size)
pasos_validacion = np.ceil(datos_validacion.samples / datos_validacion.batch_size)

tiempo_inicio = datetime.now()

historial1 = modelo.fit(
    datos_entrenamiento, 
    epochs=EPOCAS_ETAPA1, 
    verbose=1,
    steps_per_epoch=pasos_por_epoca,
    validation_data=datos_validacion,
    validation_steps=pasos_validacion
)

tiempo_transcurrido = datetime.now() - tiempo_inicio
print(f"\n Etapa 1 completada en: {tiempo_transcurrido}")
print(f"   Precisión final: {historial1.history['accuracy'][-1]:.4f}")
print(f"   Precisión validación: {historial1.history['val_accuracy'][-1]:.4f}")

# ============== ETAPA 2: AJUSTE FINO ==============
print("\n" + "=" * 80)
print("ETAPA 2: AJUSTE FINO")
print("=" * 80)
print("Descongelando últimas capas del modelo base...")

# Descongelar últimas capas del modelo base
for capa in modelo.layers[:125]:
    capa.trainable = False
for capa in modelo.layers[125:]:
    capa.trainable = True

print(f"   Parámetros entrenables: {sum([tf.keras.backend.count_params(w) for w in modelo.trainable_weights]):,}")

# Recompilar con tasa de aprendizaje más baja
modelo.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.9), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print(f"\nÉpocas: {EPOCAS_ETAPA2}")
print("Esto puede tomar 15-20 minutos...")
print("=" * 80)

tiempo_inicio = datetime.now()

historial2 = modelo.fit(
    datos_entrenamiento, 
    epochs=EPOCAS_ETAPA2, 
    verbose=1,
    steps_per_epoch=pasos_por_epoca,
    validation_data=datos_validacion,
    validation_steps=pasos_validacion
)

tiempo_transcurrido = datetime.now() - tiempo_inicio
print(f"\n Etapa 2 completada en: {tiempo_transcurrido}")
print(f"   Precisión final: {historial2.history['accuracy'][-1]:.4f}")
print(f"   Precisión validación: {historial2.history['val_accuracy'][-1]:.4f}")

# ============== VISUALIZAR RESULTADOS ==============
print("\n" + "=" * 80)
print("GENERANDO GRÁFICAS")
print("=" * 80)

# Combinar historiales
historial_combinado = {
    'loss': historial1.history['loss'] + historial2.history['loss'],
    'val_loss': historial1.history['val_loss'] + historial2.history['val_loss'],
    'accuracy': historial1.history['accuracy'] + historial2.history['accuracy'],
    'val_accuracy': historial1.history['val_accuracy'] + historial2.history['val_accuracy']
}

# Configurar matplotlib para español
plt.rcParams['font.family'] = 'sans-serif'

# Crear figura con 2 subplots
plt.figure(figsize=(14, 5))

# Gráfica de pérdida
plt.subplot(1, 2, 1)
plt.plot(historial_combinado['loss'], label='Pérdida Entrenamiento', linewidth=2)
plt.plot(historial_combinado['val_loss'], label='Pérdida Validación', linewidth=2)
plt.axvline(x=EPOCAS_ETAPA1-1, color='red', linestyle='--', label='Inicio ajuste fino')
plt.title('Pérdida Durante el Entrenamiento', fontsize=14, fontweight='bold')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Pérdida', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Gráfica de precisión
plt.subplot(1, 2, 2)
plt.plot(historial_combinado['accuracy'], label='Precisión Entrenamiento', linewidth=2)
plt.plot(historial_combinado['val_accuracy'], label='Precisión Validación', linewidth=2)
plt.axvline(x=EPOCAS_ETAPA1-1, color='red', linestyle='--', label='Inicio ajuste fino')
plt.title('Precisión Durante el Entrenamiento', fontsize=14, fontweight='bold')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
ruta_grafica = os.path.join(os.path.dirname(RUTA_MODELO), 'historial_entrenamiento.png')
plt.savefig(ruta_grafica, dpi=150, bbox_inches='tight')
print(f"Gráficas guardadas en: {ruta_grafica}")
plt.close()

# GUARDAR MODELO
print("\n" + "=" * 80)
print("GUARDANDO MODELO")
print("=" * 80)

# Guardar en formato nativo de TensorFlow
modelo.save(RUTA_MODELO)
print(f"Modelo guardado en: {RUTA_MODELO}")

# Guardar también en formato H5
ruta_h5 = f'{RUTA_MODELO}.h5'
modelo.save(ruta_h5)
print(f"Modelo H5 guardado en: {ruta_h5}")

# EVALUAR MODELO
print("\n" + "=" * 80)
print("EVALUANDO MODELO EN CONJUNTO DE VALIDACIÓN")
print("=" * 80)

perdida_val, precision_val = modelo.evaluate(datos_validacion)
print(f"\n   Pérdida en validación: {perdida_val:.4f}")
print(f"   Precisión en validación: {precision_val:.4f}")

# RESUMEN FINAL
print("\n" + "=" * 80)
print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\n📁 Archivos generados:")
print(f"   1. Modelo: {RUTA_MODELO}")
print(f"   2. Modelo H5: {ruta_h5}")
print(f"   3. Etiquetas: {ruta_etiquetas}")
print(f"   4. Gráficas: {ruta_grafica}")

print(f"\n📊 Precisión final en validación: {precision_val*100:.2f}%")

input("\nPresiona ENTER para cerrar...")
