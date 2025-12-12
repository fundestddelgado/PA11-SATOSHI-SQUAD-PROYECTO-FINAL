import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
RUTA_DATASET = './dataset'
IMAGEN_ANCHO = 150
IMAGEN_ALTO = 150
BATCH_SIZE = 32
EPOCHS = 15  # Vueltas de entrenamiento
CLASES = ['inmaduro', 'maduro', 'podrido', 'sobremaduro']

def entrenar():
    # 1. PREPARACIÓN DE DATOS
    # Usamos rescale=1./255 para normalizar colores.
    # Validation_split=0.2 separa el 20% de las imágenes para examen final.
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 
    )

    # Generador de datos para ENTRENAR (80%)
    print("Cargando imágenes de entrenamiento...")
    train_generator = datagen.flow_from_directory(
        RUTA_DATASET,
        target_size=(IMAGEN_ANCHO, IMAGEN_ALTO),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Generador de datos para VALIDAR (20%)
    print("Cargando imágenes de validación...")
    validation_generator = datagen.flow_from_directory(
        RUTA_DATASET,
        target_size=(IMAGEN_ANCHO, IMAGEN_ALTO),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. CREACIÓN DE LA RED NEURONAL (CNN)
    model = Sequential([
        # Capa 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGEN_ANCHO, IMAGEN_ALTO, 3)),
        MaxPooling2D(2, 2),
        
        # Capa 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Capa 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Aplanado y Capas Densas
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5), # Apagar neuronas para evitar sobreajuste
        Dense(4, activation='softmax') # 4 Neuronas de salida (una por cada estado)
    ])

    # 3. COMPILAR EL MODELO
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 4. ENTRENAR
    print("Iniciando entrenamiento... esto puede tardar unos minutos.")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 5. GUARDAR EL MODELO
    if not os.path.exists('./modelo'):
        os.makedirs('./modelo')
    model.save('modelo/modelo_banana.h5')
    print("Modelo guardado exitosamente en 'modelo/modelo_banana.h5'")

    # 6. GRAFICAR RESULTADOS (Para tu reporte)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Precisión de Validación')
    plt.legend(loc='lower right')
    plt.title('Precisión')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
    plt.legend(loc='upper right')
    plt.title('Pérdida')
    plt.show()

if __name__ == "__main__":
    entrenar()