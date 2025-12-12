import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# --- CONFIGURACIÓN ---
RUTA_MODELO = 'modelo/modelo_banana.h5'
IMAGEN_ANCHO = 150
IMAGEN_ALTO = 150
#   ORDEN ALFABÉTICO
CLASES = ['inmaduro', 'maduro', 'podrido', 'sobremaduro']

def predecir_imagen(ruta_imagen):
    # 1. Cargar el modelo entrenado
    print("Cargando modelo...")
    try:
        model = tf.keras.models.load_model(RUTA_MODELO)
    except OSError:
        print("Error: No se encuentra el archivo del modelo. ¿Ejecutaste entrenar.py primero?")
        return

    # 2. Cargar y procesar la imagen
    try:
        img = image.load_img(ruta_imagen, target_size=(IMAGEN_ANCHO, IMAGEN_ALTO))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Crear un lote de una sola imagen
        img_array /= 255.0 # Normalizar igual que en el entrenamiento
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return

    # 3. Realizar la predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Obtener el índice de la clase con mayor probabilidad
    indice_ganador = np.argmax(predictions[0])
    clase_ganadora = CLASES[indice_ganador]
    confianza = 100 * np.max(predictions[0])

    print("\n" + "="*30)
    print(f"RESULTADO DEL ANÁLISIS")
    print("="*30)
    print(f"Estado detectado: {clase_ganadora.upper()}")
    print(f"Confianza de la IA: {confianza:.2f}%")
    print("="*30)

    # Mostrar probabilidades de todas las clases 
    print("\nDetalle de probabilidades:")
    for i, clase in enumerate(CLASES):
        print(f"{clase}: {predictions[0][i]*100:.2f}%")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        ruta_foto = sys.argv[1]
        predecir_imagen(ruta_foto)
    else:
        
        ruta_fija = input("Introduce la ruta de la imagen a analizar (ej: prueba.jpg): ")
        predecir_imagen(ruta_fija)