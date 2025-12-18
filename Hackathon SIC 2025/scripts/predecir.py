import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import sys
import os

# CONFIGURACIÓN
RUTA_MODELO = "models\modelo_frutas"
TAMAÑO_IMAGEN = (224, 224)

# Lista de clases (15 clases en orden alfabético en español)
CLASES = [
    'banana_madura', 'banana_podrida', 'banana_verde',
    'mango_maduro', 'mango_podrido', 'mango_verde',
    'manzana_madura', 'manzana_podrida', 'manzana_verde',
    'naranja_madura', 'naranja_podrida', 'naranja_verde',
    'papaya_madura', 'papaya_podrida', 'papaya_verde'
]

# CARGAR MODELO
print("=" * 70)
print("CARGANDO MODELO")
print("=" * 70)
print(f"Ruta del modelo: {RUTA_MODELO}")

try:
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    print("Modelo cargado exitosamente")
    print(f"   Entrada: {modelo.input_shape}")
    print(f"   Salida: {modelo.output_shape}")
except Exception as e:
    print(f"\nERROR al cargar modelo: {e}")
    print("\nVerifica que:")
    print("  1. Ejecutaste train_model_es.py primero")
    print("  2. El modelo se guardó correctamente")
    print(f"  3. La ruta {RUTA_MODELO} existe")
    input("\nPresiona ENTER para salir...")
    sys.exit(1)

# FUNCIÓN DE PREDICCIÓN
def predecir_fruta(ruta_imagen):
    """
    Predice el estado de madurez de una fruta
    
    Args:
        ruta_imagen: Ruta a la imagen
        
    Returns:
        tuple: (clase_predicha, confianza, todas_las_probabilidades)
    """
    try:
        # Cargar y preprocesar imagen
        img = image.load_img(ruta_imagen, target_size=TAMAÑO_IMAGEN)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote
        
        # Realizar predicción
        predicciones = modelo.predict(img_array, verbose=0)
        indice_predicho = np.argmax(predicciones[0])
        confianza = predicciones[0][indice_predicho]
        clase_predicha = CLASES[indice_predicho]
        
        return clase_predicha, confianza, predicciones[0]
    
    except Exception as e:
        print(f"\nERROR al procesar imagen: {e}")
        return None, None, None

def formatear_nombre_clase(nombre_clase):
    """
    Formatea el nombre de la clase para mostrar
    Ejemplo: 'manzana_madura' -> 'Manzana Madura'
    """
    partes = nombre_clase.split('_')
    return ' '.join([parte.capitalize() for parte in partes])

# MAIN
if __name__ == "__main__":
    print("\n" + "=" * 70)
    
    if len(sys.argv) < 2:
        print("USO DEL SCRIPT")
        print("=" * 70)
        print("python scripts/predecir.py <ruta_a_imagen>")
        print("\nEjemplo:")
        print("  python scripts/predecir.py imagenes_prueba/banana.jpg")
        print("=" * 70)
        input("\nPresiona ENTER para salir...")
        sys.exit(1)
    
    ruta_imagen = sys.argv[1]
    
    # Verificar que existe la imagen
    if not os.path.exists(ruta_imagen):
        print(f"ERROR: No se encontró la imagen en {ruta_imagen}")
        print("\nVerifica que:")
        print("  1. La ruta es correcta")
        print("  2. El archivo existe")
        print("  3. La extensión es .jpg, .jpeg o .png")
        input("\nPresiona ENTER para salir...")
        sys.exit(1)
    
    print(f"ANALIZANDO IMAGEN")
    print("=" * 70)
    print(f"Imagen: {ruta_imagen}")
    print(f"Tamaño esperado: {TAMAÑO_IMAGEN}")
    
    # Realizar predicción
    print("\n🔄 Procesando...")
    clase_predicha, confianza, todas_probs = predecir_fruta(ruta_imagen)
    
    if clase_predicha is None:
        print("\nNo se pudo realizar la predicción")
        input("\nPresiona ENTER para salir...")
        sys.exit(1)
    
    # Mostrar resultado
    print("\n" + "=" * 70)
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 70)
    print(f"\n🎯 Predicción: {formatear_nombre_clase(clase_predicha)}")
    print(f"📊 Confianza: {confianza*100:.2f}%")
    
    # Mostrar top 5 predicciones
    print(f"\n🏆 Top 5 predicciones:")
    print("-" * 70)
    top_5_indices = np.argsort(todas_probs)[-5:][::-1]
    for i, indice in enumerate(top_5_indices, 1):
        nombre_clase = formatear_nombre_clase(CLASES[indice])
        probabilidad = todas_probs[indice] * 100
        barra = "█" * int(probabilidad / 2)  # Barra visual
        print(f"{i}. {nombre_clase:25} {probabilidad:6.2f}% {barra}")
    
    print("=" * 70)
    
    # Interpretación del resultado
    print("\n💡 INTERPRETACIÓN:")
    if confianza > 0.9:
        print("  ✅ Confianza MUY ALTA - Predicción muy confiable")
        interpretacion = "El modelo está muy seguro de esta predicción"
    elif confianza > 0.7:
        print("  ✅ Confianza ALTA - Predicción confiable")
        interpretacion = "El modelo tiene buena confianza en esta predicción"
    elif confianza > 0.5:
        print("  ⚠️ Confianza MEDIA - Predicción aceptable")
        interpretacion = "El modelo tiene dudas, considera verificar manualmente"
    else:
        print("  ❌ Confianza BAJA - El modelo no está seguro")
        interpretacion = "El modelo no está seguro, la imagen puede no ser clara"
    
    print(f"  {interpretacion}")
    
    # Consejo basado en la predicción
    print("\n💬 CONSEJO:")
    if 'verde' in clase_predicha:
        print("  La fruta aún no está madura. Espera unos días.")
    elif 'madura' in clase_predicha or 'maduro' in clase_predicha:
        print("  La fruta está en su punto óptimo. ¡Disfrútala!")
    elif 'podrida' in clase_predicha or 'podrido' in clase_predicha:
        print("  La fruta está en mal estado. No se recomienda consumir.")
    
    print("\n" + "=" * 70)
    input("\nPresiona ENTER para cerrar...")