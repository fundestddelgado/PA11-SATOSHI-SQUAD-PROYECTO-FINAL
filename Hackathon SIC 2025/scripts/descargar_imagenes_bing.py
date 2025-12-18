from bing_image_downloader import downloader
import os
import time

# CONFIGURACIÓN 
CARPETA_SALIDA = "data"
LIMITE = 100  # Descargar 100 imágenes por clase
FILTRO_ADULTO = 'on'

# Lista de búsquedas para cada clase
CONSULTAS_BUSQUEDA = [
    # Manzanas
    "manzana verde sin madurar",
    "manzana roja madura",
    "manzana podrida estropeada",
    
    # Bananas
    "banana verde sin madurar",
    "banana amarilla madura",
    "banana negra marrón podrida",
    
    # Mangos
    "mango verde duro sin madurar",
    "mango amarillo naranja maduro",
    "mango podrido estropeado",
    
    # Naranjas
    "naranja verde sin madurar",
    "naranja madura",
    "naranja podrida mohosa",
    
    # Papayas
    "papaya verde sin madurar",
    "papaya naranja madura",
    "papaya podrida estropeada"
]

# Nombres de carpetas correspondientes
NOMBRES_CARPETAS = [
    "manzana_verde",
    "manzana_madura", 
    "manzana_podrida",
    "banana_verde",
    "banana_madura",
    "banana_podrida",
    "mango_verde",
    "mango_maduro",
    "mango_podrido",
    "naranja_verde",
    "naranja_madura",
    "naranja_podrida",
    "papaya_verde",
    "papaya_madura",
    "papaya_podrida"
]

# FUNCIÓN PRINCIPAL
def descargar_todas_imagenes():
    """
    Descarga todas las imágenes para cada clase
    """
    print("🍎🍌🍊 Iniciando descarga de imágenes con BING\n")
    print(f"Total de clases: {len(CONSULTAS_BUSQUEDA)}")
    print(f"Imágenes por clase: {LIMITE}")
    print(f"Total aproximado: {len(CONSULTAS_BUSQUEDA) * LIMITE} imágenes\n")
    print("=" * 70)
    
    for i, (consulta, carpeta) in enumerate(zip(CONSULTAS_BUSQUEDA, NOMBRES_CARPETAS), 1):
        print(f"\n[{i}/{len(CONSULTAS_BUSQUEDA)}] Descargando: {consulta}")
        print(f"    📁 Carpeta destino: {carpeta}")
        
        try:
            downloader.download(
                consulta,
                limit=LIMITE,
                output_dir=CARPETA_SALIDA,
                adult_filter_off=False,
                force_replace=False,
                timeout=15,
                verbose=True
            )
            
            # Renombrar la carpeta descargada
            carpeta_original = os.path.join(CARPETA_SALIDA, consulta)
            carpeta_nueva = os.path.join(CARPETA_SALIDA, carpeta)
            
            if os.path.exists(carpeta_original):
                if os.path.exists(carpeta_nueva):
                    print(f"La carpeta {carpeta} ya existe, saltando...")
                else:
                    os.rename(carpeta_original, carpeta_nueva)
                    print(f"Carpeta renombrada a: {carpeta}")
            
            # Pausa entre descargas
            print(f"    ⏳ Esperando 3 segundos antes de la siguiente descarga...")
            time.sleep(3)
            
        except Exception as e:
            print(f"Error descargando {consulta}: {e}")
            print(f"Continuando con la siguiente...")
            continue
    
    print("\n" + "=" * 70)
    print("✅ ¡Descarga completada!")
    print(f"\nRevisa la carpeta: {os.path.abspath(CARPETA_SALIDA)}")
    print("\n SIGUIENTE PASO:")
    print("   1. Revisa manualmente cada carpeta")
    print("   2. Elimina imágenes irrelevantes")
    print("   3. Asegúrate de tener ~50-110 imágenes por carpeta")
    print("   4. Ejecuta: python scripts/validate_dataset.py")

# EJECUTAR
if __name__ == "__main__":
    # Crear directorio de salida si no existe
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    
    print("Script de Descarga de Imágenes - Bing Image Downloader")
    print("=" * 70)
    print("Tiempo estimado: 30-45 minutos\n")
    
    input("Presiona ENTER para comenzar...")
    
    # Iniciar descarga
    descargar_todas_imagenes()
    
    print("\n" + "=" * 70)
    print("Script finalizado")
    input("\nPresiona ENTER para cerrar...")
