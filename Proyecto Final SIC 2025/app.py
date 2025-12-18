import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import os

# --- CONFIGURACIÓN ---
RUTA_MODELO = 'modelo/modelo_banana.h5'
IMAGEN_ANCHO = 150
IMAGEN_ALTO = 150
CLASES = ['inmaduro', 'maduro', 'podrido', 'sobremaduro']

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Plátanos",
    page_icon="🍌",
    layout="wide"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E7D32;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .team-names {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stImage"] img {
        max-width: 300px;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo = tf.keras.models.load_model(RUTA_MODELO)
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info("💡 Asegúrate de haber ejecutado 'entrenar.py' primero.")
        return None

def procesar_imagen(imagen_cargada):
    """Procesa la imagen para la predicción"""
    try:
        # Convertir a RGB si es necesario (para PNGs con transparencia)
        if imagen_cargada.mode != 'RGB':
            imagen_cargada = imagen_cargada.convert('RGB')
        
        # Redimensionar
        img_resized = imagen_cargada.resize((IMAGEN_ANCHO, IMAGEN_ALTO))
        
        # Convertir a array y normalizar
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

def predecir(modelo, img_array):
    """Realiza la predicción"""
    try:
        predictions = modelo.predict(img_array, verbose=0)
        return predictions[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

def crear_grafica_barras(confianza):
    """Crea gráfica de barras para Accuracy y Loss"""
    perdida = 100 - confianza
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Accuracy (Confianza)', 'Loss (Pérdida)'],
            y=[confianza, perdida],
            text=[f'{confianza:.2f}%', f'{perdida:.2f}%'],
            textposition='auto',
            marker_color=['#4CAF50', '#F44336']
        )
    ])
    
    fig.update_layout(
        title='Precisión del Modelo',
        yaxis_title='Porcentaje (%)',
        yaxis_range=[0, 100],
        template='plotly_white',
        height=400
    )
    
    return fig

def crear_grafica_pastel(clase, probabilidad):
    """Crea gráfica de pastel individual para cada estado"""
    resto = 100 - probabilidad
    
    colors_dict = {
        'inmaduro': '#8BC34A',
        'maduro': '#FFC107',
        'podrido': '#795548',
        'sobremaduro': '#FF5722'
    }
    
    # Para valores muy pequeños, asegurar que sea visible (mínimo 0.5% visual)
    valor_visual = max(probabilidad, 0.5) if probabilidad < 1 else probabilidad
    resto_visual = 100 - valor_visual
    
    fig = go.Figure(data=[go.Pie(
        labels=[clase.capitalize(), 'Otros'],
        values=[valor_visual, resto_visual],
        hole=0.4,
        marker_colors=[colors_dict.get(clase, '#2196F3'), '#E0E0E0'],
        textinfo='label+percent',
        textfont_size=14,
        # Mostrar el porcentaje real en el hover
        customdata=[probabilidad, resto],
        hovertemplate='%{label}: %{customdata:.4f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=f'Estado: {clase.upper()}',
        height=300,
        showlegend=False
    )
    
    return fig

# ===================== INTERFAZ PRINCIPAL =====================

# Título y nombres del equipo
st.markdown('<h1 class="main-title">🍌 Clasificador de la Madurez del Plátano</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="team-names">Li Chao • Diego Corrales • Hassan Rada • Nayim Rodríguez • Valentín Rodríguez</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# Inicializar session state
if 'imagen_cargada' not in st.session_state:
    st.session_state.imagen_cargada = None
if 'prediccion_realizada' not in st.session_state:
    st.session_state.prediccion_realizada = False
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'nombre_archivo_actual' not in st.session_state:
    st.session_state.nombre_archivo_actual = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sección de carga de imagen
st.subheader("📤 Cargar Imagen de Plátano")

uploaded_file = st.file_uploader(
    "Arrastra o selecciona una imagen",
    type=['jpg', 'jpeg', 'png'],
    help="Formatos soportados: JPG, JPEG, PNG",
    key=f"uploader_{st.session_state.uploader_key}"
)

# Si se eliminó el archivo, limpiar todo
if uploaded_file is None and st.session_state.imagen_cargada is not None:
    st.session_state.imagen_cargada = None
    st.session_state.prediccion_realizada = False
    st.session_state.resultados = None
    st.session_state.nombre_archivo_actual = None
    st.rerun()

if uploaded_file is not None:
    # Detectar si cambió la imagen (diferente nombre de archivo)
    if st.session_state.nombre_archivo_actual != uploaded_file.name:
        # Limpiar resultados anteriores cuando se carga una imagen diferente
        st.session_state.prediccion_realizada = False
        st.session_state.resultados = None
        st.session_state.nombre_archivo_actual = uploaded_file.name
    
    # Cargar la imagen
    imagen = Image.open(uploaded_file)
    st.session_state.imagen_cargada = imagen
    
    # Mostrar la imagen con tamaño estandarizado y centrada
    st.markdown("### 🖼️ Imagen Cargada")
    
    # Crear columnas para centrar la imagen (proporciones ajustadas)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image(imagen, caption='Imagen seleccionada', use_container_width=True)
    
    # Botones de acción
    st.markdown("### 🎯 Acciones")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔮 Predecir", use_container_width=True):
            with st.spinner('🤖 Analizando imagen...'):
                # Cargar modelo
                modelo = cargar_modelo()
                
                if modelo is not None:
                    # Procesar imagen
                    img_array = procesar_imagen(imagen)
                    
                    if img_array is not None:
                        # Realizar predicción
                        predictions = predecir(modelo, img_array)
                        
                        if predictions is not None:
                            # Obtener resultados
                            indice_ganador = np.argmax(predictions)
                            clase_ganadora = CLASES[indice_ganador]
                            confianza = predictions[indice_ganador] * 100
                            
                            # Guardar resultados en session state
                            st.session_state.resultados = {
                                'clase': clase_ganadora,
                                'confianza': confianza,
                                'probabilidades': predictions * 100
                            }
                            st.session_state.prediccion_realizada = True
                            st.success("✅ ¡Predicción completada!")
                            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Escoger Otra", use_container_width=True):
            st.session_state.imagen_cargada = None
            st.session_state.prediccion_realizada = False
            st.session_state.resultados = None
            st.session_state.nombre_archivo_actual = None
            st.session_state.uploader_key += 1  # Incrementar para resetear el uploader
            st.rerun()

# Mostrar resultados solo si hay predicción Y hay imagen cargada
if st.session_state.prediccion_realizada and st.session_state.resultados is not None and st.session_state.imagen_cargada is not None:
    st.markdown("---")
    st.markdown("## 📊 Resultados del Análisis")
    
    resultados = st.session_state.resultados
    
    # Mostrar resultado principal
    st.markdown(f"""
    <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='color: #2E7D32; margin: 0;'>Estado Detectado: {resultados['clase'].upper()}</h2>
        <h3 style='color: #666; margin-top: 10px;'>Confianza: {resultados['confianza']:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfica de barras (Accuracy vs Loss)
    st.markdown("### 📈 Precisión del Modelo")
    fig_barras = crear_grafica_barras(resultados['confianza'])
    st.plotly_chart(fig_barras, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficas de pastel (4 estados)
    st.markdown("### 🥧 Detalle de Probabilidades por Estado")
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    columnas = [col1, col2, col3, col4]
    
    for i, (clase, probabilidad) in enumerate(zip(CLASES, resultados['probabilidades'])):
        with columnas[i]:
            fig_pastel = crear_grafica_pastel(clase, probabilidad)
            st.plotly_chart(fig_pastel, use_container_width=True)
            # Mostrar con notación científica si es muy pequeño
            if probabilidad < 0.01:
                st.markdown(f"<p style='text-align: center; font-weight: bold;'>{probabilidad:.2e}%</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align: center; font-weight: bold;'>{probabilidad:.2f}%</p>", unsafe_allow_html=True)

# Mensaje inicial solo si no hay imagen cargada
if uploaded_file is None and st.session_state.imagen_cargada is None:
    st.info("👆 Por favor, carga una imagen de un plátano para comenzar el análisis.")
