import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# CONFIGURACIÓN
RUTA_MODELO = "models/modelo_frutas.h5"
RUTA_ETIQUETAS = "models/etiquetas.txt"
TAMANO_IMAGEN = (224, 224)

# PÁGINA
st.set_page_config(
    page_title="Clasificador de Madurez de Frutas",
    page_icon="🍎",
    layout="wide"
)

# ESTILOS
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #2E7D32;
    font-size: 3rem;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
/* Botones extendidos con mayor padding y tamaño de texto */
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 1.3rem;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# UTILIDADES
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(RUTA_MODELO)

@st.cache_data
def cargar_clases():
    with open(RUTA_ETIQUETAS, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def procesar_imagen(imagen):
    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")
    imagen = imagen.resize(TAMANO_IMAGEN)
    img_array = np.array(imagen) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def separar_clase(nombre):
    fruta, estado = nombre.split("_")
    return fruta.capitalize(), estado.capitalize()

# Mapeo de frutas a emojis
def obtener_emoji_fruta(fruta):
    """Retorna el emoji correspondiente a cada fruta"""
    emojis = {
        "manzana": "🍎",
        "banana": "🍌",
        "mango": "🥭",
        "naranja": "🍊",
        "papaya": "🍑"
    }
    return emojis.get(fruta.lower(), "🍏")  # Default: manzana verde

# Mapeo de colores por fruta
def obtener_colores_fruta(fruta):
    """Retorna colores específicos para cada fruta manteniendo contraste"""
    colores = {
        "manzana": {
            "fondo": "#FFEBEE",      # Rojo muy claro
            "texto": "#B71C1C",       # Rojo oscuro
            "borde": "#E57373"        # Rojo medio
        },
        "banana": {
            "fondo": "#FFF9C4",       # Amarillo muy claro
            "texto": "#F57F17",       # Amarillo oscuro/dorado
            "borde": "#FDD835"        # Amarillo medio
        },
        "mango": {
            "fondo": "#FFF3E0",       # Naranja muy claro
            "texto": "#E65100",       # Naranja oscuro
            "borde": "#FB8C00"        # Naranja medio
        },
        "naranja": {
            "fondo": "#FBE9E7",       # Naranja claro (diferente al mango)
            "texto": "#D84315",       # Naranja rojizo oscuro
            "borde": "#FF7043"        # Naranja rojizo medio
        },
        "papaya": {
            "fondo": "#FFE0B2",       # Melocotón/durazno claro
            "texto": "#E65100",       # Naranja oscuro intenso
            "borde": "#FF9800"        # Naranja melocotón medio
        }
    }
    # Default: verde
    return colores.get(fruta.lower(), {
        "fondo": "#E8F5E9",
        "texto": "#1B5E20",
        "borde": "#4CAF50"
    })

# Función para gráficas de barras verticales (Top 5) con colores por fruta
def grafica_top5_barras(clases, probabilidades):
    """
    Genera una gráfica de barras verticales con el Top 5.
    Cada barra tiene el color correspondiente a su fruta.
    """
    top_idx = np.argsort(probabilidades)[-5:][::-1]
    
    labels = []
    values = []
    colors = []
    
    for i in top_idx:
        fruta, estado = separar_clase(clases[i])
        prob = probabilidades[i] * 100
        
        # Obtener color de la fruta
        colores_fruta = obtener_colores_fruta(fruta)
        
        labels.append(f"{fruta} - {estado}")
        values.append(prob)
        colors.append(colores_fruta['borde'])
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        text=[f"{v:.2f}%" for v in values],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Probabilidad: %{y:.2f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="Top 5 Predicciones",
            x=0.5,
            xanchor='center',
            font=dict(size=16, weight='bold')
        ),
        yaxis_title="Probabilidad (%)",
        yaxis=dict(range=[0, 105]),
        xaxis=dict(
            tickfont=dict(size=14),
            tickmode='linear'
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=60, b=80, l=60, r=40)
    )
    
    return fig

# TÍTULO
st.markdown('<h1 class="main-title">🍎 Clasificador de Madurez de Frutas</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Modelo IA con Transfer Learning (MobileNetV2)</p>',
    unsafe_allow_html=True
)

# SESSION STATE
if "imagen" not in st.session_state:
    st.session_state.imagen = None
if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# CARGA
st.subheader("📤 Cargar Imagen")

archivo = st.file_uploader(
    "Selecciona una imagen de fruta",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

# Limpiar resultados cuando se elimina la imagen del uploader
if archivo is None and st.session_state.imagen is not None:
    st.session_state.imagen = None
    st.session_state.resultado = None
    st.rerun()

if archivo:
    imagen = Image.open(archivo)
    st.session_state.imagen = imagen
    st.session_state.resultado = None

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image(imagen, caption="Imagen seleccionada", use_container_width=True)

    # Sección de Acciones con diseño exacto solicitado
    st.subheader("🎯 Acciones")
    
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔮 Predecir", use_container_width=True):
            with st.spinner("🤖 Analizando imagen..."):
                modelo = cargar_modelo()
                clases = cargar_clases()
                img_array = procesar_imagen(imagen)

                probs = modelo.predict(img_array, verbose=0)[0]
                idx = np.argmax(probs)

                fruta, estado = separar_clase(clases[idx])

                st.session_state.resultado = {
                    "fruta": fruta,
                    "estado": estado,
                    "confianza": probs[idx] * 100,
                    "probs": probs,
                    "clases": clases
                }
                st.success("✅ Predicción realizada")

    with col_btn2:
        if st.button("🔄 Escoger otra", use_container_width=True):
            st.session_state.imagen = None
            st.session_state.resultado = None
            st.session_state.uploader_key += 1
            st.rerun()

# RESULTADOS
if st.session_state.resultado:
    r = st.session_state.resultado

    st.markdown("---")
    st.markdown("## 📊 Resultado")

    # Bloque de resultados con colores dinámicos según la fruta
    emoji_fruta = obtener_emoji_fruta(r['fruta'])
    colores = obtener_colores_fruta(r['fruta'])
    
    st.markdown(f"""
    <div style="background-color:{colores['fondo']};padding:25px;border-radius:10px;text-align:center;border:2px solid {colores['borde']}">
        <h2 style="color:{colores['texto']};margin:10px 0;font-weight:700">{emoji_fruta} Fruta: {r['fruta']}</h2>
        <h3 style="color:{colores['texto']};margin:10px 0;font-weight:600">📌 Estado: {r['estado']}</h3>
        <h3 style="color:{colores['texto']};margin:10px 0;font-weight:600">📊 Confianza: {r['confianza']:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Gráfica de barras en lugar de pasteles individuales
    st.markdown(" ")
    st.markdown("### 📈 Distribución de Probabilidades (Top 5)")
    
    fig = grafica_top5_barras(r["clases"], r["probs"])
    st.plotly_chart(fig, use_container_width=True)

# MENSAJE INICIAL
if not archivo and st.session_state.imagen is None:
    st.info("👆 Carga una imagen de una fruta para comenzar.")
