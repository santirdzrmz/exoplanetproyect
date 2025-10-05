import streamlit as st
import time
import pandas as pd
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter

# === FUNCIONES PARA EL MODELO KERAS ===
positive = ["CP"]
negative = ["FP"]  
TARGET_LEN = 200               
PHASE_WINDOW = 0.1          

def curves(name, label, target_len=TARGET_LEN, phase_window=PHASE_WINDOW):
    search = lk.search_lightcurve(f"TIC {name}", mission="TESS", author="SPOC")
    if len(search) == 0:
        return [], []

    lc = search.download(download_dir="lightcurves_cache")
    lc = lc.remove_nans().normalize()

    trend = savgol_filter(lc.flux.value, 301, 2)
    flux_flat = lc.flux.value / trend
    t, f = lc.time.value, flux_flat

    bls = BoxLeastSquares(t, f)
    periods = np.linspace(0.5, 20, 10000)
    result = bls.power(periods, 0.05)
    best = np.argmax(result.power)
    period, t0 = result.period[best], result.transit_time[best]

    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    mask = (phase > -phase_window) & (phase < phase_window)
    phase, f = phase[mask], f[mask]

    order = np.argsort(phase)
    f = f[order]
    f = (f - np.median(f)) / np.std(f)

    mid = np.argmin(f)
    shift = len(f)//2 - mid
    f = np.roll(f, shift)

    bins = target_len
    binned = np.array_split(f, bins)
    f = np.array([np.mean(b) for b in binned])

    if len(f) > target_len:
        f = f[:target_len]
    else:
        f = np.pad(f, (0, target_len - len(f)), 'constant', constant_values=0.0)
    
    if label in positive:
       label_bin=1
    elif label in negative:
       label_bin=0
    return f.astype(np.float32), label_bin

def mc_dropout_proba(model, X, T=100, batch_size=None):
    probs = []
    for _ in range(T):
        p = model(X, training=True).numpy().ravel()  
        probs.append(p[0])
    probs = np.array(probs)
    return probs

def ci_mean_proba(probs, alpha=0.05):
    pbar = probs.mean()
    se = probs.std(ddof=1) / np.sqrt(len(probs))
    z = norm.ppf(1 - alpha/2)
    lo, hi = pbar - z*se, pbar + z*se
    return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)), float(pbar), float(se)

def predict_with_uncertainty(model, X_input, T=200, threshold=0.5):
    probs = mc_dropout_proba(model, X_input, T=T)
    lo_p, hi_p, pbar, se = ci_mean_proba(probs, alpha=0.05)
    return pbar, [lo_p, hi_p]

# --- Función para convertir imagen local a base64 ---
def get_base64_of_image(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        else:
            return ""
    except Exception as e:
        return ""

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Exopia - IA Astronómica",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTADO DE LA APLICACIÓN ---
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'transition_state' not in st.session_state:
    st.session_state.transition_state = "ready"
if 'selected_method' not in st.session_state:
    st.session_state.selected_method = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# --- FUNCIÓN MEJORADA PARA TRANSICIONES ---
def trigger_transition(new_page):
    st.session_state.transition_state = "transitioning"
    st.session_state.next_page = new_page
    st.rerun()

# --- MANEJO DE TRANSICIONES ---
if st.session_state.transition_state == "transitioning":
    # Mostrar animación de transición
    st.markdown("""
    <style>
    .transition-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .planet-loader {
        width: 80px;
        height: 80px;
        background: linear-gradient(45deg, #4fc3f7, #9575cd);
        border-radius: 50%;
        position: relative;
        animation: rotate 2s linear infinite;
    }
    
    .planet-loader::before {
        content: '';
        position: absolute;
        top: -15px;
        left: 50%;
        transform: translateX(-50%);
        width: 25px;
        height: 25px;
        background: #4fc3f7;
        border-radius: 50%;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    
    @keyframes pulse {
        0% { transform: translateX(-50%) scale(1); }
        100% { transform: translateX(-50%) scale(1.3); }
    }
    
    .loading-text {
        color: white;
        margin-top: 20px;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        animation: fadeInOut 1.5s ease-in-out infinite;
    }
    
    @keyframes fadeInOut {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    </style>
    
    <div class="transition-overlay">
        <div style="text-align: center;">
            <div class="planet-loader"></div>
            <div class="loading-text">Cargando universo...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Pequeña pausa para mostrar la animación
    time.sleep(1.5)
    
    # Cambiar a la nueva página
    st.session_state.page = st.session_state.next_page
    st.session_state.transition_state = "ready"
    st.rerun()

# --- PANTALLA DE INICIO CON PLANETAS ---
if st.session_state.page == "home":
    # Carga de imágenes locales
    fondo_base64 = get_base64_of_image("fondo.png")
    logo_base64 = get_base64_of_image("logo.png")

    background_style = f'background-image: url("data:image/png;base64,{fondo_base64}");' if fondo_base64 else 'background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 100%);'

    # CSS para pantalla de inicio con animaciones
    st.markdown(f"""
    <style>
    .stApp {{
        {background_style}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        animation: pageEnter 0.8s ease-out;
    }}
    
    @keyframes pageEnter {{
        from {{ 
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .title-container {{
        text-align: center;
        color: white;
        margin-top: 15%;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        animation: fadeInUp 1s ease-out 0.3s both;
    }}
    
    .title-container h1 {{
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }}
    
    .title-container p {{
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 25px;
    }}
    
    @keyframes fadeInUp {{
        from {{ 
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .logo {{
        display: block;
        margin: 0 auto;
        width: 200px;
        margin-top: 40px;
        animation: floatIn 1.5s ease-out 0.5s both, float 3s ease-in-out infinite 2s;
    }}
    
    @keyframes floatIn {{
        from {{ 
            opacity: 0;
            transform: translateY(-20px) scale(0.9);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0) scale(1);
        }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-10px) rotate(2deg); }}
    }}
    
    .pulse-button {{
        animation: gentlePulse 2s ease-in-out infinite;
        transition: all 0.3s ease;
    }}
    
    @keyframes gentlePulse {{
        0%, 100% {{ 
            transform: scale(1);
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
        }}
        50% {{ 
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(79, 195, 247, 0.5);
        }}
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

    # Logo centrado
    if logo_base64:
        st.markdown(f"""
        <img src="data:image/png;base64,{logo_base64}" class="logo" alt="Logo ExopiaIA">
        """, unsafe_allow_html=True)

    # Texto principal y botón
    st.markdown("""
    <div class="title-container">
        <p>Inteligencia Artificial para análisis astronómico</p>
        <h1>ExopiaIA: Descubre Exoplanetas</h1>
    </div>
    """, unsafe_allow_html=True)

    # Botón para comenzar con animación
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🚀 COMIENZA TU EXPLORACIÓN', use_container_width=True, key="start_exploration", type="primary"):
            trigger_transition("method_selection")

# --- MENÚ DE SELECCIÓN DE MÉTODO ---
elif st.session_state.page == "method_selection":
    # Cargar fondo para el menú de selección
    menu_fondo_base64 = get_base64_of_image("menu_fondo.png")
    background_style = f'background-image: url("data:image/png;base64,{menu_fondo_base64}");' if menu_fondo_base64 else 'background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 100%);'

    # CSS para el menú de selección
    st.markdown(f"""
    <style>
    .stApp {{
        {background_style}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        animation: cosmicEntrance 1s ease-out;
    }}
    
    @keyframes cosmicEntrance {{
        from {{ 
            opacity: 0;
            transform: scale(1.1);
        }}
        to {{ 
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .method-selection-container {{
        text-align: center;
        color: white;
        margin-top: 5%;
        animation: slideInFromCosmos 1.2s ease-out 0.3s both;
    }}
    
    @keyframes slideInFromCosmos {{
        from {{ 
            opacity: 0;
            transform: translateY(50px) rotateX(45deg);
        }}
        to {{ 
            opacity: 1;
            transform: translateY(0) rotateX(0);
        }}
    }}
    
    .method-title {{
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #4fc3f7, #9575cd, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(79, 195, 247, 0.5);
    }}
    
    .method-subtitle {{
        font-size: 1.4rem;
        margin-bottom: 4rem;
        color: #b3e5fc;
        font-weight: 300;
    }}
    
    .method-cards-container {{
        display: flex;
        justify-content: center;
        gap: 4rem;
        margin: 4rem 0;
        flex-wrap: wrap;
    }}
    
    .method-card {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(79, 195, 247, 0.4);
        border-radius: 25px;
        padding: 4rem 3rem;
        width: 450px;
        min-height: 500px;
        cursor: pointer;
        transition: all 0.4s ease;
        animation: cardFloat 4s ease-in-out infinite;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    
    .method-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(79, 195, 247, 0.3), transparent);
        transition: left 0.6s ease;
    }}
    
    .method-card:hover::before {{
        left: 100%;
    }}
    
    .method-card:hover {{
        transform: translateY(-20px) scale(1.05);
        border-color: #4fc3f7;
        box-shadow: 0 25px 50px rgba(79, 195, 247, 0.4);
        background: rgba(255, 255, 255, 0.15);
    }}
    
    @keyframes cardFloat {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        33% {{ transform: translateY(-15px) rotate(1deg); }}
        66% {{ transform: translateY(-8px) rotate(-1deg); }}
    }}
    
    .method-icon {{
        font-size: 6rem;
        margin-bottom: 2rem;
        animation: iconPulse 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(79, 195, 247, 0.5));
    }}
    
    @keyframes iconPulse {{
        0%, 100% {{ 
            transform: scale(1) rotate(0deg);
            filter: drop-shadow(0 0 20px rgba(79, 195, 247, 0.5));
        }}
        50% {{ 
            transform: scale(1.1) rotate(5deg);
            filter: drop-shadow(0 0 30px rgba(79, 195, 247, 0.8));
        }}
    }}
    
    .method-card h3 {{
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
        color: #e3f2fd;
        font-weight: 700;
        text-align: center;
    }}
    
    .method-card p {{
        font-size: 1.2rem;
        color: #b3e5fc;
        line-height: 1.7;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 300;
    }}
    
    .method-badge {{
        background: linear-gradient(45deg, #4fc3f7, #9575cd);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
        animation: badgeGlow 2s ease-in-out infinite alternate;
        margin-top: 1rem;
    }}
    
    @keyframes badgeGlow {{
        from {{ 
            box-shadow: 0 0 15px rgba(79, 195, 247, 0.6);
            transform: scale(1);
        }}
        to {{ 
            box-shadow: 0 0 25px rgba(79, 195, 247, 0.9);
            transform: scale(1.05);
        }}
    }}
    
    .stButton > button {{
        background: linear-gradient(45deg, #4fc3f7, #9575cd);
        color: white !important;
        border: none;
        padding: 20px 40px;
        font-size: 1.3rem;
        font-weight: 600;
        border-radius: 15px;
        transition: all 0.3s ease;
        margin: 10px;
        box-shadow: 0 8px 25px rgba(79, 195, 247, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(79, 195, 247, 0.5);
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

    # Contenido del menú de selección
    st.markdown("""
    <div class="method-cards-container">
    <!-- Tarjeta para Subir CSV -->
    <div class="method-card">
        <div class="method-icon">🌌</div>
        <h3>ANÁLISIS CON ARCHIVOS CSV</h3>
        <p>Sube datasets completos de la NASA en formato CSV para un análisis profesional y detallado de grandes volúmenes de datos astronómicos.</p>
        <div class="method-badge">POTENTE Y PRECISO</div>
    </div>

    <!-- Tarjeta para Datos Manuales -->
    <div class="method-card">
        <div class="method-icon">🔭</div>
        <h3>INGRESO MANUAL DE DATOS</h3>
        <p>Introduce parámetros específicos manualmente para análisis personalizados y experimentación con diferentes escenarios astronómicos.</p>
        <div class="method-badge">FLEXIBLE Y EDUCATIVO</div>
    </div>
    </div>
    """, unsafe_allow_html=True)


    # Botones funcionales CORREGIDOS
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            if st.button("🌌 USAR ARCHIVO CSV", use_container_width=True, key="csv_method"):
                st.session_state.selected_method = "csv"
                trigger_transition("csv_analysis")  # CORREGIDO: "analyze" → "csv_analysis"
        
        with col_right:
            if st.button("🔭 DATOS MANUALES", use_container_width=True, key="manual_method"):
                st.session_state.selected_method = "manual" 
                trigger_transition("manual_analysis")  # CORREGIDO: "analyze" → "manual_analysis"
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Botón para regresar al inicio
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← VOLVER AL INICIO", use_container_width=True, key="back_to_home"):
            trigger_transition("home")

# --- INTERFAZ PARA ANÁLISIS CON CSV ---
elif st.session_state.page == "csv_analysis":
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0c0c2e 0%, #1a1a4a 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #4fc3f7, #9575cd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(79, 195, 247, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .upload-zone {
        border: 3px dashed #4fc3f7;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(79, 195, 247, 0.05);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .model-option {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .model-option:hover {
        border-color: #4fc3f7;
        transform: translateY(-3px);
    }
    
    /* CAMBIAR COLOR DEL TEXTO DEL FILE UPLOADER */
    .stFileUploader > div > div > div {
        color: #e3f2fd !important;
    }
    
    .stFileUploader > div > div > div::before {
        color: #b3e5fc !important;
    }
    
    /* CAMBIAR COLOR DEL TEXTO DE RADIO BUTTONS */
    .stRadio > div {
        color: #e3f2fd !important;
    }
    
    .stRadio > div > label {
        color: #e3f2fd !important;
        font-size: 1.1rem !important;
    }
    
    .stRadio > div > label > div:first-child {
        background-color: rgba(79, 195, 247, 0.3) !important;
    }
    
    /* COLOR PARA LAS DESCRIPCIONES */
    .stRadio > div > label > div:nth-child(2) {
        color: #b3e5fc !important;
        font-size: 1rem !important;
    }
    
    /* COLOR DEL TEXTO DEL UPLOADER */
    .uploadedFileName {
        color: #4fc3f7 !important;
        font-weight: bold !important;
    }
    
    /* COLOR DEL BORDE Y FONDO DEL UPLOADER */
    .stFileUploader > div {
        border: 2px dashed #4fc3f7 !important;
        background: rgba(79, 195, 247, 0.05) !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* COLOR DEL TEXTO "Drag and drop file here" */
    .stFileUploader > div > div > div::before {
        content: "Arrastra tu archivo aquí o haz clic para buscar" !important;
        color: #b3e5fc !important;
        font-size: 1.1rem !important;
    }
    
    /* COLOR DEL TEXTO "Limit 200MB per file • CSV" */
    .stFileUploader > div > div > small {
        color: #90caf9 !important;
        font-size: 0.9rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # HEADER PRINCIPAL
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🌌 ANÁLISIS CON ARCHIVOS CSV/NPY</h1>
        <p style="font-size: 1.2rem; color: #b3e5fc;">Sube tus datasets de la NASA y selecciona el modelo de IA para análisis avanzado</p>
    </div>
    """, unsafe_allow_html=True)

    # SECCIÓN 1: SUBIR ARCHIVOS (CSV O NPY)
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('### 📁 SUBIR DATOS ASTRONÓMICOS')
        
        st.markdown('<p style="color: #b3e5fc; font-size: 1.1rem; margin-bottom: 1rem;">Selecciona tu archivo .npy (curva de luz) o .csv (datos tabulares) de la NASA</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            " ",
            type=['npy', 'csv'],
            key="file_upload",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            try:
                if file_extension == 'npy':
                    # Cargar archivo .npy
                    datos = np.load(uploaded_file)
                    
                    st.success(f"✅ **Archivo .npy cargado:** {uploaded_file.name}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Puntos de datos", f"{len(datos):,}")
                    with col2:
                        st.metric("📈 Tipo de datos", str(datos.dtype))
                    with col3:
                        st.metric("💾 Tamaño", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    with st.expander("👁️ **Vista previa de datos**", expanded=False):
                        st.write("Primeros 10 valores de la curva de luz:")
                        st.write(datos[:10])
                        
                else:  # file_extension == 'csv'
                    # Cargar archivo .csv
                    datos = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ **Archivo .csv cargado:** {uploaded_file.name}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Filas", f"{len(datos):,}")
                    with col2:
                        st.metric("📈 Columnas", len(datos.columns))
                    with col3:
                        st.metric("💾 Tamaño", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    with st.expander("👁️ **Vista previa de datos**", expanded=False):
                        st.dataframe(datos.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # SECCIÓN 2: SELECCIÓN DE MODELO DE IA
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('### 🤖 SELECCIONA EL MODELO DE IA')
        
        st.markdown('<p style="color: #b3e5fc; font-size: 1.1rem; margin-bottom: 1rem;">Elige el algoritmo de IA que mejor se adapte a tus datos</p>', unsafe_allow_html=True)
        
        modelos_csv = [
            "🧠 Red Neuronal Avanzada - Análisis de patrones complejos",
            "🌠 Gradient Boosting - Clasificación de exoplanetas", 
            "📊 Random Forest - Análisis multivariable"
        ]
        
        modelo_seleccionado = st.radio(
            " ",
            options=modelos_csv,
            key="modelo_csv",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # SECCIÓN 3: EJECUCIÓN
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('### 🚀 EJECUTAR ANÁLISIS')
        
        if st.session_state.uploaded_file is None:
            st.warning("📝 Primero sube un archivo para comenzar el análisis")
        else:
            if st.button("🎯 EJECUTAR ANÁLISIS CON GRADIENT BOOSTING", use_container_width=True, type="primary"):
                with st.spinner('🔭 Analizando datos con Gradient Boosting...'):
                    progress_bar = st.progress(0)
                    
                    try:
                        # === CÓDIGO GRADIENT BOOSTING ===
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        from tsfresh import extract_features
                        from tsfresh.feature_extraction import ComprehensiveFCParameters
                        from tsfresh.utilities.dataframe_functions import impute
                        from joblib import load
                        import warnings
                        warnings.filterwarnings("ignore")
                        
                        progress_bar.progress(20)
                        
                        # 1. CARGAR Y PREPARAR DATOS (CSV O NPY)
                        st.info("📁 Cargando y procesando datos...")
                        
                        file_extension = st.session_state.uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension == 'npy':
                            # Cargar .npy directamente
                            flux = np.load(st.session_state.uploaded_file)
                            
                        else:  # CSV
                            # Cargar CSV y extraer la curva de luz
                            datos_csv = pd.read_csv(st.session_state.uploaded_file)
                            
                            # Buscar columnas numéricas para la curva de luz
                            numeric_cols = datos_csv.select_dtypes(include=[np.number]).columns
                            
                            if len(numeric_cols) == 0:
                                st.error("❌ No se encontraron columnas numéricas en el CSV")
                                st.stop()
                            
                            # Usar la primera columna numérica como curva de luz
                            flux_column = numeric_cols[0]
                            flux = datos_csv[flux_column].values
                            
                            st.info(f"📋 Usando columna '{flux_column}' para el análisis")
                        
                        # 2. PREPROCESAR LA CURVA
                        flux = np.nan_to_num(flux, nan=np.nanmedian(flux))
                        flux = (flux - np.mean(flux)) / np.std(flux)
                        
                        LENGTH = 2000
                        # Ajustar longitud
                        if len(flux) > LENGTH:
                            flux = flux[:LENGTH]
                        else:
                            flux = np.pad(flux, (0, LENGTH - len(flux)), "constant", constant_values=np.nan)
                        
                        progress_bar.progress(60)
                        
                        # 3. CREAR DATAFRAME PARA tsfresh
                        df = pd.DataFrame({
                            "id": [0]*len(flux),
                            "time": np.arange(len(flux)),
                            "flux": flux
                        })
                        
                        # 4. EXTRAER FEATURES
                        st.info("🔍 Extrayendo características de la curva de luz...")
                        features = extract_features(
                            df,
                            column_id="id",
                            column_sort="time",
                            column_value="flux",
                            disable_progressbar=True,
                            default_fc_parameters=ComprehensiveFCParameters()
                        )
                        impute(features)
                        
                        progress_bar.progress(80)
                        
                        # 5. CARGAR MODELO Y SCALER
                        st.info("⚙️ Cargando modelo Gradient Boosting...")
                        
                        try:
                            # Cargar el scaler y modelo
                            scaler = load("tess_scaler.joblib")
                            model = load("tess_tsfresh_gb.joblib")
                        except FileNotFoundError:
                            st.error("❌ No se encontraron los archivos del modelo")
                            st.info("💡 Asegúrate de tener estos archivos en tu carpeta:")
                            st.info("- tess_scaler.joblib")
                            st.info("- tess_tsfresh_gb.joblib")
                            st.stop()
                        
                        # 6. ALINEAR COLUMNAS
                        expected_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else features.columns
                        
                        # Añadir columnas faltantes con 0
                        for col in expected_features:
                            if col not in features.columns:
                                features[col] = 0.0
                        
                        # Reordenar columnas
                        features = features[expected_features]
                        
                        # 7. ESCALAR Y PREDECIR
                        X_scaled = scaler.transform(features)
                        y_prob = model.predict_proba(X_scaled)[0, 1]
                        y_pred = model.predict(X_scaled)[0]
                        
                        progress_bar.progress(100)
                        
                        # 8. MOSTRAR RESULTADOS EN STREAMLIT
                        st.success("✅ Análisis completado exitosamente!")
                        
                        # === RESULTADOS VISUALES MEJORADOS ===
                        st.markdown("---")
                        
                        # Tarjetas de resultados
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if y_pred == 1:
                                st.markdown("""
                                <div style="background: linear-gradient(45deg, #4CAF50, #8BC34A); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                                    <h2>🪐</h2>
                                    <h3>EXOPLANETA DETECTADO</h3>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background: linear-gradient(45deg, #f44336, #FF9800); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                                    <h2>💨</h2>
                                    <h3>NO ES EXOPLANETA</h3>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            # Barra de probabilidad
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                                <h3>Confianza del Modelo</h3>
                                <h1 style="color: #4fc3f7; margin: 1rem 0;">{y_prob:.1%}</h1>
                                <div style="background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; margin: 1rem 0;">
                                    <div style="background: linear-gradient(45deg, #4fc3f7, #9575cd); height: 100%; width: {y_prob*100}%; border-radius: 10px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                                <h3>📊 Datos Analizados</h3>
                                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Muestras: {len(flux)}</p>
                                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Características: {len(features.columns)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # GRÁFICA DE LA CURVA DE LUZ
                        st.markdown("---")
                        st.subheader("📈 Curva de Luz Analizada")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(flux, color='#4fc3f7', linewidth=2, alpha=0.8)
                        ax.set_title(f'Curva de Luz - {st.session_state.uploaded_file.name}', 
                                   fontsize=16, color='white', pad=20)
                        ax.set_xlabel('Tiempo (puntos normalizados)', color='white', fontsize=12)
                        ax.set_ylabel('Flujo Normalizado', color='white', fontsize=12)
                        ax.grid(True, alpha=0.3, color='white')
                        ax.set_facecolor('none')
                        fig.patch.set_facecolor('none')
                        
                        # Color del gráfico para tema oscuro
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        
                        st.pyplot(fig)
                        
                        # INFORMACIÓN DETALLADA
                        with st.expander("🔍 VER DETALLES TÉCNICOS", expanded=False):
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.markdown("**📋 Información del Modelo:**")
                                st.write(f"- **Algoritmo:** Gradient Boosting")
                                st.write(f"- **Tipo:** Clasificación Binaria")
                                st.write(f"- **Características usadas:** {len(features.columns)}")
                                st.write(f"- **Preprocesamiento:** tsfresh + StandardScaler")
                            
                            with col_info2:
                                st.markdown("**📊 Métricas de la Curva:**")
                                st.write(f"- **Longitud:** {len(flux)} puntos")
                                st.write(f"- **Media:** {np.mean(flux):.3f}")
                                st.write(f"- **Desviación estándar:** {np.std(flux):.3f}")
                                st.write(f"- **Rango:** {np.min(flux):.3f} a {np.max(flux):.3f}")
                        
                        # RECOMENDACIONES BASADAS EN EL RESULTADO
                        st.markdown("---")
                        st.subheader("💡 Interpretación de Resultados")
                        
                        if y_prob > 0.8:
                            st.success("""
                            **🎯 Alta probabilidad de exoplaneta detectado!**
                            - La curva de luz muestra patrones consistentes con tránsitos planetarios
                            - Recomendamos análisis adicional con otros métodos
                            - Considerar observaciones de seguimiento
                            """)
                        elif y_prob > 0.5:
                            st.warning("""
                            **⚠️ Señal ambigua detectada**
                            - Posible candidato a exoplaneta
                            - Se recomienda análisis más profundo
                            - Verificar con datos adicionales
                            """)
                        else:
                            st.info("""
                            **🔍 No se detectaron señales fuertes de exoplaneta**
                            - La curva no muestra patrones claros de tránsito
                            - Podría ser ruido o variabilidad estelar
                            - Considerar analizar otras curvas
                            """)
                    
                    except Exception as e:
                        st.error(f"❌ Error en el análisis: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # BOTÓN PARA VOLVER
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← VOLVER AL MENÚ PRINCIPAL", use_container_width=True):
            trigger_transition("method_selection")

# --- INTERFAZ PARA ANÁLISIS MANUAL ---
elif st.session_state.page == "manual_analysis":
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2d5a 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #9575cd, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(149, 117, 205, 0.3);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

    # HEADER PRINCIPAL
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🔭 INGRESO MANUAL DE DATOS</h1>
        <p style="font-size: 1.2rem; color: #d1c4e9;">Introduce parámetros específicos para análisis personalizado</p>
    </div>
    """, unsafe_allow_html=True)

    # FORMULARIO DE DATOS
    with st.form("manual_analysis_form"):
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('### 🪐 DATOS DEL SISTEMA PLANETARIO')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌟 Datos de la Estrella")
            star_temp = st.number_input("Temperatura Estelar (K)", min_value=1000, max_value=50000, value=5778)
            star_radius = st.number_input("Radio Estelar (Soles)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        
        with col2:
            st.markdown("#### 🪐 Datos del Planeta")
            orbital_period = st.number_input("Período Orbital (días)", min_value=0.1, max_value=10000.0, value=365.25, step=0.1)
            planet_radius = st.number_input("Radio Planetario (Tierras)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # SELECCIÓN DE MODELO
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('### 🤖 MODELO DE IA')
        
        modelos_manual = [
            "🧠 Red Neuronal - Prueba de Luz",
            "📈 Gradient Boosting - Curvas de Luz", 
            "🌌 Gradient Boosting - Archive Exoplanet"
        ]
        
        modelo_seleccionado = st.radio(
            "Selecciona el modelo de IA:",
            options=modelos_manual,
            key="modelo_manual"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # BOTÓN DE EJECUCIÓN
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        analyze_clicked = st.form_submit_button(
            "🚀 EJECUTAR ANÁLISIS CON IA", 
            use_container_width=True,
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # RESULTADOS CON MODELO KERAS
    if analyze_clicked:
        with st.spinner('🔭 Analizando con Red Neuronal...'):
            progress_bar = st.progress(0)
            
            try:
                # 1. CARGAR MODELO KERAS
                st.info("🧠 Cargando modelo de Red Neuronal...")
                model = tf.keras.models.load_model("exoplanetia_model.keras")
                progress_bar.progress(30)
                
                # 2. PREPARAR DATOS PARA EL MODELO
                st.info("📊 Procesando datos de entrada...")
                
                # Usar los datos del formulario para crear entrada del modelo
                # Aquí necesitamos adaptar según lo que espera tu modelo
                # Por ahora usaremos datos de ejemplo
                X_input = np.random.randn(200).astype(np.float32)  # EJEMPLO
                X = X_input.reshape(1, 200, 1)
                
                progress_bar.progress(60)
                
                # 3. REALIZAR PREDICCIÓN CON INCERTIDUMBRE
                st.info("🎯 Realizando predicción con Monte Carlo Dropout...")
                T = 50
                thr = 0.1
                
                prob, interval = predict_with_uncertainty(model, X, T=T, threshold=thr)
                
                progress_bar.progress(100)
                
                # 4. MOSTRAR RESULTADOS
                st.success("✅ Análisis completado!")
                
                # === RESULTADOS VISUALES ===
                st.markdown("---")
                
                # Tarjetas de resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prob > 0.5:
                        st.markdown("""
                        <div style="background: linear-gradient(45deg, #4CAF50, #8BC34A); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h2>🪐</h2>
                            <h3>EXOPLANETA DETECTADO</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(45deg, #f44336, #FF9800); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h2>💨</h2>
                            <h3>NO ES EXOPLANETA</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Barra de probabilidad con intervalo de confianza
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h3>Probabilidad de Exoplaneta</h3>
                        <h1 style="color: #4fc3f7; margin: 1rem 0;">{prob:.1%}</h1>
                        <div style="background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; margin: 1rem 0; position: relative;">
                            <div style="background: linear-gradient(45deg, #4fc3f7, #9575cd); height: 100%; width: {prob*100}%; border-radius: 10px;"></div>
                        </div>
                        <p style="color: #b3e5fc; font-size: 0.9rem;">Intervalo: {interval[0]:.1%} - {interval[1]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h3>📊 Métricas del Modelo</h3>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0;">Muestras MC: {T}</p>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0;">Confianza: 95%</p>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0;">Umbral: {thr}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # INFORMACIÓN DETALLADA
                with st.expander("🔍 VER DETALLES TÉCNICOS", expanded=False):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.markdown("**📋 Información del Modelo:**")
                        st.write("- **Algoritmo:** Red Neuronal con Dropout")
                        st.write("- **Tipo:** Clasificación con Incertidumbre")
                        st.write("- **Técnica:** Monte Carlo Dropout")
                        st.write("- **Entrada:** Curvas de luz (200 puntos)")
                    
                    with col_info2:
                        st.markdown("**📊 Resultados Estadísticos:**")
                        st.write(f"- **Probabilidad media:** {prob:.3f}")
                        st.write(f"- **Intervalo inferior:** {interval[0]:.3f}")
                        st.write(f"- **Intervalo superior:** {interval[1]:.3f}")
                        st.write(f"- **Ancho del intervalo:** {interval[1]-interval[0]:.3f}")
                
                # INTERPRETACIÓN DE RESULTADOS
                st.markdown("---")
                st.subheader("💡 Interpretación de Resultados")
                
                if prob > 0.7:
                    st.success("""
                    **🎯 Alta confianza en la detección!**
                    - La red neuronal muestra alta probabilidad de exoplaneta
                    - El intervalo de confianza es consistente
                    - Recomendado para observaciones de seguimiento
                    """)
                elif prob > 0.3:
                    st.warning("""
                    **⚠️ Señal con incertidumbre moderada**
                    - Probabilidad intermedia de exoplaneta
                    - Se recomienda análisis adicional
                    - Considerar otros métodos de validación
                    """)
                else:
                    st.info("""
                    **🔍 Baja probabilidad de exoplaneta**
                    - La señal no es suficientemente convincente
                    - Podría ser ruido o falsos positivos
                    - Recomendado analizar otros candidatos
                    """)
                    
                # NOTA SOBRE LA INCERTIDUMBRE
                st.markdown("---")
                st.subheader("🎯 Sobre la Incertidumbre del Modelo")
                st.info("""
                **Monte Carlo Dropout:** Esta técnica realiza múltiples predicciones activando el dropout 
                durante la inferencia, permitiendo estimar la incertidumbre del modelo. Un intervalo más 
                amplio indica mayor incertidumbre en la predicción.
                """)
            
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Asegúrate de que el archivo 'exoplanetia_model.keras' esté en la carpeta correcta")

    # BOTÓN PARA VOLVER
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← VOLVER AL MENÚ PRINCIPAL", use_container_width=True, key="back_from_manual"):
            trigger_transition("method_selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌟 Datos de la Estrella")
            star_temp = st.number_input("Temperatura Estelar (K)", min_value=1000, max_value=50000, value=5778)
            star_radius = st.number_input("Radio Estelar (Soles)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        
        with col2:
            st.markdown("#### 🪐 Datos del Planeta")
            orbital_period = st.number_input("Período Orbital (días)", min_value=0.1, max_value=10000.0, value=365.25, step=0.1)
            planet_radius = st.number_input("Radio Planetario (Tierras)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # SELECCIÓN DE MODELO
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('### 🤖 MODELO DE IA')
        
        modelos_manual = [
            "🧠 Red Neuronal - Prueba de Luz",
            "📈 Gradient Boosting - Curvas de Luz", 
            "🌌 Gradient Boosting - Archive Exoplanet"
        ]
        
        modelo_seleccionado = st.radio(
            "Selecciona el modelo de IA:",
            options=modelos_manual,
            key="modelo_manual"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # BOTÓN DE EJECUCIÓN
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        analyze_clicked = st.form_submit_button(
            "🚀 EJECUTAR ANÁLISIS CON IA", 
            use_container_width=True,
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # RESULTADOS
    # RESULTADOS
# RESULTADOS CON MODELO KERAS
if analyze_clicked:
    with st.spinner('🔭 Analizando con Red Neuronal...'):
        progress_bar = st.progress(0)
        
        try:
            # 1. CARGAR MODELO KERAS
            st.info("🧠 Cargando modelo de Red Neuronal...")
            model = tf.keras.models.load_model("exoplanetia_model.keras")
            progress_bar.progress(30)
            
            # 2. PREPARAR DATOS PARA EL MODELO
            st.info("📊 Procesando datos de entrada...")
            
            # Usar los datos del formulario para crear entrada del modelo
            # Aquí necesitamos adaptar según lo que espera tu modelo
            # Por ahora usaremos datos de ejemplo
            X_input = np.random.randn(200).astype(np.float32)  # EJEMPLO
            X = X_input.reshape(1, 200, 1)
            
            progress_bar.progress(60)
            
            # 3. REALIZAR PREDICCIÓN CON INCERTIDUMBRE
            st.info("🎯 Realizando predicción con Monte Carlo Dropout...")
            T = 50
            thr = 0.1
            
            prob, interval = predict_with_uncertainty(model, X, T=T, threshold=thr)
            
            progress_bar.progress(100)
            
            # 4. MOSTRAR RESULTADOS
            st.success("✅ Análisis completado!")
            
            # === RESULTADOS VISUALES ===
            st.markdown("---")
            
            # Tarjetas de resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prob > 0.5:
                    st.markdown("""
                    <div style="background: linear-gradient(45deg, #4CAF50, #8BC34A); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>🪐</h2>
                        <h3>EXOPLANETA DETECTADO</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(45deg, #f44336, #FF9800); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                        <h2>💨</h2>
                        <h3>NO ES EXOPLANETA</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Barra de probabilidad con intervalo de confianza
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h3>Probabilidad de Exoplaneta</h3>
                    <h1 style="color: #4fc3f7; margin: 1rem 0;">{prob:.1%}</h1>
                    <div style="background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; margin: 1rem 0; position: relative;">
                        <div style="background: linear-gradient(45deg, #4fc3f7, #9575cd); height: 100%; width: {prob*100}%; border-radius: 10px;"></div>
                    </div>
                    <p style="color: #b3e5fc; font-size: 0.9rem;">Intervalo: {interval[0]:.1%} - {interval[1]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h3>📊 Métricas del Modelo</h3>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Muestras MC: {T}</p>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Confianza: 95%</p>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Umbral: {thr}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # INFORMACIÓN DETALLADA
            with st.expander("🔍 VER DETALLES TÉCNICOS", expanded=False):
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**📋 Información del Modelo:**")
                    st.write("- **Algoritmo:** Red Neuronal con Dropout")
                    st.write("- **Tipo:** Clasificación con Incertidumbre")
                    st.write("- **Técnica:** Monte Carlo Dropout")
                    st.write("- **Entrada:** Curvas de luz (200 puntos)")
                
                with col_info2:
                    st.markdown("**📊 Resultados Estadísticos:**")
                    st.write(f"- **Probabilidad media:** {prob:.3f}")
                    st.write(f"- **Intervalo inferior:** {interval[0]:.3f}")
                    st.write(f"- **Intervalo superior:** {interval[1]:.3f}")
                    st.write(f"- **Ancho del intervalo:** {interval[1]-interval[0]:.3f}")
            
            # INTERPRETACIÓN DE RESULTADOS
            st.markdown("---")
            st.subheader("💡 Interpretación de Resultados")
            
            if prob > 0.7:
                st.success("""
                **🎯 Alta confianza en la detección!**
                - La red neuronal muestra alta probabilidad de exoplaneta
                - El intervalo de confianza es consistente
                - Recomendado para observaciones de seguimiento
                """)
            elif prob > 0.3:
                st.warning("""
                **⚠️ Señal con incertidumbre moderada**
                - Probabilidad intermedia de exoplaneta
                - Se recomienda análisis adicional
                - Considerar otros métodos de validación
                """)
            else:
                st.info("""
                **🔍 Baja probabilidad de exoplaneta**
                - La señal no es suficientemente convincente
                - Podría ser ruido o falsos positivos
                - Recomendado analizar otros candidatos
                """)
                
            # NOTA SOBRE LA INCERTIDUMBRE
            st.markdown("---")
            st.subheader("🎯 Sobre la Incertidumbre del Modelo")
            st.info("""
            **Monte Carlo Dropout:** Esta técnica realiza múltiples predicciones activando el dropout 
            durante la inferencia, permitiendo estimar la incertidumbre del modelo. Un intervalo más 
            amplio indica mayor incertidumbre en la predicción.
            """)
        
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.info("💡 Asegúrate de que el archivo 'exoplanetia_model.keras' esté en la carpeta correcta")
    # BOTÓN PARA VOLVER
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← VOLVER AL MENÚ PRINCIPAL", use_container_width=True, key="back_from_manual"):
            trigger_transition("method_selection")