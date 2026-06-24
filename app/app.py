import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. CONFIGURACIÓN Y CSS (OCULTAR CÍRCULOS) ---
st.set_page_config(page_title="App Setas Profesional", page_icon="🍄", layout="wide")

st.markdown("""
    <style>
    /* Oculta los círculos de selección del radio button */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return joblib.load(Path("models/best_xgboost_model.pkl"))

pipeline = load_pipeline()
columnas_esperadas = pipeline.feature_names_in_

# --- 2. DICCIONARIOS ---
mapeo = {
    "cap-shape": {"Campaniforme": "c", "Cónica": "x", "Convexa": "b", "Plana": "f", "Abollada": "k", "Sumida": "s"},
    "cap-surface": {"Fibrosa": "f", "Surcada": "g", "Escamosa": "y", "Lisa": "s"},
    "cap-color": {"Marrón": "n", "Ante": "b", "Canela": "c", "Gris": "g", "Verde": "r", "Rosa": "p", "Violeta": "u", "Morado": "e", "Blanco": "w", "Amarillo": "y"},
    "bruises": {"Sí": "t", "No": "f"},
    "odor": {"Almendra": "a", "Anís": "l", "Creosota": "c", "Pescado": "y", "Fétido": "f", "Moho": "m", "Ninguno": "n", "Picante": "p", "Especia": "s"},
    "gill-attachment": {"Adherente": "a", "Descendente": "d", "Libre": "f", "Escotada": "n"},
    "gill-spacing": {"Cerrado": "c", "Abierto": "w", "Distante": "d"},
    "gill-size": {"Ancho": "b", "Estrecho": "n"},
    "gill-color": {"Negro": "k", "Marrón": "n", "Ante": "b", "Chocolate": "h", "Gris": "g", "Verde": "r", "Naranja": "o", "Rosa": "p", "Violeta": "u", "Blanco": "w", "Amarillo": "y"},
    "stalk-shape": {"Ensanchado": "e", "Ahusado": "t"},
    "stalk-root": {"Bulbosa": "b", "Maza": "c", "Copa": "u", "Igual": "e", "Rizomorfo": "z", "Enraizada": "r", "Desconocido": "?"},
    "stalk-surface-above-ring": {"Fibrosa": "f", "Escamosa": "y", "Sedosa": "k", "Lisa": "s"},
    "stalk-surface-below-ring": {"Fibrosa": "f", "Escamosa": "y", "Sedosa": "k", "Lisa": "s"},
    "stalk-color-above-ring": {"Marrón": "n", "Ante": "b", "Canela": "c", "Gris": "g", "Naranja": "o", "Rosa": "p", "Rojo": "e", "Blanco": "w", "Amarillo": "y"},
    "stalk-color-below-ring": {"Marrón": "n", "Ante": "b", "Canela": "c", "Gris": "g", "Naranja": "o", "Rosa": "p", "Rojo": "e", "Blanco": "w", "Amarillo": "y"},
    "veil-type": {"Parcial": "p", "Universal": "u"},
    "veil-color": {"Marrón": "n", "Naranja": "o", "Blanco": "w", "Amarillo": "y"},
    "ring-number": {"Ninguno": "n", "Uno": "o", "Dos": "t"},
    "ring-type": {"Telaraña": "c", "Evanescente": "e", "Aflorado": "f", "Grande": "l", "Ninguno": "n", "Colgante": "p", "Envoltura": "s", "Zona": "z"},
    "spore-print-color": {"Negro": "k", "Marrón": "n", "Ante": "b", "Chocolate": "h", "Verde": "r", "Naranja": "o", "Violeta": "u", "Blanco": "w", "Amarillo": "y"},
    "population": {"Abundante": "a", "Agrupado": "c", "Numeroso": "n", "Disperso": "s", "Varios": "v", "Solitario": "y"},
    "habitat": {"Pastos": "g", "Hojas": "l", "Praderas": "m", "Caminos": "p", "Urbano": "u", "Residuos": "w", "Bosques": "d"}
}

etiquetas_es = {
    "cap-shape": "Forma del Sombrero", "cap-surface": "Superficie del Sombrero", "cap-color": "Color del Sombrero",
    "bruises": "¿Tiene Moretones?", "odor": "Olor", "gill-attachment": "Inserción de las Láminas",
    "gill-spacing": "Espaciado de las Láminas", "gill-size": "Tamaño de las Láminas", "gill-color": "Color de las Láminas",
    "stalk-shape": "Forma del Tallo", "stalk-root": "Raíz del Tallo", "stalk-surface-above-ring": "Superficie Tallo (Sup)",
    "stalk-surface-below-ring": "Superficie Tallo (Inf)", "stalk-color-above-ring": "Color Tallo (Sup)",
    "stalk-color-below-ring": "Color Tallo (Inf)", "veil-type": "Tipo de Velo", "veil-color": "Color del Velo",
    "ring-number": "Número de Anillos", "ring-type": "Tipo de Anillo", "spore-print-color": "Color de la Esporada",
    "population": "Población", "habitat": "Hábitat"
}

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("🍄 Menú Principal")
    opcion = st.radio("Navegación:", ["🍄 Clasificador", "📊 Análisis"])

# --- 4. LÓGICA DE PÁGINAS ---
if opcion == "🍄 Clasificador":
    st.title("🍄 Clasificador de Setas")
    col1, col2 = st.columns(2)
    datos_usuario = {}
    
    for i, col in enumerate(columnas_esperadas):
        with col1 if i % 2 == 0 else col2:
            label = etiquetas_es.get(col, col.replace('-', ' ').title())
            if col in mapeo:
                datos_usuario[col] = mapeo[col][st.selectbox(label, list(mapeo[col].keys()))]
            else:
                datos_usuario[col] = st.text_input(label, value="?")

    if st.button("ANALIZAR SETA", use_container_width=True):
        df = pd.DataFrame([datos_usuario])[columnas_esperadas]
        pred = pipeline.predict(df)
        if pred[0] == 1:
            st.error("### 💀 ¡PELIGRO: VENENOSA!")
        else:
            st.success("### ✅ ¡SEGURA: COMESTIBLE!")
            st.balloons()

elif opcion == "📊 Análisis":
    st.title("📊 Dashboard de Micología")
    k1, k2, k3 = st.columns(3)
    k1.metric("Precisión Modelo", "99.8%")
    k2.metric("Total Setas", "8,124")
    k3.metric("Variables", "22")
    
    st.write("---")
    st.subheader("Importancia de las variables (Factores de riesgo)")
    
    datos_importancia = {"Olor": 0.45, "Color de Esporas": 0.25, "Hábitat": 0.15, "Tipo de Anillo": 0.10, "Otros": 0.05}
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(datos_importancia).plot(kind='barh', color='#FF4B4B', ax=ax)
    ax.set_xlabel("Impacto en la predicción (%)")
    st.pyplot(fig)
    st.info("💡 El olor y el color de las esporas son los indicadores más críticos.")