import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

st.title("📝 Clasificador de Tópicos Flexible (Zero-Shot)")

texto = st.text_area("🔹 Ingresa un texto a analizar:", 
                     "El presidente habló sobre la importancia de las nuevas reformas económicas en el país.")

etiquetas = st.text_input("🔹 Ingresa categorías separadas por comas:", 
                          "política, deportes, tecnología, economía, entretenimiento")

if st.button("Clasificar"):
    if texto and etiquetas:
        labels = [et.strip() for et in etiquetas.split(",")]
        resultado = classifier(texto, candidate_labels=labels)
        st.subheader("📊 Resultados de clasificación:")
        st.bar_chart(dict(zip(resultado["labels"], resultado["scores"])))
    else:
        st.warning("⚠️ Ingresa un texto y al menos una etiqueta.")
