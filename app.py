import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Predicción de Churn")
st.title('Predicción de Tasa de Cancelación de Suscripciones (Churn)')
st.write('Introduce la información del cliente para predecir la probabilidad de churn.')

# Rutas a los modelos (estas rutas deben ser accesibles desde el entorno de Streamlit)
# Si la aplicación se ejecuta en Colab, '/content/drive/MyDrive' es la base del Drive montado.
churn_model_path = 'modelo_churn.pkl'
preprocessing_pipeline_path = 'pipeline_preproc.pkl'

# Cargar el modelo y el pipeline una sola vez para eficiencia
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(churn_model_path)
        pipeline = joblib.load(preprocessing_pipeline_path)
        return model, pipeline
    except Exception as e:
        st.error(f"Error al cargar los recursos: {e}. Asegúrate de que los archivos .pkl existan en las rutas especificadas.")
        return None, None

model, pipeline = load_resources()

if model is not None and pipeline is not None:
    st.subheader('Características del Cliente')

    # =========================================================================
    #              ***** ¡¡IMPORTANTE: AJUSTA ESTAS CARACTERÍSTICAS!! *****
    # Estos son solo EJEMPLOS. Debes modificarlos para que coincidan EXACTAMENTE
    # con las características que tu modelo y pipeline de preprocesamiento esperan.
    # Nombres, tipos, rangos y opciones de categorías deben ser correctos.
    # =========================================================================


    # Nuevas características añadidas según tu solicitud:
    uso_mensual_km = st.number_input('Uso Mensual (Km)', min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    soporte_tickets = st.number_input('Tickets de Soporte Abiertos', min_value=0, max_value=10, value=0)
    region = st.selectbox('Región', ['Norte', 'Centro', 'Sur', 'Este', 'Oeste']) # Asegúrate de que estas opciones coincidan con las de tu modelo



    # Crear un DataFrame con las entradas del usuario
    # Los NOMBRES DE LAS COLUMNAS deben coincidir con los que el pipeline espera.
    input_data = pd.DataFrame({
        'Uso_Mensual_Km': [uso_mensual_km],
        'Soporte_Tickets': [soporte_tickets],
        'Region': [region],
    })

    if st.button('Predecir Churn'):
        try:
            # Aplicar el pipeline de preprocesamiento
            # Asegúrate de que `pipeline.transform` acepte el DataFrame directamente
            # o ajusta según cómo lo entrenaste (ej. `pipeline.fit_transform` si el pipeline no fue fit)
            processed_data = pipeline.transform(input_data)

            # Realizar la predicción de probabilidad
            churn_probability = model.predict_proba(processed_data)[:, 1][0]

            # Realizar la predicción de clase (0 o 1)
            churn_prediction = model.predict(processed_data)[0]

            st.subheader('Resultado de la Predicción')
            st.write(f'Probabilidad de Churn: **{churn_probability:.2f}**')

            if churn_prediction == 1:
                st.error('¡Alto riesgo de Churn! El cliente es propenso a cancelar la suscripción.')
            else:
                st.success('Bajo riesgo de Churn. Es probable que el cliente mantenga la suscripción.')

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}. Revisa las características de entrada y la compatibilidad con tu pipeline y modelo.")
else:
    st.warning("No se pudo cargar el modelo o el pipeline. Por favor, verifica las rutas y los archivos.")
