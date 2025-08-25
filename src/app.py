import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Estrellas - Dataset Completo",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../models/star_prediction_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar datos de estrellas
@st.cache_data
def load_star_data():
    try:
        df = pd.read_csv('../data/processed/data_stars_processed.csv')
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

# Información de tipos de estrellas
star_types_info = {
    'M': {
        'nombre': 'Enana Roja Tipo M', 
        'color_hex': '#ff6b6b', 
        'descripcion': 'Estrellas rojas pequeñas y frías, las más comunes en la galaxia',
        'color_estrella': 'Rojo',
        'clase_espectral': 'M'
    },
    'K': {
        'nombre': 'Estrella Naranja Tipo K', 
        'color_hex': '#ffa726', 
        'descripcion': 'Estrellas naranjas de tamaño mediano',
        'color_estrella': 'Naranja',
        'clase_espectral': 'K'
    },
    'G': {
        'nombre': 'Estrella Amarilla Tipo G', 
        'color_hex': '#ffd54f', 
        'descripcion': 'Estrellas amarillas como nuestro Sol',
        'color_estrella': 'Amarillo',
        'clase_espectral': 'G'
    },
    'F': {
        'nombre': 'Estrella Blanco-Amarillenta Tipo F', 
        'color_hex': '#f8f9fa', 
        'descripcion': 'Estrellas blanco-amarillentas más calientes que el Sol',
        'color_estrella': 'Blanco-Amarillento',
        'clase_espectral': 'F'
    },
    'A': {
        'nombre': 'Estrella Blanca Tipo A', 
        'color_hex': '#e3f2fd', 
        'descripcion': 'Estrellas blancas como Sirio',
        'color_estrella': 'Blanco',
        'clase_espectral': 'A'
    },
    'B': {
        'nombre': 'Estrella Azul-Blanca Tipo B', 
        'color_hex': '#90caf9', 
        'descripcion': 'Estrellas azul-blancas muy calientes',
        'color_estrella': 'Azul-Blanca',
        'clase_espectral': 'B'
    },
    'O': {
        'nombre': 'Estrella Azul Tipo O', 
        'color_hex': '#64b5f6', 
        'descripcion': 'Estrellas azules extremadamente calientes y masivas',
        'color_estrella': 'Azul',
        'clase_espectral': 'O'
    }
}

# Función para hacer predicciones
def predict_star_type(row):
    model = load_model()
    if model is None:
        return None
    
    try:
        # Asegurarse de que las características estén en el orden correcto
        features = pd.DataFrame([row])
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

# Función para obtener opciones disponibles del dataset
def get_dataset_options(df):
    options = {}
    
    if not df.empty:
        # Para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            options[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'unique_values': sorted(df[col].unique())
            }
        
        # Para columnas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            options[col] = {
                'unique_values': sorted(df[col].dropna().unique().tolist())
            }
    
    return options

# Función para mostrar información del dataset
def show_dataset_info(df):
    st.subheader("📋 Información General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Estrellas", df.shape[0])
    with col2:
        st.metric("Número de Características", df.shape[1])
    with col3:
        st.metric("Tipos de Estrellas", df['Spectral Class'].nunique())
    with col4:
        st.metric("Valores Faltantes", df.isnull().sum().sum())
    
    # Mostrar tipos de datos
    st.subheader("📊 Tipos de Datos por Columna")
    dtype_info = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de Dato': df.dtypes.astype(str),
        'Valores Únicos': [df[col].nunique() for col in df.columns],
        'Valores Faltantes': df.isnull().sum().values
    })
    st.dataframe(dtype_info, use_container_width=True)

# Función principal
def main():
    st.title("⭐ Sistema de Predicción de Tipos de Estrellas")
    
    # Cargar datos
    star_data = load_star_data()
    
    if star_data.empty:
        st.warning("No se pudieron cargar los datos. Verifica la ruta del archivo CSV.")
        return
    
    # Obtener opciones del dataset
    dataset_options = get_dataset_options(star_data)
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Selecciona una página:", 
                           ["Predicción", "Explorar Dataset", "Estadísticas", "Visualizaciones"])
    
    if page == "Predicción":
        st.header("🔮 Predicción de Tipo Estelar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Selecciona una estrella del dataset:")
            
            # Selector de estrella
            selected_index = st.selectbox(
                "Selecciona una estrella para predecir su tipo:",
                options=star_data.index,
                format_func=lambda x: f"Estrella {x} - {star_data.loc[x, 'Star color'] if 'Star color' in star_data.columns else 'Color desconocido'}"
            )
            
            if selected_index is not None:
                # Mostrar características de la estrella seleccionada
                selected_star = star_data.loc[selected_index]
                
                st.subheader("Características de la Estrella Seleccionada:")
                
                # Mostrar todas las características en columnas
                cols_per_row = 3
                features_list = list(selected_star.items())
                
                for i in range(0, len(features_list), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (feature, value) in enumerate(features_list[i:i+cols_per_row]):
                        with cols[j]:
                            st.metric(feature, value)
                
                # Botón de predicción
                if st.button("📊 Realizar Predicción", type="primary"):
                    # Preparar características para la predicción
                    features = selected_star.to_dict()
                    
                    # Hacer predicción
                    prediction = predict_star_type(features)
                    
                    if prediction is not None:
                        predicted_class = list(star_types_info.keys())[prediction]
                        star_info = star_types_info.get(predicted_class, {})
                        
                        st.success(f"🌟 **Predicción:** {star_info.get('nombre', 'Desconocido')}")
                        
                        # Mostrar información detallada
                        st.info(f"""
                        **Detalles del tipo predicho:**
                        - Clase Espectral: {predicted_class}
                        - Color: {star_info.get('color_estrella', 'Desconocido')}
                        - Descripción: {star_info.get('descripcion', 'Desconocida')}
                        """)
        
        with col2:
            st.subheader("📈 Información del Dataset")
            st.metric("Total de estrellas", len(star_data))
            st.metric("Características", len(star_data.columns))
            
            # Mostrar distribución de clases espectrales
            if 'Spectral Class' in star_data.columns:
                spectral_dist = star_data['Spectral Class'].value_counts()
                fig = px.pie(
                    values=spectral_dist.values,
                    names=spectral_dist.index,
                    title="Distribución de Clases Espectrales"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Explorar Dataset":
        st.header("🔍 Explorar Dataset Completo")
        
        # Mostrar dataset completo con posibilidad de filtrado
        st.subheader("Dataset de Estrellas")
        st.dataframe(star_data, use_container_width=True)
        
        # Opciones de filtrado
        st.subheader("Filtrar Datos")
        filter_col = st.selectbox("Seleccionar columna para filtrar:", star_data.columns)
        
        if filter_col in star_data.columns:
            if star_data[filter_col].dtype == 'object':
                unique_values = star_data[filter_col].dropna().unique()
                selected_values = st.multiselect(
                    f"Seleccionar valores de {filter_col}:",
                    options=unique_values,
                    default=unique_values[:min(3, len(unique_values))]
                )
                filtered_data = star_data[star_data[filter_col].isin(selected_values)]
            else:
                min_val = float(star_data[filter_col].min())
                max_val = float(star_data[filter_col].max())
                selected_range = st.slider(
                    f"Rango de {filter_col}:",
                    min_val, max_val, (min_val, max_val)
                )
                filtered_data = star_data[
                    (star_data[filter_col] >= selected_range[0]) & 
                    (star_data[filter_col] <= selected_range[1])
                ]
            
            st.metric("Estrellas filtradas", len(filtered_data))
            st.dataframe(filtered_data, use_container_width=True)
    
    elif page == "Estadísticas":
        st.header("📊 Estadísticas Descriptivas")
        
        # Estadísticas numéricas
        numeric_cols = star_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Variables Numéricas")
            st.dataframe(star_data[numeric_cols].describe(), use_container_width=True)
        
        # Estadísticas categóricas
        categorical_cols = star_data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("Variables Categóricas")
            for col in categorical_cols:
                st.write(f"**{col}:**")
                value_counts = star_data[col].value_counts()
                st.dataframe(pd.DataFrame({
                    'Valor': value_counts.index,
                    'Frecuencia': value_counts.values,
                    'Porcentaje': (value_counts.values / len(star_data) * 100).round(2)
                }), use_container_width=True)
    
    elif page == "Visualizaciones":
        st.header("📈 Visualizaciones de Datos")
        
        tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Scatter Plots"])
        
        with tab1:
            # Distribuciones
            numeric_cols = star_data.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Selecciona una variable para ver su distribución:", numeric_cols)
            
            if selected_col:
                fig = px.histogram(star_data, x=selected_col, title=f"Distribución de {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Matriz de correlación
            numeric_cols = star_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = star_data[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Matriz de Correlación",
                               color_continuous_scale='RdBu_r',
                               aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Se necesitan al menos 2 variables numéricas para la matriz de correlación")
        
        with tab3:
            # Scatter plots
            numeric_cols = star_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Eje X:", numeric_cols)
                with col2:
                    y_axis = st.selectbox("Eje Y:", numeric_cols)
                
                if x_axis and y_axis:
                    fig = px.scatter(star_data, x=x_axis, y=y_axis, 
                                    color='Spectral Class' if 'Spectral Class' in star_data.columns else None,
                                    title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.info("""
    Esta aplicación utiliza machine learning para predecir el tipo espectral de estrellas 
    basándose en sus características físicas. Todos los datos y opciones provienen del dataset original.
    """)

if __name__ == "__main__":
    main()
