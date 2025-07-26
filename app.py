import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Configuração da página
st.set_page_config(
    page_title="🚴 Bike Sharing Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função consolidada para carregar dados e modelo
@st.cache_resource
def load_model_and_data():
    try:
        # Carregar modelo pré-treinado (usando o PKL original do notebook)
        model = joblib.load('bike_sharing_model_pipeline.pkl')
        
        # Carregar dados
        df = pd.read_csv('day.csv')
        
        # Calcular score do modelo usando dados de teste
        # Preparar features com feature engineering conforme usado no treinamento
        df_copy = df.copy()
        df_copy['dteday'] = pd.to_datetime(df_copy['dteday'])
        df_copy['month'] = df_copy['dteday'].dt.month
        df_copy['day_of_week'] = df_copy['dteday'].dt.dayofweek
        df_copy['year'] = df_copy['dteday'].dt.year
        df_copy['is_weekend'] = df_copy['weekday'].apply(lambda x: 1 if x in [0,6] else 0)
        df_copy['temp_hum_interaction'] = df_copy['temp'] * df_copy['hum']
        
        # Features completas conforme o modelo foi treinado
        features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                   'month', 'day_of_week', 'year', 'is_weekend', 'temp_hum_interaction']
        
        X = df_copy[features]
        y = df_copy['cnt']
        
        # Score do modelo
        model_score = model.score(X, y)
        
        return model, df, model_score
        
    except FileNotFoundError as e:
        if 'bike_sharing_model_pipeline.pkl' in str(e):
            st.error("❌ Arquivo 'bike_sharing_model_pipeline.pkl' não encontrado!")
        else:
            st.error("❌ Arquivo 'day.csv' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo/dados: {str(e)}")
        st.stop()

# Carregar recursos
model, df_original, model_score = load_model_and_data()

# Header da aplicação
st.title("🚴 Preditor de Demanda - Bike Sharing")
st.markdown("### Sistema de Predição baseado em Machine Learning")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("📋 Navegação")
page = st.sidebar.selectbox(
    "Escolha uma página:",
    ["🎯 Predição", "📊 Dashboard", "🔍 Análise Exploratória", "ℹ️ Sobre o Modelo"]
)

# ================================
# PÁGINA DE PREDIÇÃO
# ================================
if page == "🎯 Predição":
    st.header("🎯 Fazer Predição")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌤️ Condições Climáticas")
        
        temp = st.slider(
            "Temperatura Normalizada", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Temperatura normalizada (0 = muito frio, 1 = muito quente)"
        )
        
        atemp = st.slider(
            "Sensação Térmica", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01
        )
        
        hum = st.slider(
            "Umidade (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Umidade normalizada (0 = seco, 1 = muito úmido)"
        )
        
        windspeed = st.slider(
            "Velocidade do Vento", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.01
        )
        
        weathersit = st.selectbox(
            "Condição Climática",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "☀️ Claro/Poucas Nuvens",
                2: "⛅ Nublado/Neblina", 
                3: "🌧️ Chuva Leve/Neve",
                4: "⛈️ Chuva Forte/Tempestade"
            }[x],
            index=0
        )
    
    with col2:
        st.subheader("📅 Informações Temporais")
        
        season = st.selectbox(
            "Estação do Ano",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "🌸 Primavera",
                2: "☀️ Verão", 
                3: "🍂 Outono",
                4: "❄️ Inverno"
            }[x],
            index=1
        )
        
        yr = st.selectbox("Ano", options=[0, 1], format_func=lambda x: str(2011 + int(x)) if isinstance(x, (int, float)) else str(x), index=1)
        
        mnth = st.selectbox(
            "Mês", 
            options=list(range(1, 13)),
            format_func=lambda x: {
                1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
                5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto", 
                9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
            }[x],
            index=5
        )
        
        weekday = st.selectbox(
            "Dia da Semana",
            options=list(range(7)),
            format_func=lambda x: ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"][int(x)] if isinstance(x, (int, float)) else str(x),
            index=1
        )
        
        holiday = st.checkbox("É feriado?")
        workingday = st.checkbox("É dia útil?", value=True)
    
    # Criar array para predição com todas as features que o modelo espera
    # Incluindo feature engineering conforme feito no notebook
    input_data = pd.DataFrame({
        'season': [season],
        'yr': [yr], 
        'mnth': [mnth],
        'holiday': [1 if holiday else 0],
        'weekday': [weekday],
        'workingday': [1 if workingday else 0],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed],
        'month': [mnth],  # Feature engineering: month duplicado
        'day_of_week': [weekday],  # Feature engineering: day_of_week duplicado  
        'year': [2011 + yr],  # Feature engineering: ano real
        'is_weekend': [1 if weekday in [0, 6] else 0],  # Feature engineering: fim de semana
        'temp_hum_interaction': [temp * hum]  # Feature engineering: interação temperatura-umidade
    })
    
    # Botão de predição
    if st.button("🚀 Fazer Predição", type="primary", use_container_width=True):
        try:
            prediction = model.predict(input_data)[0]
            
            # Garantir que a predição seja positiva
            prediction = max(0, prediction)
            
            # Resultado com destaque
            st.success(f"### 🎯 Demanda Prevista: **{prediction:.0f} bicicletas**")
            
            # Métricas adicionais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Predição", f"{prediction:.0f}", "bicicletas")
            
            with col2:
                media_historica = df_original['cnt'].mean()
                delta = prediction - media_historica
                st.metric("📊 vs Média Histórica", f"{media_historica:.0f}", f"{delta:+.0f}")
            
            with col3:
                if prediction < 2000:
                    categoria = "🟢 Baixa"
                elif prediction < 5000:
                    categoria = "🟡 Média"
                else:
                    categoria = "🔴 Alta"
                st.metric("📋 Categoria", categoria, "")
            
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")
            st.info("**💡 Dica:** Verifique se todos os campos foram preenchidos corretamente.")

# ================================
# PÁGINA DE DASHBOARD
# ================================
elif page == "📊 Dashboard":
    st.header("📊 Dashboard Analítico")
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Total de Dias", len(df_original), "")
    
    with col2:
        st.metric("🚴 Média Diária", f"{df_original['cnt'].mean():.0f}", "bicicletas")
    
    with col3:
        st.metric("📈 Máximo", f"{df_original['cnt'].max():.0f}", "bicicletas")
    
    with col4:
        st.metric("📉 Mínimo", f"{df_original['cnt'].min():.0f}", "bicicletas")
    
    st.markdown("---")
    
    # Gráficos lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição da demanda
        fig1 = px.histogram(
            df_original, 
            x='cnt', 
            nbins=30,
            title="📊 Distribuição da Demanda Diária",
            labels={'cnt': 'Contagem de Bicicletas', 'count': 'Frequência'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Demanda por estação
        season_map = {1: 'Primavera', 2: 'Verão', 3: 'Outono', 4: 'Inverno'}
        df_season = df_original.copy()
        df_season['season_name'] = df_season['season'].map(season_map)
        
        fig2 = px.box(
            df_season,
            x='season_name',
            y='cnt',
            title="🌤️ Demanda por Estação do Ano",
            labels={'cnt': 'Contagem', 'season_name': 'Estação'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico de linha temporal
    df_time = df_original.copy()
    df_time['dteday'] = pd.to_datetime(df_time['dteday'])
    
    fig3 = px.line(
        df_time,
        x='dteday',
        y='cnt',
        title="📈 Evolução da Demanda ao Longo do Tempo",
        labels={'cnt': 'Contagem de Bicicletas', 'dteday': 'Data'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ================================
# PÁGINA DE ANÁLISE EXPLORATÓRIA  
# ================================
elif page == "🔍 Análise Exploratória":
    st.header("🔍 Análise Exploratória dos Dados")
    
    # Mostrar dados brutos
    if st.checkbox("📋 Mostrar dados brutos"):
        st.dataframe(df_original.head(10))
    
    st.subheader("🌡️ Correlação entre Variáveis")
    
    # Matriz de correlação
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df_original.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Matriz de Correlação')
    st.pyplot(fig)
    
    # Análises específicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("☁️ Impacto do Clima")
        
        weather_map = {
            1: 'Claro', 
            2: 'Nublado', 
            3: 'Chuva Leve', 
            4: 'Chuva Forte'
        }
        
        df_weather = df_original.copy()
        df_weather['weather_name'] = df_weather['weathersit'].map(weather_map)
        
        fig4 = px.box(
            df_weather,
            x='weather_name', 
            y='cnt',
            title="Demanda por Condição Climática"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.subheader("📅 Padrões Temporais")
        
        weekday_map = {
            0: 'Dom', 1: 'Seg', 2: 'Ter', 3: 'Qua', 
            4: 'Qui', 5: 'Sex', 6: 'Sáb'
        }
        
        df_weekday = df_original.copy()
        df_weekday['weekday_name'] = df_weekday['weekday'].map(weekday_map)
        
        fig5 = px.bar(
            df_weekday.groupby('weekday_name')['cnt'].mean().reset_index(),
            x='weekday_name',
            y='cnt',
            title="Demanda Média por Dia da Semana"
        )
        st.plotly_chart(fig5, use_container_width=True)

# ================================
# PÁGINA SOBRE O MODELO
# ================================
elif page == "ℹ️ Sobre o Modelo":
    st.header("ℹ️ Informações do Modelo")
    
    st.subheader("🎓 Projeto Acadêmico")
    st.info("""
    **Universidade Federal do Maranhão**  
    **Centro de Ciência Exatas e Tecnologia**  
    **Disciplina:** Aprendizagem de Máquina  
    **Professor:** Alex Oliveira Barradas Filho
    """)
    
    st.subheader("👥 Equipe de Desenvolvimento")
    st.write("""
    - ALISSON EMANUEL DINIZ SANTOS
    - ANDRE VICTOR MACEDO PEREIRA  
    - HUDSON COSTA DINIZ
    - ITALO MATHEUS RODRIGUES SOUSA
    - VITOR FERREIRA NUNES
    """)
    
    st.subheader("🤖 Especificações Técnicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Dataset:**")
        st.write("- Fonte: day.csv")
        st.write(f"- Registros: {len(df_original)} dias")
        st.write("- Período: 2011-2012")
        st.write("- Features: 11 variáveis principais")
        
        st.write("**🎯 Target:**")
        st.write("- Variável: cnt (contagem diária)")
        st.write("- Tipo: Regressão")
        st.write(f"- Range: {df_original['cnt'].min()} - {df_original['cnt'].max():,} bicicletas")
    
    with col2:
        st.write("**🛠️ Modelo:**")
        st.write("- Pipeline pré-treinado")
        st.write(f"- R²: {model_score:.3f}")
        st.write("- Modelo otimizado")
        st.write("- Features simplificadas")
        
        st.write("**⚙️ Características:**")
        st.write("- Carregamento via joblib")
        st.write("- Sem feature engineering complexo")
        st.write("- Princípio YAGNI aplicado")
    
    st.subheader("📈 Performance do Modelo")
    
    # Métricas do modelo
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("🎯 R² Score", f"{model_score:.3f}", "Performance")
    
    with metrics_col2:
        st.metric("📊 Features", "11", "variáveis")
    
    with metrics_col3:
        st.metric("✅ Status", "Ativo", "Funcionando")
    
    st.subheader("🔧 Features Utilizadas")
    st.write("""
    **Variáveis Principais:**
    - season, yr, mnth, weekday
    - holiday, workingday  
    - weathersit
    - temp, atemp, hum, windspeed
    
    **Simplificações Aplicadas:**
    - Removido feature engineering complexo
    - Consolidado carregamento de dados
    - Uso de modelo pré-treinado
    """)

# Footer
st.markdown("---")
st.markdown("### 🚴 Bike Sharing Predictor | Desenvolvido com Streamlit | Versão Simplificada")