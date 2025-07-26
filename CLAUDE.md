# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bike sharing demand prediction system developed as an academic project for the Machine Learning course at Universidade Federal do Maranhão. The project consists of:

- A Jupyter notebook (`AprendizagemDeMaquina_AV3_atualizado.ipynb`) with the complete ML pipeline development
- A Streamlit web application (`app.py`) for interactive predictions
- A trained model pipeline (`bike_sharing_model_pipeline.pkl`) for deployment

## Core Architecture

### Data Pipeline
- **Dataset**: `day.csv` contains 731 daily records from 2011-2012 bike sharing system
- **Target Variable**: `cnt` (daily bike rental count)
- **Features**: 11 main variables including weather conditions, temporal data, and seasonal patterns
- **Feature Engineering**: Creates additional features like weekend indicators, temperature-humidity interactions, and extracted date components

### Model Pipeline
The system uses a complete scikit-learn pipeline that includes:
- **Preprocessing**: ColumnTransformer with separate pipelines for numerical and categorical features
- **Numerical Pipeline**: SimpleImputer (median) + StandardScaler
- **Categorical Pipeline**: SimpleImputer (most_frequent) + OneHotEncoder
- **Model**: LinearRegression with R² score of ~0.81

### Streamlit Application Structure
- **Multi-page app** with 4 main sections:
  1. 🎯 Predição: Interactive prediction interface
  2. 📊 Dashboard: Data analytics and visualizations  
  3. 🔍 Análise Exploratória: Exploratory data analysis
  4. ℹ️ Sobre o Modelo: Model documentation and team info

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv streamlit_env

# Activate environment (Windows)
streamlit_env\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start Streamlit app
streamlit run app.py

# Start on specific port
streamlit run app.py --server.port 8502
```

### Model Operations
```python
# Load trained model
import joblib
model = joblib.load('bike_sharing_model_pipeline.pkl')

# Make predictions (expects DataFrame with 11 features)
prediction = model.predict(input_data)
```

## Key Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **scikit-learn**: ML pipeline and model
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plots
- **joblib**: Model serialization

## Important Notes

### File Dependencies
The application requires these files to be present:
- `day.csv`: Original dataset for analysis
- `bike_sharing_model_pipeline.pkl`: Trained model pipeline
- `app.py`: Main Streamlit application

### Model Input Format
The model expects exactly 11 features in this order:
- season, yr, mnth, holiday, weekday, workingday
- weathersit, temp, atemp, hum, windspeed

### Data Caching
The Streamlit app uses `@st.cache_resource` for loading model and data to improve performance.

## Academic Context
- **Institution**: Universidade Federal do Maranhão - Centro de Ciência Exatas e Tecnidade
- **Course**: Aprendizagem de Máquina
- **Professor**: Alex Oliveira Barradas Filho
- **Team**: 5 students (names listed in README.md and app.py)

## Claude Interaction Guidelines
- sempre responda em portugues brasil