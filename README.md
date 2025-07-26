# 🚴 Bike Sharing Predictor - Streamlit App

Sistema de predição de demanda para compartilhamento de bicicletas desenvolvido com Streamlit e Machine Learning.

## 📋 Pré-requisitos

### 1. Python
Certifique-se de que o Python está instalado no seu sistema:
- **Download:** https://www.python.org/downloads/
- **Versão mínima:** Python 3.8+

Para verificar se o Python está instalado:
```cmd
python --version
```

### 2. Arquivos necessários
Certifique-se de que os seguintes arquivos estão no diretório:
- `day.csv` (dataset original)
- `bike_sharing_model_pipeline.pkl` (modelo treinado)
- `app.py` (aplicação Streamlit)
- `requirements.txt` (dependências)

## 🚀 Instalação e Execução

### Método 1: Automático (Recomendado)

1. **Execute o setup:**
   ```cmd
   setup.bat
   ```
   Este script irá:
   - Criar ambiente virtual Python
   - Instalar todas as dependências
   - Configurar o projeto

2. **Execute a aplicação:**
   ```cmd
   run_app.bat
   ```

3. **Acesse no navegador:**
   ```
   http://localhost:8501
   ```

### Método 2: Manual

1. **Criar ambiente virtual:**
   ```cmd
   python -m venv streamlit_env
   ```

2. **Ativar ambiente virtual:**
   ```cmd
   streamlit_env\Scripts\activate.bat
   ```

3. **Instalar dependências:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Executar aplicação:**
   ```cmd
   streamlit run app.py
   ```

## 📊 Funcionalidades

### 🎯 Predição
- Interface interativa para fazer predições
- Formulário com todos os parâmetros do modelo
- Resultado em tempo real com métricas

### 📈 Dashboard
- Visualizações dos dados históricos
- Métricas estatísticas
- Gráficos interativos com Plotly

### 🔍 Análise Exploratória
- Matriz de correlação
- Análise de padrões climáticos
- Distribuições temporais

### ℹ️ Documentação
- Informações do projeto acadêmico
- Especificações técnicas do modelo
- Performance e métricas

## 🛠️ Tecnologias Utilizadas

- **Streamlit:** Interface web interativa
- **scikit-learn:** Pipeline de ML e modelo
- **Pandas:** Manipulação de dados
- **Plotly:** Visualizações interativas
- **Matplotlib/Seaborn:** Gráficos estáticos

## 📦 Dependências

```
streamlit==1.29.0
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
joblib==1.3.2
numpy==1.25.2
```

## 🎓 Projeto Acadêmico

**Universidade Federal do Maranhão**  
**Centro de Ciência Exatas e Tecnologia**  
**Disciplina:** Aprendizagem de Máquina  
**Professor:** Alex Oliveira Barradas Filho

### 👥 Equipe:
- ALISSON EMANUEL DINIZ SANTOS
- ANDRE VICTOR MACEDO PEREIRA
- HUDSON COSTA DINIZ
- ITALO MATHEUS RODRIGUES SOUSA
- VITOR FERREIRA NUNES

## 🔧 Solução de Problemas

### Python não encontrado
1. Instale Python em: https://www.python.org/downloads/
2. Marque a opção "Add Python to PATH" durante a instalação
3. Reinicie o terminal/prompt de comando

### Erro ao instalar dependências
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Arquivos não encontrados
Verifique se `day.csv` e `bike_sharing_model_pipeline.pkl` estão no mesmo diretório que `app.py`

### Porta já em uso
Se a porta 8501 estiver ocupada:
```cmd
streamlit run app.py --server.port 8502
```

## 📱 Como Usar

1. **Acesse a aplicação** no navegador
2. **Escolha uma página** no menu lateral:
   - 🎯 Predição: Faça predições interativas
   - 📊 Dashboard: Veja análises dos dados
   - 🔍 Análise Exploratória: Explore padrões
   - ℹ️ Sobre o Modelo: Documentação técnica

3. **Na página de Predição:**
   - Ajuste os parâmetros climáticos
   - Configure informações temporais
   - Clique em "Fazer Predição"
   - Veja o resultado em tempo real

## 🌐 Deploy

Para deploy em produção, considere:
- **Streamlit Cloud:** https://streamlit.io/cloud
- **Heroku:** Plataforma gratuita
- **Railway:** Deploy automático

---

**Desenvolvido com ❤️ usando Streamlit**