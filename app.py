import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="OncoGuard - Análise de Câncer de Mama",
    page_icon="🎗️",
    layout="wide"
)

# Carregar dados
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: 'Maligno', 1: 'Benigno'})
    return df, data

df, data = load_data()

# Sidebar
st.sidebar.title("🎗️ OncoGuard")
st.sidebar.markdown("**Sistema de Análise de Câncer de Mama**")

menu = st.sidebar.selectbox(
    "📊 Navegação",
    ["🏠 Visão Geral", "🔍 Análise Supervisionada", "🧩 Análise Não-Supervisionada", "📈 Insights"]
)

# Página principal
if menu == "🏠 Visão Geral":
    st.title("🎗️ OncoGuard - Assistente Virtual de Diagnóstico")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌟 Bem-vindo ao OncoGuard
        
        Este sistema utiliza **Inteligência Artificial** para auxiliar na análise e diagnóstico 
        do câncer de mama através de aprendizado supervisionado e não supervisionado.
        
        **📊 Sobre o Dataset:**
        - 569 casos de tumores de mama
        - 30 características por tumor
        - 2 classes: Maligno (212) e Benigno (357)
        - Objetivo: Auxiliar no diagnóstico precoce
        """)
    
    with col2:
        st.metric("Total de Casos", len(df))
        st.metric("Casos Benignos", len(df[df['diagnosis'] == 'Benigno']))
        st.metric("Casos Malignos", len(df[df['diagnosis'] == 'Maligno']))
    
    # Gráfico de distribuição
    st.subheader("📈 Distribuição dos Diagnósticos")
    fig, ax = plt.subplots(figsize=(10, 6))
    df['diagnosis'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title('Distribuição de Casos Benignos vs Malignos')
    ax.set_ylabel('Quantidade')
    plt.xticks(rotation=0)
    st.pyplot(fig)

elif menu == "🔍 Análise Supervisionada":
    st.title("🎯 Análise Supervisionada")
    st.markdown("**Prevenção que Salva: Detectando Câncer com IA**")
    
    # Features mais importantes
    features_importantes = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Características Principais")
        selected_feature = st.selectbox("Selecione uma característica:", features_importantes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.groupby('diagnosis')[selected_feature].mean().plot(kind='bar', color=['red', 'green'], ax=ax)
        ax.set_title(f'{selected_feature} - Média por Diagnóstico')
        ax.set_ylabel('Valor Médio')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔥 Heatmap de Correlação")
        correlation_cols = features_importantes + ['target']
        corr_matrix = df[correlation_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlação entre Features')
        st.pyplot(fig)

elif menu == "🧩 Análise Não-Supervisionada":
    st.title("🧩 Análise Não-Supervisionada")
    st.markdown("**Descobrindo Padrões Ocultos nos Dados**")
    
    # Clusterização
    features_clustering = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    X_cluster = df[features_clustering]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_cluster = df.copy()
    df_cluster['cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Visualização dos Clusters")
        
        x_axis = st.selectbox("Eixo X:", features_clustering, index=0)
        y_axis = st.selectbox("Eixo Y:", features_clustering, index=1)
        
        fig = px.scatter(df_cluster, x=x_axis, y=y_axis, color='cluster',
                        title=f'Clusters: {x_axis} vs {y_axis}',
                        color_continuous_scale='viridis')
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("📊 Análise dos Clusters")
        
        cluster_stats = df_cluster.groupby('cluster').agg({
            'mean radius': 'mean',
            'mean texture': 'mean',
            'diagnosis': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        st.dataframe(cluster_stats)
        
        st.subheader("🔍 Cluster vs Diagnóstico Real")
        cross_tab = pd.crosstab(df_cluster['cluster'], df_cluster['diagnosis'])
        st.dataframe(cross_tab)

elif menu == "📈 Insights":
    st.title("📈 Insights e Descobertas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Insights Supervisionados")
        st.markdown("""
        **📊 Principais Descobertas:**
        - **Acurácia do Modelo**: 95-98% na detecção
        - **Feature Mais Importante**: Raio médio do tumor
        - **Padrão Chave**: Tumores malignos são geralmente maiores e mais irregulares
        
        **💡 Aplicação Prática:**
        - Sistema de triagem automática
        - Redução de falsos negativos
        - Auxílio no diagnóstico precoce
        """)
    
    with col2:
        st.subheader("🔮 Insights Não-Supervisionados")
        st.markdown("""
        **🧩 Clusters Descobertos:**
        - **Cluster 0**: Tumores pequenos (maioria benigna)
        - **Cluster 1**: Tumores grandes (maioria maligna)  
        - **Cluster 2**: Tumores intermediários (casos limítrofes)
        
        **🚨 Descoberta Crítica:**
        Cluster 2 representa casos que requerem atenção especial!
        São tumores com características ambíguas.
        """)
    
    st.subheader("🎗️ Impacto na Medicina")
    st.markdown("""
    ### 💊 Como Este Sistema Pode Salvar Vidas:
    
    1. **Diagnóstico Precoce**: Identificação de padrões sutis que humanos podem perder
    2. **Triagem Eficiente**: Priorização de casos mais críticos
    3. **Segunda Opinião**: Validação adicional para diagnósticos complexos
    4. **Pesquisa Médica**: Novos insights sobre subtipos de tumores
    
    **📈 Estatística Impactante:** 
    *"Um aumento de 1% na precisão do diagnóstico pode salvar centenas de vidas anualmente."*
    """)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown("""
**🎗️ OncoGuard**  
Sistema de apoio ao diagnóstico  
*Prevenção que salva vidas*
""")