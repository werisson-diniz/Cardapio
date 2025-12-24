import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Configuração e Estilo
st.set_page_config(page_title="NutriIA - Planejador Inteligente", layout="wide")
st.title("🥗 NutriIA: Cardápio Personalizado com Machine Learning")

# 2. Carregamento de Dados
@st.cache_data
def load_data():
    # Carrega o seu dataset original
    df = pd.read_csv('meu_dataframe_refeicoes.csv')
    return df

df = load_data()

# 3. Sidebar: Cadastro e Biometria
with st.sidebar:
    st.header("👤 Perfil do Usuário")
    nome = st.text_input("Nome Completo")
    idade = st.number_input("Idade", min_value=1, max_value=110, value=25)
    peso = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0)
    altura = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=1.70)
    sexo = st.radio("Sexo", ["Masculino", "Feminino"])
    
    st.divider()
    st.header("🎯 Configurações de Saúde")
    objetivo = st.selectbox("Objetivo", ["Emagrecer", "Massa Muscular", "Manter Peso"])
    diabetico = st.toggle("Sou Diabético", value=False)
    
    # Parâmetro do K-Means
    k_clusters = st.slider("Refinamento da IA (Clusters)", 2, 10, 5)

# 4. Cálculos Biométricos
if altura > 0:
    imc = peso / (altura ** 2)
    
    # Taxa Metabólica Basal (Mifflin-St Jeor)
    if sexo == "Masculino":
        tmb = (10 * peso) + (625 * altura) - (5 * idade) + 5
    else:
        tmb = (10 * peso) + (625 * altura) - (5 * idade) - 161
    
    # Meta Calórica baseada no objetivo
    if objetivo == "Emagrecer":
        meta_calorica = tmb * 1.2 - 500
    elif objetivo == "Massa Muscular":
        meta_calorica = tmb * 1.5 + 300
    else:
        meta_calorica = tmb * 1.3

# 5. Inteligência Artificial: K-Means
# Selecionando colunas chave para o agrupamento
features = ['energia_kcal', 'proteina_g', 'carboidrato_g', 'lipideos_g', 'fibra_g']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['cluster_ia'] = kmeans.fit_predict(df_scaled)

# 6. Painel Principal - Resultados Biométricos
if nome:
    st.subheader(f"Análise de Saúde: {nome}")
    c1, c2, c3, c4 = st.columns(4)
    
    # Classificação do IMC
    status_imc = "Normal"
    if imc < 18.5: status_imc = "Abaixo do peso"
    elif imc >= 25: status_imc = "Sobrepeso/Obesidade"
    
    c1.metric("Seu IMC", f"{imc:.1f}", help=status_imc)
    c2.metric("TMB (Repouso)", f"{tmb:.0f} kcal")
    c3.metric("Meta Diária Sugerida", f"{meta_calorica:.0f} kcal")
    c4.metric("Condição", "Diabético" if diabetico else "Saudável")

# 7. Recomendação Baseada em Clusters e Objetivos
def recomendar_alimentos(df_input):
    df_f = df_input.copy()
    
    # Restrição para Diabéticos: Filtra carboidratos altos e remove doces
    if diabetico:
        df_f = df_f[df_f['categoria'] != 'Produtos açucarados']
        df_f = df_f[df_f['carboidrato_g'] < 40] # Limite de segurança por item
    
    # Lógica por Objetivo usando Clusters
    # Identificamos qual cluster tem a característica desejada
    cluster_stats = df_f.groupby('cluster_ia')[features].mean()
    
    if objetivo == "Massa Muscular":
        # Busca o cluster com maior média de proteína
        alvo = cluster_stats['proteina_g'].idxmax()
        df_f = df_f[df_f['cluster_ia'] == alvo]
    elif objetivo == "Emagrecer":
        # Busca o cluster com menor média de calorias e boa fibra
        alvo = cluster_stats['energia_kcal'].idxmin()
        df_f = df_f[df_f['cluster_ia'] == alvo]

    # Sorteia itens para cada refeição
    cardapio = []
    for ref in ['Café da Manhã', 'Almoço', 'Merenda', 'Jantar']:
        opcoes = df_f[df_f['Refeicoes'].str.contains(ref, na=False)]
        if not opcoes.empty:
            item = opcoes.sample(1)
            item['Refeição'] = ref
            cardapio.append(item)
            
    return pd.concat(cardapio) if cardapio else pd.DataFrame()

# 8. Exibição do Cardápio
if st.button("✨ Gerar Cardápio Personalizado"):
    with st.spinner('A IA está analisando os grupos de alimentos...'):
        meu_cardapio = recomendar_alimentos(df)
        
        if not meu_cardapio.empty:
            st.success("Cardápio calculado com sucesso!")
            exibir = meu_cardapio[['Refeição', 'descricao', 'categoria', 'energia_kcal', 'proteina_g', 'carboidrato_g', 'fibra_g']]
            st.table(exibir)
            
            # Totais do dia
            st.info(f"**Total do dia:** {meu_cardapio['energia_kcal'].sum():.0f} kcal | "
                    f"Proteína: {meu_cardapio['proteina_g'].sum():.1f}g | "
                    f"Fibras: {meu_cardapio['fibra_g'].sum():.1f}g")
        else:
            st.error("Não encontramos alimentos que atendam a todos os filtros simultaneamente. Tente ajustar o nível de IA.")

st.divider()
st.caption("Nota: Este sistema utiliza K-Means para agrupar perfis nutricionais similares.")