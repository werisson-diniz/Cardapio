import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# CONFIGURAÇÃO DA PÁGINA
# =============================
st.set_page_config(
    page_title="Dashboard Nutricional por Clusters",
    layout="wide"
)

st.title("🥗 Dashboard Nutricional por Clusters (K-Means)")
st.markdown(
    "Exploração de alimentos agrupados por **perfil nutricional**, "
    "exibindo **apenas valores reais** (kcal e gramas)."
)

# =============================
# CARREGAMENTO DOS DADOS
# =============================
@st.cache_data
def carregar_dados():
    return pd.read_csv("taco_atualizado.csv")

df = carregar_dados()

# =============================
# COLUNAS NUTRICIONAIS (REAIS)
# =============================
colunas_nutricionais = [
    'energia_kcal',
    'proteina_g',
    'lipideos_g',
    'carboidrato_g',
    'fibra_g',
    'cinzas_g',
    'calcio_mg',
    'magnesio_mg'
]

# =============================
# GARANTIR DADOS INTERPRETÁVEIS
# =============================
df[colunas_nutricionais] = df[colunas_nutricionais].clip(lower=0)

# =============================
# VALIDAÇÃO DE COLUNAS ESSENCIAIS
# =============================
colunas_necessarias = [
    'nome',
    'cluster',
    'perfil_nutricional',
    'energia_kcal',
    'proteina_g',
    'lipideos_g',
    'carboidrato_g',
    'fibra_g'
]

faltando = [c for c in colunas_necessarias if c not in df.columns]
if faltando:
    st.error(f"Colunas ausentes no dataset: {faltando}")
    st.stop()

# =============================
# SIDEBAR — FILTRO ÚNICO
# =============================
st.sidebar.header("🔎 Filtros")

perfil_selecionado = st.sidebar.multiselect(
    "Perfil nutricional",
    options=sorted(df['perfil_nutricional'].unique()),
    default=sorted(df['perfil_nutricional'].unique())
)

# =============================
# APLICAR FILTRO
# =============================
df_filtrado = df[
    df['perfil_nutricional'].isin(perfil_selecionado)
]

# =============================
# GRÁFICO — DISTRIBUIÇÃO DOS PERFIS
# =============================
st.subheader("📊 Distribuição dos Perfis Nutricionais")

contagem = df['perfil_nutricional'].value_counts()

fig, ax = plt.subplots()
contagem.plot(kind='bar', ax=ax)
ax.set_xlabel("Perfil Nutricional")
ax.set_ylabel("Quantidade de Alimentos")
plt.xticks(rotation=45, ha='right')

for i, v in enumerate(contagem.values):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

st.pyplot(fig)

# =============================
# TABELA — ALIMENTOS FILTRADOS
# =============================
st.subheader("📋 Alimentos filtrados (valores reais)")
st.write(f"Total de alimentos encontrados: **{len(df_filtrado)}**")

st.dataframe(
    df_filtrado[
        [
            'nome',
            'perfil_nutricional',
            'energia_kcal',
            'proteina_g',
            'lipideos_g',
            'carboidrato_g',
            'fibra_g'
        ]
    ]
    .round(2)
    .sort_values(by='proteina_g', ascending=False),
    use_container_width=True
)

# =============================
# LISTAR ALIMENTOS POR CLUSTER
# =============================
st.subheader("📦 Alimentos agrupados por Cluster (valores reais)")

for cluster_id, grupo in df.groupby('cluster'):
    perfil = grupo['perfil_nutricional'].iloc[0]

    st.markdown(f"### Cluster {cluster_id} — {perfil}")
    st.write(f"Quantidade de alimentos: **{len(grupo)}**")

    st.dataframe(
        grupo[
            [
                'nome',
                'energia_kcal',
                'proteina_g',
                'lipideos_g',
                'carboidrato_g',
                'fibra_g'
            ]
        ]
        .round(2)
        .sort_values(by='energia_kcal'),
        use_container_width=True
    )

# =============================
# ESTATÍSTICAS POR PERFIL
# =============================
st.subheader("📊 Estatísticas médias por Perfil Nutricional")

perfil_stats = (
    df.groupby('perfil_nutricional')[
        ['energia_kcal', 'proteina_g', 'lipideos_g', 'carboidrato_g', 'fibra_g']
    ]
    .mean()
    .round(2)
)

st.caption(
    "📌 Valores exibidos em **kcal e gramas reais**. "
    "Dados padronizados foram usados apenas internamente no K-Means."
)

st.dataframe(perfil_stats, use_container_width=True)

# =============================
# RODAPÉ
# =============================
st.markdown("---")
st.markdown(
    "Projeto educacional de **Machine Learning aplicado à Nutrição** usando K-Means."
)
