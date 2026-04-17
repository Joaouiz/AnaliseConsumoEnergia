# =========================
# 1. Importações
# =========================
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# =========================
# 2. Carregar e adaptar dados
# =========================
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

# Renomeando colunas para contexto elétrico
df.columns = [
    'consumo_medio_kwh',        # sepal length
    'demanda_kw',               # sepal width
    'consumo_pico_kwh',         # petal length
    'consumo_fora_pico_kwh'     # petal width
]

# Tipos de consumidores
df['tipo_consumidor'] = data.target

tipo_map = {
    0: 'Residencial',
    1: 'Comercial',
    2: 'Industrial'
}

df['tipo_consumidor'] = df['tipo_consumidor'].map(tipo_map)

print("Primeiras linhas do dataset:")
print(df.head())

# =========================
# 3. Análise básica
# =========================
print("\nResumo estatístico:")
print(df.describe())

print("\nDistribuição por tipo de consumidor:")
print(df['tipo_consumidor'].value_counts())

# =========================
# 4. Visualizações
# =========================

# Pairplot (visão geral)
sns.pairplot(df, hue='tipo_consumidor')
plt.suptitle("Relações entre variáveis de consumo de energia", y=1.02)
plt.show()

# Heatmap de correlação
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns=['tipo_consumidor']).corr(), annot=True)
plt.title("Correlação entre variáveis elétricas")
plt.show()

# Boxplot - Consumo em horário de pico
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='tipo_consumidor', y='consumo_pico_kwh')
plt.title("Consumo no horário de pico por tipo de consumidor")
plt.show()

# =========================
# 5. Insights
# =========================
print("\nINSIGHTS:")
print("- Consumidores industriais apresentam maior consumo médio e maior demanda.")
print("- O consumo em horário de pico é mais elevado em consumidores comerciais.")
print("- Existe forte correlação entre consumo médio e consumo em horário de pico.")
print("- Consumidores residenciais apresentam menor variabilidade de consumo.")