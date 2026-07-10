import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Carregar os dados (ajuste o nome do arquivo se necessário)
df = pd.read_csv("serpro_precos.tsv", sep="\t")


# 2. Funções para converter strings (ex: "R$ 0,00180" e "14,00") para formato numérico (float)
def limpar_moeda(x):
    if pd.isna(x):
        return x
    return float(str(x).replace("R$", "").replace(" ", "").replace(",", "."))


def limpar_numero(x):
    if pd.isna(x):
        return x
    return float(str(x).replace(".", "").replace(",", "."))


# Aplicar a limpeza nas colunas que usaremos
df["Tamanho (BP)"] = df["Tamanho (BP)"].apply(limpar_numero)
df["Tokens por segundo IN"] = df["Tokens por segundo IN"].apply(limpar_numero)
df["Tokens por segundo OUT"] = df["Tokens por segundo OUT"].apply(limpar_numero)
df["Preço 1k Tokens IN"] = df["Preço 1k Tokens IN"].apply(limpar_moeda)
df["Preço 1k Tokens OUT"] = df["Preço 1k Tokens OUT"].apply(limpar_moeda)

# 3. Separar dados de Treino (modelos com preço) e Teste (modelos sem preço)
df_treino = df.dropna(subset=["Preço 1k Tokens IN", "Preço 1k Tokens OUT"])
df_prever = df[df["Nome na API Serpro"].isin(["qwen3.5-35b", "magistral-small"])]

# --- REGRESSÃO PARA O PREÇO IN ---
# 2 Variáveis independentes e 1 dependente
X_in = df_treino[["Tamanho (BP)", "Tokens por segundo IN"]]
y_in = df_treino["Preço 1k Tokens IN"]

modelo_in = LinearRegression()
modelo_in.fit(X_in, y_in)

# --- REGRESSÃO PARA O PREÇO OUT ---
# 2 Variáveis independentes e 1 dependente
X_out = df_treino[["Tamanho (BP)", "Tokens por segundo OUT"]]
y_out = df_treino["Preço 1k Tokens OUT"]

modelo_out = LinearRegression()
modelo_out.fit(X_out, y_out)

# 4. Fazer as previsões para os modelos sem preço
X_in_prever = df_prever[["Tamanho (BP)", "Tokens por segundo IN"]]
X_out_prever = df_prever[["Tamanho (BP)", "Tokens por segundo OUT"]]

previsoes_in = modelo_in.predict(X_in_prever)
previsoes_out = modelo_out.predict(X_out_prever)

# 5. Exibir os resultados
print("=== PREVISÕES DE PREÇO ===")
for i, nome in enumerate(df_prever["Nome na API Serpro"]):
    print(f"\nModelo: {nome}")
    print(f"  Preço 1k Tokens IN:  R$ {previsoes_in[i]:.8f}")
    print(f"  Preço 1k Tokens OUT: R$ {previsoes_out[i]:.8f}")
