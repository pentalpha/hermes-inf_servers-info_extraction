import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker

df_path = "results/df_simples.csv"

#Markers for each model type (only filled formats)
model_type_to_format = {
    "Gliner v1": "o",
    "Gliner v2": "s",
    "NuExtract": "^",
    "GPT": "d",
    "Gemini": "p",
    "Others": "h"
}

model_type_to_colors = {
    "Gliner v1": "red",
    "Gliner v2": "darkred",
    "NuExtract": "green",
    "GPT": "purple",
    "Gemini": "orange",
    "Others": "gray"
}

def get_ner_model_type(ner_name):
    '''
    Gliner v1
    Gliner v2
    NuExtract
    GPT (including oN): 
    Gemini'''

    if "gliner" in ner_name.lower() or 'nuner' in ner_name.lower():
        if "gliner2" in ner_name.lower():
            return "Gliner v2"
        else:
            return "Gliner v1"
    elif "gpt" in ner_name.lower():
        return "GPT"
    elif ner_name.lower().startswith("o") and (ner_name.lower()[-1].isdigit() or ner_name.lower()[1].isdigit()):
        #o3, o4, o4.1...
        return "GPT"
    elif "gemini" in ner_name.lower():
        return "Gemini"
    elif "nuextract" in ner_name.lower():
        return "NuExtract"
    else:
        print("Other is", ner_name)
        return "Others"

def sep_model_types(df):
    #Adds a column for type of LLM
    df["Tipo de Modelo"] = df["ner_model"].apply(get_ner_model_type)
    return df

def simple_hf_name(df):
    #If the model is a HF model, simplify the name to the last part
    df["simple_name"] = df["ner_model"].apply(lambda x: x.split("/")[-1] if "/" in x else x)
    return df

def model_subtype(df):
    #Replace - and _ with ' ', identify the first word
    #Remove first word, remove _ / - from start of string
    def to_subtype(ner_name):
        words = ner_name.replace("-", " ").replace("_", " ").split(" ")
        first_w = words[0]
        if len(words) > 0:
            if first_w in ['o3', 'o4', 'o4.1']:
                return ner_name
            return ner_name.replace(first_w, "").strip(" -_")
        else:
            return ner_name

    df["subtype"] = df["simple_name"].apply(to_subtype)
    return df   

def plot_cost_vs_latency_vs_quality(df):
    '''
    X axis is cost, Y axis is latency, color is quality
    Good quality should be blue, bad quality should be red
    '''
    fig, ax = plt.subplots(figsize=(10, 6))

    #If the model is Gliner v1, Gliner v2 or Others, keep only the rows where the clf model is knowledgator/gliclass-large-v3.0
    df_gliners = df[(df["Tipo de Modelo"] == "Gliner v1") | (df["Tipo de Modelo"] == "Gliner v2") | (df["Tipo de Modelo"] == "Others")]
    df_gliners = df_gliners[df_gliners["classification_model"] == "knowledgator/gliclass-large-v3.0"]

    df_not_gliners = df[(df["Tipo de Modelo"] != "Gliner v1") & (df["Tipo de Modelo"] != "Gliner v2") & (df["Tipo de Modelo"] != "Others")]

    df = pd.concat([df_gliners, df_not_gliners])

    #keep onlythe top 18 results, using general quality
    df = df.nlargest(18, "Qualidade Geral")

    tipos_bons = df["Tipo de Modelo"].unique()
    print(tipos_bons)

    #Predefine colors in RdBu scale, to be able to plot types separatly,
    #but only from smaller quality to larger quality
    low = df["Qualidade Geral"].min()
    high = df["Qualidade Geral"].max()

    #Make custom scale
    norm = plt.Normalize(low, high)
    cmap = plt.cm.RdBu

    df["color"] = df["Qualidade Geral"].apply(lambda x: cmap(norm(x)))

    for model_type, type_df in df.groupby("Tipo de Modelo"):
        scatter = ax.scatter(type_df["total_cost"], type_df["total_runtime"], c=type_df["color"], 
            marker=model_type_to_format[model_type], label=model_type, linewidths=1, edgecolors="black",
            s=200, alpha=0.92)

    for index, row in df.iterrows():
        print(row["Tipo de Modelo"], row["subtype"], row["Qualidade Geral"])
    
    ax.set_xlabel("Custo do Teste ($)")
    ax.set_ylabel("Latência Total (minutos)")
    ax.set_title("Custo vs Latência vs Qualidade")

    ax.set_xscale("log")
    ax.set_yscale("log")

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    # Optional: Set a limit for when scientific notation should be used (e.g., set to (0,0) to include all numbers)
    #formatter.set_powerlimits((0, 0)) 

    # 5. Apply the formatter to both major and minor ticks for the x and y axes
    ax.xaxis.set_major_formatter(formatter)
    #ax.xaxis.set_minor_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    #ax.yaxis.set_minor_formatter(formatter)

    #Add colorbar to figure, but using the custom scale
    # Cria um objeto mapeável explicitamente com as suas regras de cor
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Evita warnings em algumas versões do Matplotlib
    
    # Adiciona a barra de cores usando o ScalarMappable e os valores reais para os ticks
    cbar = fig.colorbar(sm, ax=ax, ticks=[low, (low+high)/2, high])
    
    cbar.set_label("Qualidade Geral")
    cbar.ax.set_yticklabels([f"{low:.2f}", f"{(low+high)/2:.2f}", f"{high:.2f}"])

    #Legend outside plot of model types, with custom markers
    #Placed below graph to not overlap colorbar
    legend_elements = []
    for model_type, marker in model_type_to_format.items():
        if model_type not in tipos_bons:
            continue
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
            label=model_type, markerfacecolor='k', markersize=8))
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)

    #Add text labels for all results
    for i, row in df.iterrows():
        #Centered text, semi-transparent, small font
        ax.annotate(row["subtype"], (row["total_cost"], row["total_runtime"]),
            ha="center", va="bottom", fontsize=8, alpha=0.85)
    
    #Save plot
    plt.savefig("results/cost_vs_latency_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.close()

def ner_vs_clf_quality(df):
    #Scatter all the results, but name only the 20 best and gliner v2
    df_gliner_v2 = df[df["Tipo de Modelo"] == "Gliner v2"]
    df_best = df.nlargest(20, "Qualidade Geral")
    df_named = pd.concat([df_gliner_v2, df_best])

    tipos_encontrados = df["Tipo de Modelo"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlabel("Reconhecimento de Entidades")
    ax.set_ylabel("Classificação")
    ax.set_title("Qualidade de Extração de Informações")

    

    #Scatter all the results, but name only the 20 best and gliner v2
    for model_type, type_df in df.groupby("Tipo de Modelo"):
        color = model_type_to_colors[model_type]
        scatter = ax.scatter(type_df["Qualidade Reconhecimento de Entidades"], 
            type_df["Qualidade Classificação"], c=color,
            marker=model_type_to_format[model_type], label=model_type, #linewidths=1, edgecolors="black",
            s=170, alpha=0.8)


    #Legend outside plot of model types, with custom markers
    #Placed below graph to not overlap colorbar
    '''legend_elements = []
    for model_type, marker in model_type_to_format.items():
        if model_type not in tipos_encontrados:
            continue
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
            label=model_type, markerfacecolor='k', markersize=8))'''
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)

    #Add text labels for all results
    for i, row in df_named.iterrows():
        #Centered text, semi-transparent, small font
        ax.annotate(row["subtype"], (row["Qualidade Reconhecimento de Entidades"], row["Qualidade Classificação"]),
            ha="center", va="bottom", fontsize=8, alpha=0.85)
    
    #Save plot
    plt.savefig("results/ner_vs_clf_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    

    

if __name__ == "__main__":
    df = pd.read_csv(df_path)

    #Converter latência para minutos
    df["total_runtime"] = df["total_runtime"] / 60

    df = sep_model_types(df)
    df = simple_hf_name(df)
    df = model_subtype(df)
    plot_cost_vs_latency_vs_quality(df)
    ner_vs_clf_quality(df)