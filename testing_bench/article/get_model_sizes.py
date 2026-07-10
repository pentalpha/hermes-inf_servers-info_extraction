import sys
import transformers as hf_tf
import pandas as pd
import os
import torch

hf_token = os.getenv("HF_TOKEN")
models_csv_path = sys.argv[1]
name_col = "Nome Completo"
df = pd.read_csv(models_csv_path)
# number of parameters
model_sizes = []
# original names
model_names = []

from tqdm import tqdm

bar = tqdm(total=len(df[name_col]))
for name in df[name_col]:
    if "/" not in name:
        # Cannot be a HF repo name
        pass
    try:
        print(f"[Rede] Calculando tamanho de {name}...")
        total_params = None

        try:

            config = hf_tf.AutoConfig.from_pretrained(
                name, trust_remote_code=True, token=hf_token
            )

            with torch.device("meta"):
                model = None

                # Estratégia 1: Tentar a inicialização pela classe exata da arquitetura
                if hasattr(config, "architectures") and config.architectures:
                    arch_name = config.architectures[0]
                    if hasattr(hf_tf, arch_name):
                        model_cls = getattr(hf_tf, arch_name)
                        try:
                            # O SEGREDO AQUI: Usar Classe(config) e não Classe.from_config()
                            model = model_cls(config)
                        except Exception:
                            pass  # Falhou, passa para o próximo fallback

                # Estratégia 2: Fallback para CausalLMs (Ideal para códigos remotos não locais)
                if model is None:
                    try:
                        model = hf_tf.AutoModelForCausalLM.from_config(
                            config, trust_remote_code=True
                        )
                    except Exception:
                        pass

                # Estratégia 3: Fallback genérico final (Para modelos de Embedding, MPNet, etc)
                if model is None:
                    model = hf_tf.AutoModel.from_config(config, trust_remote_code=True)

            # Conta os parâmetros matematicamente na memória virtual
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[Sucesso] {name} tem {total_params} parâmetros.")
        except Exception as e_tf:
            # --- TENTATIVA 2: Fallback Direcionado para GLiNER ---
            try:
                from gliner import GLiNER

                # Instancia o modelo pela biblioteca oficial (fará o download dos ~400MB)
                gliner_model = GLiNER.from_pretrained(name, load_tokenizer=False)

                total_params = sum(p.numel() for p in gliner_model.parameters())

                # Força a liberação da memória RAM imediatamente após a contagem
                del gliner_model

            except ImportError:
                print(
                    f"[Aviso] Para processar {name}, pare o script e instale: pip install gliner"
                )
                continue
            except Exception as e_gl:
                print(
                    f"Falha dupla no modelo {name}.\nTF Error: {e_tf}\nGLiNER Error: {e_gl}"
                )
                continue

        if total_params is not None:
            model_names.append(name)
            model_sizes.append(total_params)
        else:
            print(f"[Falha] Não foi possível carregar o modelo {name}")
    except Exception as e:
        print(f"[Falha] Não foi possível carregar o modelo {name}: {e}")
    bar.update(1)
bar.close()
# convert to billion parameters
model_sizes = [size / 1e9 for size in model_sizes]

# create a dataframe with the results
results_df = pd.DataFrame(
    {name_col: model_names, "Tamanho do Modelo (BP)": model_sizes}
)
results_dir = os.path.dirname(models_csv_path)
results_path = os.path.join(results_dir, "model_sizes.csv")
# save the results to a csv file
results_df.to_csv(results_path, index=False)
