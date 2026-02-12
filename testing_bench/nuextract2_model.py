import time
import json
import sys
from typing import List
from collections import deque
from itertools import islice
from typing import List
import numpy as np

import torch
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
from transformers import AutoProcessor
from tqdm import tqdm

def load_nuextract_cuda(model_name_str):
    torch.cuda.empty_cache() 
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_name_str, 
                                               trust_remote_code=True, 
                                               torch_dtype=torch.bfloat16,
                                               #attn_implementation="flash_attention_2",
                                               ).to("cuda")
        processor = AutoProcessor.from_pretrained(model_name_str, 
                                          trust_remote_code=True, 
                                          padding_side='left',
                                          use_fast=False).to("cuda")
        print("Loaded nuextract model")
    except Exception as e:
        print(f"Error loading nuextract model with CUDA: {e}")
        raise (e)

    return model, processor


def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def sliding_window_over_paragraph(text, n_words=160, sobreposicao=12, fillvalue=""):
    windows = [
        " ".join(x).strip()
        for x in sliding_window(
            text.split(" "),
            size=n_words,
            step=n_words - sobreposicao,
            fillvalue=fillvalue,
        )
    ]
    return windows


def join_entity_predictions(entity_dicts: List[dict]) -> dict:
    combined = None

    for entities in entity_dicts:
        if combined is None:
            combined = entities
        else:
            for ent_key in entities.keys():
                new_values = entities[ent_key]
                previous = []
                if ent_key in combined:
                    previous = combined[ent_key]
                value_points = {}
                for value, points in new_values + previous:
                    if value in value_points:
                        if points > value_points[value]:
                            value_points[value] = points
                    else:
                        value_points[value] = points
                updated_values = [(key, p) for key, p in value_points.items()]
                updated_values.sort(key=lambda xy: xy[1], reverse=True)
                combined[ent_key] = updated_values
    return combined


class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    def initialize(self, args):
        """
        Carrega o modelo da Hugging Face.
        """
        if "model_id" in args:
            self.model_id = args["model_id"]
        else:
            self.model_id = "numind/NuExtract-2.0-4B"

        print(f"Carregando modelo {self.model_id}...")
        self.model, self.processor = load_nuextract_cuda(self.model_id)

    def find_labels(self, labels_str, schema_str, transcript):
        try:
            labels_list = json.loads(labels_str.replace('\"', '"'))
        except json.JSONDecodeError as err:
            labels_list = labels_str.split(",")
            labels_list = {
                label.strip(): label.strip().replace('_', ' ').title() 
                for label in labels_list if label.strip() != ""
            }

        try:
            classification_schema = json.loads(schema_str.replace('\\"', '"'))
        except json.JSONDecodeError as err:
            raise Exception("Invalid classification schema: " + str(err))
        nuextract_schema = {}
        for label_name, label_desc in labels_list.items():
            nuextract_schema[label_name] = "verbatim-string"
        
        for label_group, labels_list in classification_schema.items():
            for label_raw in labels_list:
                parts = label_raw.split("::")
                label_name = parts[0]
                options = parts[1].strip('[').strip(']').split(',')
                data_type = parts[2]
                desc = parts[3]
                
                nuextract_schema[label_name] = options

        
        #print(labels_list, file=sys.stderr)
        req_start = time.time()
        
        template = json.dumps(nuextract_schema, indent=2, ensure_ascii=False)
        document = transcript

        # prepare the user message content
        messages = [
            {
                "role": "system",
                "content": "You are NuExtract, an information extraction tool created by NuMind." 
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": document}]
            }
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            template=template, # template is specified here
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

        generated_ids = self.model.generate(
            **inputs,
            **generation_config
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_dict = json.loads(output_text[0])
        
        infer_finish = time.time()
        req_duration = infer_finish - req_start

        entities_dict = {}
        entities = output_dict
        for ent_label, entity_values in entities.items():
            for entity in entity_values:
                new_val = entity.strip().replace("\n", " ").replace("\r", "")
                if new_val != "":
                    if not ent_label in entities_dict:
                        entities_dict[ent_label] = []
                    entities_dict[ent_label].append(
                        (new_val, 1.0)
                    )

        for key in entities_dict:
            non_redundant_lower = []
            non_redundant = []
            for val, score in entities_dict[key]:
                lower = val.lower()
                if lower not in non_redundant_lower:
                    non_redundant_lower.append(lower)
                    non_redundant.append((val, score))
            entities_dict[key] = non_redundant

        return entities_dict, req_duration

    def execute(self, requests):
        """
        Chamado para cada BATCH de inferência.

        Processa cada request individualmente (mas dentro do batch).
        """
        responses = []

        # O Triton agrupa N requests (definido pelo dynamic_batching).
        # Iteramos por cada request no batch.
        for request in requests:
            try:
                proc_start = time.time()
                # 2. EXTRAIR OS INPUTS DO TRITON
                # Extrai o PROMPT (texto)
                input_text_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                # Acessa o valor do array NumPy e decodifica.
                # Para shape [1, 1], o valor está em [0, 0].
                transcript = input_text_tensor.as_numpy()[0, 0].decode("utf-8")

                # Extrai a LABEL_LIST (string JSON)
                labels_tensor = pb_utils.get_input_tensor_by_name(request, "LABEL_LIST")
                # Acessa o valor do array NumPy e decodifica.
                labels_str = labels_tensor.as_numpy()[0, 0].decode("utf-8")

                # Extrai a CLASSIFICATION_SCHEMA (string JSON)
                classification_schema_str = pb_utils.get_input_tensor_by_name(
                    request, "CLASSIFICATION_SCHEMA"
                )
                # Acessa o valor do array NumPy e decodifica.
                classification_schema_str = (
                    classification_schema_str.as_numpy()[0, 0]
                    .decode("utf-8")
                )

                new_entities, req_duration2 = self.find_labels(
                    labels_str, classification_schema_str, transcript
                )

                # Contando comprimento do input e output
                n_input_tokens = len(transcript.split())
                output_str = json.dumps(new_entities, ensure_ascii=False)
                n_output_tokens = len(output_str.split())

                total_duration = time.time() - proc_start
                post_processing_duration = (
                    total_duration - req_duration1 - req_duration2
                )
                meta = {
                    "processing_time": req_duration1 + req_duration2,
                    "no_gpu_time": post_processing_duration,
                    "input_tokens": n_input_tokens,
                    "output_tokens": n_output_tokens,
                    "model_name": "nuextract",
                }
                meta_str = json.dumps(meta, ensure_ascii=False)

                # CRIAR OS TENSORES DE SAÍDA
                # Converte as strings de saída de volta para tensores Triton
                output_entities_tensor = pb_utils.Tensor(
                    "ENTITIES_JSON", np.array([output_str], dtype=np.object_)
                )
                output_meta_tensor = pb_utils.Tensor(
                    "META_INFO", np.array([meta_str], dtype=np.object_)
                )

                # CRIAR A RESPOSTA
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_entities_tensor, output_meta_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                # Se algo der errado (ex: JSON mal formatado), retorna um erro
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                )
                responses.append(error_response)

        # 6. RETORNAR A LISTA DE RESPOSTAS
        return responses

    def finalize(self):
        """
        Chamado quando o modelo é descarregado.
        """
        print("Limpando o modelo...")
        self.model = None
        torch.cuda.empty_cache()
        print("Finalizado.")
