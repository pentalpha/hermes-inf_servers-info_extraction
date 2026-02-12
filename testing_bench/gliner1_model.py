import time
import json
import sys
from typing import List
from collections import deque
from itertools import islice
from typing import List
import numpy as np
from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
from tqdm import tqdm

#MAX_TOKENS = 128
#MAX_TOKENS = 384
MAX_TOKENS = 340

def load_gliner_cuda(model_name_str):
    word_splitter_name = 'spacy'
    word_splitter = WordsSplitter(splitter_type=word_splitter_name)
    
    try:
        gliner_large_model = GLiNER.from_pretrained(model_name_str).to('cuda')
        print('Loaded GLiNER model')
    except Exception as e:
        print(f"Error loading GLiNER model with CUDA: {e}")
        raise(e)
    
    return gliner_large_model, word_splitter

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
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def sliding_window_over_paragraph(text, n_words=160, sobreposicao=12, fillvalue=''):
    windows = [' '.join(x).strip()
               for x in sliding_window(text.split(' '), 
                    size=n_words, step=n_words-sobreposicao, fillvalue=fillvalue)]
    return windows

def join_entity_predictions(entity_dicts: List[dict]) -> dict:
    combined = None

    for entities in entity_dicts:
        if combined is None:
            combined = entities
        else:
            for ent_key in entities.keys():
                new_values = entities[ent_key]
                if type(new_values) == list:
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
                elif type(new_values) in [int, float]:
                    if ent_key in combined:
                        old_value = combined[ent_key]
                        if old_value != old_value or old_value is None:
                            old_value = -1
                    else:
                        old_value = -1
                    if new_values > old_value:
                        combined[ent_key] = new_values

    return combined

def join_entity_predictions_mean(entity_dicts: List[dict]) -> dict:
    combined = {}
    list_accumulators = {}
    scalar_accumulators = {}

    for entities in entity_dicts:
        for ent_key, new_values in entities.items():
            if type(new_values) == list:
                if ent_key not in list_accumulators:
                    list_accumulators[ent_key] = {}
                for value, points in new_values:
                    if value not in list_accumulators[ent_key]:
                        list_accumulators[ent_key][value] = []
                    list_accumulators[ent_key][value].append(points)
            elif type(new_values) in [int, float]:
                if ent_key not in scalar_accumulators:
                    scalar_accumulators[ent_key] = []
                scalar_accumulators[ent_key].append(new_values)

    for ent_key, values_map in list_accumulators.items():
        updated_values = []
        for value, points_list in values_map.items():
            mean_points = sum(points_list) / len(points_list)
            updated_values.append((value, mean_points))
        updated_values.sort(key=lambda xy: xy[1], reverse=True)
        combined[ent_key] = updated_values

    for ent_key, values_list in scalar_accumulators.items():
        if values_list:
            combined[ent_key] = sum(values_list) / len(values_list)

    return combined

class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    DEFAULT_GLINERX = "knowledgator/gliner-x-large"

    def initialize(self, args):
        """
        Carrega o modelo da Hugging Face.
        """
        if "model_id" in args:
            self.model_id = args["model_id"]
        else:
            self.model_id = TritonPythonModel.DEFAULT_GLINERX

        if "clf_model_id" in args:
            self.clf_model_id = args["clf_model_id"]
        else:
            self.clf_model_id = "knowledgator/gliclass-large-v3.0"

        print(f"Carregando modelo {self.model_id}...")
        self.model, self.word_splitter = load_gliner_cuda(self.model_id)

        self.clf_model = GLiClassModel.from_pretrained(self.clf_model_id)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(self.clf_model_id)

        self.clf_pipeline = ZeroShotClassificationPipeline(
            self.clf_model, self.clf_tokenizer, 
            classification_type='multi-label', device='cuda:0',
            progress_bar=False
        )
        #self.clf_pipeline.set_progress_bar_config(disable=True)

    def find_labels(self, labels_list, transcript):
        
        labels_list = list(labels_list.keys()) if type(labels_list) == dict else labels_list
        req_start = time.time()
        entities = self.model.predict_entities(transcript, labels_list, 
                                                threshold=0.5, flat_ner=False)
        infer_finish = time.time()
        req_duration = infer_finish - req_start

        entities_dict = {label: [] for label in labels_list}
        for entity in entities:
            new_val = entity["text"].strip().replace('\n', ' ').replace('\r', '')
            if new_val != '':
                ent_label = entity["label"]
                entities_dict[ent_label].append((new_val, round(entity['score']*100, 3)))

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

    def classify_by_schema_ml(self, classification_schema, transcript, examples):

        try:
            req_start = time.time()
            name_translator = {n.replace(' ?', ''): n for n in classification_schema}
            no_interrogation_labels = list(name_translator.keys())
            examples = []
            if len(examples) == 0:
                results = self.clf_pipeline(transcript, no_interrogation_labels, threshold=0.001)[0]
            else:
                results = self.clf_pipeline(transcript, no_interrogation_labels, 
                    threshold=0.001, rac_examples=examples)[0]
            '''new_entities = self.model.extract_json(
                transcript,
                classification_schema,
                threshold=0.01,
                include_confidence=True,
                include_spans=False,
            )'''
            new_entities = [
                    {'label': name_translator[r['label']], 
                    'confidence': r['score']} 
                for r in results]
            #print(new_entities)
            infer_finish = time.time()
            req_duration = infer_finish - req_start

            return new_entities, req_duration
        except Exception as err:
            raise Exception(
                "Error classifying transcript: "
                + str(err)
                + "\nWith clf schema:\n"
                + str(classification_schema)
            )

    def classify_by_schema_hierarchical(self, classification_schema, transcript):
        #requires version 0.1.14 of gliclass, but has very worse results
        try:
            req_start = time.time()
            name_translator = {n.replace(' ?', ''): n for n in classification_schema.keys()}
            no_interrogation_labels = {k.replace(' ?', ''): v 
                for k, v in classification_schema.items()}
            results = self.clf_pipeline(transcript, no_interrogation_labels, 
                threshold=0.001)[0]
            infer_finish = time.time()
            req_duration = infer_finish - req_start

            results2 = {}
            for infer in results:
                label_raw = infer['label']
                score = infer['score']
                label_group = label_raw.split('.')[0]
                label_group = name_translator[label_group]
                label_name = label_raw.split('.')[1]
                if label_group not in results2:
                    results2[label_group] = {}
                results2[label_group][label_name] = score
            #print(classification_schema, file=sys.stderr)
            #print(results, file=sys.stderr)
            #print(results2, file=sys.stderr)
            #quit(1)

            return results2, req_duration
        except Exception as err:
            raise Exception(
                "Error classifying transcript: "
                + str(err)
                + "\nWith clf schema:\n"
                + str(classification_schema)
            )

    def single_inference(self, transcript_full, classification_schema_str, examples=[]):

        classification_schema_dict = json.loads(classification_schema_str)
        transcript_parts = sliding_window_over_paragraph(transcript_full, n_words=MAX_TOKENS, sobreposicao=5)
        to_join = []
        gpu_usage_secs = 0.0
        for transcript in transcript_parts:
            if "entities" in classification_schema_dict:
                entities_schema = classification_schema_dict["entities"]
                entities_dict, req_duration1 = self.find_labels(entities_schema, transcript)
            else:
                entities_dict = {}
                req_duration1 = 0
            gpu_usage_secs += req_duration1

            if "boolean" in classification_schema_dict:
                classification_schema_raw = classification_schema_dict["boolean"]

                classification_schema_1 = {}
                multilabel_schemas = {}
                for key, value in classification_schema_raw.items():
                    entity_names = []
                    for entity_name, info in value.items():
                        nome_sim = info["sim"]
                        nome_nao = info["nao"]
                        classification_schema_1[entity_name] = [nome_sim, nome_nao]
                        entity_names.append(entity_name)
                    multilabel_schemas[key] = entity_names

                #print(classification_schema_1, file=sys.stderr)
                results_multilabel0 = []
                for ml_schema_name, ml_categories in multilabel_schemas.items():
                    new_entities_raw, req_duration2 = self.classify_by_schema_ml(
                        ml_categories, transcript, examples)
                    gpu_usage_secs += req_duration2
                    categories_found = [label_info["label"] for label_info in new_entities_raw]
                    not_found = [label for label in ml_categories if label not in categories_found]
                    for label in not_found:
                        new_entities_raw.append({"label": label, "confidence": 0.0})
                    results_multilabel0 += new_entities_raw

                '''hq_entities, req_duration3 = self.classify_by_schema_hierarchical(classification_schema_1, transcript)
                gpu_usage_secs += req_duration3

                for label_name, option_scores in hq_entities.items():
                    yes_name = classification_schema_1[label_name][0]
                    no_name = classification_schema_1[label_name][1]
                    if yes_name in option_scores:
                        yes_score = option_scores[yes_name]
                    else:
                        yes_score = 0.0
                    if no_name in option_scores:
                        no_score = option_scores[no_name]
                    else:
                        no_score = 0.0
                    if yes_score > no_score:
                        results_multilabel0.append({"label": label_name, "confidence": yes_score})
                    else:
                        results_multilabel0.append({"label": label_name, "confidence": 1.0-no_score})'''
                    
                results_final = {}
                for label_info in results_multilabel0:
                    label_name = label_info["label"]
                    label_conf = label_info["confidence"]
                    if label_name not in results_final:
                        results_final[label_name] = []
                    results_final[label_name].append(label_conf)
                
                results_media = {}
                for label_name, label_confs in results_final.items():
                    results_media[label_name] = sum(label_confs) / len(label_confs)
                results_sorted = sorted([(k, results_media[k]) for k,v in results_final.items()], key=lambda x: results_media[x[0]], reverse=True)
                for label, conf in results_sorted:
                    entities_dict[label] = conf
            to_join.append(entities_dict)

        joined_entities = join_entity_predictions_mean(to_join)
        
        return joined_entities, gpu_usage_secs

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
                labels_str = labels_tensor.as_numpy()[0, 0].decode("utf-8").replace('\"', '"')

                entities_dict, req_duration1 = self.single_inference(transcript, labels_str)

                # Contando comprimento do input e output
                token_generator_input = self.word_splitter.splitter(transcript)
                n_input_tokens = len(list(token_generator_input))
                output_str = json.dumps(entities_dict, ensure_ascii=False)
                token_generator_output = self.word_splitter.splitter(output_str)
                n_output_tokens = len(list(token_generator_output))

                total_duration = time.time() - proc_start
                post_processing_duration = (
                    total_duration - req_duration1
                )
                meta = {
                    "processing_time": req_duration1,
                    "no_gpu_time": post_processing_duration,
                    "input_tokens": n_input_tokens,
                    "output_tokens": n_output_tokens,
                    "model_name": "gliner",
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
