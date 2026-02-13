import time
import os
import json
import sys
from typing import List
from typing import List
from multiprocessing import Pool
import gc
import re
from copy import copy
import numpy as np
from tqdm import tqdm
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance.JaroWinkler import (
    normalized_similarity as normalized_similarity1,
)
from rapidfuzz.distance.Levenshtein import (
    normalized_similarity as normalized_similarity2,
)
from unidecode import unidecode


entity_redundancies = {
    "rua": "rua_ou_logradouro",
    "municipio": "cidade",
    "street_number": "numero",
    "number": "numero",
    "endereço_complemento": "complemento",
}

SIM_TH = 0.71
NAME_SIM_TH = 0.7
NUMBER_TH = 0.95
USE_PERC = 0.96

HONORIFICS = [
    r"\bdra\b",
    r"\bdr\b",
    r"\bdr\.\b",
    r"\bsr\b",
    r"\bsr\.\b",
    r"\bsra\b",
    r"\bsra\.\b",
    r"\bsenhor\b",
    r"\bsenhora\b",
    r"\bprof\b",
    r"\bprof\.\b",
]


def basic_norm(t: str, name=None):
    t2 = copy(t)
    t2 = t2.lower()
    t2 = unidecode(t2).replace("...", "").strip(".,;:-").strip()
    t3 = []
    for w in t2.split():
        repeated = False
        if len(t3) > 0:
            if w.strip(".,") == t3[-1].strip(".,"):
                repeated = True
        if not repeated:
            t3.append(w)
    t2 = " ".join(t3)

    if name != None:
        if "pessoa" in name or "nome" in name:
            for h in HONORIFICS:
                t2 = re.sub(h, "", t2)
        if "numero" in name:
            t2 = name + ": " + t2
            t2 = t2.replace("numero ", "")
    # t = t.replace('dr.', 'doutor')
    return t2


diffs_printed = set()


def normalized_similarity_custom(str1: str, str2: str, field_name: str = None) -> float:
    str1_norm = str1
    str2_norm = basic_norm(str2, field_name)
    if "nome" in field_name or "pessoa" in field_name:
        sim = fuzz.partial_ratio(str2_norm, str1_norm) / 100.0
        th = NAME_SIM_TH
    else:
        sim1 = normalized_similarity1(str1_norm, str2_norm)
        sim2 = normalized_similarity2(str1_norm, str2_norm)
        sim = (sim1 + sim2) / 2
        th = SIM_TH
    if "numero" in field_name:
        th = NUMBER_TH

    equals = sim >= th
    if abs(sim - th) < 0.04:
        if not (str1_norm, str2_norm, field_name) in diffs_printed:
            diffs_printed.add((str1_norm, str2_norm, field_name))
            print(
                f"{field_name} | {str1_norm} | {str2} -> {str2_norm} | {sim} | {equals}"
            )

    return sim, equals


def _process_single_column_fmax(args):
    """
    Worker function to process a single column.
    Must be top-level to be picklable by multiprocessing.
    """
    score_col, label_col, col_id = args

    # Optimization: Pre-calculate total positives for recall denominator
    # This avoids summing the label column 150 times
    total_positives = np.sum(label_col)

    # If no positive labels exist for this class, F1 is 0 by definition
    if total_positives == 0:
        return col_id, {
            "f1": 1.0,
            "th": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "recall_at_good_precision": 1.0,
        }

    thresholds = np.linspace(0, 1, 150)
    best_f1 = 0.0
    best_th = 0.0
    best_recall = 0.0
    best_precision = 0.0

    best_recall_at_good_precision = 0.0
    good_precision = 0.8

    # Optimization: We can broadcast the threshold comparison if memory allows,
    # but iterating 150 times with raw numpy is fast enough and memory-safe.
    for th in thresholds:
        # Create binary predictions
        pred_bin = score_col > th

        # Fast Numpy Calculation of TP, FP, FN
        # True Positives: Where both pred and label are 1
        tp = np.sum(pred_bin & (label_col == 1))

        # Predicted Positives (TP + FP)
        pred_pos_count = np.sum(pred_bin)

        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN) -> TP / total_positives

        if pred_pos_count == 0:
            prec = 0.0
        else:
            prec = tp / pred_pos_count

        rec = tp / total_positives

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_recall = rec
            best_precision = prec

        if prec >= good_precision and rec > best_recall_at_good_precision:
            best_recall_at_good_precision = rec

    result_dict = {
        "f1": best_f1,
        "th": best_th,
        "recall": best_recall,
        "precision": best_precision,
        "recall_at_good_precision": best_recall_at_good_precision,
    }

    return col_id, result_dict


def find_fmax_per_col_parallel(
    scores_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    col_ids: list,
    n_jobs=12,
) -> dict:

    if scores_matrix.shape != labels_matrix.shape:
        raise ValueError("Scores and labels matrices must have the same shape.")

    n_samples, n_labels = scores_matrix.shape

    print("scores_matrix.shape", scores_matrix.shape)
    print("labels_matrix.shape", labels_matrix.shape)
    print("col_ids", col_ids)

    if len(col_ids) != n_labels:
        raise ValueError("Column IDs must match number of labels.")

    # print(f"Starting parallel processing on {n_jobs} cores...")

    # Prepare arguments as tuples for the worker function
    # We transpose (.T) to iterate over columns easily
    tasks = zip(scores_matrix.T, labels_matrix.T, col_ids)

    fmax_per_col = {}
    recalls_at_good_precisions = {}
    recalls = {}
    precisions = {}
    best_thresholds = {}

    # Use multiprocessing Pool
    with Pool(processes=n_jobs) as pool:
        # imap_unordered is often faster as it yields results as soon as they finish
        # We wrap it in tqdm for a progress bar
        results = list(
            tqdm(pool.imap(_process_single_column_fmax, tasks), total=n_labels)
        )

    # Unpack results
    print("Number of results", len(results))
    for r in results:
        col_id = r[0]
        result_dict = r[1]
        fmax_per_col[col_id] = result_dict["f1"]
        best_thresholds[col_id] = result_dict["th"]
        recalls_at_good_precisions[col_id] = result_dict["recall_at_good_precision"]
        recalls[col_id] = result_dict["recall"]
        precisions[col_id] = result_dict["precision"]

        # print(f"Best threshold for {col_id}: {th} (F1: {f1})")

    return (
        fmax_per_col,
        recalls_at_good_precisions,
        recalls,
        precisions,
        best_thresholds,
    )


def filter_entities_by_score(entities_with_scores: list, threshold: float) -> list:
    """
    Filters entities based on their score.

    Args:
        entities_with_scores: List of tuples (entity, score)
        threshold: Minimum score to keep the entity

    Returns:
        List of entities that meet the threshold
    """
    return [e for e, s in entities_with_scores if s >= threshold]


def compare_values(
    pred_entities: List[str], true_entities: List[str], field_name: str = None
) -> bool:
    """
    Compares predicted entities with true entities.

    Args:
        pred_entities: List of predicted entities
        true_entities: List of true entities

    Returns:
        True if predicted entities match true entities, False otherwise
    """

    true_entities_norm = [basic_norm(t, field_name) for t in true_entities]
    true_entities_norm = [t for t in true_entities_norm if t != ""]

    if len(true_entities_norm) == 0 and len(pred_entities) == 0:
        return 1.0, 0, 0, 0  # sim, fp, tp, fn
    elif len(true_entities_norm) == 0 and len(pred_entities) > 0:
        return 0.0, len(pred_entities), 0, 0  # sim, fp, tp, fn
    elif len(true_entities_norm) > 0 and len(pred_entities) == 0:
        return 0.0, 0, 0, len(true_entities_norm)  # sim, fp, tp, fn
    else:
        best_matches = []
        true_with_equals = []
        true_without_equals = []

        preds_no_match = set(pred_entities)

        for true_entity in true_entities_norm:
            best_score = 0
            best_match = None
            equal_found = False
            for pred_entity in pred_entities:
                score, equals = normalized_similarity_custom(
                    true_entity, pred_entity, field_name
                )
                if score > best_score:
                    best_score = score
                    best_match = pred_entity
                    equal_found = equals
            if best_match in preds_no_match:
                preds_no_match.remove(best_match)
            best_matches.append(best_score)
            if equal_found:
                true_with_equals.append(true_entity)
            else:
                true_without_equals.append(true_entity)

        fp = len(preds_no_match)
        tp = len(true_with_equals)
        fn = len(true_without_equals)

        return np.mean(best_matches), fp, tp, fn


def find_max_jw_sim(pred_values, true_values, field_name: str = None):
    thresholds = np.linspace(0, 1, 150)
    best_threshold = 0
    best_sim = 0
    th_results = []
    for th in thresholds:
        filtered_pred_values = [
            filter_entities_by_score(pred_line, th) for pred_line in pred_values
        ]
        metrics = [
            compare_values(pred, t, field_name)
            for pred, t in zip(filtered_pred_values, true_values)
        ]
        sims = [m[0] for m in metrics]
        fp = sum([m[1] for m in metrics])
        tp = sum([m[2] for m in metrics])
        fn = sum([m[3] for m in metrics])

        # avoid float division by zero
        if tp + fp == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        sim = np.mean(sims)

        th_results.append((th, sim, fp, tp, fn, precision, recall, f1))

    th_results_by_f1 = sorted(th_results, key=lambda x: x[7], reverse=True)
    fmax = th_results_by_f1[0][7]
    th_results_by_jw_sim = sorted(th_results, key=lambda x: x[1], reverse=True)
    jw_sim_max = th_results_by_jw_sim[0][1]

    recall_at_fmax = th_results_by_f1[0][6]
    precision_at_fmax = th_results_by_f1[0][5]
    recall_at_jw_sim = th_results_by_jw_sim[0][6]
    precision_at_jw_sim = th_results_by_jw_sim[0][5]

    recall = max(recall_at_fmax, recall_at_jw_sim)
    precision = max(precision_at_fmax, precision_at_jw_sim)

    return fmax, jw_sim_max, recall, precision


def get_testing_inputs():
    text1 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
        Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. """
    text2 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, 
        perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, 
        moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text3 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Ela está inconsciente. Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text4 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Sim, o marido violento espancou ela!."""
    text5 = """Socorro, preciso de uma viatura rápido! Estão atirando em mim! Quando? Agora mesmo! 
        Ele tem uma pistola! Meu nome é Maria Oliveira. Meu telefone é 99999-3888, moro na cidade de Natal. 
        Repito: o Claudio começou a atirar em mim com uma pistola, do nada"""
    text6 = """Quero avisar de um risco de tumulto. Meu nome é Maria Oliveira. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Há uma manifestação em frente ao shopping. Há muitas pessoas e veículos.
    Parece que vai começar uma briga. """
    text7 = """Alô? É da ambulancia? Meu pai levou um tiro na rua. Ele chegou em casa sangrando e disse que foi tiro. 
    Não, ele não sabe quem foi. Aconteceu a mais ou menos 1 hora. Meu nome é Pedro. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Meu pai é o Claudio. ele disse que foi na rua salgado filho, número 123, perto do shopping.
    Mas nós moramos em outro lugar. O endereço da nossa casa? É rua capitão mor golveia, número 450.
    """
    all_texts = [text1, text2, text3, text4, text5, text6, text7]

    fatoOcorrendoNesteMomento = "Fato Ocorrendo Neste Momento ?"
    autorDoFatoNoLocal = "Autor Do Fato No Local ?"
    autorDoFatoArmado = "Autor Do Fato Armado ?"
    feridosComRiscoDeMorte = "Feridos Com Risco de Morte ?"
    riscoDeTumulto = "Risco De Tumulto ?"
    leiMariaPenha = "Lei Maria da Penha ?"

    clfnames = [
        fatoOcorrendoNesteMomento,
        autorDoFatoNoLocal,
        autorDoFatoArmado,
        feridosComRiscoDeMorte,
        riscoDeTumulto,
        leiMariaPenha,
    ]
    clfnames_hf = [
        "fatoOcorrendoNesteMomento",
        "autorDoFatoNoLocal",
        "autorDoFatoArmado",
        "feridosComRiscoDeMorte",
        "riscoDeTumulto",
        "leiMariaPenha",
    ]

    correct_clfs = [
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: False,
            leiMariaPenha: True,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: True,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: True,
            autorDoFatoArmado: True,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: True,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: True,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: True,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: False,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: True,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
    ]

    texts_hf = []
    correct_clfs_hf = []
    true_entities_hf = []

    hf_df = pl.read_parquet("input/dataset_filtrado.parquet")
    for row in hf_df.iter_rows(named=True):
        roteiro_str = "\n".join([" - ".join(p) for p in row["roteiro_segmentado"]])
        texts_hf.append(roteiro_str)
        clfs_line = np.array([int(row[c]) for c in clfnames_hf])
        correct_clfs_hf.append(clfs_line)

        pessoas = []
        for part in row["participacoes"]:
            new_person = part["pessoa"].replace("pessoa:", "").strip()
            if len(new_person) > 1:
                pessoas.append(new_person)
        true_entities = {"pessoa": pessoas}

        name_pairs = [
            ("nome_solicitante", "nome_do_solicitante"),
            ("rua", "rua_ou_logradouro"),
            ("complemento", "complemento"),
            ("numero", "numero"),
            ("bairro", "bairro"),
            ("cidade", "cidade"),
            ("estado", "estado"),
            ("ponto_de_referencia", "ponto_de_referencia"),
        ]

        for df_col, true_entity_name in name_pairs:
            true_entities[true_entity_name] = []
            if type(row[df_col]) == str:
                if len(row[df_col]) > 0:
                    true_entities[true_entity_name].append(row[df_col])
        # print(true_entities)
        true_entities_hf.append(true_entities)

    """for txt, clfs in zip(all_texts, correct_clfs):
        texts_hf.append(txt)
        correct_clfs_hf.append([int(clfs[c]) for c in clfnames])"""

    correct_clfs_hf = np.asarray(correct_clfs_hf)

    """"natureza_da_ocorrencia": {
        "Fato Ocorrendo Neste Momento": {"desc": "Se o fato relatado está ocorrendo neste momento", 
            "sim": "Fato Ocorrendo Neste Momento", "nao": "Não Ha Fato Ocorrendo Neste Momento"}, 
        "Autor Do Fato No Local": {"desc": "Se o autor (culpado/acusado) do fato está no local", 
            "sim": "Autor Do Fato No Local", "nao": "Não Ha Autor Do Fato No Local"}, 
        "Autor Do Fato Armado": {"desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado", 
            "sim": "Autor Do Fato Armado", "nao": "Não Ha Autor Do Fato Armado"}, 
        "Feridos Com Risco de Morte": {"desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Feridos Com Risco de Morte", "nao": "Não Ha Risco de Morte"}, 
        "Risco De Tumulto": {"desc": "Se a ocorrência envolve um risco de tumulto", 
            "sim": "Risco De Tumulto", "nao": "Não Ha Risco De Tumulto"}, 
        "Lei Maria da Penha": {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal", 
            "sim": "É um caso de Maria da Penha", "nao": "Não é um caso de Maria da Penha"
        },
    }"""

    clfs_bool = {
        fatoOcorrendoNesteMomento: {
            "desc": "Se o fato relatado está ocorrendo neste momento",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoNoLocal: {
            "desc": "Se o autor (culpado/acusado) do fato está no local",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoArmado: {
            "desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado",
            "sim": "Sim",
            "nao": "Não",
        },
        feridosComRiscoDeMorte: {
            "desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Sim",
            "nao": "Não",
        },
        riscoDeTumulto: {
            "desc": "Se a ocorrência envolve um risco de tumulto",
            "sim": "Sim",
            "nao": "Não",
        },
        leiMariaPenha: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "Sim",
            "nao": "Não",
        },
    }

    clfs_named = {
        fatoOcorrendoNesteMomento: {
            "desc": "Se o fato relatado está ocorrendo neste momento",
            "sim": "Fato Ocorrendo Neste Momento",
            "nao": "Não Ha Fato Ocorrendo Neste Momento",
        },
        autorDoFatoNoLocal: {
            "desc": "Se o autor (culpado/acusado) do fato está no local",
            "sim": "Autor Do Fato No Local",
            "nao": "Não Ha Autor Do Fato No Local",
        },
        autorDoFatoArmado: {
            "desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado",
            "sim": "Autor Do Fato Armado",
            "nao": "Não Ha Autor Do Fato Armado",
        },
        feridosComRiscoDeMorte: {
            "desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Feridos Com Risco de Morte",
            "nao": "Não Ha Risco de Morte",
        },
        riscoDeTumulto: {
            "desc": "Se a ocorrência envolve um risco de tumulto",
            "sim": "Risco De Tumulto",
            "nao": "Não Ha Risco De Tumulto",
        },
        leiMariaPenha: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "É um caso de Maria da Penha",
            "nao": "Não é um caso de Maria da Penha",
        },
    }

    full_schema_dict = {
        "entities": {
            "rua_ou_logradouro": "Nome de rua, logradouro ou avenida",
            "rua": "Nome de rua",
            "bairro": "Nome de bairro",
            "municipio": "Nome de município",
            "cidade": "Nome de cidade",
            "ponto_de_referencia": "Nome de ponto de referência do endereço",
            "nome_do_solicitante": "Nome do solicitante (pessoa que está fazendo o chamado)",
            "pessoa": "Nome de pessoa ou participante",
            "numero": "Número do endereço",
            "street_number": "Street Number",
            "number": "Número",
            "complemento": "Complemento do endereço (Exemplos: apt 301, casa A, etc)",
            "endereço_complemento": "Número ou código da casa, apartamento ou loja naquele endereço",
        },
        "boolean": {"natureza_da_ocorrencia": clfs_bool},
    }

    rac_perc = 1.0 - USE_PERC
    # set default seed:
    np.random.seed(1337)
    rac_indexes = np.random.choice(
        len(texts_hf), int(len(texts_hf) * rac_perc), replace=False
    )

    rac_texts = [texts_hf[i] for i in rac_indexes]
    rac_clfs_true = np.asarray([correct_clfs_hf[i] for i in rac_indexes])
    print("rac_indexes", rac_indexes)

    pred_indexes = [i for i in range(len(texts_hf)) if i not in rac_indexes]

    print("pred_indexes", pred_indexes)

    texts_hf = [texts_hf[i] for i in pred_indexes]
    correct_clfs_hf = np.asarray([correct_clfs_hf[i] for i in pred_indexes])
    true_entities_hf = [true_entities_hf[i] for i in pred_indexes]

    examples = []
    for txt, clfs in zip(rac_texts, rac_clfs_true):
        all_clfs = [c.replace(" ?", "") for c in clfnames]
        true_examples = [c for n, c in enumerate(all_clfs) if int(clfs[n]) > 0]
        # print(true_examples)
        examples.append(
            {"text": txt, "true_labels": true_examples, "all_labels": all_clfs}
        )

    ml_categories = list(full_schema_dict["boolean"]["natureza_da_ocorrencia"].keys())

    return (
        ml_categories,
        texts_hf,
        clfnames,
        correct_clfs_hf,
        full_schema_dict,
        true_entities_hf,
    )


def calc_metadata(input_strs, output_strs, gpu_usage_secs):
    input_tokens = sum([len(s.split()) for s in input_strs])
    output_tokens = sum([len(s.split()) for s in output_strs])
    return input_tokens / gpu_usage_secs, output_tokens / gpu_usage_secs


def test_gliner_model(model) -> dict:
    ml_categories, all_texts, clfnames, clfs_true, full_schema_dict, entities_true = (
        get_testing_inputs()
    )
    schema_str = json.dumps(full_schema_dict, ensure_ascii=False, indent=2)
    # print("classification schema:", schema_str)

    entity_names = list(full_schema_dict["entities"].keys())
    non_redundant_entity_names = [
        name for name in entity_names if name not in entity_redundancies.keys()
    ]

    clfs_scores = []
    entities_found = []
    gpu_usage_total = 0.0
    output_strs = []
    for transcript in tqdm(all_texts):
        # for transcript in [text5]:
        # print('\n\nTexto: ', transcript)
        new_entities, req_duration1 = model.single_inference(transcript, schema_str)
        output_strs.append(json.dumps(new_entities, ensure_ascii=False, indent=2))
        gpu_usage_total += req_duration1
        # print('\tLabels encontradas: ', new_entities)
        scores_line = [new_entities[c] if c in new_entities else 0.0 for c in clfnames]
        clfs_scores.append(scores_line)
        entities_line = {
            name: new_entities[name] for name in entity_names if name in new_entities
        }
        for redundant, correct_name in entity_redundancies.items():
            if redundant in entities_line:
                if not correct_name in entities_line:
                    entities_line[correct_name] = []
                entities_line[correct_name].extend(entities_line[redundant])
                del entities_line[redundant]
        entities_found.append(entities_line)
        # print(entities_found[-1])

    tokens_per_second_in, tokens_per_second_out = calc_metadata(
        all_texts, output_strs, gpu_usage_total
    )

    fmax_per_col, recalls_at_good_precisions, recalls, precisions, best_thresholds = (
        find_fmax_per_col_parallel(
            np.array(clfs_scores),
            clfs_true,
            clfnames,
            n_jobs=4,
        )
    )

    similarities_per_entity = {}
    for entity_name in non_redundant_entity_names:
        pred_values = [
            entities_found[i][entity_name] if entity_name in entities_found[i] else []
            for i in range(len(entities_found))
        ]
        true_values = [entities_true[i][entity_name] for i in range(len(entities_true))]

        """print(entity_name)
        print('\tpreds', pred_values)
        print('\ttrue', true_values)"""

        fmax, jw_sim_max, recall, precision = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name
        )
        similarities_per_entity[entity_name] = jw_sim_max
        fmax_per_col[entity_name] = fmax
        recalls[entity_name] = recall
        precisions[entity_name] = precision

    result_dicts = {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "precisions": precisions,
        "best_thresholds": best_thresholds,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "gpu_seconds": gpu_usage_total,
        },
    }

    return result_dicts


if __name__ == "__main__":
    from gliner2_model import TritonPythonModel as TritonPythonModelGliner2
    from gliner1_model import TritonPythonModel as TritonPythonModelGliner1
    import torch

    results = []

    results_path = sys.argv[1]

    prev_res = None
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            prev_res = json.load(f)

    if prev_res is None:
        prev_res = []

    for prev_calc in prev_res:
        results.append(prev_calc)

    models_calculated_quality = [r["model"] for r in prev_res]

    gliclass_names = [
        #'knowledgator/gliclass-modern-base-v2.0',
        #'knowledgator/gliclass-modern-large-v2.0',
        "knowledgator/gliclass-llama-1.3B-v1.0",
        "knowledgator/gliclass-qwen-1.5B-v1.0",
        "knowledgator/gliclass-x-base",
        "BioMike/gliclass-large-reddit-1m-6k",
        "knowledgator/gliclass_msmarco_merged",
        "knowledgator/gliclass-base-v3.0",
        "knowledgator/gliclass-large-v3.0",
        "knowledgator/gliclass-base-v2.0-rac-init",
        #'knowledgator/gliclass-modern-large-v3.0',
    ]

    glinerv1_names = [
        "knowledgator/gliner-x-large",
        "nvidia/gliner-pii",
        "knowledgator/gliner-pii-large-v1.0",
        "gliner-community/gliner_xxl-v2.5",
        "gliner-community/gliner_large-v2.5",
        "gretelai/gretel-gliner-bi-large-v1.0",
        "knowledgator/gliner-bi-large-v2.0",
        "knowledgator/gliner-bi-large-v1.0",
        "knowledgator/gliner-multitask-v1.0",
    ]

    glinerv2_names = ["fastino/gliner2-multi-v1", "fastino/gliner2-large-v1"]

    for glinerv1_name in glinerv1_names:
        for gliclass_name in gliclass_names:
            comb_name = glinerv1_name + " + " + gliclass_name
            if comb_name in models_calculated_quality:
                continue
            try:
                model_gli1 = TritonPythonModelGliner1()
                model_gli1.initialize(
                    {"clf_model_id": gliclass_name, "model_id": glinerv1_name}
                )

                result_dicts = test_gliner_model(model_gli1)

                for col_name, fmax in result_dicts["fmax_per_col"].items():
                    print(f"\tFmax para {col_name}: {fmax}")

                for col_name, sim in result_dicts["similarities_per_entity"].items():
                    print(f"\tJW Sim para {col_name}: {sim}")

                mean_fmax = np.mean(list(result_dicts["fmax_per_col"].values()))
                mean_jw_sim = np.mean(
                    list(result_dicts["similarities_per_entity"].values())
                )
                mean_good_recall = np.mean(
                    list(result_dicts["recalls_at_good_precisions"].values())
                )
                mean_recall = np.mean(list(result_dicts["recalls"].values()))
                mean_precision = np.mean(list(result_dicts["precisions"].values()))

                print(f"\tMedia Fmax: {mean_fmax}")
                print(f"\tMedia JW Sim: {mean_jw_sim}")
                print(f"\tMedia Good Recall: {mean_good_recall}")
                print(f"\tMedia Recall: {mean_recall}")
                print(f"\tMedia Precision: {mean_precision}")
                results.append(
                    {
                        "model": comb_name,
                        "mean_metrics": {
                            "fmax": mean_fmax,
                            "jw_sim": mean_jw_sim,
                            # "recall_at_95_precision": mean_good_recall,
                            "recall": mean_recall,
                            "precision": mean_precision,
                        },
                        "meta": result_dicts["meta"],
                        "fmax": result_dicts["fmax_per_col"],
                        "jw_sim": result_dicts["similarities_per_entity"],
                        # "recall_at_95_precision": result_dicts["recalls_at_good_precisions"],
                        "recall": result_dicts["recalls"],
                        "precision": result_dicts["precisions"],
                        "best_thresholds": result_dicts["best_thresholds"],
                    }
                )
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

                del model_gli1
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with model {gliclass_name}: {e}")
                del model_gli1
                gc.collect()
                torch.cuda.empty_cache()
                continue

    for m_name in glinerv2_names:
        if m_name in models_calculated_quality:
            continue
        model_gli2 = TritonPythonModelGliner2()
        model_gli2.initialize({"model_id": m_name})

        result_dicts = test_gliner_model(model_gli2)

        del model_gli2
        gc.collect()
        torch.cuda.empty_cache()
        for col_name, fmax in result_dicts["fmax_per_col"].items():
            print(f"\tFmax para {col_name}: {fmax}")

        mean_fmax = np.mean(list(result_dicts["fmax_per_col"].values()))
        mean_jw_sim = np.mean(list(result_dicts["similarities_per_entity"].values()))
        mean_good_recall = np.mean(
            list(result_dicts["recalls_at_good_precisions"].values())
        )
        mean_recall = np.mean(list(result_dicts["recalls"].values()))
        mean_precision = np.mean(list(result_dicts["precisions"].values()))

        print(f"\tMedia Fmax: {mean_fmax}")
        print(f"\tMedia JW Sim: {mean_jw_sim}")
        # print(f"\tMedia Good Recall: {mean_good_recall}")
        print(f"\tMedia Recall: {mean_recall}")
        print(f"\tMedia Precision: {mean_precision}")

        results.append(
            {
                "model": m_name,
                "mean_metrics": {
                    "fmax": mean_fmax,
                    "jw_sim": mean_jw_sim,
                    # "recall_at_95_precision": mean_good_recall,
                    "recall": mean_recall,
                    "precision": mean_precision,
                },
                "meta": result_dicts["meta"],
                "fmax": result_dicts["fmax_per_col"],
                "jw_sim": result_dicts["similarities_per_entity"],
                # "recall_at_95_precision": result_dicts["recalls_at_good_precisions"],
                "recall": result_dicts["recalls"],
                "precision": result_dicts["precisions"],
                "best_thresholds": result_dicts["best_thresholds"],
            }
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    # Save results to json:

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
