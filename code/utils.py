import os
import json
import pytrec_eval
import torch
import torch.nn as nn
import tqdm
import gzip
import lzma
import itertools
from torch import Tensor
from typing import Dict, List, Tuple


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_metric_eval(qrel: str, run: str, metric: str):

    assert os.path.exists(qrel)
    assert os.path.exists(run)
    assert metric in pytrec_eval.supported_measures

    with open(qrel, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)

    with open(run, 'r') as f_run:
        runs = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})

    #print(runs['tqa:relative%20ages%20of%20rocks'])

    results = evaluator.evaluate(runs)
    #print(results)

    measure = pytrec_eval.compute_aggregated_measure(metric, [query_measures[metric] for query_measures in results.values()])

    return measure


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(
                    q_id + ' Q0 ' + str(value[0]) + ' ' + str(rank + 1) + ' ' + str(value[1][0][0]) + ' BERT\n')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(save_path, model):
    if save_path is None:
        return

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    print(f'Model loaded from <== {load_path}')


def prepare_result_dict(query_id, doc_id, batch_score, label, result_dict):

    for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
        if q_id not in result_dict:
            result_dict[q_id] = dict()
        if d_id not in result_dict[q_id] or (type(result_dict[q_id][d_id])==list and b_s > result_dict[q_id][d_id][0]):
            result_dict[q_id][d_id] = [b_s, l]
        else:
            if d_id not in result_dict[q_id] or (type(result_dict[q_id][d_id])==float and b_s > result_dict[q_id][d_id]):
                result_dict[q_id][d_id] = [b_s, l]

    return result_dict


def evaluate(model, data_loader, device):
    rst_dict = {}
    valid_loss = 0.0
    model.eval()

    num_batch = len(data_loader)

    with torch.no_grad():
        for dev_batch in tqdm.tqdm(data_loader, total=num_batch):
            if dev_batch is not None:
                query_id, entity_id, label = dev_batch['query_id'], dev_batch['entity_id'], dev_batch['label']
                #print(dev_batch['query_emb'].shape)
                #print(dev_batch['ent_emb'].shape)

                batch_score = model(
                    query_emb=dev_batch['query_emb'].to(device),
                    entity_emb=dev_batch['ent_emb'].to(device),
                    neighbors=dev_batch['entity_neighbors']
                )

                valid_loss += batch_score.item()

                batch_score = batch_score.detach().cpu().tolist()
                for (q_id, d_id, b_s, l) in zip(query_id, entity_id, batch_score, label):
                    if q_id not in rst_dict:
                        rst_dict[q_id] = {}
                    if d_id not in rst_dict[q_id] or (type(rst_dict[q_id][d_id])==list and b_s > rst_dict[q_id][d_id][0]):
                            rst_dict[q_id][d_id] = [b_s, l]
                    else:
                        if d_id not in rst_dict[q_id] or (type(rst_dict[q_id][d_id])==float and b_s > rst_dict[q_id][d_id]):
                            rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict, valid_loss


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_task_string(task: str) -> str:
    if task == 'retrieval':
        task_str = 'Given a web search query, retrieve relevant passages that answer the query'
        return task_str
    return None

def convert_queries_json_to_jsonl(queries_file_path: str) -> List[Dict[str, str]]:
    
    queries_converted_data = []
    
    with open(queries_file_path, 'rt', encoding='utf-8') as reader:
        data = json.load(reader)
        
    for key, value in data.items():
        temp_dict = dict()
        temp_dict['query_id'] = key
        temp_dict['query_text'] = value
        queries_converted_data.append(temp_dict)
        
    return queries_converted_data

def filter_queries_data(queries_data: List[Dict[str, str]],
                        index: List[int]) -> Dict[str, str]:
    filtered_data = dict()
    
    for ind in index:
        data = queries_data[ind]
        filtered_data[data['query_id']] = data['query_text']
        
    return filtered_data


def split_dict(data, chunk_size=500):
    """Splits a dictionary into chunks of specified size."""
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k: data[k] for k in itertools.islice(it, chunk_size)}
        
        
def read_embeddings_file(folder: str) -> Dict[str, List]:
    
    files = os.listdir(folder)
    merged_data = dict()
    
    for file_name in files:
        file_path = os.path.join(folder, file_name)
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as reader:
            data = json.load(reader)
            merged_data.update(data)
            
    return merged_data


def mistral_last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]