#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:23:03 2024

@author: poojaoza
"""
import argparse
import lzma
import gzip
import json
import torch
import random
import tqdm
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
from utils import get_detailed_instruct, get_task_string, mistral_last_token_pool

cuda_device = 'cuda:0'
device = ''





def _read_query_input_file(input_file: str) -> Dict[str, str]:
    
    data = dict()
    
    with open(input_file, 'rt', encoding='utf-8') as reader:
        data = json.load(reader)
        
    return data

def _read_document_input_file(input_file: str) -> Dict[str, str]:
    
    data = dict()
    
    with lzma.open(input_file, 'rt', encoding='utf-8') as reader:
        for line in reader:
            item = json.loads(line)
            paragraphs_data = item[1]
            
            for para in paragraphs_data:
                paragraph_id = para['paragraph_id']
                paragraph_text = para['text']
                
                if paragraph_id not in data:
                    data[paragraph_id] = paragraph_text
    return data
                
def _read_symbols_input_file(input_file: str,
                             dataset: str) -> Dict[str, str]:
    
    data = dict()
    
    if dataset == 'car':
    
        with gzip.open(input_file, 'rt', encoding='utf-8') as reader:
            for line in reader:
                item = json.loads(line)
                symbol_id = item['qid']
                symbol_text = item['question']
                
                if symbol_id not in data:
                    data[symbol_id] = symbol_text
    else:
        
        with gzip.open(input_file, 'rt', encoding='utf-8') as reader:
            for line in reader:
                item = json.loads(line)
                
                for element in item['items']:
                    symbol_id = element['question_id']
                    symbol_text = element['question_text']
                    
                    if symbol_id not in data:
                        data[symbol_id] = symbol_text

    return data     


def _read_answers_input_file(input_file: str,
                             dataset: str) -> Dict[str, str]:
    
    '''
    return:
        {query_id/section_id:{
            'paraid1':{
                    'answerkeyid1':'answertext'
                }
            }
        }
    '''
    data = dict()
    
    if dataset == 'car':
    
        with open(input_file, 'rt', encoding='utf-8') as reader:
            for line in reader:
                item = json.loads(line)
                query_id = item[0]
                autograd_items = item[1]
                
                for paragraph_data in autograd_items:
                    for exam_data in paragraph_data['exam_grades']:
                        if exam_data['prompt_info']['prompt_class'] == 'QuestionCompleteConciseUnanswerablePromptWithChoices':
                            for answers in exam_data['answers']:
                                split_id = answers[0].split('/')
                                answers_id = split_id[0]+'/'+split_id[1]+'/'+paragraph_data['paragraph_id']+'/'+split_id[2]
                                symbol_id = answers_id
                                symbol_text = answers[1]
                                if symbol_id not in data:
                                    data[symbol_id] = symbol_text
    else:
        
        with open(input_file, 'rt', encoding='utf-8') as reader:
            for line in reader:
                item = json.loads(line)
                query_id = item[0]
                autograd_items = item[1]
                
                for paragraph_data in autograd_items:
                    for exam_data in paragraph_data['exam_grades']:
                        if exam_data['prompt_info']['prompt_class'] == 'QuestionCompleteConciseUnanswerablePromptWithChoices':
                            for answers in exam_data['answers']:
                                split_id = answers[0].split('/')
                                answers_id = split_id[0]+'/'+paragraph_data['paragraph_id']+'/'+split_id[1]
                                symbol_id = answers_id
                                symbol_text = answers[1]
                                if symbol_id not in data:
                                    data[symbol_id] = symbol_text

    return data 

def _get_mistral_model_embeddings(input_texts: List,
                                  tokenizer: AutoTokenizer,
                                  model: AutoModel):
    max_length = 1024
    
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
    
    outputs = model(**batch_dict)
    embeddings = mistral_last_token_pool(outputs.hidden_states[1], batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def _get_model_embeddings(input_texts: List,
                         tokenizer: AutoTokenizer,
                         model: AutoModel):
        
    batch_dict = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    output = model.encoder(input_ids=batch_dict['input_ids'],
                                attention_mask=batch_dict['attention_mask'],
                                return_dict=True)
    #pooled_sentence = (output.last_hidden_state * batch_dict['attention_mask'].unsqueeze(-1)).sum(dim=-2) / batch_dict['attention_mask'].sum(dim=-1)
    pooled_sentence = output.last_hidden_state
    
    pooled_sentence = torch.mean(pooled_sentence, dim=1)
    #output.detach().cpu()
    #del output
    #del batch_dict
    
    return pooled_sentence


def _write_to_file(data: Dict[str, str],
                   output_file: str,
                   i: int):
    
    #for i, chunk in enumerate(split_dict(data), 1):
    with gzip.open(f'{output_file}-{i}.json.gz', 'wt', encoding='utf-8') as writer:
        json.dump(data, writer)
        print(f'Finished writing {output_file}-{i}.json.gz file')

def _process_embeddings(data: Dict[str, str],
                        output_file: str,
                        flag: str,
                        chunk_size: int,
                        input_model: str):
    
    model_name = 'google/flan-t5-large'
    
    if input_model == 'mistral':
        model_name = 'intfloat/e5-mistral-7b-instruct'
    
    if input_model == 'flan':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModel.from_pretrained(model_name)
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
                return_dict=True
            )
        #model.to(device)
    
    embeddings = dict()
    
    if flag == 'query':
        task_string = get_task_string('retrieval')
        counter = 0
        file_number = 1
        
        for qid, qtext in tqdm.tqdm(data.items(), total=len(data)):
            counter += 1
            query = [get_detailed_instruct(task_string, qtext)]
            if input_model == 'flan':
                embedding = _get_model_embeddings(query, tokenizer, model)
            else:
                embedding = _get_mistral_model_embeddings(query, tokenizer, model)
            embeddings[qid] = embedding.detach().cpu().tolist()
            if counter == chunk_size:
                _write_to_file(embeddings, output_file, file_number)
                file_number += 1
                counter = 0
                embeddings = dict()
            #embedding.detach().cpu()
            #del embedding
        if counter > 0:
            _write_to_file(embeddings, output_file, file_number)
    else:
        counter = 0
        file_number = 1
        
        for docid, doctext in tqdm.tqdm(data.items(), total=len(data)):
            counter += 1
            text = [doctext]
            if input_model == 'flan':
                embedding = _get_model_embeddings(text, tokenizer, model)
            else:
                embedding = _get_mistral_model_embeddings(text, tokenizer, model)
            embeddings[docid] = embedding.detach().cpu().tolist()
            if counter == chunk_size:
                _write_to_file(embeddings, output_file, file_number)
                file_number += 1
                counter = 0
                embeddings = dict()
            #embedding.detach().cpu()
            #del embedding
        if counter > 0:
            _write_to_file(embeddings, output_file, file_number)
    #_write_to_file(embeddings, output_file)

def embeddings_retrieval(input_file: str,
         output_file: str,
         flag: str,
         chunk_size: int,
         model: str,
         dataset: str):
    
    data = dict()

    print('Start to process')
    
    if flag == 'query':
        data = _read_query_input_file(input_file)
    elif flag == 'document':
        data = _read_document_input_file(input_file)
    elif flag == 'answers':
        data = _read_answers_input_file(input_file, dataset)
    else:
        data = _read_symbols_input_file(input_file, dataset)

    print(f'Done reading input file. Total length of data {len(data)}')
        
    _process_embeddings(data, output_file, flag, chunk_size, model)
    
    
def main():
    parser = argparse.ArgumentParser("Script to retrieve embeddings.")
    parser.add_argument('--file', help='Enter json file path', type=str, required=True)
    parser.add_argument('--save', help='Enter output file path', type=str, required=True)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    parser.add_argument('--seed', help='Random seed initialization',type=int, default=49500)
    parser.add_argument('--dataset', help='Which dataset to process the data of 1) dl and 2) trec, default=trec', default='trec', required=True)
    parser.add_argument('--chunk-size', help='chunk size for the dictionaries. Default: 1000',type=int, default=1000)
    parser.add_argument('--flag', help='select the type of data to process, 1) query, 2) document, 3) symbols and 4) answers, default=document', choices=['query', 'document', 'symbols', 'answers'])
    parser.add_argument('--model', help='select the model to get the embeddings of: 1) flan 2) mistral, default=flan', choices=['flan', 'mistral'])
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    global cuda_device
    
    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    global device
    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )
    
    embeddings_retrieval(args.file, args.save, args.flag, args.chunk_size, args.model, args.dataset)
    
    
if __name__ == "__main__":
    main()
