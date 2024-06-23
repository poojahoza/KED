#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:02:05 2024

@author: poojaoza
"""

import lzma
import json
from typing import Dict
from ..utils import _get_multiple_ranking_data


def _retrieve_dl20_samples(test_qrel_file_path: str,
              train_data: Dict) -> Dict:
    
    positive_passages = dict()
    
    with open(test_qrel_file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = line.split()
            query_id = data[0]
            passage_id = data[2]
            passage_relevance = int(data[3])
            
            if query_id in train_data:
                if query_id in positive_passages:
                    temp_data = positive_passages[query_id]
                    if passage_relevance > 0:
                        positive = temp_data['positive']
                        positive.append(passage_id)
                        pos_rel = temp_data['positive_rel']
                        pos_rel.append(passage_relevance)
                    else:
                        negative = temp_data['negative']
                        negative.append(passage_id)
                    positive_passages[query_id] = temp_data
                else:
                    temp_data = dict()
                    temp_data['positive'] = []
                    temp_data['positive_rel'] = []
                    temp_data['negative'] = []
                    if passage_relevance > 0:
                        positive = temp_data['positive']
                        positive.append(passage_id)
                        pos_rel = temp_data['positive_rel']
                        pos_rel.append(passage_relevance)
                    else:
                        negative = temp_data['negative']
                        negative.append(passage_id)
                    positive_passages[query_id] = temp_data

    print(len(list(positive_passages.keys())))

    for key, val in positive_passages.items():
        print(key+' '+str(len(val['positive']))+' '+str(len(val['negative'])))
        
    return positive_passages


def _retrieve_trec_y3_section_samples(autograder_file_path: str,
                                      qrels_data: Dict) -> Dict:
    
    samples = dict()
    
    with open(autograder_file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            paragraphs_data = data[1]
            
            for item in paragraphs_data:
                paragraph_id = item['paragraph_id']
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if rate['question_id'].startswith("tqa2:"):
                                query_id = rate['question_id'].rsplit('/', 1)[0]
                                
                                if query_id in qrels_data:
                                    if query_id in samples:
                                        temp_data = samples[query_id]
                                        if paragraph_id in qrels_data[query_id]:
                                            positive = temp_data['positive']
                                            positive.append(paragraph_id)
                                            pos_rel = temp_data['positive_rel']
                                            pos_rel.append(qrels_data[query_id][paragraph_id])
                                            temp_data['positive'] = positive
                                            temp_data['positive_rel'] = pos_rel
                                        else:
                                            negative = temp_data['negative']
                                            negative.append(paragraph_id)
                                            temp_data['negative'] = negative
                                        samples[query_id] = temp_data
                                    else:
                                        temp_data = dict()
                                        temp_data['positive'] = []
                                        temp_data['positive_rel'] = []
                                        temp_data['negative'] = []
                                        if paragraph_id in qrels_data[query_id]:
                                            positive = temp_data['positive']
                                            positive.append(paragraph_id)
                                            pos_rel = temp_data['positive_rel']
                                            pos_rel.append(qrels_data[query_id][paragraph_id])
                                        else:
                                            negative = temp_data['negative']
                                            negative.append(paragraph_id)
                                        samples[query_id] = temp_data
                                        
    return samples

def _retrieve_trec_y3_section_ranking_samples(qrels_data: Dict,
                                      ratings_data: Dict) -> Dict:
    
    positive_passages = dict()
    #print(test_qrel_file_path)
    
    for query, para in qrels_data.items():
        
        if query in ratings_data:
            #if query in positive_passages:
            temp_ratings_data = ratings_data[query] 
            qrel_para_keys = list(para.keys())
            ratings_para_keys = list(temp_ratings_data.keys())
            
            inside_query = dict()
            inside_query['positive'] = []
            inside_query['positive_rel'] = []
            inside_query['negative'] = []
            positive_passages[query] = inside_query

            for paraid in ratings_para_keys:
                temp_data = positive_passages[query]
                if paraid in qrel_para_keys and para[paraid] > 0:
                    positive = temp_data['positive']
                    positive.append(paraid)
                    pos_rel = temp_data['positive_rel']
                    pos_rel.append(para[paraid])
                else:
                    negative = temp_data['negative']
                    negative.append(paraid)
                positive_passages[query] = temp_data
    print(len(list(positive_passages.keys())))

    for key, val in positive_passages.items():
        print(key+' '+str(len(val['positive']))+' '+str(len(val['negative'])))
        
    return positive_passages                    
                                            


def _retrieve_trecy3_samples(autograder_file_path: str,
                            qrels_data: Dict) -> Dict:
    
    samples = dict()
    
    with open(autograder_file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            paragraphs_data = data[1]
            
            for item in paragraphs_data:
                paragraph_id = item['paragraph_id']
                
                if query_id in qrels_data:               
                    if query_id in samples:
                        temp_data = samples[query_id]
                        if paragraph_id in qrels_data[query_id]:
                            positive = temp_data['positive']
                            positive.append(paragraph_id)
                            pos_rel = temp_data['positive_rel']
                            pos_rel.append(qrels_data[query_id][paragraph_id])
                        else:
                            negative = temp_data['negative']
                            negative.append(paragraph_id)
                        samples[query_id] = temp_data
                    else:
                        temp_data = dict()
                        temp_data['positive'] = []
                        temp_data['positive_rel'] = []
                        temp_data['negative'] = []
                        if paragraph_id in qrels_data[query_id]:
                            positive = temp_data['positive']
                            positive.append(paragraph_id)
                            pos_rel = temp_data['positive_rel']
                            pos_rel.append(qrels_data[query_id][paragraph_id])
                        else:
                            negative = temp_data['negative']
                            negative.append(paragraph_id)
                        samples[query_id] = temp_data
                    
    return samples


