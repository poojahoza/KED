#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:14:58 2024

@author: poojaoza
"""

import json

from typing import Dict, List

from .sampling import _retrieve_pointwise_dl_samples, _retrieve_pointwise_trecy3_samples, _retrieve_pointwise_trecy3_section_samples, _retrieve_pointwise_trecy3_section_ranking_samples
from ..utils import read_queries, read_dl_generated_questions, read_autograder_file, read_trec_y3_generated_questions, read_trec_y3_qrels, get_neighbors_data, read_trec_y3_section_generated_questions, read_trec_y3_section_autograder_data, get_ranking_neighbors_data, read_trec_y3_section_ranking_autograder_data, get_multiple_ranking_neighbors_data, read_dl_ranking_autograder_file, read_generated_answers, get_dl_answers_neighbors_data, get_trec_car_answers_neighbors_data


def make_pointwise(test_qrel_file_path: str,
                   train_data: Dict,
                   dataset: str,
                   re_rank: bool,
                   re_rank_method: str) -> List[Dict]:
    
    test_passages = dict()
    test_pointwise_passages_data = []
    
    if dataset == 'dl' or dataset == 'dl-rank':
        test_passages = _retrieve_pointwise_dl_samples(test_qrel_file_path, train_data)
    elif dataset == 'car-section':
        test_passages = _retrieve_pointwise_trecy3_section_samples(test_qrel_file_path, train_data, re_rank, re_rank_method)
    elif dataset == 'car-section-rank':
        test_passages = _retrieve_pointwise_trecy3_section_ranking_samples(test_qrel_file_path, train_data, re_rank, re_rank_method)
    else:
        test_passages = _retrieve_pointwise_trecy3_samples(test_qrel_file_path, train_data)

    for key, val in test_passages.items():
        for pass_id, pass_rel in val.items():
            temp_dict = dict()
            temp_dict['query'] = {'query_id': key}
            temp_dict['entity'] = {'entity_id':pass_id, 'entity_relevance':pass_rel}
            test_pointwise_passages_data.append(temp_dict)

    return test_pointwise_passages_data


def update_pointwise_data(queries_data,
                        questions_data,
                        passages_data,
                        ratings_data,
                        pointwise_data,
                        queries_embed,
                        passages_embed,
                        symbols_embed,
                        flag):
    
    updated_pointwise_data = []
    
    for item in pointwise_data:
        if item['query']['query_id'] in queries_data:
            query_text = queries_data[item['query']['query_id']]
            query_embed = queries_embed[item['query']['query_id']]
            ent_text = passages_data[item['query']['query_id']][item['entity']['entity_id']]
            ent_embed = passages_embed[item['entity']['entity_id']]
            
            if flag == 'car-section-rank' or flag == 'dl-rank':
                # ent_neighbors = get_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                #                                    questions_data[item['query']['query_id']],
                #                                    item['entity']['entity_id'],
                #                                   ent_text,
                #                                   symbols_embed=symbols_embed,
                #                                   ent_embed=ent_embed)
                
                # This is for the questions data
                # ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                #                                    questions_data[item['query']['query_id']],
                #                                    item['entity']['entity_id'],
                #                                   ent_text,
                #                                   symbols_embed=symbols_embed,
                #                                   ent_embed=ent_embed)
                
                # This is for the answers data
                ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   questions_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   item['entity']['entity_id'],
                                                  ent_text,
                                                  symbols_embed,
                                                  ent_embed,
                                                  flag)
                
                
            elif flag == 'dl': 
                # This is for the questions data
                # ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                #                                    questions_data[item['query']['query_id']],
                #                                    item['entity']['entity_id'],
                #                                   ent_text,
                #                                   symbols_embed=symbols_embed,
                #                                   ent_embed=ent_embed)    
                
                # This is for the answers data
                ent_neighbors = get_dl_answers_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   questions_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   item['entity']['entity_id'],
                                                  ent_text,
                                                  symbols_embed=symbols_embed,
                                                  ent_embed=ent_embed)
            else:
                # This is for the questions data
                # ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                #                                    questions_data[item['query']['query_id']],
                #                                    item['entity']['entity_id'],
                #                                   ent_text,
                #                                   symbols_embed=symbols_embed,
                #                                   ent_embed=ent_embed)    
                
                # This is for the answers data
                ent_neighbors = get_trec_car_answers_neighbors_data(ratings_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   questions_data[item['query']['query_id']][item['entity']['entity_id']],
                                                   item['entity']['entity_id'],
                                                  ent_text,
                                                  symbols_embed=symbols_embed,
                                                  ent_embed=ent_embed)
            
            item['query']['query_text'] = query_text
            item['query']['query_embed'] = query_embed
            item['entity']['entity_text'] = ent_text
            item['entity']['entity_embed'] = ent_embed
            item['entity']['entity_neighbors'] = ent_neighbors
            
            updated_pointwise_data.append(item)
        
    
    
    # for item in pointwise_data:
    #     updated_pointwise_data.append(json.dumps(item))
    
    return updated_pointwise_data
    
    
def to_pointwise_data(queries_path: str,
                      generated_questions_path: str,
                      autograder_file: str,
                      test_qrel: str,
                      dataset: str,
                      kfold: bool,
                      queries_test_data: Dict[str, str],
                      queries_embed: Dict[str, List],
                      para_embed: Dict[str, List],
                      symb_embed: Dict[str, List],
                      re_rank: bool,
                      re_rank_method: str):
    
    final_data = []
    
    if kfold:
        queries_data = queries_test_data
    else:
        queries_data = read_queries(queries_path)
        print(len(queries_data))
    
    if dataset == 'dl':
    
        
        #questions_data = read_dl_generated_questions(generated_questions_path)
        questions_data = read_generated_answers(generated_questions_path) # This is for using generated answers as the symbols
        ratings_data, passages_data = read_autograder_file(autograder_file, dataset)
        pointwise_data = make_pointwise(test_qrel, queries_data, dataset, re_rank, re_rank_method)
    
        final_data = update_pointwise_data(queries_data,
                                         questions_data,
                                         passages_data,
                                         ratings_data,
                                         pointwise_data,
                                         queries_embed,
                                         para_embed,
                                         symb_embed,
                                         dataset)
    elif dataset == 'dl-rank':
    
        
        #questions_data = read_dl_generated_questions(generated_questions_path)
        questions_data = read_generated_answers(generated_questions_path) # This is for using generated answers as the symbols
        ratings_data, passages_data = read_dl_ranking_autograder_file(autograder_file, dataset)
        pointwise_data = make_pointwise(test_qrel, queries_data, dataset, re_rank, re_rank_method)
    
        final_data = update_pointwise_data(queries_data,
                                         questions_data,
                                         passages_data,
                                         ratings_data,
                                         pointwise_data,
                                         queries_embed,
                                         para_embed,
                                         symb_embed,
                                         dataset)
    elif dataset == 'car-section':
        #questions_data = read_trec_y3_section_generated_questions(generated_questions_path)
        questions_data = read_generated_answers(generated_questions_path) # This is for using generated answers as the symbols
        print(len(questions_data))
        ratings_data, passages_data = read_trec_y3_section_autograder_data(queries_data, autograder_file)
        print(len(ratings_data))
        print(len(passages_data))
        qrels_data = read_trec_y3_qrels(test_qrel)
        #passing autograder file here as test_qrel since for trec car we are
        # taking all the samples from autograder file only
        pointwise_data = make_pointwise(autograder_file, qrels_data, dataset, re_rank, re_rank_method)
        
        final_data = update_pointwise_data(queries_data,
                                         questions_data,
                                         passages_data,
                                         ratings_data,
                                         pointwise_data,
                                         queries_embed,
                                         para_embed,
                                         symb_embed,
                                         dataset)
    elif dataset == 'car-section-rank':
        #questions_data = read_trec_y3_section_generated_questions(generated_questions_path)
        questions_data = read_generated_answers(generated_questions_path) # This is for using generated answers as the symbols
        print(len(questions_data))
        ratings_data, passages_data = read_trec_y3_section_ranking_autograder_data(queries_data, autograder_file)
        print(len(ratings_data))
        print(len(passages_data))
        qrels_data = read_trec_y3_qrels(test_qrel)
        #passing autograder file here as test_qrel since for trec car we are
        # taking all the samples from autograder file only
        pointwise_data = make_pointwise(autograder_file, qrels_data, dataset, re_rank, re_rank_method)
        
        final_data = update_pointwise_data(queries_data,
                                         questions_data,
                                         passages_data,
                                         ratings_data,
                                         pointwise_data,
                                         queries_embed,
                                         para_embed,
                                         symb_embed,
                                         dataset)    
    
    else:
        questions_data = read_trec_y3_generated_questions(generated_questions_path)
        ratings_data, passages_data = read_autograder_file(autograder_file, dataset)
        
        #passing autograder file here as test_qrel since for trec car we are
        # taking all the samples from autograder file only
        pointwise_data = make_pointwise(autograder_file, queries_data, dataset)
    
        final_data = update_pointwise_data(queries_data,
                                         questions_data,
                                         passages_data,
                                         ratings_data,
                                         pointwise_data,
                                         queries_embed,
                                         para_embed,
                                         symb_embed,
                                         dataset)

    return final_data