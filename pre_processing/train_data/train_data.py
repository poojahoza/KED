#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:06:46 2024

@author: poojaoza
"""

import json

from typing import Dict, List

from .sampling import _retrieve_dl20_samples, _retrieve_trecy3_samples, _retrieve_trec_y3_section_samples, _retrieve_trec_y3_section_ranking_samples
from ..utils import read_queries, read_trec_y3_generated_questions, read_trec_y3_section_generated_questions, read_trec_y3_qrels, read_dl_generated_questions, read_autograder_file, get_neighbors_data, read_trec_y3_section_autograder_data, get_ranking_neighbors_data, read_trec_y3_section_ranking_autograder_data, get_multiple_ranking_neighbors_data, read_dl_ranking_autograder_file, read_generated_answers, get_dl_answers_neighbors_data, get_trec_car_answers_neighbors_data

def make_pairs(test_qrel_file_path: str,
              train_data: Dict,
              flag: str) -> List[Dict]:
    
    samples = dict()
    test_pairwise_passages_data = []
    
    if flag == 'dl' or flag == 'dl-rank':
        samples = _retrieve_dl20_samples(test_qrel_file_path,
                                              train_data)
    elif flag == 'car-section':
        samples = _retrieve_trec_y3_section_samples(test_qrel_file_path,
                                              train_data)
    elif flag == 'car-section-rank':
         samples = _retrieve_trec_y3_section_ranking_samples(test_qrel_file_path,
                                               train_data)
    else:
        samples = _retrieve_trecy3_samples(test_qrel_file_path,
                                              train_data)


    
    for key, val in samples.items():
        positive_len = len(val['positive'])
        negative_len = len(val['negative'])

        k = positive_len
        if negative_len < positive_len:
            k = negative_len

        query_relevant_passages = val['positive'][:k]
        query_relevance = val['positive_rel'][:k]
        query_irrelevant_passages = val['negative'][:k]

        
        ''' 
        for pos, neg, pos_rel in zip(query_relevant_passages, query_irrelevant_passages, query_relevance):
            temp_dict = dict()
            temp_dict['query'] = {'query_id': key}
            temp_dict['ent_pos'] = {'entity_id':pos, 'entity_relevance':pos_rel}
            temp_dict['ent_neg'] = {'entity_id':neg, 'entity_relevance':0}
            test_pairwise_passages_data.append(temp_dict)
        
        '''
        # All pairs
        for pos, pos_rel in zip(query_relevant_passages, query_relevance):
            for neg in query_irrelevant_passages:
                temp_dict = dict()
                temp_dict['query'] = {'query_id': key}
                temp_dict['ent_pos'] = {'entity_id':pos, 'entity_relevance':pos_rel}
                temp_dict['ent_neg'] = {'entity_id':neg, 'entity_relevance':0}
                test_pairwise_passages_data.append(temp_dict)
        
    
    print(len(test_pairwise_passages_data))
    return test_pairwise_passages_data
    
        
def update_pairwise_data(queries_data,
                        questions_data,
                        passages_data,
                        ratings_data,
                        pairwise_data,
                        queries_embed,
                        passages_embed,
                        symbols_embed,
                        dataset):
    
    updated_pairwise_data = []
    
    for item in pairwise_data:
        if item['query']['query_id'] in queries_data:
            query_text = queries_data[item['query']['query_id']]
            pos_ent_text = passages_data[item['query']['query_id']][item['ent_pos']['entity_id']]
            neg_ent_text = passages_data[item['query']['query_id']][item['ent_neg']['entity_id']]
            query_embed = queries_embed[item['query']['query_id']]
            pos_ent_embed = passages_embed[item['ent_pos']['entity_id']]
            neg_ent_embed = passages_embed[item['ent_neg']['entity_id']]
            
            if dataset == 'car-section-rank' or dataset == 'dl-rank':
                # pos_ent_neighbors = get_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_pos']['entity_id'],
                #                                       pos_ent_text,
                #                                       symbols_embed,
                #                                       pos_ent_embed)
                # neg_ent_neighbors = get_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_neg']['entity_id'],
                #                                       neg_ent_text,
                #                                       symbols_embed,
                #                                       neg_ent_embed)
                
                
                # This is for the questions data
                
                # pos_ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_pos']['entity_id'],
                #                                       pos_ent_text,
                #                                       symbols_embed,
                #                                       pos_ent_embed)
                # neg_ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_neg']['entity_id'],
                #                                       neg_ent_text,
                #                                       symbols_embed,
                #                                       neg_ent_embed)
                
                # This is for the answers data
                
                pos_ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      item['ent_pos']['entity_id'],
                                                      pos_ent_text,
                                                      symbols_embed,
                                                      pos_ent_embed,
                                                      dataset)
                neg_ent_neighbors = get_multiple_ranking_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      item['ent_neg']['entity_id'],
                                                      neg_ent_text,
                                                      symbols_embed,
                                                      neg_ent_embed,
                                                      dataset)
                
            elif dataset == 'dl':
                # This is for the questions data
                # pos_ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_pos']['entity_id'],
                #                                       pos_ent_text,
                #                                       symbols_embed,
                #                                       pos_ent_embed)
                # neg_ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_neg']['entity_id'],
                #                                       neg_ent_text,
                #                                       symbols_embed,
                #                                       neg_ent_embed)
                
                # This is for the answers data
                pos_ent_neighbors = get_dl_answers_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      item['ent_pos']['entity_id'],
                                                      pos_ent_text,
                                                      symbols_embed,
                                                      pos_ent_embed)
                neg_ent_neighbors = get_dl_answers_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      item['ent_neg']['entity_id'],
                                                      neg_ent_text,
                                                      symbols_embed,
                                                      neg_ent_embed)
            else:
                # This is for the questions data
                # pos_ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_pos']['entity_id'],
                #                                       pos_ent_text,
                #                                       symbols_embed,
                #                                       pos_ent_embed)
                # neg_ent_neighbors = get_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                #                                       questions_data[item['query']['query_id']],
                #                                       item['ent_neg']['entity_id'],
                #                                       neg_ent_text,
                #                                       symbols_embed,
                #                                       neg_ent_embed)
                
                # This is for the answers data
                pos_ent_neighbors = get_trec_car_answers_neighbors_data(ratings_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_pos']['entity_id']],
                                                      item['ent_pos']['entity_id'],
                                                      pos_ent_text,
                                                      symbols_embed,
                                                      pos_ent_embed)
                neg_ent_neighbors = get_trec_car_answers_neighbors_data(ratings_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      questions_data[item['query']['query_id']][item['ent_neg']['entity_id']],
                                                      item['ent_neg']['entity_id'],
                                                      neg_ent_text,
                                                      symbols_embed,
                                                      neg_ent_embed)
            
            item['query']['query_text'] = query_text
            item['query']['query_embed'] = query_embed
            item['ent_pos']['entity_text'] = pos_ent_text
            item['ent_pos']['entity_embed'] = pos_ent_embed
            item['ent_neg']['entity_text'] = neg_ent_text
            item['ent_neg']['entity_embed'] = neg_ent_embed
            item['ent_pos']['entity_neighbors'] = pos_ent_neighbors
            item['ent_neg']['entity_neighbors'] = neg_ent_neighbors
            
            updated_pairwise_data.append(item)
        
        
    return updated_pairwise_data
    
    
def to_pairwise_data(queries_path: str,
                    generated_questions_path: str,
                    autograder_file: str,
                    test_qrel: str,
                    dataset: str,
                    kfold: bool,
                    queries_train_data: Dict[str, str],
                    queries_embed: Dict[str, List],
                    para_embed: Dict[str, List],
                    symb_embed: Dict[str, List]):
    
    final_data = []
    
    if kfold:
        queries_data = queries_train_data
    else:
        queries_data = read_queries(queries_path)
        print(len(queries_data))
    
    if dataset == 'dl': 
        #questions_data = read_dl_generated_questions(generated_questions_path)
        #print(len(questions_data))
        questions_data = read_generated_answers(generated_questions_path) # This is for the generated answers used as symbols
        print(len(questions_data))
        ratings_data, passages_data = read_autograder_file(autograder_file, dataset)
        print(len(ratings_data))
        print(len(passages_data))
        pairwise_data = make_pairs(test_qrel, queries_data, dataset)
        print(len(pairwise_data))
        final_data = update_pairwise_data(queries_data,
                                     questions_data,
                                     passages_data,
                                     ratings_data,
                                     pairwise_data,
                                     queries_embed,
                                     para_embed,
                                     symb_embed,
                                     dataset)
    elif dataset == 'dl-rank': 
        #questions_data = read_dl_generated_questions(generated_questions_path)
        #print(len(questions_data))
        questions_data = read_generated_answers(generated_questions_path) # This is for the generated answers used as symbols
        print(len(questions_data))
        ratings_data, passages_data = read_dl_ranking_autograder_file(autograder_file, dataset)
        print(len(ratings_data))
        print(len(passages_data))
        pairwise_data = make_pairs(test_qrel, queries_data, dataset)
        print(len(pairwise_data))
        final_data = update_pairwise_data(queries_data,
                                     questions_data,
                                     passages_data,
                                     ratings_data,
                                     pairwise_data,
                                     queries_embed,
                                     para_embed,
                                     symb_embed,
                                     dataset)
    
    elif dataset == 'car-section':
        #questions_data = read_trec_y3_section_generated_questions(generated_questions_path)
        #print(len(questions_data))
        questions_data = read_generated_answers(generated_questions_path) # This is for the generated answers used as symbols
        print(len(questions_data))
        ratings_data, passages_data = read_trec_y3_section_autograder_data(queries_data, autograder_file)
        print(len(ratings_data))
        print(len(passages_data))
        qrels_data = read_trec_y3_qrels(test_qrel)
        pairwise_data = make_pairs(autograder_file, qrels_data, dataset)
        print(len(pairwise_data))
        final_data = update_pairwise_data(queries_data,
                                     questions_data,
                                     passages_data,
                                     ratings_data,
                                     pairwise_data,
                                     queries_embed,
                                     para_embed,
                                     symb_embed,
                                     dataset)
        
    elif dataset == 'car-section-rank':
        #questions_data = read_trec_y3_section_generated_questions(generated_questions_path)
        #print(len(questions_data))
        questions_data = read_generated_answers(generated_questions_path) # This is for the generated answers used as symbols
        print(len(questions_data))
        ratings_data, passages_data = read_trec_y3_section_ranking_autograder_data(queries_data, autograder_file)
        print(len(ratings_data))
        print(len(passages_data))
        qrels_data = read_trec_y3_qrels(test_qrel)
        pairwise_data = make_pairs(qrels_data, ratings_data, dataset)
        print(len(pairwise_data))
        final_data = update_pairwise_data(queries_data,
                                      questions_data,
                                      passages_data,
                                      ratings_data,
                                      pairwise_data,
                                      queries_embed,
                                      para_embed,
                                      symb_embed,
                                      dataset)
    
    else:
        questions_data = read_trec_y3_generated_questions(generated_questions_path)
        print(len(questions_data))
        ratings_data, passages_data = read_autograder_file(autograder_file, dataset)
        print(len(ratings_data))
        print(len(passages_data))
        qrels_data = read_trec_y3_qrels(test_qrel)
        pairwise_data = make_pairs(autograder_file, qrels_data, dataset)
        print(len(pairwise_data))
        final_data = update_pairwise_data(queries_data,
                                     questions_data,
                                     passages_data,
                                     ratings_data,
                                     pairwise_data,
                                     queries_embed,
                                     para_embed,
                                     symb_embed,
                                     dataset)
    
    print(len(final_data))
    #print(final_data[0])
    return final_data
