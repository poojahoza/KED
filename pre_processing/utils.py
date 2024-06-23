#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:43:29 2024

@author: poojaoza
"""

import gzip
import json
import lzma
from typing import Dict, Tuple, List

TREC_RANKING_METHOD = 'dangnt-nlp'

def read_trec_y3_generated_questions(questions_path:str) -> Dict[str, Dict[str, str]]:
    questions_data = dict()
    
    with gzip.open(questions_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data['query_id']
            
            if query_id in questions_data:
                temp_data = questions_data[query_id]
                temp_data[data['qid']] = data['question']
                questions_data[query_id] = temp_data
            else:
                temp_data = dict()
                temp_data[data['qid']] = data['question']
                questions_data[query_id] = temp_data
            
    return questions_data

def read_trec_y3_section_generated_questions(questions_path:str) -> Dict[str, Dict[str, str]]:
    questions_data = dict()
    
    with gzip.open(questions_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            #query_id = data['query_id']
            section_level_query = data['query_id']+'/'+data['facet_id']
            
            if section_level_query in questions_data:
                temp_data = questions_data[section_level_query]
                temp_data[data['qid']] = data['question']
                questions_data[section_level_query] = temp_data
            else:
                temp_data = dict()
                temp_data[data['qid']] = data['question']
                questions_data[section_level_query] = temp_data
            
    return questions_data

def read_trec_y3_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    qrels_data = dict()
    
    with open(qrels_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = line.split(' ')
            query_id = data[0]
            paragraph_id = data[2]
            relevance = int(data[3])
            
            if query_id in qrels_data:
                temp_data = qrels_data[query_id]
                temp_data[paragraph_id] = relevance
                qrels_data[query_id] = temp_data
            else:
                temp_data = dict()
                temp_data[paragraph_id] = relevance
                qrels_data[query_id] = temp_data
    return qrels_data


def read_queries(queries_path:str) -> Dict[str, str]:
    queries_data = dict()
    
    with open(queries_path, 'rt', encoding='utf-8') as reader:
        queries_data = json.load(reader)
        
    return queries_data

def read_trec_y3_section_queries(queries_path:str) -> Dict[str, str]:
    queries_data = dict()
    
    with gzip.open(queries_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            section_level_query = data['query_id']+'/'+data['facet_id']
            queries_data[section_level_query] = data['query_text']
        
    return queries_data

def read_dl_generated_questions(questions_path:str) -> Dict[str, Dict[str, str]]:
    questions_data = dict()
    
    with gzip.open(questions_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data['query_id']
            questions = data['items']
            temp_dict = dict()
            for item in questions:
                temp_dict[item['question_id']] = item['question_text']
            questions_data[query_id] = temp_dict
            
    return questions_data

def read_generated_answers(answers_path:str) -> Dict[str, Dict[str, Dict[str, str]]]:
    '''

    Parameters
    ----------
    answers_path : str
        DESCRIPTION.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]]
        DESCRIPTION. {'queryid':{
                            'paragraphid1':{
                                'original_answerid1':'answertext',
                                'original_answerid2':'answertext',
                                'paragraph_answerid1':'original_answerid1'},
                            'paragraphid2':{
                                'original_answerid1':'answertext',
                                'original_answerid2':'answertext',
                                'paragraph_answerid1':'original_answerid1'}
                        }
                    }

    '''
    
    answers_data = dict()
    
    with gzip.open(answers_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data['query_id']
            paragraph_id = data['paragraph_id']
            original_answer_id = data['original_answer_id']
            answer_text = data['answer_text']
            paragraph_answer_id = data['answer_id']
            
            if query_id in answers_data:
                temp_query_data = answers_data[query_id]
                
                if paragraph_id in temp_query_data:
                    temp_para_data = temp_query_data[paragraph_id]
                    temp_para_data[original_answer_id] = answer_text
                    temp_para_data[paragraph_answer_id] = original_answer_id
                else:
                    temp_answer_data = dict()
                    temp_answer_data[original_answer_id] = answer_text
                    temp_answer_data[paragraph_answer_id] = original_answer_id
                    temp_query_data[paragraph_id] = temp_answer_data
                
            else:
                temp_answer_data = dict()
                temp_answer_data[original_answer_id] = answer_text
                temp_answer_data[paragraph_answer_id] = original_answer_id
                
                temp_para_data = dict()
                temp_para_data[paragraph_id] = temp_answer_data
                answers_data[query_id] = temp_para_data
            
            
    return answers_data

def read_autograder_file(file_path:str,
                         dataset: str) -> Tuple:
    
    '''
    return format of questions_rating:
        {query_id:
             {
            paragraph_id1:
                  {
                      question_id1: rating,
                      question_id2: rating,
                  },
            paragraph_id2:
                {
                    question_id1: rating,
                    question_id2: rating,
                }
              }
        }
            
    return format of paragraphs_data:
        {query_id1:
             {
                 paragraph_id1: paragraph_text,
                 paragraph_id2: paragraph_text
            },
        query_id2:
            {
                paragraph_id1: paragraph_text,
                paragraph_id2: paragraph_text
            }
        }
    '''
    
    paragraphs_data = dict()
    questions_rating = dict()
    
    with open(file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            rating_data = data[1]
            
            query_ratings_data = dict()
                
            if query_id in questions_rating:
                query_ratings_data = questions_rating[query_id]
            
            for item in rating_data:
                paragraph_id = item['paragraph_id']
                
                if query_id in paragraphs_data:
                    temp_data = paragraphs_data[query_id]
                    temp_data[paragraph_id] = item['text']
                    paragraphs_data[query_id] = temp_data
                else:
                    temp_data = dict()
                    temp_data[paragraph_id] = item['text']
                    paragraphs_data[query_id] = temp_data
                    
                para_ratings = dict()
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if dataset == 'car':
                                if rate['question_id'].startswith("tqa2:"):
                                    para_ratings[rate['question_id']] = rate['self_rating']/5
                            else:
                                para_ratings[rate['question_id']] = rate['self_rating']/5
                
                query_ratings_data[paragraph_id] = para_ratings
            questions_rating[query_id] = query_ratings_data
            
    return questions_rating, paragraphs_data

def read_dl_ranking_autograder_file(file_path:str,
                         dataset: str) -> Tuple:
    
    '''
    return format of questions_rating:
        {query_id:
             {
            paragraph_id1:
                  {
                      question_id1: rating,
                      question_id2: rating,
                  },
            paragraph_id2:
                {
                    question_id1: rating,
                    question_id2: rating,
                }
              }
        }
            
    return format of paragraphs_data:
        {query_id1:
             {
                 paragraph_id1: paragraph_text,
                 paragraph_id2: paragraph_text
            },
        query_id2:
            {
                paragraph_id1: paragraph_text,
                paragraph_id2: paragraph_text
            }
        }
    '''
    
    paragraphs_data = dict()
    questions_rating = dict()
    
    with open(file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            rating_data = data[1]
            
            query_ratings_data = dict()
                
            if query_id in questions_rating:
                query_ratings_data = questions_rating[query_id]
            
            for item in rating_data:
                paragraph_id = item['paragraph_id']
                
                if query_id in paragraphs_data:
                    temp_data = paragraphs_data[query_id]
                    temp_data[paragraph_id] = item['text']
                    paragraphs_data[query_id] = temp_data
                else:
                    temp_data = dict()
                    temp_data[paragraph_id] = item['text']
                    paragraphs_data[query_id] = temp_data
                    
                para_ratings = dict()
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            pos_rel = [rate['self_rating']/5]
                            pos_rel.extend(_get_multiple_ranking_data(item, query_id,"dl-rank"))
                            para_ratings[rate['question_id']] = pos_rel
                
                query_ratings_data[paragraph_id] = para_ratings
            questions_rating[query_id] = query_ratings_data
            
    return questions_rating, paragraphs_data

def read_trec_y3_section_autograder_data(queries_data: Dict[str, str],
                                         file_path: str) -> Tuple:
    
    '''
    return format of questions_rating:
        {query_id:
             {
            paragraph_id1:
                  {
                      question_id1: rating,
                      question_id2: rating,
                  },
            paragraph_id2:
                {
                    question_id1: rating,
                    question_id2: rating,
                }
              }
        }
            
    return format of paragraphs_data:
        {query_id1:
             {
                 paragraph_id1: paragraph_text,
                 paragraph_id2: paragraph_text
            },
        query_id2:
            {
                paragraph_id1: paragraph_text,
                paragraph_id2: paragraph_text
            }
        }
    '''
    
    paragraphs_data = dict()
    queries_data = dict()
    
    with open(file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            rating_data = data[1]
            
            query_ratings_data = dict()
            
            for item in rating_data:
                paragraph_id = item['paragraph_id']
                paragraph_text = item['text']
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if rate['question_id'].startswith("tqa2:"):
                                query_id = rate['question_id'].rsplit('/', 1)[0]
                                
                                if query_id in queries_data:
                                    query_ratings_data = queries_data[query_id]
                                
                                    if paragraph_id in query_ratings_data:
                                        temp_data = query_ratings_data[paragraph_id]
                                        temp_data[rate['question_id']] = rate['self_rating']
                                        query_ratings_data[paragraph_id] = temp_data
                                    else:
                                        temp_data = dict()
                                        temp_data[rate['question_id']] = rate['self_rating']
                                        query_ratings_data[paragraph_id] = temp_data
                                        
                                    queries_data[query_id] = query_ratings_data
                                else:
                                    temp_para_data = dict()
                                    temp_rating_data = dict()
                                    
                                    temp_rating_data[rate['question_id']] = rate['self_rating']
                                    temp_para_data[paragraph_id] = temp_rating_data
                                    
                                    queries_data[query_id] = temp_para_data
                                    
                                if query_id in paragraphs_data:
                                    paragraphs = paragraphs_data[query_id]
                                    paragraphs[paragraph_id] = paragraph_text
                                    paragraphs_data[query_id] = paragraphs
                                else:
                                    paragraphs = dict()
                                    paragraphs[paragraph_id] = paragraph_text
                                    paragraphs_data[query_id] = paragraphs
                                    
    return queries_data, paragraphs_data


def read_trec_y3_section_ranking_autograder_data(queries_data: Dict[str, str],
                                         file_path: str) -> Tuple:
    
    '''
    return format of questions_rating:
        {query_id:
             {
            paragraph_id1:
                  {
                      question_id1: rating,
                      question_id2: rating,
                  },
            paragraph_id2:
                {
                    question_id1: rating,
                    question_id2: rating,
                }
              }
        }
            
    return format of paragraphs_data:
        {query_id1:
             {
                 paragraph_id1: paragraph_text,
                 paragraph_id2: paragraph_text
            },
        query_id2:
            {
                paragraph_id1: paragraph_text,
                paragraph_id2: paragraph_text
            }
        }
    '''
    
    paragraphs_data = dict()
    queries_data = dict()
    
    with open(file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            rating_data = data[1]
            
            query_ratings_data = dict()
            
            for item in rating_data:
                paragraph_id = item['paragraph_id']
                paragraph_text = item['text']
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if rate['question_id'].startswith("tqa2:"):
                                query_id = rate['question_id'].rsplit('/', 1)[0]
                                
                                if query_id in queries_data:
                                    query_ratings_data = queries_data[query_id]
                                
                                    if paragraph_id in query_ratings_data:
                                        temp_data = query_ratings_data[paragraph_id]
                                        
                                        pos_rel = [rate['self_rating']/5]
                                        pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                            
                                        temp_data[rate['question_id']] = pos_rel
                                        query_ratings_data[paragraph_id] = temp_data
                                    else:
                                        temp_data = dict()
                                        
                                        pos_rel = [rate['self_rating']/5]
                                        pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                        
                                        temp_data[rate['question_id']] = pos_rel
                                        query_ratings_data[paragraph_id] = temp_data
                                        
                                    queries_data[query_id] = query_ratings_data
                                else:
                                    temp_para_data = dict()
                                    temp_rating_data = dict()
                                    
                                    pos_rel = [rate['self_rating']/5]
                                    pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                    
                                    temp_rating_data[rate['question_id']] = pos_rel
                                    temp_para_data[paragraph_id] = temp_rating_data
                                    
                                    queries_data[query_id] = temp_para_data
                                    
                                if query_id in paragraphs_data:
                                    paragraphs = paragraphs_data[query_id]
                                    paragraphs[paragraph_id] = paragraph_text
                                    paragraphs_data[query_id] = paragraphs
                                else:
                                    paragraphs = dict()
                                    paragraphs[paragraph_id] = paragraph_text
                                    paragraphs_data[query_id] = paragraphs
                                    
    print(f"*********** {len(queries_data)} {len(paragraphs_data)}")
    return queries_data, paragraphs_data
                    

def get_neighbors_data(passage_ratings_data,
                      query_questions_data,
                      entity_id,
                      entity_text,
                      symbols_embed,
                      ent_embed):
    
    neighbors_data = []
    
    for neighbors_id, relevance in passage_ratings_data.items():
        temp_neighbors_data = dict()
        temp_neighbors_data['paraid'] = neighbors_id
        temp_neighbors_data['paratext'] = query_questions_data[neighbors_id]
        temp_neighbors_data['parascore'] = [relevance]
        temp_neighbors_data['paraembed'] = symbols_embed[neighbors_id]
        
        neighbors_data.append(temp_neighbors_data)
    
    temp_neighbors_data = dict()
    temp_neighbors_data['entid'] = entity_id
    temp_neighbors_data['enttext'] = entity_text
    temp_neighbors_data['entscore'] = [5]
    temp_neighbors_data['entembed'] = ent_embed
    neighbors_data.append(temp_neighbors_data)
    
    return neighbors_data


def get_dl_answers_neighbors_data(passage_ratings_data,
                      query_questions_data,
                      entity_id,
                      entity_text,
                      symbols_embed,
                      ent_embed):
    
    neighbors_data = []
    
    for neighbors_id, relevance in passage_ratings_data.items():
        temp_neighbors_data = dict()
        temp_neighbors_data['paraid'] = neighbors_id
        temp_neighbors_data['paratext'] = query_questions_data[neighbors_id]
        temp_neighbors_data['parascore'] = [relevance]
        split_id = neighbors_id.split('/')
        paragraph_answers_id = split_id[0]+'/'+entity_id+'/'+split_id[1]
        
        #split_id = neighbors_id.split('/')
        #paragraph_answers_id = split_id[0]+'/'+split_id[1]+'/'+'/'+entity_id+'/'+split_id[2]
        temp_neighbors_data['paraembed'] = symbols_embed[paragraph_answers_id]
        
        neighbors_data.append(temp_neighbors_data)
    
    temp_neighbors_data = dict()
    temp_neighbors_data['entid'] = entity_id
    temp_neighbors_data['enttext'] = entity_text
    temp_neighbors_data['entscore'] = [5]
    temp_neighbors_data['entembed'] = ent_embed
    neighbors_data.append(temp_neighbors_data)
    
    return neighbors_data

def get_trec_car_answers_neighbors_data(passage_ratings_data,
                      query_questions_data,
                      entity_id,
                      entity_text,
                      symbols_embed,
                      ent_embed):
    
    neighbors_data = []
    
    for neighbors_id, relevance in passage_ratings_data.items():
        temp_neighbors_data = dict()
        temp_neighbors_data['paraid'] = neighbors_id
        temp_neighbors_data['paratext'] = query_questions_data[neighbors_id]
        temp_neighbors_data['parascore'] = [relevance]
        #split_id = neighbors_id.split('/')
        #paragraph_answers_id = split_id[0]+'/'+entity_id+'/'+split_id[1]
        
        split_id = neighbors_id.split('/')
        paragraph_answers_id = split_id[0]+'/'+split_id[1]+'/'+entity_id+'/'+split_id[2]
        temp_neighbors_data['paraembed'] = symbols_embed[paragraph_answers_id]
        
        neighbors_data.append(temp_neighbors_data)
    
    temp_neighbors_data = dict()
    temp_neighbors_data['entid'] = entity_id
    temp_neighbors_data['enttext'] = entity_text
    temp_neighbors_data['entscore'] = [5]
    temp_neighbors_data['entembed'] = ent_embed
    neighbors_data.append(temp_neighbors_data)
    
    return neighbors_data

def get_ranking_neighbors_data(passage_ratings_data,
                      query_questions_data,
                      entity_id,
                      entity_text,
                      symbols_embed,
                      ent_embed):
    
    neighbors_data = []
    
    for neighbors_id, relevance in passage_ratings_data.items():
        temp_neighbors_data = dict()
        temp_neighbors_data['paraid'] = neighbors_id
        temp_neighbors_data['paratext'] = query_questions_data[neighbors_id]
        temp_neighbors_data['parascore'] = relevance
        temp_neighbors_data['paraembed'] = symbols_embed[neighbors_id]
        
        neighbors_data.append(temp_neighbors_data)
    
    temp_neighbors_data = dict()
    temp_neighbors_data['entid'] = entity_id
    temp_neighbors_data['enttext'] = entity_text
    temp_neighbors_data['entscore'] = [5/5, 1]
    temp_neighbors_data['entembed'] = ent_embed
    neighbors_data.append(temp_neighbors_data)
    
    return neighbors_data

def get_multiple_ranking_neighbors_data(passage_ratings_data,
                      query_questions_data,
                      entity_id,
                      entity_text,
                      symbols_embed,
                      ent_embed,
                      dataset):
    
    neighbors_data = []
    
    for neighbors_id, relevance in passage_ratings_data.items():
        temp_neighbors_data = dict()
        temp_neighbors_data['paraid'] = neighbors_id
        temp_neighbors_data['paratext'] = query_questions_data[neighbors_id]
        temp_neighbors_data['parascore'] = relevance
        
        # This is for the answers symbol
        
        if dataset == 'dl-rank':
            split_id = neighbors_id.split('/')
            neighbors_id = split_id[0]+'/'+entity_id+'/'+split_id[1]
        else:
            split_id = neighbors_id.split('/')
            neighbors_id = split_id[0]+'/'+split_id[1]+'/'+entity_id+'/'+split_id[2]
        
        temp_neighbors_data['paraembed'] = symbols_embed[neighbors_id]
        
        neighbors_data.append(temp_neighbors_data)
    
    temp_neighbors_data = dict()
    temp_neighbors_data['entid'] = entity_id
    temp_neighbors_data['enttext'] = entity_text
    temp_neighbors_data['entscore'] = [5/5, 1, 1, 1]
    temp_neighbors_data['entembed'] = ent_embed
    neighbors_data.append(temp_neighbors_data)
    
    return neighbors_data


def _get_ranking_relevance_data(item: Dict,
                                query_id: str,
                                rate: Dict):
    counter = 0
    pos_rel = []
    
    for ranking in item['paragraph_data']['rankings']:
        if ranking['method'] == TREC_RANKING_METHOD and ranking['queryId'] == query_id:
            pos_rel.append([rate['self_rating']/5, (1/ranking['rank'])])
            counter = 1
    
    if counter == 0:
        pos_rel.append([rate['self_rating']/5, 0])
        
    return pos_rel


def _get_multiple_ranking_data(ranking_dict: Dict,
                      queryId: str,
                      dataset: str) -> List:
    
    '''
    This returns a list of ranks of the top 3 rankings
    1) dangnt-nlp
    2) ReRnak3_BERT
    3) ReRnak2_BERT
    
    If the ranking does not exist for the particular paragraph in the rankings,
    then 0 is added
    '''
    
    TREC_RANKING_METHOD1 = 'dangnt-nlp'
    TREC_RANKING_METHOD2 = 'ReRnak3_BERT'
    TREC_RANKING_METHOD3 = 'ReRnak2_BERT'
    
    if dataset == 'dl-rank':
        TREC_RANKING_METHOD1 = 'p_d2q_rm3_duo'
        TREC_RANKING_METHOD2 = 'p_d2q_bm25_duo'
        #TREC_RANKING_METHOD3 = 'p_bm25rm3_duo'
        TREC_RANKING_METHOD3 = 'bigIR-T5-R'
        
    
    counter = 0
    ranking_data = []
    
    for ranking in ranking_dict['paragraph_data']['rankings']:
        if ranking['method'] == TREC_RANKING_METHOD1 and ranking['queryId'] == queryId:
            ranking_data.append((1/ranking['rank']))
            counter = 1
            
    if counter == 0:
        ranking_data.append(0)
        
    counter = 0
    for ranking in ranking_dict['paragraph_data']['rankings']:
        if ranking['method'] == TREC_RANKING_METHOD2 and ranking['queryId'] == queryId:
            ranking_data.append((1/ranking['rank']))
            counter = 1
                       
    if counter == 0:
        ranking_data.append(0)
        
        
    counter = 0
    for ranking in ranking_dict['paragraph_data']['rankings']:
        if ranking['method'] == TREC_RANKING_METHOD3 and ranking['queryId'] == queryId:
            ranking_data.append((1/ranking['rank']))
            counter = 1
                       
    if counter == 0:
        ranking_data.append(0)
        
    return ranking_data