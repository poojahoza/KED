#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:13:42 2024

@author: poojaoza
"""

import json
from typing import Dict, List
from ..utils import _get_multiple_ranking_data


def _retrieve_pointwise_dl_samples(test_qrel_file_path: str,
              train_data: Dict) -> List[Dict]:
    test_passages = dict()
    
    with open(test_qrel_file_path, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = line.split()
            query_id = data[0]
            passage_id = data[2]
            passage_relevance = int(data[3])
            
            if query_id in train_data:
                if query_id in test_passages:
                    temp_data = test_passages[query_id]
                    temp_data[passage_id] = passage_relevance
                    test_passages[query_id] = temp_data
                else:
                    temp_data = dict()
                    temp_data[passage_id] = passage_relevance
                    test_passages[query_id] = temp_data
    return test_passages

def _retrieve_pointwise_trecy3_section_samples(autograde_file: str,
                                      qrel_data: Dict,
                                      re_rank: bool,
                                      re_rank_method: str) -> List[Dict]:
    import lzma
    
    test_passages = dict()
    
    with open(autograde_file, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            paragraphs = data[1]
            
            for item in paragraphs:
                paragraph_id = item['paragraph_id']
                
                is_ranked = False
                
                if re_rank:                
                    for ranking in item['paragraph_data']['rankings']:
                        if ranking['method'].lower() == re_rank_method.lower():
                            is_ranked = True
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if rate['question_id'].startswith("tqa2:"):
                                query_id = rate['question_id'].rsplit('/', 1)[0]
                                
                                if re_rank:
                                    if is_ranked:
                                        if query_id in qrel_data:
                                            if query_id in test_passages:
                                                if paragraph_id in qrel_data[query_id]:
                                                    temp_data = test_passages[query_id]
                                                    temp_data[paragraph_id] = qrel_data[query_id][paragraph_id]
                                                    test_passages[query_id] = temp_data
                                                else:
                                                    temp_data = test_passages[query_id]
                                                    temp_data[paragraph_id] = 0
                                                    test_passages[query_id] = temp_data
                                            else:
                                                if paragraph_id in qrel_data[query_id]:
                                                    temp_data = dict()
                                                    temp_data[paragraph_id] = qrel_data[query_id][paragraph_id]
                                                    test_passages[query_id] = temp_data
                                                else:
                                                    temp_data = dict()
                                                    temp_data[paragraph_id] = 0
                                                    test_passages[query_id] = temp_data
                                else:
                                    if query_id in qrel_data:
                                        if query_id in test_passages:
                                            if paragraph_id in qrel_data[query_id]:
                                                temp_data = test_passages[query_id]
                                                temp_data[paragraph_id] = qrel_data[query_id][paragraph_id]
                                                test_passages[query_id] = temp_data
                                            else:
                                                temp_data = test_passages[query_id]
                                                temp_data[paragraph_id] = 0
                                                test_passages[query_id] = temp_data
                                        else:
                                            if paragraph_id in qrel_data[query_id]:
                                                temp_data = dict()
                                                temp_data[paragraph_id] = qrel_data[query_id][paragraph_id]
                                                test_passages[query_id] = temp_data
                                            else:
                                                temp_data = dict()
                                                temp_data[paragraph_id] = 0
                                                test_passages[query_id] = temp_data
                                
                        
    return test_passages


def _retrieve_pointwise_trecy3_section_ranking_samples(autograde_file: str,
                                      qrel_data: Dict,
                                      re_rank: bool,
                                      re_rank_method: str) -> List[Dict]:
    import lzma
    
    test_passages = dict()
    
    with open(autograde_file, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            paragraphs = data[1]
            
            for item in paragraphs:
                paragraph_id = item['paragraph_id']
                
                is_ranked = False
                
                if re_rank:                
                    for ranking in item['paragraph_data']['rankings']:
                        if ranking['method'].lower() == re_rank_method.lower():
                            is_ranked = True
                
                for grade in item['exam_grades']:
                    if grade['prompt_info']['is_self_rated'] and grade['prompt_info']['prompt_class'] == 'QuestionSelfRatedUnanswerablePromptWithChoices':
                        for rate in grade['self_ratings']:
                            if rate['question_id'].startswith("tqa2:"):
                                query_id = rate['question_id'].rsplit('/', 1)[0]
                                
                                if re_rank:
                                    if is_ranked:
                                        if query_id in qrel_data:
                                            if query_id in test_passages:
                                                if paragraph_id in qrel_data[query_id]:
                                                    temp_data = test_passages[query_id]
                                                    
                                                    pos_rel = [qrel_data[query_id][paragraph_id]/5]
                                                    pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                    temp_data[paragraph_id] = pos_rel
                                                    
                                                    test_passages[query_id] = temp_data
                                                else:
                                                    temp_data = test_passages[query_id]
                                                    
                                                    pos_rel = [0]
                                                    pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                    temp_data[paragraph_id] = pos_rel
                                                    
                                                    test_passages[query_id] = temp_data
                                            else:
                                                if paragraph_id in qrel_data[query_id]:
                                                    temp_data = dict()
                                                    
                                                    pos_rel = [qrel_data[query_id][paragraph_id]/5]
                                                    pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                    temp_data[paragraph_id] = pos_rel
                                                    
                                                    test_passages[query_id] = temp_data
                                                else:
                                                    temp_data = dict()
                                                    
                                                    pos_rel = [0]
                                                    pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                    temp_data[paragraph_id] = pos_rel
                                                    
                                                    test_passages[query_id] = temp_data
                                else:
                                    if query_id in qrel_data:
                                        if query_id in test_passages:
                                            if paragraph_id in qrel_data[query_id]:
                                                temp_data = test_passages[query_id]
                                               
                                                pos_rel = [qrel_data[query_id][paragraph_id]/5]
                                                pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                temp_data[paragraph_id] = pos_rel
                                                test_passages[query_id] = temp_data
                                            else:
                                                temp_data = test_passages[query_id]
                                                
                                                pos_rel = [0]
                                                pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                temp_data[paragraph_id] = pos_rel
                                                test_passages[query_id] = temp_data
                                        else:
                                            if paragraph_id in qrel_data[query_id]:
                                                temp_data = dict()
                                                
                                                pos_rel = [qrel_data[query_id][paragraph_id]/5]
                                                pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                temp_data[paragraph_id] = pos_rel
                                                test_passages[query_id] = temp_data
                                            else:
                                                temp_data = dict()
                                                
                                                pos_rel = [0]
                                                pos_rel.extend(_get_multiple_ranking_data(item, query_id, "car-section-rank"))
                                                temp_data[paragraph_id] = pos_rel
                                                test_passages[query_id] = temp_data
                                
                        
    return test_passages



def _retrieve_pointwise_trecy3_samples(autograde_file: str,
                                      qrel_data: Dict) -> List[Dict]:
    import lzma
    
    test_passages = dict()
    
    with open(autograde_file, 'rt', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            query_id = data[0]
            paragraphs = data[1]
            
            for para in paragraphs:
                paragraph_id = para['paragraph_id']
                
                if query_id in qrel_data:
                    if paragraph_id in qrel_data[query_id]:
                        temp_data = test_passages[query_id]
                        temp_data[paragraph_id] = qrel_data[query_id][paragraph_id]
                        test_passages[query_id] = temp_data
                    else:
                        temp_data = dict()
                        temp_data[paragraph_id] = 0
                        test_passages[query_id] = temp_data
                        
    return test_passages