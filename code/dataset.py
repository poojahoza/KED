from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EntityRankingDataset(Dataset):
    def __init__(
            self,
            dataset,
            train: bool,
            experiment: str,
            kfold: bool
    ):
        self._dataset = dataset
        self._train = train
        self._experiment = experiment
        self._kfold = kfold
        self._read_data()
        

        self._count = len(self._examples)

    def _read_data(self):
        if (self._experiment == "parasymbtoken" or self._experiment == "parasymbembedding" or self._experiment == 'parasymbtunedbert' or self._experiment == 'parasymbonlyselfratingwithoutgnn') and self._kfold:
            #self._examples = [json.loads(line) for line in self._dataset]
            self._examples = self._dataset
        else:
            with open(self._dataset, 'r') as f:
                self._examples = [json.loads(line) for i, line in enumerate(f)]

    def __len__(self) -> int:
        return self._count


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._train:
            # Create the entity inputs
            if self._experiment == 'paragrn' or self._experiment == 'paragat' or self._experiment == 'paragrntrans' or self._experiment == 'paragattrans' or self._experiment == 'paratunedberttrans' or self._experiment == 'paraentitytrans' or self._experiment == 'paragrnneighborstrans' or self._experiment == 'paragatasp' or self._experiment == 'paragrnasp':
                query_emb = np.array(example['query']['query_embed'])
                query_id = example['query']['query_id']
                ent_emb_pos = np.array(example['ent_pos']['entity_embedding'])
                ent_emb_neg = np.array(example['ent_neg']['entity_embedding'])
                pos_ent_neighbors = example['ent_pos']['entity_neighbors']
                neg_ent_neighbors = example['ent_neg']['entity_neighbors']
                ent_pos_label = 1
                ent_neg_label = 0
                ent_pos_id = example['ent_pos']['entity_id']
                ent_neg_id = example['ent_neg']['entity_id']

                return {
                    'query_emb': query_emb,
                    'query_id': query_id,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id
                }
            elif self._experiment == 'parasymbtoken':
                
                query_id = example['query']['query_id']
                pos_ent_neighbors = example['ent_pos']['entity_neighbors']
                neg_ent_neighbors = example['ent_neg']['entity_neighbors']
                ent_pos_label = example['ent_pos']['entity_relevance']
                ent_neg_label = example['ent_neg']['entity_relevance']
                ent_pos_id = example['ent_pos']['entity_id']
                ent_neg_id = example['ent_neg']['entity_id']
                pos_ent_text = example['ent_pos']['entity_text']
                neg_ent_text = example['ent_neg']['entity_text']
                query_text = example['query']['query_text']

                return {
                    'query_id': query_id,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text
                }
            elif self._experiment == 'parasymbembedding' or self._experiment == 'parasymbtunedbert':
                
                query_id = example['query']['query_id']
                query_emb = np.array(example['query']['query_embed'])
                pos_ent_neighbors = example['ent_pos']['entity_neighbors']
                neg_ent_neighbors = example['ent_neg']['entity_neighbors']
                ent_pos_label = example['ent_pos']['entity_relevance']
                ent_neg_label = example['ent_neg']['entity_relevance']
                ent_pos_id = example['ent_pos']['entity_id']
                ent_neg_id = example['ent_neg']['entity_id']
                pos_ent_text = example['ent_pos']['entity_text']
                neg_ent_text = example['ent_neg']['entity_text']
                query_text = example['query']['query_text']
                ent_emb_pos = np.array(example['ent_pos']['entity_embed'])
                ent_emb_neg = np.array(example['ent_neg']['entity_embed'])

                return {
                    'query_id': query_id,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text,
                    'query_emb': query_emb,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                }
            else:
                query_emb = np.array(example['query']['query_embed'])
                query_id = example['query']['query_id']
                ent_emb_pos = np.array(example['ent_pos']['entity_embedding'])
                ent_emb_neg = np.array(example['ent_neg']['entity_embedding'])
                ent_pos_label = 1
                ent_neg_label = 0
                pos_ent_neighbors = example['ent_pos']['entity_neighbors']
                neg_ent_neighbors = example['ent_neg']['entity_neighbors']
                ent_pos_id = example['ent_pos']['entity_id']
                ent_neg_id = example['ent_neg']['entity_id']
                pos_ent_text = example['ent_pos']['entity_text']
                neg_ent_text = example['ent_neg']['entity_text']
                query_text = example['query']['query_text']

                return {
                    'query_emb': query_emb,
                    'query_id': query_id,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text
                }
        else:
            if self._experiment == 'paragrn' or self._experiment == 'paragat' or self._experiment == 'paragrntrans' or self._experiment == 'paragattrans' or self._experiment == 'paratunedberttrans' or self._experiment == 'paraentitytrans' or self._experiment == 'paragrnneighborstrans' or self._experiment == 'paragatasp' or self._experiment == 'paragrnasp':
                query_emb = np.array(example['query']['query_embed'])
                ent_emb = np.array(example['entity']['entity_embedding'])
                neighbors = example['entity']['entity_neighbors']

                return{
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': example['label'],
                    'query_id': example['query_id'],
                    'entity_id': example['entity']['entity_id'],
                    'entity_neighbors': neighbors
                }
            elif self._experiment == 'parasymbtoken':
                neighbors = example['entity']['entity_neighbors']
                entity_text = example['entity']['entity_text']
                query_text = example['query']['query_text']
                
                return {
                    'label': example['entity']['entity_relevance'],
                    'query_id': example['query']['query_id'],
                    'entity_id': example['entity']['entity_id'],
                    'entity_neighbors': neighbors,
                    'entity_text': entity_text,
                    'query_text': query_text
                }
            elif self._experiment == 'parasymbembedding' or self._experiment == 'parasymbtunedbert' or self._experiment == 'parasymbonlyselfratingwithoutgnn':
                neighbors = example['entity']['entity_neighbors']
                query_emb = np.array(example['query']['query_embed'])
                ent_emb = np.array(example['entity']['entity_embed'])
                
                return {
                    'label': example['entity']['entity_relevance'],
                    'query_id': example['query']['query_id'],
                    'entity_id': example['entity']['entity_id'],
                    'entity_neighbors': neighbors,
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                }
            else:
                query_emb = np.array(example['query']['query_embed'])
                ent_emb = np.array(example['entity']['entity_embedding'])
                neighbors = example['entity']['entity_neighbors']
                entity_text = example['entity']['entity_text']
                query_text = example['query']['query_text']

                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': example['label'],
                    'query_id': example['query_id'],
                    'entity_id': example['entity']['entity_id'],
                    'entity_neighbors': neighbors,
                    'entity_text': entity_text,
                    'query_text': query_text
                }

    def collate(self, batch):
        if self._train:
            if self._experiment == 'paragrn' or self._experiment == 'paragat' or self._experiment == 'paragrntrans' or self._experiment == 'paragattrans' or self._experiment == 'paratunedberttrans' or self._experiment == 'paraentitytrans' or self._experiment == 'paragrnneighborstrans' or self._experiment == 'paragatasp' or self._experiment == 'paragrnasp':
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb_pos = torch.from_numpy(np.array([item['ent_emb_pos'] for item in batch])).float()
                ent_emb_neg = torch.from_numpy(np.array([item['ent_emb_neg'] for item in batch])).float()
                pos_ent_neighbors = [item['pos_ent_neighbors'] for item in batch]
                neg_ent_neighbors = [item['neg_ent_neighbors'] for item in batch]
                query_id = [item['query_id'] for item in batch]
                ent_pos_label = [item['ent_pos_label'] for item in batch]
                ent_neg_label = [item['ent_neg_label'] for item in batch]
                ent_pos_id = [item['ent_pos_id'] for item in batch]
                ent_neg_id = [item['ent_neg_id'] for item in batch]

                return{
                    'query_emb': query_emb,
                    'query_id': query_id,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id
                }
            elif self._experiment == 'parasymbtoken':
                
                
                pos_ent_neighbors = [item['pos_ent_neighbors'] for item in batch]
                neg_ent_neighbors = [item['neg_ent_neighbors'] for item in batch]
                query_id = [item['query_id'] for item in batch]
                ent_pos_label = [item['ent_pos_label'] for item in batch]
                ent_neg_label = [item['ent_neg_label'] for item in batch]
                ent_pos_id = [item['ent_pos_id'] for item in batch]
                ent_neg_id = [item['ent_neg_id'] for item in batch]
                pos_ent_text = [item['pos_ent_text'] for item in batch]
                neg_ent_text = [item['neg_ent_text'] for item in batch]
                query_text = [item['query_text'] for item in batch]
                
                return {
                    'query_id': query_id,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text
                }
            elif self._experiment == 'parasymbembedding' or self._experiment == 'parasymbtunedbert' or self._experiment == 'parasymbonlyselfratingwithoutgnn':
                
                
                pos_ent_neighbors = [item['pos_ent_neighbors'] for item in batch]
                neg_ent_neighbors = [item['neg_ent_neighbors'] for item in batch]
                query_id = [item['query_id'] for item in batch]
                ent_pos_label = [item['ent_pos_label'] for item in batch]
                ent_neg_label = [item['ent_neg_label'] for item in batch]
                ent_pos_id = [item['ent_pos_id'] for item in batch]
                ent_neg_id = [item['ent_neg_id'] for item in batch]
                pos_ent_text = [item['pos_ent_text'] for item in batch]
                neg_ent_text = [item['neg_ent_text'] for item in batch]
                query_text = [item['query_text'] for item in batch]
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb_pos = torch.from_numpy(np.array([item['ent_emb_pos'] for item in batch])).float()
                ent_emb_neg = torch.from_numpy(np.array([item['ent_emb_neg'] for item in batch])).float()
                
                return {
                    'query_id': query_id,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text,
                    'query_emb': query_emb,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg
                }
            else:
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb_pos = torch.from_numpy(np.array([item['ent_emb_pos'] for item in batch])).float()
                ent_emb_neg = torch.from_numpy(np.array([item['ent_emb_neg'] for item in batch])).float()
                pos_ent_neighbors = [item['pos_ent_neighbors'] for item in batch]
                neg_ent_neighbors = [item['neg_ent_neighbors'] for item in batch]
                query_id = [item['query_id'] for item in batch]
                ent_pos_label = [item['ent_pos_label'] for item in batch]
                ent_neg_label = [item['ent_neg_label'] for item in batch]
                ent_pos_id = [item['ent_pos_id'] for item in batch]
                ent_neg_id = [item['ent_neg_id'] for item in batch]
                pos_ent_text = [item['pos_ent_text'] for item in batch]
                neg_ent_text = [item['neg_ent_text'] for item in batch]
                query_text = [item['query_text'] for item in batch]

                return {
                    'query_emb': query_emb,
                    'query_id': query_id,
                    'ent_emb_pos': ent_emb_pos,
                    'ent_emb_neg': ent_emb_neg,
                    'pos_ent_neighbors': pos_ent_neighbors,
                    'neg_ent_neighbors': neg_ent_neighbors,
                    'ent_pos_label': ent_pos_label,
                    'ent_neg_label': ent_neg_label,
                    'ent_pos_id': ent_pos_id,
                    'ent_neg_id': ent_neg_id,
                    'pos_ent_text': pos_ent_text,
                    'neg_ent_text': neg_ent_text,
                    'query_text': query_text
                }
        else:
            if self._experiment == 'paragrn' or self._experiment == 'paragat' or self._experiment == 'paragrntrans' or self._experiment == 'paragattrans' or self._experiment == 'paratunedberttrans' or self._experiment == 'paraentitytrans' or self._experiment == 'paragrnneighborstrans' or self._experiment == 'paragatasp' or self._experiment == 'paragrnasp':
                query_id = [item['query_id'] for item in batch]
                entity_id = [item['entity_id'] for item in batch]
                label = [item['label'] for item in batch]
                entity_neighbors = [item['entity_neighbors'] for item in batch]
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb = torch.from_numpy(np.array([item['ent_emb'] for item in batch])).float()

                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': label,
                    'query_id': query_id,
                    'entity_id': entity_id,
                    'entity_neighbors': entity_neighbors
                }
            elif self._experiment == 'parasymbtoken':
                query_id = [item['query_id'] for item in batch]
                entity_id = [item['entity_id'] for item in batch]
                label = [item['label'] for item in batch]
                entity_neighbors = [item['entity_neighbors'] for item in batch]
                entity_text = [item['entity_text'] for item in batch]
                query_text = [item['query_text'] for item in batch]

                return {
                    'label': label,
                    'query_id': query_id,
                    'entity_id': entity_id,
                    'entity_neighbors': entity_neighbors,
                    'entity_text': entity_text,
                    'query_text': query_text
                }
            elif self._experiment == 'parasymbembedding' or self._experiment == 'parasymbtunedbert' or self._experiment == 'parasymbonlyselfratingwithoutgnn':
                query_id = [item['query_id'] for item in batch]
                entity_id = [item['entity_id'] for item in batch]
                label = [item['label'] for item in batch]
                entity_neighbors = [item['entity_neighbors'] for item in batch]
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb = torch.from_numpy(np.array([item['ent_emb'] for item in batch])).float()

                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': label,
                    'query_id': query_id,
                    'entity_id': entity_id,
                    'entity_neighbors': entity_neighbors
                }
            else:
                query_id = [item['query_id'] for item in batch]
                entity_id = [item['entity_id'] for item in batch]
                label = [item['label'] for item in batch]
                entity_neighbors = [item['entity_neighbors'] for item in batch]
                query_emb = torch.from_numpy(np.array([item['query_emb'] for item in batch])).float()
                ent_emb = torch.from_numpy(np.array([item['ent_emb'] for item in batch])).float()
                entity_text = [item['entity_text'] for item in batch]
                query_text = [item['query_text'] for item in batch]

                return {
                    'query_emb': query_emb,
                    'ent_emb': ent_emb,
                    'label': label,
                    'query_id': query_id,
                    'entity_id': entity_id,
                    'entity_neighbors': entity_neighbors,
                    'entity_text': entity_text,
                    'query_text': query_text
                }
