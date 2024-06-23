import torch
import tqdm
import utils

class Validate(object):
    def __init__(self, model, data_loader, device, experiment):
        self._model = model
        self._data_loader = data_loader
        self._device = device
        self._result_dict = dict()
        self._experiment = experiment

    def _validation_step(self, batch):

        query_id, entity_id, label = batch['query_id'], batch['entity_id'], batch['label']

        if self._experiment == 'grnall' or self._experiment == 'gatall':
            batch_score = self._model(
                    query_emb=batch['query_emb'].to(self._device),
                    entity_emb=batch['ent_emb'].to(self._device),
                    neighbors=batch['entity_neighbors'],
                    entity_text=batch['entity_text'],
                    query_text=batch['query_text']
            )
        elif self._experiment == 'parasymbtoken':
            batch_score = self._model(neighbors=batch['entity_neighbors'], 
                                      entity_text=batch['entity_text'], 
                                      query_text=batch['query_text'], 
                                      task='retrieval')
        else:
            batch_score = self._model(
                query_emb=batch['query_emb'].to(self._device),
                entity_emb=batch['ent_emb'].to(self._device),
                neighbors=batch['entity_neighbors']
            )

        return batch_score

    def validate(self):
        valid_loss = []
        self._model.eval()

        self._result_dict = dict()

        num_data = len(self._data_loader)

        with torch.no_grad():
            for eval_batch in tqdm.tqdm(self._data_loader, total=num_data):
                if eval_batch is not None:
                    eval_score = self._validation_step(eval_batch)
                    #valid_loss += eval_score.item()
                    eval_score = eval_score.detach().cpu().tolist()
                    valid_loss.extend(eval_score)
                    utils.prepare_result_dict(eval_batch['query_id'], 
                            eval_batch['entity_id'],
                            eval_score,
                            eval_batch['label'],
                            self._result_dict)

        valid_loss = torch.tensor(valid_loss).sum().item()
        return self._result_dict, valid_loss

