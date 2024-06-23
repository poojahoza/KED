import torch
import tqdm
import utils

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, data_loader, device, experiment):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._data_loader = data_loader
        self._device = device
        self._result_dict = dict()
        self._experiment = experiment

    def make_train_step(self):
        # Builds function that performs a step in the train loop
        def train_step(train_batch):
            # Sets model to TRAIN mode
            self._model.train()

            # Zero the gradients
            self._optimizer.zero_grad()

            # Makes predictions and compute loss
            if self._experiment == 'grnall' or self._experiment == 'gatall':
                batch_score_pos = self._model(
                        query_emb = train_batch['query_emb'].to(self._device),
                        entity_emb = train_batch['ent_emb_pos'].to(self._device),
                        neighbors = train_batch['pos_ent_neighbors'],
                        entity_text = train_batch['pos_ent_text'],
                        query_text = train_batch['query_text']
                )

                batch_score_neg = self._model(
                        query_emb = train_batch['query_emb'].to(self._device),
                        entity_emb  = train_batch['ent_emb_neg'].to(self._device),
                        neighbors = train_batch['neg_ent_neighbors'],
                        entity_text = train_batch['neg_ent_text'],
                        query_text = train_batch['query_text']
                )
            elif self._experiment == 'parasymbtoken':
                batch_score_pos = self._model(
                        neighbors = train_batch['pos_ent_neighbors'],
                        entity_text = train_batch['pos_ent_text'],
                        query_text = train_batch['query_text'],
                        task = 'retrieval'
                )

                batch_score_neg = self._model(
                        neighbors = train_batch['neg_ent_neighbors'],
                        entity_text = train_batch['neg_ent_text'],
                        query_text = train_batch['query_text'],
                        task = 'retrieval'
                )
            else:
                batch_score_pos = self._model(
                    query_emb = train_batch[ 'query_emb'].to(self._device),
                    entity_emb = train_batch[ 'ent_emb_pos'].to(self._device),
                    neighbors = train_batch['pos_ent_neighbors']
                )
                #print(batch_score_pos)

                batch_score_neg = self._model(
                    query_emb=train_batch['query_emb'].to(self._device),
                    entity_emb=train_batch['ent_emb_neg'].to(self._device),
                    neighbors=train_batch['neg_ent_neighbors']
                )

            batch_loss = self._criterion(
                batch_score_pos.tanh(),
                batch_score_neg.tanh(),
                torch.ones(batch_score_pos.size()).to(self._device)
            )
            #print(batch_loss)

            batch_score_pos = batch_score_pos.detach().cpu().tolist()
            batch_score_neg = batch_score_neg.detach().cpu().tolist()

            batch_queryid = train_batch['query_id']
            batch_pos_entity_id = train_batch['ent_pos_id']
            batch_neg_entity_id = train_batch['ent_neg_id']
            batch_pos_label = train_batch['ent_pos_label']
            batch_neg_label = train_batch['ent_neg_label']

            utils.prepare_result_dict(batch_queryid, batch_pos_entity_id, batch_score_pos, batch_pos_label, self._result_dict)

            utils.prepare_result_dict(batch_queryid, batch_neg_entity_id, batch_score_neg, batch_neg_label, self._result_dict)

            # Computes gradients
            batch_loss.backward()

            # Updates parameters
            self._optimizer.step()
            self._scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)
        self._result_dict = dict()

        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            batch_loss = train_step(batch)
            epoch_loss += batch_loss

        return epoch_loss, self._result_dict


