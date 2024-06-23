import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import utils
import metrics
import warnings
import argparse
import numpy as np
import pandas as pd
import random
from typing import Tuple
from dataloader import EntityRankingDataLoader
from model import NeuralECMModel, GRNECMModel, TunedBERTModel, ParaNeuralECMModel, NeuralECMTokenModel, ParaTransformerNeuralECMModel,ParaTransformerTunedBertModel, ParaTransformerNeighborsNeuralECMModel, ParaAspectNeuralECMModel, ParaTransformerParaEntityModel, ParaSymbolsTokenModel, ParaSymbolsEmbeddingModel, ParaSymbolsSelfRatingWithoutGNNEmbeddingModel
from transformers import get_linear_schedule_with_warmup
from dataset import EntityRankingDataset
from trainer import Trainer
from validate import Validate
from utils import read_embeddings_file


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save, run_file,
          eval_every, device, loss_csv, train_run_file, validator):
    best_valid_metric = 0.0

    total_training_loss = []
    total_validation_loss = []
    total_validation_metric = []

    for epoch in range(epochs):

        # Train
        start_time = time.time()
        train_loss, train_result = trainer.train()
        total_training_loss.append(train_loss)

        utils.save_trec(os.path.join(save_path, train_run_file), train_result)

        # Validate
        if (epoch + 1) % eval_every == 0:
            res_dict, validation_loss = validator.validate()
            #res_dict, validation_loss = utils.evaluate(model, valid_loader, device)
            #print(validation_loss)
            #print(res_dict)
            total_validation_loss.append(validation_loss)

            utils.save_trec(os.path.join(save_path, run_file), res_dict)
            valid_metric = utils.get_metric_eval(qrels, os.path.join(save_path, run_file), metric)

            total_validation_metric.append(valid_metric)

            if valid_metric >= best_valid_metric:
                best_valid_metric = valid_metric
                utils.save_checkpoint(os.path.join(save_path, save), model)
                utils.save_trec(os.path.join(save_path, run_file+'.best.txt'), res_dict)

            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(
                    f'\t Train Loss: {train_loss:.3f}| Validation Loss: {validation_loss: .3f} | Val. Metric: {valid_metric:.4f} | Best Val. Metric: {best_valid_metric:.4f}')

    loss_data = pd.DataFrame(list(zip(total_training_loss, total_validation_loss, total_validation_metric)), columns=['Training Loss', 'Validation Loss', 'Validation Metric'])
    loss_data.to_csv(loss_csv, index=False)



def main():
    parser = argparse.ArgumentParser("Script to train a model.")
    parser.add_argument('--experiment', help='Which experiment to run? choices: grn, gat, grnecm, tunedbert, paragrn, paragat, grnall, gatall, paragrnasp, paragatasp, paragrntrans, paragattrans, paratunedberttrans, paraentitytrans, paragrnneighborstrans, parasymbtoken', choices=['grn', 'gat', 'grnecm', 'tunedbert', 'paragrn', 'paragat', 'grnall', 'gatall', 'paragrnasp', 'paragatasp', 'paragrntrans', 'paragattrans', 'paratunedberttrans', 'paraentitytrans', 'paragrnneighborstrans', 'parasymbtoken', 'parasymbembedding', 'parasymbtunedbert', 'parasymbonlyselfratingwithoutgnn'], required=True)
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where model is saved.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save. Default: neuralecm.bin', default='neuralecm.bin', type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load. Default: None', default=None, type=str)
    parser.add_argument('--run', help='Output run file in TREC format.', required=True,type=str)
    parser.add_argument('--train-run', help='Train output run file in TREC format', required=True, type=str)
    parser.add_argument('--metric', help='Metric to use for evaluation. Default: map', default='map', type=str)
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 2e-5.', type=float, default=2e-5)
    parser.add_argument('--n-warmup-steps', help='Number of warmup steps for scheduling. Default: 1000.', type=int,
                        default=1000)
    parser.add_argument('--eval-every', help='Evaluate every number of epochs. Default: 1', type=int, default=1)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--query-in-emb-dim', help='Dimension of query input embedding.', required=True, type=int)
    parser.add_argument('--ent-in-emb-dim', help='Dimension of entity input embedding.', required=True, type=int)
    parser.add_argument('--para-in-emb-dim', help='Dimension of paragraph input embedding.', required=True, type=int)
    parser.add_argument('--model', help='Model to retreive query and paragraph representations', default='distilbert-base-uncased', type=str)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    parser.add_argument('--kfold', help='Whether or not to use KFold cross validation. Default: False.', action='store_true')
    parser.add_argument('--queries', help='Queries file in JSON format {id: text}', type=str)
    parser.add_argument('--generated-symbols', help='Questions | Nuggets file in jsonl.gz format', type=str)
    parser.add_argument('--autograder-file', help='File generated through autograder project', type=str)
    parser.add_argument('--queries-embed', help='Folder path containing queries embeddings file', type=str)
    parser.add_argument('--para-embed', help='Folder path containing paragraph embeddings file', type=str)
    parser.add_argument('--symb-embed', help='Folder path containing generated symbools embeddings file', type=str)
    parser.add_argument('--dataset', help='Which dataset to process for para symbols token? dl or car or car-section. Default=car', type=str, default='car')
    parser.add_argument('--seed', help='Random seed initialization',type=int, default=49500)
    parser.add_argument('--re-rank', help='Whether or not to re-rank any method. Default: False.', action='store_true')
    parser.add_argument('--re-rank-method', help='Which method to re-rank?', type=str)
    #parser.add_argument('--layer-flag', help='Flag to select the GNN model (1: GRN| 2:GAT), default = 1', type=int, default=1)
    parser.add_argument('--loss-csv', help='csv file path to store all the losses for each epoch',type=str)
    parser.add_argument('--para-aggrg', help='Which aggregation to use to select the feature for para neural ecm (max, linear, prod, entrank)',choices=['max','linear', 'prod', 'entrank'])
    parser.add_argument('--asp-aggrg', help='Which function to use to learn aspect for para neural ecm (linear, prod, bilinear)',choices=['linear', 'prod', 'bilinear'])
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = json.dumps({
        'Experiment': args.experiment,
        'Metric': args.metric,
        'Epochs': args.epoch,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Warmup Steps': args.n_warmup_steps,
    })
    config_file: str = os.path.join(args.save_dir, 'config.json')
    with open(config_file, 'w') as f:
        f.write("%s\n" % model_config)

    
    model = None

    print('Experiment: ' + args.experiment)
    if args.experiment == 'grn' or args.experiment == 'gat':
        model = NeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim, 
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment)
    elif args.experiment == 'grnecm':
        model = GRNECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'tunedbert':
        model = TunedBERTModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'paratunedberttrans':
        model = ParaTransformerTunedBertModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'paraentitytrans':
        model = ParaTransformerParaEntityModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'paragrn' or args.experiment == 'paragat':
        model = ParaNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment,
                feature_selection=args.para_aggrg)
    elif args.experiment == 'paragrntrans' or args.experiment == 'paragattrans':
        model = ParaTransformerNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment,
                feature_selection=args.para_aggrg)
    elif args.experiment == 'paragrnneighborstrans' or args.experiment == 'paragatneighborstrans':
        model = ParaTransformerNeighborsNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment,
                feature_selection=args.para_aggrg)
    elif args.experiment == 'grnall' or args.experiment == 'gatall':
        model = NeuralECMTokenModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment)
    elif args.experiment == 'paragatasp' or args.experiment == 'paragrnasp':
        model = ParaAspectNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment,
                feature_selection=args.para_aggrg,
                aspects_method=args.asp_aggrg)
    elif args.experiment == 'parasymbtunedbert':
        model = TunedBERTModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'parasymbtoken':
        
        # from accelerate import dispatch_model, infer_auto_device_map
        # from accelerate.utils import get_balanced_memory
        # from torch.cuda.amp import autocast
        
        model = ParaSymbolsTokenModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment,
                batch_size=args.batch_size)
        
        # max_memory = get_balanced_memory(
        #     model,
        #     max_memory = None,
        #     no_split_module_classes = ['MistralDecoderLayer', 'MistralAttention', 'MistralMLP', 'MistralRMSNorm', 'Linear', 'GRN'],
        #     dtype='float16',
        #     low_zero=False,
        # )
        
        # device_map = infer_auto_device_map(
        #     model,
        #     max_memory = max_memory,
        #     no_split_module_classes = ['MistralDecoderLayer', 'MistralAttention', 'MistralMLP', 'MistralRMSNorm', 'Linear', 'GRN'],
        #     dtype='float16')
        
        # model = dispatch_model(model, device_map=device_map)
        
        # for i in model.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")
    elif args.experiment == 'parasymbembedding':
        
        # from accelerate import dispatch_model, infer_auto_device_map
        # from accelerate.utils import get_balanced_memory
        # from torch.cuda.amp import autocast
        
        model = ParaSymbolsEmbeddingModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment)
        
        # max_memory = get_balanced_memory(
        #     model,
        #     max_memory = None,
        #     no_split_module_classes = ['MistralDecoderLayer', 'MistralAttention', 'MistralMLP', 'MistralRMSNorm', 'Linear', 'GRN'],
        #     dtype='float16',
        #     low_zero=False,
        # )
        
        # device_map = infer_auto_device_map(
        #     model,
        #     max_memory = max_memory,
        #     no_split_module_classes = ['MistralDecoderLayer', 'MistralAttention', 'MistralMLP', 'MistralRMSNorm', 'Linear', 'GRN'],
        #     dtype='float16')
        
        # model = dispatch_model(model, device_map=device_map)
        
        # for i in model.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")
        
    elif args.experiment == 'parasymbonlyselfratingwithoutgnn':
        model = ParaSymbolsSelfRatingWithoutGNNEmbeddingModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment)
        

    loss_fn = nn.MarginRankingLoss(margin=1)

    if args.checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint))
        print('[Done].')


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    

    print('Using device: {}'.format(device))
    model.to(device)
    loss_fn.to(device)

    
    
    if args.kfold:
        
        import pathlib
        from sklearn.model_selection import KFold
        from utils import convert_queries_json_to_jsonl, filter_queries_data
        from pre_processing.train_data.train_data import to_pairwise_data
        from pre_processing.test_data.test_data import to_pointwise_data
        
        queries_data = convert_queries_json_to_jsonl(args.queries)
        
        print('Reading Embeddings files...')
        
        queries_embeddings = read_embeddings_file(args.queries_embed)
        print('queries embeddings done...')
        para_embeddings = read_embeddings_file(args.para_embed)
        print('paragraphs embeddings done...')
        symbols_embeddings = read_embeddings_file(args.symb_embed)
        print('symbols embeddings done...')
        
        kf = KFold(n_splits=5, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(kf.split(queries_data)):
            
            print(f'Processing Fold {i}')
            
            fold_name = 'Fold'+str(i)
            path_name = args.save_dir+fold_name
            path = pathlib.Path(path_name)
            path.mkdir(parents=True, exist_ok=True)
            
            test_run_file_save_path = path_name+'/'+args.run.split('/')[-1]
            loss_csv_save_path = path_name+'/'+args.loss_csv.split('/')[-1]
            train_run_file_save_path = path_name+'/'+args.train_run.split('/')[-1]
            model_save_path = path_name+'/'+args.save
            
            print('Generating training data')
            
            train_filtered_queries = filter_queries_data(queries_data, train_index)
            train_data = to_pairwise_data(args.queries,
                                          args.generated_symbols,
                                          args.autograder_file,
                                          args.qrels,
                                          args.dataset,
                                          args.kfold,
                                          train_filtered_queries,
                                          queries_embeddings,
                                          para_embeddings,
                                          symbols_embeddings)
            
            print('Generating test data')
            
            test_filtered_queries = filter_queries_data(queries_data, test_index)
            test_data = to_pointwise_data(args.queries,
                                          args.generated_symbols,
                                          args.autograder_file,
                                          args.qrels,
                                          args.dataset,
                                          args.kfold,
                                          test_filtered_queries,
                                          queries_embeddings,
                                          para_embeddings,
                                          symbols_embeddings,
                                          args.re_rank,
                                          args.re_rank_method)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.n_warmup_steps,
                num_training_steps=len(train_data) * args.epoch // args.batch_size)
            
            print(f'Reading train data for fold {i}...')
            train_set = EntityRankingDataset(
                dataset=train_data,
                train=True,
                experiment=args.experiment,
                kfold=args.kfold
            )
            print('[Done].')
            print(f'Reading dev data for fold {i}...')
            dev_set = EntityRankingDataset(
                dataset=test_data,
                train=False,
                experiment=args.experiment,
                kfold=args.kfold
            )
            print('[Done].')

            print('Creating data loaders...')
            print('Number of workers = ' + str(args.num_workers))
            print('Batch Size = ' + str(args.batch_size))
            train_loader = EntityRankingDataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            dev_loader = EntityRankingDataLoader(
                dataset=dev_set,
                shuffle=False,
                batch_size=1,
                num_workers=args.num_workers
            )
            print('[Done].')
            print("Starting to train...")
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=loss_fn,
                scheduler=scheduler,
                data_loader=train_loader,
                device=device,
                experiment=args.experiment
            )

            validator = Validate(
                    model=model,
                    data_loader=dev_loader,
                    device=device,
                    experiment=args.experiment
            )
            
            
            
            train(
                model=model,
                trainer=trainer,
                epochs=args.epoch,
                metric=args.metric,
                qrels=args.qrels,
                valid_loader=dev_loader,
                save_path=path_name,
                save=args.save,
                run_file=test_run_file_save_path,
                eval_every=args.eval_every,
                device=device,
                loss_csv=loss_csv_save_path,
                train_run_file = train_run_file_save_path,
                validator=validator
            )
            print(f'Finished training Fold {i}')
            
            
        
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=len(train_set) * args.epoch // args.batch_size)
        print('Reading train data...')
        train_set = EntityRankingDataset(
            dataset=args.train,
            train=True,
            experiment=args.experiment,
            kfold=args.kfold
        )
        print('[Done].')
        print('Reading dev data...')
        dev_set = EntityRankingDataset(
            dataset=args.dev,
            train=False,
            experiment=args.experiment,
            kfold=args.kfold
        )
        print('[Done].')

        print('Creating data loaders...')
        print('Number of workers = ' + str(args.num_workers))
        print('Batch Size = ' + str(args.batch_size))
        train_loader = EntityRankingDataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        dev_loader = EntityRankingDataLoader(
            dataset=dev_set,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print('[Done].')
        print("Starting to train...")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=loss_fn,
            scheduler=scheduler,
            data_loader=train_loader,
            device=device,
            experiment=args.experiment
        )

        validator = Validate(
                model=model,
                data_loader=dev_loader,
                device=device,
                experiment=args.experiment
        )
        train(
            model=model,
            trainer=trainer,
            epochs=args.epoch,
            metric=args.metric,
            qrels=args.qrels,
            valid_loader=dev_loader,
            save_path=args.save_dir,
            save=args.save,
            run_file=args.run,
            eval_every=args.eval_every,
            device=device,
            loss_csv=args.loss_csv,
            train_run_file = args.train_run,
            validator=validator
        )

    print('Training complete.')


if __name__ == '__main__':
    main()
