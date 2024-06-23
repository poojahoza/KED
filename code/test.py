import torch
import utils
import argparse
from dataloader import EntityRankingDataLoader
from model import NeuralECMModel, ParaNeuralECMModel, GRNECMModel, TunedBERTModel, ParaNeuralECMModel, NeuralECMTokenModel, ParaTransformerNeuralECMModel, ParaTransformerTunedBertModel, ParaTransformerNeighborsNeuralECMModel, ParaAspectNeuralECMModel
from dataset import EntityRankingDataset
from validate import Validate


def test(validator, run_file):
    res_dict, valid_loss = validator.validate()

    print('Writing run file...')
    utils.save_trec(run_file, res_dict)
    print('[Done].')


def main():
    parser = argparse.ArgumentParser("Script to test a model.")
    parser.add_argument('--experiment', help='Which experiment do you want to run? (grn, gat, grnecm, tunedbert, paragrn, paragat, grnall, gatall, paragrntrans, paragattrans, paratunedbertrans, paragrnneighborstrans, paragatasp, paragrnasp)', choices= ['grn', 'gat', 'grnecm', 'tunedbert', 'paragrn', 'paragat', 'grnall', 'gatall', 'paragrntrans', 'paragattrans', 'paratunedberttrans', 'paragrnneighborstrans', 'paragatasp', 'paragrnasp'], required=True)
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--run', help='Test run file.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--query-in-emb-dim', help='Dimension of input embedding.', required=True, type=int)
    parser.add_argument('--ent-in-emb-dim', help='Dimension of output embedding.', required=True, type=int)
    parser.add_argument('--para-in-emb-dim', help='Dimension of para embedding.',required=True, type=int)
    parser.add_argument('--para-aggrg', help='Which paragraph aggregator to use for para neural ecm',type=str, choices=['max', 'linear', 'prod', 'entrank'])
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    parser.add_argument('--asp-aggrg', help='Which function to use to learn aspect for para neural ecm (linear, prod, bilinear)',choices=['linear', 'prod', 'bilinear'])
    args = parser.parse_args()

    model = None
    experiment = args.experiment

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )


    print('Reading test data...')
    test_set = EntityRankingDataset(
        dataset=args.test,
        experiment=args.experiment,
        train=False
    )
    print('[Done].')

    print('Creating data loader...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))

    test_loader = EntityRankingDataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print('[Done].')

    if experiment == 'grn' or experiment == 'gat':
        model = NeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=args.experiment)
    elif experiment == 'grnecm':
        model = GRNECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif experiment == 'tunedbert':
        model = TunedBERTModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif args.experiment == 'paratunedberttrans':
        model = ParaTransformerTunedBertModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device)
    elif experiment == 'paragrn' or experiment == 'paragat':
        model = ParaNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=experiment,
                feature_selection=args.para_aggrg)
    elif experiment == 'paragrntrans' or experiment == 'paragattrans':
        model = ParaTransformerNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=experiment,
                feature_selection=args.para_aggrg)
    elif experiment == 'paragrnneighborstrans' or experiment == 'paragatneighborstrans':
        model = ParaTransformerNeighborsNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=experiment,
                feature_selection=args.para_aggrg)
    elif experiment == 'grnall' or experiment == 'gatall':
        model = NeuralECMTokenModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=experiment)
    elif experiment == 'paragatasp' or experiment == 'paragrnasp':
        model = ParaAspectNeuralECMModel(ent_input_emb_dim=args.ent_in_emb_dim,
                query_input_emb_dim=args.query_in_emb_dim,
                para_input_emb_dim=args.para_in_emb_dim,
                device=device,
                experiment=experiment,
                feature_selection=args.para_aggrg,
                aspects_method=args.asp_aggrg)

    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print('[Done].')


    print('Using device: {}'.format(device))
    model.to(device)

    validator = Validate(
            model=model,
            data_loader=test_loader,
            device=device,
            experiment=experiment)

    print("Starting to test...")

    test(
        validator=validator,
        run_file=args.run
    )

    print('Test complete.')


if __name__ == '__main__':
    main()
