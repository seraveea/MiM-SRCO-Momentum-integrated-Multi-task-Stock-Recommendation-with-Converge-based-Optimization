"""
here are two tasks in the multi-task learning framework
one is regression.
one is classification
in this new mto file, we use scale_algorithm and project algorithm
we have two projection algorithm: no projection/PCGrad
we have several scale algorithm: sum, MGDA, UC, GradNorm
"""
import sys
sys.path.insert(0, sys.path[0]+"/../")
import torch
import torch.optim as optim
from utils.utils import DotDict
import os
import json
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys

sys.path.insert(0, sys.path[0] + "/../")
from models.model import MLP, HIST, GRU, LSTM, GAT, RSR
from models.sub_task_models import regression_submodel, classification_submodel
from utils.utils import generate_label
from utils.dataloader import create_mto_loaders
import warnings
import logging


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12
warnings.filterwarnings('ignore')


def get_model(model_name):
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`' % model_name)


global_log_file = None


def pprint(*args):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1


def inference(args, model, model_c, model_r, data_loader, stock2concept_matrix=None, stock2stock_matrix=None):
    model.eval()
    model_c.eval()
    model_r.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, label_2, market_value, stock_index, index, mask = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                rep = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                rep = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                rep = model(feature)
            out_1 = model_c(rep)
            out_2 = model_r(rep)
            pred_label, true_label = generate_label(out_1, label)
            preds.append(pd.DataFrame({'pred_class': pred_label.cpu().numpy(),
                                       'pred_score': out_2.cpu().numpy(),
                                       'ground_truth_class': true_label.cpu().numpy(),
                                       'ground_truth_score': label_2.cpu().numpy()}, index=index))
    preds = pd.concat(preds, axis=0)
    return preds


def prediction(args, model_path, device):
    param_dict = json.load(open(model_path+'/info.json'))['config']
    param_dict['model_dir'] = model_path
    train_loader, valid_loader, test_loader = create_mto_loaders(args, device=device)
    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
    if param_dict['model_name'] == 'RSR':
        rep_len = param_dict['hidden_size']*2
    else:
        rep_len = param_dict['hidden_size']  # overload the rep_len for HIST model
    print('load model ', param_dict['model_name'])
    if param_dict['model_name'] == 'GRU':
        model = get_model(param_dict['model_name'])(DotDict(param_dict), d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
    elif param_dict['model_name'] == 'RSR':
        num_relation = stock2stock_matrix.shape[2]
        model = get_model(param_dict['model_name'])(DotDict(param_dict), num_relation=num_relation)
    elif param_dict['model_name'] == 'HIST':
        model = get_model(param_dict['model_name'])(DotDict(param_dict))
    else:
        model = get_model(param_dict['model_name'])(DotDict(param_dict), d_feat=param_dict["d_feat"], num_layers=param_dict["num_layers"])
    model.to(device)
    model_r = regression_submodel(seq_len=rep_len)
    model_c = classification_submodel(seq_len=rep_len, num_group=int(param_dict['num_class']))

    model.load_state_dict(torch.load(param_dict['model_dir'] + '/model_best.bin', map_location=device))
    model_r.load_state_dict(torch.load(param_dict['model_dir']+'/model_r_best.bin', map_location=device))
    model_c.load_state_dict(torch.load(param_dict['model_dir']+'/model_c_best.bin', map_location=device))
    pred = inference(DotDict(param_dict), model, model_c, model_r, test_loader, stock2concept_matrix, stock2stock_matrix)
    return pred


def main(args):
    model_path = args.model_path
    pd.to_pickle(prediction(args, model_path, device), args.pkl_path)


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`' % filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():
    parser = argparse.ArgumentParser()
    # dataloader parameters
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='./data/csi300_multi_stock2stock.npy')
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to22.pkl')
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--mtm_source_path', default='./data/original_mtm.pkl')
    parser.add_argument('--mtm_column', default='mtm0604')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')
    parser.add_argument('--device', default='cuda:1')
    # input and output
    parser.add_argument('--model_path', default='./output/RSR')
    parser.add_argument('--pkl_path', default='./pred_output/RSR.pkl',
                        help='location to save the pred dictionary file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
