"""
different with classical single-step stock forecasting task
we treat stock forecasting as a multi-classification approx ndcg task
we will devide stocks into at least 3 different class according to their ranking in a single day
optimize with cross-entropy first(then try approxNDCG, we have plan B)
if classify, we use task name multi-class
"""
import torch
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, sys.path[0]+"/../")
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, RSR
from utils.utils import cross_entropy, generate_label, evaluate_mc, class_approxNDCG
from utils.dataloader import create_mtm_loaders
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

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`'%model_name)


global_log_file = None


def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1


def train_epoch(epoch, model, optimizer, train_loader, writer, args,
                stock2concept_matrix=None, stock2stock_matrix = None):
    """
    train epoch function
    :param epoch: number of epoch of training
    :param model: model that will be used
    :param optimizer:
    :param train_loader:
    :param writer:
    :param args:
    :param stock2concept_matrix:
    :return:
    """

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value, stock_index, _, mask = train_loader.get(slc)
        # we get feature and label, pred in classification is a tensor, first
        if args.model_name == 'HIST':
            # if HIST is used, take the stock2concept_matrix and market_value
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            # the stock2concept_matrix[stock_index] is a matrix, shape is (the number of stock index, predefined con)
            # the stock2concept_matrix has been sort to the order of stock index
        elif args.model_name == 'RSR':
            pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
        elif args.model_name == 'PatchTST':
            # new added
            pred = model(feature, mask)
        else:
            # other model only use feature as input
            # for multi-class, here we get a [B, N]
            pred = model(feature)
        if args.loss_type == 'cross_entropy':
            loss = cross_entropy(pred, label)
        elif args.loss_type == 'mixed':
            loss = args.beta * (cross_entropy(pred, label) +
                                (1 - args.beta) * class_approxNDCG(args, pred, label))
        else:
            loss = class_approxNDCG(args, pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index, mask = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif args.model_name == 'PatchTST':
                # new added
                pred = model(feature, mask)
            else:
                pred = model(feature)

            if args.loss_type == 'cross_entropy':
                loss = cross_entropy(pred, label)
            elif args.loss_type == 'mixed':
                loss = args.beta * (cross_entropy(pred, label) +
                                    (1 - args.beta) * class_approxNDCG(args, pred, label))
            else:
                loss = class_approxNDCG(args, pred, label)
            pred_label, true_label = generate_label(pred, label)
            preds.append(pd.DataFrame({'pred': pred_label.cpu().numpy(),
                                       'ground_truth': true_label.cpu().numpy(),}, index=index))

        losses.append(loss.item())

    # evaluate
    preds = pd.concat(preds, axis=0)
    # use metric_fn to compute precision, recall, ic and rank ic
    acc, average_precision, f1_micro, f1_macro = evaluate_mc(preds)
    scores = acc  # score is the opposite loss

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, acc, average_precision, f1_micro, f1_macro


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index, mask = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred = model(feature)
            pred_label, true_label = generate_label(pred,label)
            preds.append(pd.DataFrame({'pred': pred_label.cpu().numpy(),
                                       'ground_truth': true_label.cpu().numpy(),}, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = '../ouput/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_mtm_loaders(args, device=device)

    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    if args.model_name == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    if args.model_name == 'RSR':
        stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
        num_relation = stock2stock_matrix.shape[2]

    all_acc = []
    all_macrof1 = []
    all_precision = []
    all_microf1 = []
    for times in range(args.repeat):
        pprint('create model...')
        if args.model_name == 'HIST':
            model = get_model(args.model_name)(args)
        elif args.model_name == 'RSR':
            model = get_model(args.model_name)(args, num_relation=num_relation)
        else:
            model = get_model(args.model_name)(args, d_feat=args.d_feat, num_layers=args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model.state_dict())
        for epoch in range(args.n_epochs):
            pprint('Running', times, 'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix)
            # save model  after every epoch
            # -------------------------------------------------------------------------

            params_ckpt = copy.deepcopy(model.state_dict())

            pprint('evaluating...')
            # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
            train_loss, train_score, train_acc, train_avg_precision, train_f1_micro, train_f1_macro = \
                test_epoch(epoch, model, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Train')
            val_loss, val_score, val_acc, val_avg_precision, val_f1_micro, val_f1_macro = \
                test_epoch(epoch, model, valid_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Valid')
            test_loss, test_score, test_acc, test_avg_precision, test_f1_micro, test_f1_macro = \
                test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            # score equals to ic here
            # pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            pprint('train_acc %.6f, valid_acc %.6f, test_acc %.6f'%(train_acc, val_acc, test_acc))
            pprint('train_avg_precision %.6f, valid_avg_precision %.6f, test_avg_precision %.6f'%(train_avg_precision,
                                                                                                  val_avg_precision,
                                                                                                  test_avg_precision))
            pprint('train_macro_f1 %.6f, valid_macro_f1 %.6f, test_macro_f1 %.6f' % (train_f1_macro,
                                                                                     val_f1_macro,
                                                                                     test_f1_macro))
            pprint('train_micro_f1 %.6f, valid_micro_f1 %.6f, test_micro_f1 %.6f' % (train_f1_micro,
                                                                                     val_f1_micro,
                                                                                     test_f1_micro))
            # load back the current parameters
            # model.load_state_dict(params_ckpt)
            if val_score > best_score:
                # the model performance is increasing
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(params_ckpt)
            else:
                # the model performance is not increasing
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin')

        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            # do prediction on train, valid and test data
            pred = inference(model, eval(name+'_loader'), stock2concept_matrix=stock2concept_matrix,
                            stock2stock_matrix=stock2stock_matrix)
            acc, average_precision, f1_micro, f1_macro = evaluate_mc(pred)
            pprint(name, ': Accuracy ', acc)
            pprint(name, ': Avg Precision ', average_precision)
            pprint(name, ':Micro F1', f1_micro)
            pprint(name, ':Macro F1', f1_macro)

            res[name+'-ACC'] = acc
            res[name + '-Avg Precision'] = average_precision
            res[name + '-Micro F1'] = f1_micro
            res[name + '-Macro F1'] = f1_macro

        all_acc.append(acc)

        all_macrof1.append(f1_macro)
        all_microf1.append(f1_micro)
        all_precision.append(average_precision)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint('Accuracy: %.4f (%.4f)' % (np.array(all_acc).mean(), np.array(all_acc).std()))
    pprint('F1 Macro: %.4f (%.4f)' % (np.array(all_macrof1).mean(), np.array(all_macrof1).std()))
    pprint('F1 Micro: %.4f (%.4f)' % (np.array(all_microf1).mean(), np.array(all_microf1).std()))
    pprint('Avg Precision: %.4f (%.4f)' % (np.array(all_precision).mean(), np.array(all_precision).std()))

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='HIST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--K', type=int, default=1)

    # for loss function setting
    parser.add_argument('--loss_type', default='mixed')
    parser.add_argument('--gumble', default=False)
    parser.add_argument('--num_class', default=5, help='the number of class of stock sequence')
    parser.add_argument('--topk', default=50, help='the number of computing NDCG@k')
    parser.add_argument('--adaptive_k', default=True)
    parser.add_argument('--ndcg', default='approx', help='choose neural or approx')
    parser.add_argument('--class_weight_method', default='square', help='provide square and exp')
    parser.add_argument('--beta', default=0.5, help='the ratio of entropy in mixed loss function')
    parser.add_argument('--approxalpha', default=1, type=float, help='the knob of approxNDCG')
    parser.add_argument('--primary', default=False)

    # training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='acc')
    parser.add_argument('--task_name', type=str, default='multi-class', help='task setup')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--repeat', type=int, default=3)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to22.pkl')
    parser.add_argument('--mtm_source_path', default='./data/original_mtm.pkl')
    parser.add_argument('--mtm_column', default='mtm0604')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='./data/csi300_multi_stock2stock.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')
    parser.add_argument('--outdir', default='./output/mc/PatchTST_mtm0604')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
