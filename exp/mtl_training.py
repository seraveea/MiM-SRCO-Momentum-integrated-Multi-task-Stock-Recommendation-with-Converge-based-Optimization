"""
here are two tasks in the multi-task learning framework
one is regression.
one is classification
in this new mto file, we use scale_algorithm and project algorithm
we have two projection algorithm: no projection/PCGrad
we have several scale algorithm: sum, MGDA, UC, GradNorm
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
from collections import defaultdict
sys.path.insert(0, sys.path[0] + "/../")
from models.model import HIST, GRU, LSTM, GAT, RSR
from models.sub_task_models import regression_submodel, classification_submodel
from utils.utils import cross_entropy, generate_label, evaluate_mc, class_approxNDCG, \
    mse, metric_fn_mto, DoubleBuffer, pair_wise_loss
from utils.dataloader import create_mto_loaders
from utils.weight_methods import METHODS, WeightMethods
import warnings
import logging

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12
warnings.filterwarnings('ignore')


def get_model(model_name):

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`' % model_name)


def extract_weight_method_parameters_from_args(input_args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=input_args.update_weights_every,
                optim_niter=input_args.nashmtl_optim_niter,
            ),
            cagrad=dict(c=input_args.c),
            dwa=dict(temp=input_args.dwa_temp),
        )
    )
    return weight_methods_parameters


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


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    # output is a single dim tensor
    return mse(pred[mask], label[mask])


global_step = -1


def train_epoch(epoch, weight_method, model, model_c, model_r, optimizer, train_loader, writer, args,
                relative_loss_drop_r, relative_loss_drop_c,
                stock2concept_matrix=None, stock2stock_matrix=None):
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
    model_c.train()
    model_r.train()
    c_grad, r_grad, cosine = [], [], []
    if args.method == 'ourmethod' or args.method == 'ourmethod_ii' or args.method == 'ourmethod_iii':
        beta = None
    for iter, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label_1, label_2, market_value, stock_index, _, mask = train_loader.get(slc)
        optimizer.zero_grad()
        if args.model_name == 'HIST':
            rep = model(feature, stock2concept_matrix[stock_index], market_value)
        elif args.model_name == 'RSR':
            rep = model(feature, stock2stock_matrix[stock_index][:, stock_index])
        else:
            rep = model(feature)
        #  task one
        out_1 = model_c(rep)
        if args.loss_type == 'cross_entropy':
            loss_1 = cross_entropy(out_1, label_1)
        elif args.loss_type == 'mixed':
            loss_1 = args.beta * (cross_entropy(out_1, label_1) +
                                  (1 - args.beta) * class_approxNDCG(args, out_1, label_1))
        elif args.loss_type == 'pair_wise':
            loss_1 = args.beta * (cross_entropy(out_1, label_1) +
                                  (1 - args.beta) * pair_wise_loss(args, out_1, label_1))
        else:
            loss_1 = class_approxNDCG(args, out_1, label_1)
        #  task two
        out_2 = model_r(rep)
        loss_2 = loss_fn(label_2, out_2)
        losses = torch.stack((loss_1, loss_2))
        model_params = []
        for m in [model_c, model_r]:
            model_params += m.parameters()
        loss, extra_outputs = weight_method.backward(
            losses=losses,
            shared_parameters=list(model.parameters()),  # rep model's parameters
            task_specific_parameters=model_params,  # model_c and model_r parameters
            last_shared_parameters=list(model.last_shared_parameters()),  # this is remain uncleared
            representation=rep,
            relative_loss_drop=[relative_loss_drop_c, relative_loss_drop_r],
            optimizer=optimizer
        )
        optimizer.step()
        if args.analysis_mode:
            c_grad.append(extra_outputs[0])
            r_grad.append(extra_outputs[1])
            cosine.append(extra_outputs[2])
        if args.method == 'our_method':
            beta = extra_outputs

    if args.method == 'our_method':
        pprint(beta)

    if args.analysis_mode:
        return [np.array(c_grad), np.array(r_grad), np.array(cosine)]
    else:
        return None


def test_epoch(epoch, model, model_c, model_r, test_loader, writer, args, stock2concept_matrix=None,
               stock2stock_matrix=None, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """

    model.eval()
    model_c.eval()
    model_r.eval()

    losses_s1 = []
    losses_s2 = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, label_2, market_value, stock_index, index, mask = test_loader.get(slc)

        with (torch.no_grad()):
            if args.model_name == 'HIST':
                rep = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                rep = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                rep = model(feature)
            out_1 = model_c(rep)
            out_2 = model_r(rep)
            if args.loss_type == 'cross_entropy':
                loss_s1 = cross_entropy(out_1, label)
            elif args.loss_type == 'mixed':
                loss_s1 = args.beta * cross_entropy(out_1, label)
                + (1 - args.beta) * class_approxNDCG(args, out_1, label)
            else:
                loss_s1 = args.beta * (cross_entropy(out_1, label) +
                                      (1 - args.beta) * pair_wise_loss(args, out_1, label))
            loss_s2 = loss_fn(out_2, label_2)
            pred_label, true_label = generate_label(out_1, label)
            preds.append(pd.DataFrame({'pred_class': pred_label.cpu().numpy(),
                                       'pred_score': out_2.cpu().numpy(),
                                       'ground_truth_class': true_label.cpu().numpy(),
                                       'ground_truth_score': label_2.cpu().numpy()}, index=index))

        #  here we record the gradient of each subtask
        losses_s1.append(loss_s1.item())
        losses_s2.append(loss_s2.item())

    # -----------------------------------------------------------------------------------------------
    # evaluate
    preds = pd.concat(preds, axis=0)
    """
    for classification task， we need acc， average precision, f1_micro, f1_macro
    for regression task, we need 
    """
    acc, average_precision, f1_micro, f1_macro = evaluate_mc(preds, pred_column='pred_class',
                                                             gt_column='ground_truth_class')
    precision, recall, ic, rank_ic, ndcg = metric_fn_mto(preds, pred_column='pred_score',
                                                         gt_column='ground_truth_score')

    score_c = acc
    score_r = rank_ic

    writer.add_scalar(prefix + '/Classification Loss', np.mean(losses_s1), epoch)
    writer.add_scalar(prefix + '/std(Classification Loss)', np.std(losses_s1), epoch)
    writer.add_scalar(prefix + '/Regression Loss', np.mean(losses_s2), epoch)
    writer.add_scalar(prefix + '/std(Regression Loss)', np.std(losses_s2), epoch)
    writer.add_scalar(prefix + '/classification score', np.mean(score_c), epoch)
    writer.add_scalar(prefix + '/std(classification score)', np.std(score_c), epoch)
    writer.add_scalar(prefix + '/regression score', np.mean(score_r), epoch)
    writer.add_scalar(prefix + '/std(regression score)', np.std(score_r), epoch)

    return np.mean(losses_s1), np.mean(losses_s2), score_c, score_r, acc, average_precision, f1_micro, f1_macro, \
        ic, rank_ic, precision, recall, ndcg


def inference(model, model_c, model_r, data_loader, stock2concept_matrix=None, stock2stock_matrix=None):
    model.eval()
    model_c.eval()
    model_r.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, label_2, market_value, stock_index, index, mask = data_loader.get(slc)
        with torch.no_grad():
            # if args.model_name == 'HIST':
            #     pred = model(feature, stock2concept_matrix[stock_index], market_value)
            # elif args.model_name in relation_model_dict:
            #     pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            # elif args.model_name in time_series_library:
            # new added
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


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = '../ouput/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path + '/' + 'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_mto_loaders(args, device=device)

    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    if args.model_name == 'RSR':
        stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
        num_relation = stock2stock_matrix.shape[2]


    all_acc = []
    all_macrof1 = []
    all_precision = []
    all_microf1 = []
    all_precisionN = []
    all_recallN = []
    all_ic = []
    all_rank_ic = []
    all_ndcg = []
    if args.model_name == 'RSR':
        rep_len = args.hidden_size * 2
    else:
        rep_len = args.hidden_size
    global_best_score = -np.inf
    for times in range(args.repeat):

        weight_method_params = extract_weight_method_parameters_from_args(args)
        weight_method = WeightMethods(
            args.method,
            n_tasks=2,
            device=device,
            **weight_method_params[args.method],
        )
        pprint('create model...')
        if args.model_name == 'HIST':
            model = get_model(args.model_name)(args)
        elif args.model_name == 'RSR':
            model = get_model(args.model_name)(args, num_relation=num_relation)
        else:
            model = get_model(args.model_name)(args, d_feat=args.d_feat, num_layers=args.num_layers)

        # create two sub-model
        model_r = regression_submodel(seq_len=rep_len)
        model_c = classification_submodel(seq_len=rep_len, num_group=int(args.num_class))

        model.to(device)
        model_c.to(device)
        model_r.to(device)
        model_params = []
        for m in [model, model_c, model_r]:
            model_params += m.parameters()

        optimizer = optim.Adam([
            dict(params=model_params, lr=args.lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
            ],)

        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model.state_dict())
        best_param_c = copy.deepcopy(model_c.state_dict())
        best_param_r = copy.deepcopy(model_r.state_dict())
        relative_loss_drop_r = 1
        relative_loss_drop_c = 1
        valid_rloss_buffer = DoubleBuffer(capacity=6)
        train_rloss_buffer = DoubleBuffer(capacity=6)
        valid_closs_buffer = DoubleBuffer(capacity=6)
        train_closs_buffer = DoubleBuffer(capacity=6)
        grads_norm_r, grads_norm_c, cosine_list = [], [], []
        for epoch in range(args.n_epochs):
            pprint('Running', times, 'Epoch:', epoch)

            pprint('training...')
            if args.analysis_mode:
                ana_value = train_epoch(epoch, weight_method,  model, model_c, model_r, optimizer,
                                        train_loader, writer, args,
                                        relative_loss_drop_r, relative_loss_drop_c,
                                        stock2concept_matrix, stock2stock_matrix)
                grads_norm_c.append(ana_value[0])
                grads_norm_r.append(ana_value[1])
                cosine_list.append(ana_value[2])
            else:
                train_epoch(epoch, weight_method, model, model_c, model_r, optimizer, train_loader, writer, args,
                            relative_loss_drop_r, relative_loss_drop_c,
                            stock2concept_matrix, stock2stock_matrix)

            params_ckpt_rep = copy.deepcopy(model.state_dict())
            params_ckpt_c = copy.deepcopy(model_c.state_dict())
            params_ckpt_r = copy.deepcopy(model_r.state_dict())

            pprint('evaluating...')

            train_loss_c, train_loss_r, train_score_c, train_score_r, train_acc, train_avg_precision, train_f1_micro, train_f1_macro, \
                train_ic, train_rank_ic, train_precisionN, train_recallN, train_ndcg = \
                test_epoch(epoch, model, model_c, model_r, train_loader, writer, args,
                           stock2concept_matrix, stock2stock_matrix, prefix='Train')
            val_loss_c, val_loss_r, val_score_c, val_score_r, val_acc, val_avg_precision, val_f1_micro, val_f1_macro, \
                val_ic, val_rank_ic, val_precisionN, val_recallN, val_ndcg = \
                test_epoch(epoch, model, model_c, model_r, valid_loader, writer, args,
                           stock2concept_matrix, stock2stock_matrix, prefix='Valid')
            test_loss_c, test_loss_r, test_score_c, test_scor_r, test_acc, test_avg_precision, test_f1_micro, test_f1_macro, \
                test_ic, test_rank_ic, test_precisionN, test_recallN, test_ndcg = \
                test_epoch(epoch, model, model_c, model_r, test_loader, writer, args,
                           stock2concept_matrix, stock2stock_matrix, prefix='Test')

            pprint('CLASSIFICATION: train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss_c,
                                                                                         val_loss_c, test_loss_c))
            pprint('REGRESSION: train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss_r,
                                                                                     val_loss_r, test_loss_r))
            pprint('train_acc %.6f, valid_acc %.6f, test_acc %.6f' % (train_acc, val_acc, test_acc))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (train_rank_ic, val_rank_ic, test_rank_ic))

            pprint('train_avg_precision %.6f, valid_avg_precision %.6f, test_avg_precision %.6f' % (train_avg_precision,
                                                                                                    val_avg_precision,
                                                                                                    test_avg_precision))
            pprint('train_macro_f1 %.6f, valid_macro_f1 %.6f, test_macro_f1 %.6f' % (train_f1_macro,
                                                                                     val_f1_macro,
                                                                                     test_f1_macro))
            pprint('train_micro_f1 %.6f, valid_micro_f1 %.6f, test_micro_f1 %.6f' % (train_f1_micro,
                                                                                     val_f1_micro,
                                                                                     test_f1_micro))

            valid_rloss_buffer.add_value(val_loss_r)
            train_rloss_buffer.add_value(train_loss_r)
            valid_closs_buffer.add_value(val_loss_c)
            train_closs_buffer.add_value(train_loss_c)

            if valid_rloss_buffer.check_capacity():
                # it's time to compute relative improvement!
                relative_loss_drop_r = valid_rloss_buffer.compute_loss_drop() / train_rloss_buffer.compute_loss_drop()
                relative_loss_drop_c = valid_closs_buffer.compute_loss_drop() / train_closs_buffer.compute_loss_drop()
                pprint("relative loss drop on regression: ", str(relative_loss_drop_r))
                pprint("relative loss drop on classification: ", str(relative_loss_drop_c))
                # compute classification task relative improvement

            if epoch == 0:
                base_score_r = val_score_r
                base_score_c = val_score_c

            val_score_r = (val_score_r - base_score_r)/abs(base_score_r)
            val_score_c = (val_score_c - base_score_c)/abs(base_score_c)

            if args.primary:
                # only care the regression
                if val_score_r > best_score:
                    # the model performance is increasing
                    best_score = val_score_r
                    stop_round = 0
                    best_epoch = epoch
                    # best_param = copy.deepcopy(avg_params)
                    best_param = copy.deepcopy(params_ckpt_rep)
                    best_param_c = copy.deepcopy(params_ckpt_c)
                    best_param_r = copy.deepcopy(params_ckpt_r)
                else:
                    # the model performance is not increasing
                    stop_round += 1
                    if stop_round >= args.early_stop:
                        pprint('early stop')
                        break
            else:
                if val_score_r + val_score_c > best_score:
                    # the model performance is increasing
                    best_score = val_score_r + val_score_c
                    stop_round = 0
                    best_epoch = epoch
                    best_param = copy.deepcopy(params_ckpt_rep)
                    best_param_c = copy.deepcopy(params_ckpt_c)
                    best_param_r = copy.deepcopy(params_ckpt_r)
                else:
                    # the model performance is not increasing
                    stop_round += 1
                    if stop_round >= args.early_stop:
                        pprint('early stop')
                        break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        model_c.load_state_dict(best_param_c)
        model_r.load_state_dict(best_param_r)

        np.save(open(output_path+'/regression_norm_grads.npy', 'wb'), grads_norm_r)
        np.save(open(output_path+'/classification_norm_grads.npy', 'wb'), grads_norm_c)
        np.save(open(output_path+'cosine_bwt_2grads.npy', 'wb'), cosine_list)

        torch.save(best_param, output_path + '/model.bin')
        torch.save(best_param_c, output_path + '/model_c.bin')
        torch.save(best_param_r, output_path + '/model_r.bin')
        if best_score > global_best_score:
            torch.save(best_param, output_path + '/model_best.bin')
            torch.save(best_param_c, output_path + '/model_c_best.bin')
            torch.save(best_param_r, output_path + '/model_r_best.bin')
            global_best_score = best_score

        #  ------------------------------------------------------------------------------------------------------
        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            # do prediction on train, valid and test data
            pred = inference(model, model_c, model_r, eval(name + '_loader'), stock2concept_matrix=stock2concept_matrix,
                             stock2stock_matrix=stock2stock_matrix)
            acc, average_precision, f1_micro, f1_macro = evaluate_mc(pred, pred_column='pred_class',
                                                                     gt_column='ground_truth_class')
            precision, recall, ic, rank_ic, ndcg = metric_fn_mto(pred, pred_column='pred_score',
                                                                 gt_column='ground_truth_score')
            # acc, average_precision, f1_micro, f1_macro = evaluate_mc(pred)
            pprint(name, ': Accuracy ', acc)
            pprint(name, ': IC ', ic)
            pprint(name, ': RankIC', rank_ic)
            pprint(name, ': Avg Precision ', average_precision)
            pprint(name, ':Micro F1', f1_micro)
            pprint(name, ':Macro F1', f1_macro)

            res[name + '-ACC'] = acc
            res[name + '-IC'] = ic
            res[name + '-RankIC'] = rank_ic
            res[name + '-Avg Precision'] = average_precision
            res[name + '-Micro F1'] = f1_micro
            res[name + '-Macro F1'] = f1_macro

            # pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

        all_acc.append(acc)
        all_macrof1.append(f1_macro)
        all_microf1.append(f1_micro)
        all_precision.append(average_precision)
        all_precisionN.append(list(precision.values()))
        all_recallN.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)
        all_ndcg.append(list(ndcg.values()))

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/' + key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path + '/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint('Accuracy: %.4f (%.4f)' % (np.array(all_acc).mean(), np.array(all_acc).std()))
    pprint('IC: %.4f (%.4f)' % (np.array(all_ic).mean(), np.array(all_ic).std()))
    pprint('RankIC: %.4f (%.4f)' % (np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    pprint('F1 Macro: %.4f (%.4f)' % (np.array(all_macrof1).mean(), np.array(all_macrof1).std()))
    pprint('F1 Micro: %.4f (%.4f)' % (np.array(all_microf1).mean(), np.array(all_microf1).std()))
    pprint('Avg Precision: %.4f (%.4f)' % (np.array(all_precision).mean(), np.array(all_precision).std()))
    precision_meanN = np.array(all_precisionN).mean(axis=0)
    precision_stdN = np.array(all_precisionN).std(axis=0)
    recall_meanN = np.array(all_recallN).mean(axis=0)
    recall_stdN = np.array(all_recallN).std(axis=0)
    ndcg_meanN = np.array(all_ndcg).mean(axis=0)
    ndcg_stdN = np.array(all_ndcg).std(axis=0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint('Precision@%d: %.4f (%.4f)' % (N[k], precision_meanN[k], precision_stdN[k]))
        pprint('NDCG@%d: %.4f (%.4f)' % (N[k], ndcg_meanN[k], ndcg_stdN[k]))

    pprint('finished.')


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

    # model
    parser.add_argument('--model_name', default='PatchTST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--K', type=int, default=1)

    # for loss function setting
    parser.add_argument('--loss_type', default='mixed', help='choose mixed or pair_wise')
    parser.add_argument('--num_class', default=5, help='the number of class of stock sequence')
    parser.add_argument('--topk', default=50, help='the number of computing NDCG@k, works when adaptive_k False')
    parser.add_argument('--adaptive_k', default=True)
    parser.add_argument('--ndcg', default='approx')
    parser.add_argument('--class_weight_method', default='square', help='provide square or no')
    parser.add_argument('--beta', default=0.5, help='the ratio of entropy in mixed loss function')
    parser.add_argument('--approxalpha', default=1, type=float, help='the knob of approxNDCG')
    parser.add_argument('--primary', default=False)

    # for ts lib model
    parser.add_argument('--task_name', type=str, default='rep_learning', help='task setup')
    parser.add_argument('--seq_len', type=int, default=60)

    # training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--analysis_mode', default=False)

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
    parser.add_argument('--outdir', default='./output/k_print_test')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:2')

    # mtl methods
    parser.add_argument("--method", type=str, default='our_method', choices=list(METHODS.keys()), help="MTL weight method")
    parser.add_argument("--update_weights_every", type=int, default=1, help="update task weights every x iterations.")
    parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
    parser.add_argument("--dwa_temp", type=float, default=2.0,
                        help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.")
    parser.add_argument("--method_params_lr", type=float, default=0.025,
                        help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",)
    parser.add_argument("--nashmtl_optim_niter", type=int, default=20, help="number of nashmtl iterations")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
