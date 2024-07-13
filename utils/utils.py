import torch
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from torchmetrics import RetrievalNormalizedDCG
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import sys
sys.path.append("..")
def mse(pred, label):
    loss = (pred - label) ** 2
    return torch.mean(loss)


def cal_cos_similarity(x, y):  # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
    cos_similarity = xy / x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


def np_relu(x):
    return x * (x > 0)


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    ndcg = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    temp2 = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='label', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)
    if len(temp2.index[0]) > 2:
        temp2 = temp2.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = (temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k)
                        / temp2.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k)).mean()

        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()
        ndcg[k] = preds.groupby(level='datetime').apply(lambda x: ndcg_score([np_relu(x.score)],
                                                                             [np_relu(x.label)], k=k)).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    return precision, recall, ic, rank_ic, ndcg


def metric_fn_mto(preds, pred_column='score', gt_column='label'):
    preds = preds[~np.isnan(preds[gt_column])]
    precision = {}
    recall = {}
    ndcg = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by=pred_column, ascending=False))
    temp2 = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by=gt_column, ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)
    if len(temp2.index[0]) > 2:
        temp2 = temp2.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = (temp.groupby(level='datetime').apply(lambda x: (x[gt_column][:k] > 0).sum() / k)
                        / temp2.groupby(level='datetime').apply(lambda x: (x[gt_column][:k] > 0).sum() / k)).mean()

        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x[gt_column][:k] > 0).sum() / (x[gt_column] > 0).sum()).mean()
        ndcg[k] = preds.groupby(level='datetime').apply(lambda x: ndcg_score([np_relu(x[pred_column])],
                                                                             [np_relu(x[gt_column])], k=k)).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x[gt_column].corr(x[pred_column])).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x[gt_column].corr(x[pred_column], method='spearman')).mean()
    return precision, recall, ic, rank_ic, ndcg


def loss_ic(pred, label):
    """
    directly use 1-ic as the loss function
    """
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    res = torch.stack([pred, label])
    return 1 - torch.corrcoef(res)[0, 1]


def pair_wise_loss(args, pred, label):
    """
    original loss function in RSR
    use it to replace NDCG methods
    """
    pred_p, label_p = class_NDCG_generation(args, pred, label)
    all_one = torch.ones(pred_p.shape[0], device=pred_p.device, dtype=torch.float32)
    pred_diff = torch.matmul(all_one, pred_p.float()) - torch.matmul(pred_p.float().T, all_one.T)
    label_diff = torch.matmul(all_one, label_p.float()) - torch.matmul(label_p.float().T, all_one.T)
    pair_wise = torch.mean(torch.nn.ReLU()(-pred_diff * label_diff))
    return pair_wise


def NDCG_loss(pred, label, alpha=0.05, k=100):
    """
    NDCG loss function
    """
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    index = torch.zeros(label.shape, dtype=torch.int64, device=pred.device)
    ndcg = RetrievalNormalizedDCG(k=k)
    ndcg_loss = alpha * ndcg(pred, label, indexes=index)
    point_wise = mse(pred, label)
    # point-wise decrease, model better
    # ndcg increase, model better
    return point_wise - ndcg_loss


def approxNDCGLoss_cutk(y_pred, y_true, eps=1, alpha=1., k=20):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor

    # the alpha should be large to better fit the indicator.
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    # padded_mask = y_true == padded_value_indicator
    # y_pred[padded_mask] = float("-inf")
    # y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    # print(y_pred_sorted.grad_fn)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    # 按照pred的顺序来排列
    true_sorted_by_preds = torch.gather(y_true, dim=0, index=indices_pred)
    # true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    # just like relu, let all values that below 0 equal to 0
    true_sorted_by_preds.clamp_(min=0.0000)
    y_true_sorted.clamp_(min=0.)
    if k == -1:
        # not perform cut
        true_sort_by_preds_k = true_sorted_by_preds
        y_pred_sorted_k = y_pred_sorted
        y_true_sorted_k = y_true_sorted
    else:
        true_sort_by_preds_k = true_sorted_by_preds[:k]
        y_pred_sorted_k = y_pred_sorted[:k]
        y_true_sorted_k = y_true_sorted[:k]
    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_true_sorted_k.shape[0] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted_k) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sort_by_preds_k) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = y_pred_sorted_k[:, None].repeat(1, y_pred_sorted_k.shape[0]) - \
                   y_pred_sorted_k[None, :].repeat(y_pred_sorted_k.shape[0], 1)
    # scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    # when sigmoid = 0.5, then the diff = 0, which is the original value of itself
    sig = torch.sigmoid(-alpha * scores_diffs)
    # this have some problems, may introduce some non-differentiable operations.
    approx_pos = 1. + torch.sum(sig * (sig > 0.5), dim=-1)
    # approx_pos = 1. + torch.sum(sig, dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)
    # return -torch.mean(approx_NDCG)
    # add a exp to make it to positive!
    return torch.exp(-torch.mean(approx_NDCG))


def ApproxNDCG_loss(pred, label, alpha=0.5, k=100):
    mask = ~torch.isnan(label)
    pred = pred[mask]
    label = label[mask]
    ndcg_part = approxNDCGLoss_cutk(pred, label, k=k) * alpha
    point_wise = mse(pred, label)
    return point_wise + ndcg_part


def class_NDCG_generation(args, pred, label):
    """
    issues: if we use NDCG, get weight*prob, then most will drop into middle part, this is not helpful
            for example, with [3,2,1,0] related weights, we have two prob [0,0.5,0.5,0] and [0.5,0,0,0.5]
            those two have THE SAME weights, this is not RIGHT!
    :param pred: the output of model [B,N]
    :param label: the ground truth [B]
    :return:
    """
    m = torch.nn.Softmax(dim=1)
    pred_n = m(pred)  # [B,N]
    if args.class_weight_method == 'square':
        weights = [w * w for w in range(pred_n.shape[1])]
        default_weight = torch.tensor(weights, device=pred_n.device, dtype=torch.float32)
        x = pred_n @ default_weight
        mc_label = torch.square(label)
    elif args.class_weight_method == 'no':
        weights = [w for w in range(pred_n.shape[1])]
        default_weight = torch.tensor(weights, device=pred_n.device, dtype=torch.float32)
        x = pred_n @ default_weight
        mc_label = label
    else:
        return None
    return x, mc_label


def cross_entropy(pred, label):
    """
    :param pred: the prediction result from model shape [B,N]
    :param label: the label from dataset is the score, we will divide those score into 4 groups as their label
    :return:

    [update]: no Norm and split all stocks into several bins,
    D5+, D5, D4, D3, D2, D1, U1, U2, U3, U4, U5, U5+
    if the stock drop more than 20% or rise more than 20%, then they are abnormal stocks
    """
    ce_loss = torch.nn.CrossEntropyLoss()
    # group = pred.shape[1]
    # mc_label = torch.zeros(label.shape[0], device=pred.device, dtype=torch.long)  # shape [B]
    # totally 12 bins
    # for i in range(1, group):
    #     indices = torch.topk(label, int(label.shape[0] * i / group)).indices.to(device=pred.device)
    #     mc_label[indices] += 1
    return ce_loss(pred, label)


def class_approxNDCG(args, pred, label):
    """
    :param pred: the prediction result from model shape [B,N]
    :param label: the label from dataset is the score, we will divide those score into 4 groups as their label
    we do softmax on pred, then times the [3,2,1,0] matrix to get the final weights, the label weights are from
    0 to 3, and compute the approxNDCG
    """
    soft_x, mc_label = class_NDCG_generation(args, pred, label)  # shape [B]

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

    if args.adaptive_k:
        unique_values, counts = torch.unique(label, return_counts=True)
        sorted_indices = torch.argsort(unique_values, descending=True)
        sorted_counts = counts[sorted_indices]
        acculated = [sum(sorted_counts[:i + 1]) for i in range(len(sorted_counts))]
        k_value = 0
        for i in acculated:
            if i / acculated[-1] > 0.2:
                k_value = i.item()
                break
    else:
        k_value = args.topk

    # pprint('K value is: start token'+str(k_value)+'end token')
    return approxNDCGLoss_cutk(soft_x, mc_label.float(), k=k_value, alpha=args.approxalpha)


def generate_label(pred, label):
    # group = pred.shape[1]
    # mc_label = torch.zeros(label.shape[0], device=pred.device, dtype=torch.long)  # shape [B]
    # for i in range(1, group):
    #     indices = torch.topk(label, int(label.shape[0] * i / group)).indices
    #     mc_label[indices] += 1  # shape B
    # pred_label = torch.topk(pred, 1).indices.squeeze()  # shape B
    pred_label = torch.topk(pred, 1).indices.squeeze()  # shape B
    return pred_label, label


def evaluate_mc(preds, pred_column='pred', gt_column='ground_truth'):
    # default precision setting is micro
    acc = preds.groupby(level='datetime').apply(lambda x: accuracy_score(x[gt_column], x[pred_column])).mean()
    average_precision = preds.groupby(level='datetime') \
        .apply(lambda x: precision_score(x[gt_column], x[pred_column], average='micro')).mean()
    f1_micro = preds.groupby(level='datetime'). \
        apply(lambda x: f1_score(x[gt_column], x[pred_column], average='micro')).mean()
    f1_macro = preds.groupby(level='datetime'). \
        apply(lambda x: f1_score(x[gt_column], x[pred_column], average='macro')).mean()
    # roc_auc = preds.groupby(level='datetime').\
    #     apply(lambda x: roc_auc_score(x['ground_truth'], x['pred_array'], average='macro')).mean()

    return acc, average_precision, f1_micro, f1_macro


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs

MAX_ITER = 250
STOP_CRIT = 1e-5


def _min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def _min_norm_2d(vecs, dps):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = 1e8
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if (i, j) not in dps:
                dps[(i, j)] = 0.0
                for k in range(len(vecs[i])):
                    dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                dps[(j, i)] = dps[(i, j)]
            if (i, i) not in dps:
                dps[(i, i)] = 0.0
                for k in range(len(vecs[i])):
                    dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
            if (j, j) not in dps:
                dps[(j, j)] = 0.0
                for k in range(len(vecs[i])):
                    dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
            c, d = _min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
            if d < dmin:
                dmin = d
                sol = [(i, j), c, d]
    return sol, dps


def _projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return np.maximum(y - tmax_f, np.zeros(y.shape))


def _next_point(cur_val, grad, n):
    proj_grad = grad - (np.sum(grad) / n)
    tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
    t = 1
    if len(tm1[tm1 > 1e-7]) > 0:
        t = np.min(tm1[tm1 > 1e-7])
    if len(tm2[tm2 > 1e-7]) > 0:
        t = min(t, np.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point


class DoubleBuffer:
    """
    create two buffer list for training
    """

    def __init__(self, capacity=6):
        self.capacity = capacity
        self.buffer_A = []
        self.buffer_B = []

    def add_value(self, value):
        self.buffer_B.append(value)

        if len(self.buffer_A) < self.capacity:
            self.buffer_A.append(self.buffer_B.pop(0))
        elif len(self.buffer_B) > self.capacity:
            self.buffer_A.append(self.buffer_B.pop(0))
            self.buffer_A.pop(0)

    def get_buffers(self):
        return self.buffer_A, self.buffer_B

    def get_latest(self):
        return self.buffer_B[-1]

    def check_capacity(self):
        if len(self.buffer_B) == self.capacity and len(self.buffer_A) == self.capacity:
            return True
        else:
            return False

    def compute_loss_drop(self):
        # larger the return value, better train in those epochs
        return sum(self.buffer_B[-2:-1])/len(self.buffer_B[-2:-1]) - sum(self.buffer_A)/len(self.buffer_A)


