import torch
import math
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import sys
sys.path.append("..")
from utils.utils import cal_cos_similarity


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=1, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d' % i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d' % i, nn.Linear(
                360 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d' % i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()


class HIST(nn.Module):
    def __init__(self, args):
        super().__init__()

        d_feat = args.d_feat
        hidden_size = args.hidden_size
        self.d_feat = d_feat
        self.task_name = args.task_name

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout,
        )

        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)

        # self.fc_out_ps = nn.Linear(hidden_size, 1)
        # self.fc_out_hs = nn.Linear(hidden_size, 1)
        # self.fc_out_indi = nn.Linear(hidden_size, 1)
        if self.task_name == 'regression':
            self.fc_out = nn.Linear(hidden_size, 1)
        elif self.task_name == 'multi-class':
            self.fc_out = nn.Linear(hidden_size, args.num_class)
        self.K = args.K

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def rep(self, x, concept_matrix, market_value):
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # device = torch.device(torch.get_device(x))
        device = x.device
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]
        # get the last layer embeddings

        # Predefined Concept Module

        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, concept_matrix.shape[1])
        # make the market value matrix the same size as the concept matrix by repeat
        # market value matrix shape: (N, number of pre define concepts)
        stock_to_concept = concept_matrix * market_value_matrix
        # torch.sum generate (1, number of pre define concepts) -> repeat (N, number of predefine concepts)
        # 对应每个concept 得到其相关所有股票市值的和, sum在哪个维度上操作，哪个维度被压缩成1
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        # mul得到结果 （N，number of predefine concepts），每个股票对应的概念不再是0或1，而是0或其相关股票市值之和
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        # 所有位置+1，防止除法报错
        stock_to_concept_sum = stock_to_concept_sum + (
            torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device))
        # 做除法，得到每个股票对应的在concepts上的权重，对应公式4
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        # stock_to_concept transpose (number of predefine concept, N) x_hidden(N, the output of gru)
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        # hidden here is the embeddings of all predefine concept (number of concept, the output of gru)
        # 至此concept的embeddings初始化完成，对应论文中公式5
        hidden = hidden[hidden.sum(1) != 0]
        # stock_to_concept (N, number of concept) 对应embeddings相乘相加
        stock_to_concept = x_hidden.mm(torch.t(hidden))
        # stock_to_concept = cal_cos_similarity(x_hidden, hidden)
        # 对dim0作softmax， stock_to_concept (N, number of concept)，得到不同股票在同一concept上的权重
        stock_to_concept = self.softmax_s2t(stock_to_concept)
        # hidden shape (number of concept, output of gru) now hidden have the embedding of all concepts
        # 使用新得到的权重更新hidden中concept的embeddings
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        # 计算x_hidden和hidden的cos sim concept_to_stock shape (N, number of concept)
        concept_to_stock = cal_cos_similarity(x_hidden, hidden)
        # softmax on dim1, (N, number of concept) 得到同一股票在不同concept上的权重，公式6
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        # p_shared_info (N, output of gru) 公式7的累加部分
        # 过三个不同的linear层输出三个不同的tensor
        # output_ps 通过leaky_relu，公式7
        p_shared_info = concept_to_stock.mm(hidden)
        p_shared_info = self.fc_ps(p_shared_info)

        p_shared_back = self.fc_ps_back(p_shared_info)
        output_ps = self.fc_ps_fore(p_shared_info)
        output_ps = self.leaky_relu(output_ps)

        # pred_ps = self.fc_out_ps(output_ps).squeeze()

        # Hidden Concept Module
        h_shared_info = x_hidden - p_shared_back
        hidden = h_shared_info
        # compute the cos sim between stocks and h_con(h_con generated from stocks, so cos sim with itself)
        h_stock_to_concept = cal_cos_similarity(h_shared_info, hidden)

        dim = h_stock_to_concept.shape[0]
        diag = h_stock_to_concept.diagonal(0)
        # delete itself
        h_stock_to_concept = h_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        # row = torch.linspace(0,dim-1,dim).to(device).long()
        # column = h_stock_to_concept.argmax(1)
        # split dim-1 into dim pieces, then reshape to (dim, 1) -> repeat (dim, K) -> reshape (1, dim*K)
        row = torch.linspace(0, dim - 1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        # found column index of topk value, and reshape to (1, dim*K)
        column = torch.topk(h_stock_to_concept, self.K, dim=1)[1].reshape(1, -1)
        mask = torch.zeros([h_stock_to_concept.shape[0], h_stock_to_concept.shape[1]], device=h_stock_to_concept.device)
        # set the topk position mask to 1
        mask[row, column] = 1
        h_stock_to_concept = h_stock_to_concept * mask
        # add the original embedding h_stock_to_concept (N,N)
        h_stock_to_concept = h_stock_to_concept + torch.diag_embed((h_stock_to_concept.sum(0) != 0).float() * diag)
        # hidden shape (the length of embedding, N)*(N,N) -> transpose (N, the length of embedding)
        hidden = torch.t(h_shared_info).mm(h_stock_to_concept).t()
        # delete concepts that have no connections
        hidden = hidden[hidden.sum(1) != 0]

        h_concept_to_stock = cal_cos_similarity(h_shared_info, hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock)
        h_shared_info = h_concept_to_stock.mm(hidden)
        h_shared_info = self.fc_hs(h_shared_info)

        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)
        output_hs = self.leaky_relu(output_hs)
        # pred_hs = self.fc_out_hs(output_hs).squeeze()

        # Individual Information Module
        individual_info = x_hidden - p_shared_back - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        # pred_indi = self.fc_out_indi(output_indi).squeeze()
        # Stock Trend Prediction
        all_info = output_ps + output_hs + output_indi
        # pred_all = self.fc_out(all_info).squeeze()
        return all_info

    def regression(self, x):
        return self.fc_out(x).squeeze()

    def classification(self, x):
        output = self.fc_out(x)
        return self.softmax_t2s(output)

    def forward(self, x, concept_matrix, market_value):
        all_info = self.rep(x, concept_matrix, market_value)
        if self.task_name == 'regression':
            return self.regression(all_info)  # shape [B, N]
        elif self.task_name == 'multi-class':
            return self.classification(all_info)  # shape [B,N]
        elif self.task_name == 'rep_learning':
            return all_info
        else:
            return None

    def last_shared_parameters(self):
        return self.fc_indi.parameters()


class GRU(nn.Module):
    def __init__(self, args, d_feat=6, num_layers=2, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=args.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.task_name = args.task_name
        if self.task_name == 'regression':
            self.fc = nn.Linear(args.hidden_size, 1)
        elif self.task_name == 'multi-class':
            self.fc = nn.Linear(args.hidden_size, args.num_class)
            self.softmax = torch.nn.Softmax(dim=1)

        self.d_feat = d_feat

    def rep(self, x):
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.gru(x)
        return out[:, -1, :]

    def regression(self, x):
        return self.fc(x).squeeze()

    def classification(self, x):
        return self.softmax(self.fc(x))

    def forward(self, x):
        # x shape N, F*T
        rep = self.rep(x)
        if self.task_name == 'regression':
            return self.regression(rep)
        elif self.task_name == 'classification':
            return self.classification(rep)
        # deliver the last layer as output
        else:
            return rep

    def last_shared_parameters(self):
        return self.gru.parameters()


class LSTM(nn.Module):
    def __init__(self, args, d_feat=6, num_layers=2, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=args.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.task_name = args.task_name
        if self.task_name == 'regression':
            self.fc = nn.Linear(args.hidden_size, 1)
        elif self.task_name == 'multi-class':
            self.fc = nn.Linear(args.hidden_size, args.num_class)
            self.softmax = torch.nn.Softmax(dim=1)

        self.d_feat = d_feat

    def last_shared_parameters(self):
        return self.lstm.parameters()

    def forward(self, x):
        # x shape (N, F*T)
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.lstm(x)
        if self.task_name == 'regression':
            return self.fc(out[:, -1, :]).squeeze()
        elif self.task_name == 'classification':
            return self.softmax(self.fc(out[:, -1, :]))
        # deliver the last layer as output
        else:
            return out[:, -1, :]


class GAT(nn.Module):
    def __init__(self, args, d_feat=6, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=args.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=args.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = args.hidden_size
        self.d_feat = d_feat
        self.transfer = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.task_name = args.task_name
        if self.task_name == 'regression':
            self.fc_out = nn.Linear(args.hidden_size, 1)
        elif self.task_name == 'multi-class':
            self.fc_out = nn.Linear(args.hidden_size, args.num_class)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def last_shared_parameters(self):
        return self.fc.parameters()

    def self_attention(self, x):
        # compute attention between each stock in the day
        x = self.transfer(x)
        stock_num = x.shape[0]
        hidden_size = x.shape[1]
        e_x = x.expand(stock_num, stock_num, hidden_size)  # shape N*N*h
        e_y = torch.transpose(e_x, 0, 1)  # shape N*N*h
        attention_in = torch.cat((e_x, e_y), 2).view(-1, hidden_size * 2)  # shape N*N*2h -> 2N*2h
        self.a_t = torch.t(self.a)  # shape 1*2h
        attention_out = self.a_t.mm(torch.t(attention_in)).view(stock_num, stock_num)  # shape 1*2N -> N*N
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x shape （N，F*T）
        x = x.reshape(len(x), self.d_feat, -1)  # N, F, T
        x = x.permute(0, 2, 1)  # N, T, F
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # N*h
        att_weight = self.self_attention(hidden)  # N*N
        hidden = att_weight.mm(hidden) + hidden  # N*h + N*h
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        if self.task_name == 'regression':
            return self.fc_out(hidden).squeeze()
        elif self.task_name == 'multi-class':
            return self.softmax(self.fc_out(hidden))
        else:
            return hidden


class RSR(nn.Module):
    def __init__(self, args, num_relation, base_model="GRU"):
        super().__init__()

        self.d_feat = args.d_feat
        self.hidden_size = args.hidden_size

        self.rnn = nn.GRU(
            input_size=args.d_feat,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout,
        )
        self.W = nn.Linear((args.hidden_size * 2) + num_relation, 1)
        torch.nn.init.xavier_uniform_(self.W.weight)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.task_name = args.task_name
        if self.task_name == 'regression':
            self.fc = nn.Linear(args.hidden_size * 2, 1)
        elif self.task_name == 'classification':
            self.fc = nn.Linear(args.hidden_size * 2, args.num_class)

    def rep(self, x, relation_matrix):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # device = torch.device(torch.get_device(x))
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        # get the last layer embeddings
        # update embedding using relation_matrix
        # relation matrix shape [N, N]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)  # matrix shape N,N,64+关系数
        # weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.W(matrix).squeeze(2)
        weight = self.leaky_relu(weight)  # relu layer
        index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
        mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        mask[index[0], index[1]] = 1
        # valid_weight = mask*weight
        # valid_weight = self.softmax1(valid_weight)
        temp_weight = mask * weight
        index_2 = torch.t((temp_weight == 0).nonzero())
        temp_weight[index_2[0], index_2[1]] = -10000
        valid_weight = self.softmax1(temp_weight)  # N,N
        valid_weight = valid_weight * mask
        hidden = torch.matmul(valid_weight, x_hidden)
        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        # pred_all = self.fc(hidden).squeeze()
        return hidden

    def regression(self, hidden):
        return self.fc(hidden).squeeze()

    def classification(self, hidden):
        return self.softmax1(self.fc(hidden))

    def forward(self, x, relation_matrix):
        # x shape N, F*T
        rep = self.rep(x, relation_matrix)
        if self.task_name == 'regression':
            return self.regression(rep)
        elif self.task_name == 'classification':
            return self.classification(rep)
        # deliver the last layer as output
        else:
            return rep

    def last_shared_parameters(self):
        return self.W.parameters()


