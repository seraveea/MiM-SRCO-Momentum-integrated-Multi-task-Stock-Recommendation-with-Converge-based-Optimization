import torch
import numpy as np
import pandas as pd
import datetime


class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, pin_memory=True,
                 start_index=0, device=None, task=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            if task == 'multi-class':
                self.df_label = torch.tensor(self.df_label, dtype=torch.long, device=device)
            else:
                self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            # self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            # this is the default situation
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):
        """
        :return:  number of days in the dataloader
        """
        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i + self.batch_size]  # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        """
        : yield an index and a slice, that from the day
        """
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]
        mask = self.padding_mask(outs[0])

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc], mask, )

    def padding_mask(self, features, max_len=None):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        padding_masks = self._padding_mask(torch.tensor(lengths, dtype=torch.int16, device=self.device), max_len=max_len)
        # (batch_size, padded_length) boolean tensor, "1" means keep
        return padding_masks

    @staticmethod
    def _padding_mask(lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class DataLoader_mto:
    """
    there will be two labels in mto_dataloader
    label is the category of time series sequence
    lable_2 is the normalized recommendation score
    """
    def __init__(self, df_feature, df_label, df_label_2, df_market_value, df_stock_index, batch_size=800, pin_memory=True,
                 start_index=0, device=None, task=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_label_2 = df_label_2.values
        self.df_market_value = df_market_value
        self.df_stock_index = df_stock_index
        self.device = device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.long, device=device)
            self.df_label_2 = torch.tensor(self.df_label_2, dtype=torch.float, device=device)
            # self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_stock_index = torch.tensor(self.df_stock_index, dtype=torch.long, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            # this is the default situation
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):
        """
        :return:  number of days in the dataloader
        """
        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i + self.batch_size]  # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        """
        : yield an index and a slice, that from the day
        """
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_label_2[slc][:, 0],\
            self.df_market_value[slc], self.df_stock_index[slc]
        # outs = self.df_feature[slc], self.df_label[slc]
        mask = self.padding_mask(outs[0])

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, device=self.device) for x in outs)

        return outs + (self.index[slc], mask, )

    def padding_mask(self, features, max_len=None):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        padding_masks = self._padding_mask(torch.tensor(lengths, dtype=torch.int16, device=self.device), max_len=max_len)
        # (batch_size, padded_length) boolean tensor, "1" means keep
        return padding_masks

    @staticmethod
    def _padding_mask(lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


def create_mto_loaders(args, device):
    """
    load qlib alpha360 data and split into train, validation and test loader
    :param args:
    :return:
    """
    df_original = pd.read_pickle(args.mtm_source_path)

    # split those three dataset into train, valid and test
    # import pickle5 as pickle
    import pickle
    with open(args.market_value_path, "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)
    # df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value / 1000000000
    # market value of every day from 07 to 20
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train = df_original[slc]
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = len(stock_index)-1
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(len(stock_index)-1).astype(int)
    # the market value and stock_index added to each line
    train_loader = DataLoader_mto(df_train["feature"], df_train[args.mtm_column], df_train["label"], df_train['market_value'],
                              df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory,
                              start_index=start_index, device=device,task='multi-class')

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid = df_original[slc]
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = len(stock_index)-1
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(len(stock_index)-1).astype(int)

    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader_mto(df_valid["feature"], df_valid[args.mtm_column], df_valid["label"], df_valid['market_value'],
                              df_valid['stock_index'], pin_memory=True, start_index=start_index,
                              device=device, task='multi-class')

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test = df_original[slc]
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = len(stock_index)-1
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(len(stock_index)-1).astype(int)
    start_index += len(df_test.groupby(level=0).size())
    test_loader = DataLoader_mto(df_test["feature"], df_test[args.mtm_column], df_test["label"], df_test['market_value'],
                             df_test['stock_index'], pin_memory=True, start_index=start_index,
                             device=device, task='multi-class')

    return train_loader, valid_loader, test_loader


def create_mtm_loaders(args, device):
    """
    the difference from mtm loader is that in this mto loader, we need to create two ground truth for two
    tasks: regression and mtm_loaders
    """
    df_original = pd.read_pickle(args.mtm_source_path)

    # split those three dataset into train, valid and test
    # import pickle5 as pickle
    import pickle
    with open(args.market_value_path, "rb") as fh:
        # load market value
        df_market_value = pickle.load(fh)
    # df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value / 1000000000
    # market value of every day from 07 to 20
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train = df_original[slc]
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    # the market value and stock_index added to each line
    train_loader = DataLoader(df_train["feature"], df_train[args.mtm_column], df_train['market_value'],
                              df_train['stock_index'],batch_size=args.batch_size, pin_memory=args.pin_memory,
                              start_index=start_index,device=device,task='multi-class')

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid = df_original[slc]
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid[args.mtm_column], df_valid['market_value'],
                              df_valid['stock_index'],pin_memory=True, start_index=start_index,
                              device=device,task='multi-class')

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test = df_original[slc]
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())
    test_loader = DataLoader(df_test["feature"], df_test[args.mtm_column], df_test['market_value'],
                             df_test['stock_index'],pin_memory=True, start_index=start_index,
                             device=device,task='multi-class')

    return train_loader, valid_loader, test_loader


def bin_helper(x):
    if x < -0.05:
        return 0
    elif x < 0:
        return 1
    elif x < 0.05:
        return 2
    else:
        return 3
    # if x < -0.05:
    #     return 0
    # elif x < -0.04:
    #     return 1
    # elif x < -0.03:
    #     return 2
    # elif x < -0.02:
    #     return 3
    # elif x < -0.01:
    #     return 4
    # elif x < 0:
    #     return 5
    # elif x < 0.01:
    #     return 6
    # elif x < 0.02:
    #     return 7
    # elif x < 0.03:
    #     return 8
    # elif x < 0.04:
    #     return 9
    # elif x < 0.05:
    #     return 10
    # else:
    #     return 11