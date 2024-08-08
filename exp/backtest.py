import pandas as pd
from pprint import pprint
import qlib
import pandas as pd
from qlib.utils.time import Freq
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
import pyfolio as pf
from tqdm import tqdm


def backtest_helper(data, start_date, end_date, topk, n_drop, model_name):
    qlib.init(provider_uri="../qlib_data/cn_data")
    CSI300_BENCH = "SH000300"
    FREQ = "day"
    STRATEGY_CONFIG = {
        "topk": topk,
        "n_drop": n_drop,
        # pred_score, pd.Series
        "signal": data,
    }

    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }

    backtest_config = {
        "start_time": start_date,
        "end_time": end_date,
        "account": 100000000,
        "benchmark": CSI300_BENCH,  # "benchmark": NASDAQ_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.00005,
            "close_cost": 0.0003,
            "min_cost": 5,
        },
    }

    # strategy object
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
    print(pf.tears.create_simple_tear_sheet(returns=report_normal['return'], benchmark_rets=report_normal['bench']))

    # analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"]
                                                           - report_normal["bench"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"]
                                                        - report_normal["bench"]
                                                        - report_normal["cost"], freq=analysis_freq)
    analysis["excess_return_with_cost"].rename(columns={'risk': model_name + '_' + start_date[:4] + 'top' + str(topk)},
                                               inplace=True)
    df = analysis["excess_return_with_cost"].transpose()
    return df


def backtest_module(symbol_file, start_date, end_date, topk, drop_n, model_name):
    data = pd.read_pickle(symbol_file)
    def helper(x):
        #         return x['pred_score']
        if x['pred_class'] > 2:
            return x['pred_score'] + 0.1 * abs(x['pred_score'])
        elif x['pred_class'] < 2:
            return x['pred_score'] - 0.1 * abs(x['pred_score'])
        else:
            return x['pred_score']

    if 'pred_class' in data.columns:
        data['symbol'] = data.apply(lambda x: helper(x), axis=1)
        # data.columns=[['score']]
        data = data[['symbol']]
        data.columns = [['score']]
    else:
        data = data[['score']]
        data.columns = [['score']]
    slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
    data = data[slc]
    df = backtest_helper(data, start_date, end_date, topk, drop_n, model_name)
    return df


if __name__ == '__main__':
    year_list = [('2020-07-01', '2020-09-30')]
    # year_list = [('2017-01-01', '2017-12-31'), ('2018-01-01', '2018-12-31')]
    strategy_list = [10]
    file_list = ['./pred_output/RSR_str.pkl', './pred_output/GRU_str.pkl',
                 './pred_output/HIST_str.pkl']
    name_list = ['RSR', 'GRU', 'HIST']
    df_list = []
    for f in tqdm(file_list):
        for y in year_list:
            for s in strategy_list:
                df_list.append(backtest_module(f,y[0], y[1], s, 0, name_list[file_list.index(f)]))
    pd.concat(df_list)