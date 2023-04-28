# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Generator, Optional, Tuple, Union, cast

import pandas as pd

from qlib.backtest.decision import BaseTradeDecision
from qlib.backtest.report import Indicator

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.executor import BaseExecutor

from tqdm.auto import tqdm

from ..utils.time import Freq


PORT_METRIC_TYPE = Dict[str, Tuple[pd.DataFrame, dict]]
INDICATOR_METRIC_TYPE = Dict[str, Tuple[pd.DataFrame, Indicator]]
PORT_METRIC_KEY = "portfolio_dict"
INDICATOR_KEY = "indicator_dict"


def backtest_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
) -> Tuple[PORT_METRIC_TYPE, INDICATOR_METRIC_TYPE]:
    """backtest function for the interaction of the outermost strategy and executor in the nested decision execution

    please refer to the docs of `collect_data_loop`

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator
    """
    return_value: dict = {}
    for _decision in collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value):
        pass

    portfolio_dict = cast(PORT_METRIC_TYPE, return_value.get(PORT_METRIC_KEY))
    indicator_dict = cast(INDICATOR_METRIC_TYPE, return_value.get(INDICATOR_KEY))

    return portfolio_dict, indicator_dict


def collect_data_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
    return_value: dict | None = None,
) -> Generator[BaseTradeDecision, Optional[BaseTradeDecision], None]:
    """Generator for collecting the trade decision data for rl training

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]), setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    trade_strategy : BaseStrategy
        the outermost portfolio strategy
    trade_executor : BaseExecutor
        the outermost executor
    return_value : dict
        used for backtest_loop

    Yields
    -------
    object
        trade decision
    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())

    with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
        _execute_result = None
        while not trade_executor.finished():
            _trade_decision: BaseTradeDecision = trade_strategy.generate_trade_decision(_execute_result)
            _execute_result = yield from trade_executor.collect_data(_trade_decision, level=0)
            trade_strategy.post_exe_step(_execute_result)
            bar.update(1)
        trade_strategy.post_upper_level_exe_step()

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()
        all_portfolio_metrics = {
            "{}{}".format(*Freq.parse(_executor.time_per_step)): _executor.trade_account.get_portfolio_metrics()
            for _executor in all_executors
            if _executor.trade_account.is_port_metr_enabled()
        }
        all_indicators = {}
        for _executor in all_executors:
            key = "{}{}".format(*Freq.parse(_executor.time_per_step))
            all_indicators[key] = (
                _executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe(),
                _executor.trade_account.get_trade_indicator(),
            )
            # all_indicators[
            #     key
            # ] = _executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
            # all_indicators[key + "_obj"] = _executor.trade_account.get_trade_indicator()
        return_value.update({PORT_METRIC_KEY: all_portfolio_metrics, INDICATOR_KEY: all_indicators})
