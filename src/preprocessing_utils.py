import numpy as np
import pandas as pd
import numba
from typing import Dict, List


@numba.jit(forceobj=True)
def triple_dot(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:
    """ accelerated intermediate calculations in the table separating process """
    return c1 * c2 * c3


def separate_and_save(df, names, sep_col='symbol',
                      path_to_save='data/'):
    """
    Splits the table into several small tables according to the value
    of the ``sep_col`` column.

    :param df: pd.DataFrame: a large table, which must be divided into several smaller ones
    :param names: List[str]: the values of the ``sep_col`` column for which the tables
     are to be saved
    :param sep_col: str: the name of the column by which we want to split the table
    :param path_to_save: str: the address where we want to save the tables
    :return: nothing is returned, separated tables by ``sep_col`` column values
     from the ``names`` list are saved to the address ``path_to_save`` in csv format
    """
    df_grouped = df.groupby(by=sep_col)
    for name in names:
        t = df_grouped.get_group(name)
        # accelerated intermediate calculations
        t['money_buy'] = triple_dot(t['price'].to_numpy(),
                                    t['quantity'].to_numpy(),
                                    t['is_buy'].to_numpy())
        t['money_sell'] = triple_dot(t['price'].to_numpy(),
                                     t['quantity'].to_numpy(),
                                     (1 - t['is_buy']).to_numpy())
        t['is_not_buy'] = 1 - t['is_buy']
        t.to_csv(f'{path_to_save}/{name}.csv')
    return


def load_tables(names,
                path_from):
    """
    Reads tables from ``path_from`` address with names from the list ``names`` into the dictionary.

    :param names: List[str]: table names
    :param path_from: str: path to the tables
    :return: Dict[str, pd.DataFrame]: dictionary, its keys are ``names`` list items, values are loaded
     tables from ``path_from/name.csv``
    """
    df_dict = {}
    for name in names:
        df_dict[name] = pd.read_csv(f'{path_from}/{name}.csv',
                                    index_col='event_time')
        if 'Unnamed: 0' in df_dict[name].columns:
            df_dict[name] = df_dict[name].drop(['Unnamed: 0'], axis=1)

    return df_dict


def save_tables(df_dict,
                names,
                path_to):
    """
    Saves tables from ``df_dict`` dictionary with keys from `names`` list.

    :param df_dict: Dict[str, pd.DataFrame]: dictionary with dataframes
    :param names: List[str]: a subset of the ``df_dict`` keys for which the tables are to be saved
    :param path_to: str: the address where we want to save the tables
    """
    for name in names:
        df_dict[name].to_csv(f'{path_to}/{name}.csv')


def quantize_table(df, freq='300ms'):
    """
    Returns a dataframe quantized by ``freq``-sized windows. Inside each window it calculates
    some statistics (mean, median, standard deviation, and others).

    :param df: pd.DataFrame: unprocessed dataframe
    :param freq: str: quantization window width
    :return: pd.DataFrame: dataframe quantized by ``freq``-sized windows
    """
    if 'event_time' not in df.columns:
        df.reset_index(inplace=True)
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.set_index('event_time')

    return df.groupby(pd.Grouper(freq=freq, origin='start_day')).agg(
        price_mean=('price', np.mean),
        price_median=('price', np.median),
        price_std=('price', np.std),
        buy_price_sum=('money_buy', np.sum),
        sell_price_sum=('money_sell', np.sum),
        nonzero_count=('price', np.count_nonzero),
        buy_count=('is_buy', np.sum),
        sell_count=('is_not_buy', np.sum),
    )
