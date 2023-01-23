import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from typing import Dict, Optional, List, Tuple


def timestamp_to_features(t):
    """
    Extract 4 numeric characteristics from the time feature:
    hour, minute, second, millisecond.

    :param t: str: timestamp string
    :return: Tuple[int, int, int, int]: time, minute, second, and millisecond of timestamp t
    """
    t = pd.to_datetime(t)
    return t.hour, t.minute, t.second, t.microsecond//1000


def bcv_extract_features(
        df,
        n_blocks,
        target_col,
        n_jobs=1,
        n_windows=5,
        window_size=20,
        lags = None,
        mode='default',
        fc_parameters=None,
):
    """
    Implement the process of block cross validation of time series with
    counting of window features within each block:

    - divide the entire dataframe evenly into ``n_tests`` blocks

    - inside each block calculate statistics on ``n_windows`` windows of size ``windows_size``

    - as a target column make % change in ``target_col`` to simplify the task for the tree based models

    - creates lag features with numbers from ``lags``


    Depending on the ``mode``, window feature evaluations are satisfactorily
    either using the ``tsfresh`` methods (parallel mode) or using the
    ``window_featurizing`` method (default mode). The difference is that the first method can be parallelized, but in
    the process of execution, the size of memory can increase many times over
    (due to ``roll_time_series`` function). In the second method, the amount of
    memory does not grow, but it will not work to parallelize the process.

    **Note**

    - all missing values are simply deleted, thus one validation block may contain non-sequential data;

    - the process of evaluating the features itself is parallel in any case, the only difference in mode is how the
      windows is formed;

    :param df: pd.DataFrame: table with data for which it is necessary to carry out block cross-validation with counting
              window features

    :param n_blocks: int: number of blocks for block cross validation

    :param target_col: str: the name of the column with the target variable
    :param n_jobs: int: number of cores for parallel calculations (for extracting/rolling)
    :param n_windows: int: the number of windows for which it is necessary to calculate window functions within each block
    :param window_size: int: number of elements to be used in counting each window function
    :param lags: Optional[List[int]]: numbers for which it is necessary to create lag features
    :param mode: str: windowing mode for feature extract
    :param fc_parameters: Optional[Dict[str, Optional[List[str]]]]: a dictionary containing information about which window functions should be calculated
     and with  what parameters
    :return: List[pd.DataFrame]: list of ``n_tests`` dataframes of ``n_windows`` size with the addition of new features
     (window functions, lags, 'target' column)
    """

    assert mode == 'default' or mode == 'parallel', \
        f'mode must be "default" or "parallel", not {mode}!'

    if lags is None:
        lags = [1]
        
    for lag in lags:
        df[f'price_lag{lag}'] = df[target_col].shift(lag)

    df['target'] = 100 * (df[target_col].shift(-1) -
                          df[target_col]) / df[target_col]

    if 'event_time' not in df.columns:
        df = df.dropna().reset_index()
    else:
        df = df.dropna().reset_index(drop=True)

    n = df.shape[0]
    fold_size = n // n_blocks

    assert fold_size >= window_size + n_windows - 1, f'the parameters n_tests, ' \
        f'train_size, window_size, n_windows are inconsistent, there is ' \
        f'not enough space to count window features in the fold: \n' \
        f'fold_size={fold_size} < {window_size + n_windows - 1}=window_size+n_windows-1 '

    assert max(lags) <= fold_size, f'data leak, max(lags)={max(lags)} is too much'

    df[['hour', 'min', 'sec', 'ms']] = [timestamp_to_features(date) for date in df.event_time]
    blocks = []

    for i in range(n_blocks, 0, -1):

        print(f'current block: {n_blocks - i + 1}/{n_blocks}')
        print('==' * int(20 * (n_blocks - i + 1) / n_blocks) +
              '--' * int(20 * ((i - 1) / n_blocks)))

        end_block = n - fold_size * (i - 1) - 1
        block = df.loc[end_block - window_size + 1 - n_windows +
                       1:end_block].reset_index(drop=True)

        if mode == 'parallel':
            # take advantage of the parallel execution feature of tsfresh,
            # but it makes the memory size grow a lot!
            block['id'] = i
            rolled_block = roll_time_series(block,
                                            column_id="id",
                                            column_sort="event_time",
                                            max_timeshift=window_size - 1,
                                            n_jobs=n_jobs,
                                            min_timeshift=window_size - 1)

            new_features = extract_features(timeseries_container=rolled_block,
                                            column_id="id",
                                            n_jobs=n_jobs,
                                            column_sort="event_time",
                                            column_value=target_col,
                                            impute_function=impute,
                                            show_warnings=False,
                                            default_fc_parameters=fc_parameters
                                            )

            block_featurized = block.loc[window_size - 1:].reset_index(drop=True)
            block_featurized[new_features.columns] = new_features.values
            block_featurized.drop(['id'], axis=1, inplace=True)

        elif mode == 'default':
            # let's calculate the window functions through our function
            block_featurized = window_featurize(block,
                                                n_windows=n_windows,
                                                window_size=window_size,
                                                target_col=target_col,
                                                n_jobs=n_jobs,
                                                fc_parameters=fc_parameters)
        else:
            raise Exception('Wrong mode!')

        # because of the timestamp_to_features call, this feature is no longer needed
        block_featurized.drop(['event_time'], axis=1, inplace=True)

        blocks.append(block_featurized)

    return blocks


def window_featurize(df,
                     target_col,
                     n_windows=5,
                     window_size=20,
                     n_jobs=1,
                     fc_parameters=None
                     ):
    """
    To calculate features from ``tsfresh``, we need to
    have some piece of the time series. The
    ``tsfresh.utilities.dataframe_functions.roll_time_series``
    method makes the dataset larger by an amount equal to the size
    of the window. To avoid this problem, it will process each
    window separately.

    :param df: pd.DataFrame: table with data for which it is necessary to calculate
     window features
    :param target_col: str: the name of the column with the target variable
    :param n_windows: int: the number of windows for which it is necessary
     to calculate window functions
    :param window_size: int: number of elements to be used in counting each window function
    :param n_jobs: int: number of cores for parallel execution
    :param fc_parameters: Dict[str, Optional[List[str]]]: a dictionary containing information about which window functions
     should be calculated and with what parameters
    :return: pd.DataFrame: dataframe of ``num_windows`` rows with counted window functions
    """

    if fc_parameters is None:
        fc_parameters = EfficientFCParameters()
    if 'event_time' not in df.columns:
        df = df.dropna().reset_index()
    else:
        df = df.dropna().reset_index(drop=True)

    assert 'event_time' in df.columns, f'even_time there no exist, df.columns:\n{df.columns}'
    n = df.shape[0]
    assert n >= window_size + n_windows - 1, 'small df'

    new_features = pd.DataFrame()
    for i in range(n_windows):
        end_window = n - i - 1
        window = df.loc[end_window - window_size + 1:end_window]
        window['id'] = i
        new_features = pd.concat([
            extract_features(timeseries_container=window,
                             column_id="id",
                             column_sort="event_time",
                             column_value=target_col,
                             impute_function=impute,
                             show_warnings=False,
                             n_jobs=n_jobs,
                             default_fc_parameters=fc_parameters),
            new_features
        ])

    return pd.concat([
        df.loc[n - n_windows:].reset_index(drop=True),
        new_features.reset_index(drop=True)],
        axis=1)
