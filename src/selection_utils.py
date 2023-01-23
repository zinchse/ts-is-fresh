import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import re
import shap
import xgboost
from xgboost import XGBRegressor
from tsfresh.feature_selection.relevance import calculate_relevance_table


def stats_select_features(relevance_table):
    """
    Using a table with the statistical significance of each feature,
    returns only low-correlated relevant features.

    It is assumed that the correlated attributes are calls of the same
    function with different parameters. Therefore, all the features are
    factorized by the values of the function arguments, and from each class
    the representative with the lowest ``p_value`` is selected. Because the
    table is sorted by ``p_value``, factorization is easy to implement
    through a set.

    :param relevance_table: pd.DataFrame: a table with the calculated features and their statistical significance
    :return: List[str]: a list of names of relevant low-correlated features from ``relevance_table``.
    """
    seen, res = set(), set()
    size_set = 0
    # to delete all parameter values in the feature name
    reg = re.compile(r'[-0-9]|"[^"]*"')

    for cur_iter in range(relevance_table.shape[0]):
        name, relevant = relevance_table[['feature', 'relevant']].iloc[cur_iter]

        if relevant == 'False':
            break

        normalized_name = reg.sub('', name)
        if normalized_name not in seen:
            seen.add(normalized_name)
            res.add(name)
            size_set += 1

    return list(res)


def get_fitted_models(
        train_list,
        n_jobs=8):
    """
    Returns the trained model for each ``train_list`` dataframe.

    :param train_list: List[pd.DataFrame]: training data list
    :param n_jobs: int: number of cores for parallel learning
    :return: List[xgboost.sklearn.XGBRegressor]: list fitted ``XGBRegressor`` models
    """
    models = []
    n_models = len(train_list)
    for i in range(n_models):
        print(f'current model: {i + 1}/{n_models}')
        print('**' * int(20 * (i + 1) / n_models) +
              '..' * int(20 * ((n_models - i - 1) / n_models)))

        assert 'target' in train_list[i].columns, \
            f'train[{i}] must contain a target column!' \
            '\ntrain.columns:\n{train[i].columns}'

        train_x, train_y = train_list[i].drop(['target'],
                                              axis=1), train_list[i].loc[:, 'target']

        models.append(
            XGBRegressor(njobs=n_jobs,
                         objective='reg:squarederror',
                         random_state=i,
                         n_estimators=1000).fit(train_x, train_y))

    return models


def get_importance(
        models,
        train_list,
        mode='all',
):
    """
    Using the built-in feature importance estimation methods within ``XGBRegressor``
    and the shap algorithm, it calculates the importance of the features on all
    training data, normalizes and averages them.

    :param models: List[xgboost.sklearn.XGBRegressor]: the list of trained models
    :param train_list: List[pd.DataFrame]: the list of training data
    :param mode: str:  importance calculating mode
    :return: Dict[str, float]: dictionary, its keys are the features from the training data,
     and the values are the calculated importance
    """
    feature_names = train_list[0].columns
    importance_dict = defaultdict(float)

    possible_modes = ['gain', 'weight', 'cover', 'total_gain', 'total_cover', 'all', 'shap']
    assert mode in possible_modes, \
        f'The mode must be one of {possible_modes}, not {mode}!'

    if mode == 'all':
        importance_type = ['gain', 'weight', 'cover', 'total_gain', 'total_cover']
    elif mode == 'shap':
        importance_type = []
    else:
        importance_type = [mode]

    for t in importance_type:
        for i, model in enumerate(models):
            importance_ti = model.get_booster().get_score(importance_type=t)
            s = sum(importance_ti.values())
            for k in importance_ti.keys():
                # normalize importance (list item level)
                importance_dict[k] += importance_ti[k] / s

    if mode == 'shap' or mode == 'all':
        for i, model in enumerate(models):
            train_x, train_y = train_list[i].drop(['target'],
                                                  axis=1), train_list[i].loc[:, 'target']

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train_x)
            row_shap_importance = abs(shap_values).sum(axis=0)
            s = sum(row_shap_importance)
            for j, shap_imp in enumerate(row_shap_importance):
                importance_dict[feature_names[j]] += shap_imp / s

    # normalize importance (list level)
    for k in importance_dict.keys():
        shap_used = (mode == 'shap') or (mode == 'all')
        importance_dict[k] /= len(train_list) * (len(importance_type) + shap_used)
    return importance_dict


def importance_select_features(importance_dict,
                               portion=0.8):
    """
    According to the values of the importance of the attributes selects
    the best of them, which contain the ``portion`` % of the importance
    of all the features.

    :param importance_dict: Dict[str, float]: a dictionary with the importance of each feature
    :param portion: float: portion of the importance of all the features to be ensured
    :return: List[Tuple[str, float]]: a minimum number of features, the overall importance of which >= ``portion``
    """
    assert 0.0 <= portion <= 1.0, f'portion must be in [0;1], not {portion}!'
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    val = .0
    size = 0
    while val < portion:
        val += sorted_features[size][1]
        size += 1
    return sorted_features[:size]


def get_stats(
        blocks,
        n_jobs=1
):
    """
    Using statistical criteria, calculates the significance of the features
    for each block in the list. Then the obtained ``p_value`` s are averaged.

    :param blocks: List[pd.DataFrame]: list of datas with ``target`` column and the same scheme
    :param n_jobs: int: the number of cores that can be used in the calculation of stat values
    :return: pd.DataFrame: df with calculated ``p_value`` for each of the attributes
    """

    x = pd.DataFrame()
    for t in blocks:
        x = pd.concat([x, t.drop(['target'], axis=1)], axis=0)

    y = pd.Series()
    for t in blocks:
        y = pd.concat([y, t['target']], axis=0)

    relevance_table = calculate_relevance_table(X=x, y=y, n_jobs=n_jobs)
    return relevance_table
