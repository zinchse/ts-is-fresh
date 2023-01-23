import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_toy_ts() -> pd.DataFrame:
    data = [0, 5, 7, 6, 4, 3, 1, 2, 8, 11, 9, 10, 13, 12]
    time = np.arange(0, 14)
    toy_ts = pd.DataFrame()
    toy_ts['price_mean'] = data
    toy_ts['event_time'] = time
    toy_ts['id'] = time
    return toy_ts.set_index(['event_time'])


def plot_toy_ts(toy_ts: pd.DataFrame) -> None:
    plt.ylim((-1, 15))
    toy_ts.loc[1:12]['price_mean'].plot(
        figsize=(8, 5),
        title='TOY EXAMPLE',
        ylabel='PRICE',
        kind='line',
        color='orange',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        markersize=20,
    )
    plt.savefig('../docs/pictures/toy_ts.png', bbox_inches='tight')
    plt.show()


def plot_blocks(toy_ts: pd.DataFrame) -> None:
    plt.ylim((-1, 15))
    toy_ts.loc[1:6]['price_mean'].plot(
        figsize=(12, 6),
        title='BLOCK CROSS VALIDATION ($n\_blocks$=2)',
        ylabel='PRICE',
        kind='area',
        color='#cdfad4',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        label='first block'
    )

    toy_ts.loc[7:12]['price_mean'].plot(
        figsize=(8, 5),
        title='BLOCK CROSS VALIDATION ($n\_blocks$=2)',
        ylabel='PRICE',
        kind='area',
        color='#facde4',
        grid='true',
        style='-',
        xlabel='TIME',
        linewidth=4,
        label='second block'
    )

    toy_ts.loc[1:12]['price_mean'].plot(
        figsize=(8, 5),
        title='BLOCK CROSS VALIDATION ($n\_blocks$=2)',
        ylabel='PRICE',
        kind='line',
        color='orange',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        markersize=20,
        label='price'
    )

    toy_ts.loc[0:1]['price_mean'].plot(
        figsize=(8, 5),
        title='BLOCK CROSS VALIDATION ($n\_blocks$=2)',
        ylabel='PRICE',
        kind='line',
        color='gray',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        markersize=20,
        label='removed (Nan)'
    )

    toy_ts.loc[12:13]['price_mean'].plot(
        figsize=(8, 5),
        title='BLOCK CROSS VALIDATION ($n\_blocks$=2)',
        ylabel='PRICE',
        kind='line',
        color='gray',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        markersize=20,
        label='removed (Nan)'
    )

    x1 = [0, 1]
    y1 = [0, 5]
    plt.fill_between(x1, y1, alpha=.3, hatch='o', color="gray")

    x2 = [12, 13]
    y2 = [13, 12]
    plt.fill_between(x2, y2, alpha=.3, hatch='o', color="gray")

    plt.legend()
    plt.savefig('../docs/pictures/toy_ts_blocks.png',  bbox_inches='tight')
    plt.show()


def plot_windows(toy_ts: pd.DataFrame) -> None:
    plt.ylim((-1, 15))

    toy_ts.loc[1:6]['price_mean'].plot(
        figsize=(12, 6),
        title='BLOCK CROSS VALIDATION ($window\_size=3$, $n\_windows=3$)',
        ylabel='PRICE',
        kind='area',
        color='#cdfad4',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        label='first block')

    toy_ts.loc[1:6]['price_mean'].plot(
        figsize=(8, 5),
        title='BLOCK CROSS VALIDATION ($window\_size=3$, $n\_windows=3$)',
        ylabel='PRICE',
        kind='line',
        color='orange',
        grid='true',
        style='.-',
        xlabel='TIME',
        linewidth=4,
        markersize=20,
        label='price'
    )

    x1 = [4, 5, 6]
    y1 = [4, 3, 1]
    plt.fill_between(x1, y1, alpha=.3, hatch='o', color="#db5c5c")

    x2 = [3, 4, 5]
    y2 = [6, 4, 3]
    plt.fill_between(x2, y2, alpha=.3, hatch='o', color="#5c69db")

    x3 = [2, 3, 4]
    y3 = [7, 6, 4]
    plt.fill_between(x3, y3, alpha=.3, hatch='o', color="#d87aeb")

    plt.vlines(x=[4.01, 5.99],
               ymin=[0, 0],
               ymax=[3.9, .9],
               linestyles='dashed',
               label='the 1st window (for point 6)',
               color='red',
               linewidth=3)

    plt.vlines(x=[3.01, 4.99],
               ymin=[0, 0],
               ymax=[5.9, 2.9],
               linestyles='dashed',
               label='the 2nd window (for point 5)',
               color='blue',
               linewidth=3)

    plt.vlines(x=[2.01, 3.99],
               ymin=[0, 0],
               ymax=[6.9, 3.9],
               linestyles='dashed',
               label='the 3rd window (for point 4)',
               color='purple',
               linewidth=3)

    plt.legend()
    plt.savefig('../docs/pictures/toy_ts_windows.png', bbox_inches='tight')
    plt.show()
