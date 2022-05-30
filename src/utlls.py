import pandas as pd


def prepare_data(path):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(path, sep=',')

    y = df.iloc[:, -1]
    x = df.iloc[:, :-1]

    return x, y
