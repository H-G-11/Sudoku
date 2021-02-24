import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..Utils.utils import PATH_TO_CSV


def custom_encoder(array_of_grids):
    """ transform numpy array of sudoku grids in one hot
    array of dimension (len(arr), 9, 9, 9), with one channel
    for each value from 1 to 9. """

    if len(array_of_grids.shape) == 2:
        array_of_grids = np.array([array_of_grids])
    shape_encoded = (*array_of_grids.shape, 9)
    encoded = np.zeros(shape_encoded)
    for i, grid in enumerate(array_of_grids):
        for value in range(9):
            encoded[i][value] = (grid == value + 1) * 1
    return encoded


def read_transform(NROWS=100, encode=False):
    """ If encode is true, also splits in train test and eval. """

    data = pd.read_csv(PATH_TO_CSV, usecols=["puzzle", "solution"],
                       nrows=NROWS)

    _data_X = data["puzzle"].apply(lambda x: [int(i) if i != '.'
                                              else 0 for i in x])
    _data_Y = data["solution"].apply(lambda x: [int(i) for i in x])

    data_X = np.stack(_data_X.to_numpy()).reshape((len(data), 9, 9))
    data_Y = np.stack(_data_Y.to_numpy()).reshape((len(data), 9, 9))

    if not encode:
        return data_X, data_Y

    data_X_encoded = custom_encoder(data_X)
    data_Y_encoded = custom_encoder(data_Y)

    _X_train, X_test, _Y_train, Y_test = train_test_split(
        data_X_encoded, data_Y_encoded, test_size=0.1, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(
        _X_train, _Y_train, test_size=0.1, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
