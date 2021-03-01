from tensorflow.keras import layers
from tensorflow import math, exp

SIZE = 3
PATH_TO_CSV = 'Sudoku/assets/data.csv'


class SoftmaxMap(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = exp(x - math.reduce_max(x, axis=self.axis, keepdims=True))
        s = math.reduce_sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape


class UnsolvableError(ValueError):
    pass


class FillTerminalGrid(ValueError):
    pass
