from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
    Concatenate
from utils import SoftmaxMap
from .data_transform import read_transform


def train_model():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_transform()
    model = build_model()
    # --- compile and fit the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train, Y_train, epochs=3, batch_size=128,
              validation_data=(X_val, Y_val))

    model.save("policy_network")


def build_model():
    # Model definition
    input = Input(shape=(9, 9, 9))

    x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                activation='tanh')(input)
    x2 = Conv2D(32, kernel_size=(1, 9), strides=(1, 1), padding='same',
                activation='tanh')(input)
    x3 = Conv2D(32, kernel_size=(9, 1), strides=(1, 1), padding='same',
                activation='tanh')(input)
    x4 = Conv2D(32, kernel_size=(9, 9), strides=(1, 1), padding='same',
                activation='tanh')(input)
    x = Concatenate()([x1, x2, x3, x4])
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(9, 9), strides=(1, 1), padding='same',
               activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same',
               activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)

    x = Concatenate()([x, input])
    x = Conv2D(9, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    outputs_play = SoftmaxMap()(x)

    # Model instantiation
    model = Model(input, outputs_play)
    print(model.summary())

    return model
