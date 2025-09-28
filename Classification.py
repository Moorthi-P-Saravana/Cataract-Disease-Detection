from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Flatten, Dense, LSTM, GRU, \
    concatenate, Bidirectional, MaxPooling2D, AveragePooling2D, MaxPool2D, Concatenate, Multiply
import numpy as np


def multi_confu_matrix(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')
    r2 = r2_score(y_test, y_predict)
    mcc = float(matthews_corrcoef(y_test, y_predict))
    kappa = float(cohen_kappa_score(y_test, y_predict))
    h_loss = hamming_loss(y_test, y_predict)
    jaccard = float(jaccard_score(y_test, y_predict, average='weighted'))
    return [accuracy, precision, recall, f1, r2, mcc, kappa, h_loss, jaccard]


def residual_block(x, filters):
    """
    Residual Block with Separable Convolutions and Channel Attention and depth wise separable
    """
    res = x
    x = SeparableConv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    # Channel Attention
    ca = GlobalAveragePooling2D()(x)
    ca = Dense(filters // 2, activation="relu")(ca)
    ca = Dense(filters, activation="sigmoid")(ca)
    x = x * keras.ops.expand_dims(keras.ops.expand_dims(ca, 1), 1)

    x = Add()([x, res])  # Residual Connection
    return x


def build_ocarnet(input_shape, num_classes):
    """
    Constructs the Optimized Channel Attention Residual Network (O-CARNet).
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="O-CARNet")
    return model


def proposed(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):

    input_shape = X_train[1].shape
    num_classes = len(set(y_train))

    # Build & Compile Model
    model = build_ocarnet(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    pred = np.argmax(model.predict(X_test), axis=1)

    met = multi_confu_matrix(y_test, pred)

    return pred, met, history


def cnn(x_train, y_train, x_test, y_test):

    input_shape = x_train[1].shape
    num_classes = len(set(y_train))

    inputs = Input(shape=input_shape)

    cnn = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.3)(cnn)

    flat = Flatten()(cnn)

    out = Dense(num_classes, activation='softmax')(flat)

    # Build Model
    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    predicted = np.argmax(model.predict(x_test), axis=1)

    met = multi_confu_matrix(y_test, predicted)

    return predicted, met


def alexnet(x_train, y_train, x_test, y_test):
    input_shape = x_train[1].shape
    num_classes = len(set(y_train))

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    Alexnet_model = Model(inputs=inputs, outputs=output)
    Alexnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    Alexnet_model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)

    pred = np.argmax(Alexnet_model.predict(x_test), axis=1)
    met = multi_confu_matrix(y_test, pred)

    return pred, met


def Residual_block(x, filters):
    res = x
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])  # Residual connection
    x = ReLU()(x)
    return x


def Resnet(x_train, y_train, x_test, y_test):
    input_shape = x_train[1].shape
    num_classes = len(set(y_train))

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)

    x = Residual_block(x, 32)
    x = Residual_block(x, 32)
    x = Residual_block(x, 32)

    x = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

    pred = np.argmax(model.predict(x_test), axis=1)
    met = multi_confu_matrix(y_test, pred)

    return pred, met


def inception_module(x, filters):
    f1, f3_r, f3, f5_r, f5, fpool = filters

    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    conv3 = Conv2D(f3_r, (1, 1), padding='same', activation='relu')(x)
    conv3 = Conv2D(f3, (1, 1), padding='same', activation='relu')(conv3)

    conv5 = Conv2D(f5_r, (1, 1), padding='same', activation='relu')(x)
    conv5 = Conv2D(f5, (1, 1), padding='same', activation='relu')(conv5)

    pool = MaxPooling2D((1, 1), strides=(1, 1), padding='same')(x)
    pool = Conv2D(fpool, (1, 1), padding='same', activation='relu')(pool)

    output = Concatenate(axis=-1)([conv1, conv3, conv5, pool])
    return output


def inception_v3(x_train, y_train, x_test, y_test):

    num_classes = len(set(y_train))

    input_shape = x_train[1].shape

    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(inputs)
    x = Conv2D(32, kernel_size=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(x)

    # Inception modules
    x = inception_module(x, filters=[64, 128, 32, 32, 96, 64])
    x = inception_module(x, filters=[128, 192, 96, 64, 32, 56])
    x = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(x)

    # Global average pooling
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(10, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

    pred = np.argmax(model.predict(x_test), axis=1)

    return pred, multi_confu_matrix(y_test, pred)

def san_cnn(x_train, y_train, x_test, y_test):


    input_shape = x_train[1].shape
    num_classes = len(set(y_train))

    inputs = Input(shape=input_shape)
    x = Conv2D(16, (1, 1), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((1, 1))(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 1))(x)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    attention = Conv2D(32, (1, 1), activation='relu')(x)
    x = Multiply()([x, attention])
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, x, name="SAN_CNN_Feature_Extractor")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(x_test, y_test))

    pred = np.argmax(model.predict(x_test), axis=1)
    met = multi_confu_matrix(y_test, pred)

    return pred, met




import tensorflow as tf
from tensorflow.keras import layers, models

# Swish activation
def swish(x):
    return x * tf.nn.sigmoid(x)

# MBConv Block
def mb_conv_block(inputs, in_channels, out_channels, expansion_factor, stride, se_ratio):
    x = inputs
    hidden_dim = in_channels * expansion_factor

    # Expansion phase
    if expansion_factor != 1:
        x = layers.Conv2D(hidden_dim, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(swish)(x)

    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    # Squeeze and Excitation
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, hidden_dim))(se)
    se = layers.Conv2D(int(in_channels * se_ratio), 1, activation='swish', padding='same')(se)
    se = layers.Conv2D(hidden_dim, 1, activation='sigmoid', padding='same')(se)
    x = layers.Multiply()([x, se])

    # Projection
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    if stride == 1 and in_channels == out_channels:
        x = layers.Add()([inputs, x])

    return x

# EfficientNet-B0 blocks config
def efficientnet_b0(x_train, y_train, x_test, y_test):
    input_shape = x_train[1].shape
    num_classes = max(y_train) + 1

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    # [in, out, expand, stride, repeats]
    blocks = [
        [32, 16, 1, 1, 1],
        [16, 24, 6, 2, 2],
        [24, 40, 6, 2, 2],
        [40, 80, 6, 2, 3],
        [80, 112, 6, 1, 3],
        [112, 192, 6, 2, 4],
        [192, 320, 6, 1, 1],
    ]

    for in_channels, out_channels, exp, stride, repeats in blocks:
        for i in range(repeats):
            s = stride if i == 0 else 1
            x = mb_conv_block(x, in_channels, out_channels, exp, s, se_ratio=0.25)
            in_channels = out_channels

    x = layers.Conv2D(1280, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(swish)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(x_test, y_test))

    pred = np.argmax(model.predict(x_test), axis=1)
    met = multi_confu_matrix(y_test, pred)

    return pred, met

