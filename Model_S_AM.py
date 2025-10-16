import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten
from Evaluation import evaluation

def Model_S_AM(train_data, train_target, test_data, test_target, m):
    out, model = SelfAttention_train(train_data, train_target, test_data)
    pred = np.asarray(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred

def SelfAttention_train(trainX, trainY, testX):
    # Reshape data to (samples, timesteps, features)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))

    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=trainX.shape[2])(input_layer, input_layer)
    attn_output = Dropout(0.1)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + input_layer)  # Residual connection

    # Feedforward
    ff = Dense(64, activation='relu')(attn_output)
    ff = Dense(trainY.shape[1])(ff)
    ff = Flatten()(ff)  # Flatten for output

    model = Model(inputs=input_layer, outputs=ff)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    testPredict = model.predict(testX)
    return testPredict, model
