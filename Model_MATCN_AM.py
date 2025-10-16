import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Add, Activation, Flatten, Concatenate, Multiply
from keras.optimizers import Adam

from Evaluation import evaluation


# Residual TCN block
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.0):
    prev_x = x
    conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(conv)
    conv = Dropout(dropout_rate)(conv)

    # Residual connection
    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters, 1, padding='same')(prev_x)
    out = Add()([prev_x, conv])
    out = Activation('relu')(out)
    return out

# Simple attention mechanism
def attention_block(x):
    attn_weights = Dense(x.shape[-1], activation='softmax')(x)
    out = Multiply()([x, attn_weights])
    return out

def Model_MATCN_AM(F1, F2, F3, F4, target, m, sol=None):
    # Default sol values if not provided
    if sol is None:
        sol = [64, 0.001, 100]  # [hidden_neurons, learning_rate, steps_per_epoch]

    hidden_neurons = int(sol[0])      # number of neurons in dense layers
    learning_rate = float(sol[1])     # learning rate for optimizer
    steps_per_epoch = int(sol[2])     # training epochs

    # Inputs
    inputs_F1 = Input(shape=(1, F1.shape[1]))
    inputs_F2 = Input(shape=(1, F2.shape[1]))
    inputs_F3 = Input(shape=(1, F3.shape[1]))
    inputs_F4 = Input(shape=(1, F4.shape[1]))

    # Multiscale convolutions
    conv1 = Conv1D(32, kernel_size=1, activation='relu')(inputs_F1)
    conv2 = Conv1D(32, kernel_size=2, activation='relu')(inputs_F2)
    conv3 = Conv1D(32, kernel_size=3, activation='relu')(inputs_F3)
    conv4 = Conv1D(32, kernel_size=4, activation='relu')(inputs_F4)

    # Combine scales
    merged = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])

    # TCN blocks
    tcn = residual_block(merged, filters=64, kernel_size=2, dilation_rate=1)
    tcn = residual_block(tcn, filters=64, kernel_size=2, dilation_rate=2)
    tcn = residual_block(tcn, filters=64, kernel_size=2, dilation_rate=4)

    # Attention
    attn = attention_block(tcn)
    attn = Flatten()(attn)

    # Output with variable hidden layer
    hidden = Dense(hidden_neurons, activation='relu')(attn)
    output = Dense(target.shape[1], activation='sigmoid')(hidden)

    # Define model
    model = Model(inputs=[inputs_F1, inputs_F2, inputs_F3, inputs_F4], outputs=output)

    # Compile with adaptive learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Reshape data for input
    F1 = np.reshape(F1, (F1.shape[0], 1, F1.shape[1]))
    F2 = np.reshape(F2, (F2.shape[0], 1, F2.shape[1]))
    F3 = np.reshape(F3, (F3.shape[0], 1, F3.shape[1]))
    F4 = np.reshape(F4, (F4.shape[0], 1, F4.shape[1]))

    # Train model
    model.fit([F1, F2, F3, F4], target, epochs=steps_per_epoch, batch_size=1, verbose=2)
    testPredict = model.predict([F1, F2, F3, F4])

    pred = np.asarray(testPredict)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, target)
    return Eval
