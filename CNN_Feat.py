from keras import Sequential, Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


def CNN_Feat(signal_data, IMG_SIZE=28):
    """
    Extract deep features from a CNN for given 1D signal data.

    Args:
        signal_data: numpy array of shape (num_samples, signal_length)
        IMG_SIZE: size to reshape signals into 2D (IMG_SIZE x IMG_SIZE)

    Returns:
        features: numpy array of deep features from CNN
    """
    # Reshape signal into 2D image-like format
    num_samples = signal_data.shape[0]
    X = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1))
    for i in range(num_samples):
        temp = np.resize(signal_data[i], (IMG_SIZE * IMG_SIZE, 1))
        X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    X = X.astype('float32') / 255  # normalize

    # Define CNN
    Hidden = 128  # you can change
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(Hidden, activation='relu'),  # features extracted from this layer
        Dropout(0.5),
        Dense(10, activation='softmax')  # dummy output, not used
    ])

    # Create a feature extraction model
    feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)  # output from Dense(Hidden)

    # Extract features
    features = feature_model.predict(X, batch_size=64)

    return features
