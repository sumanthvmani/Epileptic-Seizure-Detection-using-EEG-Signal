
import numpy as np
from scipy.stats import kurtosis, skew

def extract_statistical_features(signals):
    """
    Extract statistical features from 1D signals.

    Args:
        signals: numpy array of shape (num_samples, signal_length)

    Returns:
        features: numpy array of shape (num_samples, 9)
                  [min, max, mean, median, std, var, kurtosis, skewness, correlation]
    """
    num_samples = signals.shape[0]
    features = []

    for i in range(num_samples):
        sig = signals[i]
        min_val = np.min(sig)
        max_val = np.max(sig)
        mean_val = np.mean(sig)
        median_val = np.median(sig)
        std_val = np.std(sig)
        var_val = np.var(sig)
        kurt_val = kurtosis(sig)
        skew_val = skew(sig)

        # Correlation coefficient: for single 1D signal, correlation with shifted version (lag=1)
        if len(sig) > 1:
            corr_val = np.corrcoef(sig[:-1], sig[1:])[0, 1]
        else:
            corr_val = 0  # default if signal has length 1

        features.append([min_val, max_val, mean_val, median_val, std_val,
                         var_val, kurt_val, skew_val, corr_val])

    return np.array(features)
