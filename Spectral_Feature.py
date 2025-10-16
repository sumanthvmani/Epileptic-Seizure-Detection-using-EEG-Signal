import numpy as np


def extract_spectral_features(signals, fs=1.0):
    """
    Extract spectral features from 1D signals.

    Args:
        signals: numpy array of shape (num_samples, signal_length)
        fs: sampling frequency (default=1.0)

    Returns:
        features: numpy array of shape (num_samples, 5)
                  [spectral centroid, bandwidth, roll-off 85%, dominant freq, mean power]
    """
    num_samples = signals.shape[0]
    features = []

    for i in range(num_samples):
        sig = signals[i]

        # Compute FFT
        freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
        fft_vals = np.abs(np.fft.rfft(sig))

        # Power spectrum
        psd = fft_vals ** 2
        psd_sum = np.sum(psd) + 1e-12  # avoid divide by zero

        # Spectral centroid
        centroid = np.sum(freqs * psd) / psd_sum

        # Spectral bandwidth (variance around centroid)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum)

        # Spectral roll-off (85% of cumulative energy)
        cumulative_energy = np.cumsum(psd)
        rolloff_idx = np.where(cumulative_energy >= 0.85 * cumulative_energy[-1])[0][0]
        rolloff = freqs[rolloff_idx]

        # Dominant frequency
        dom_freq = freqs[np.argmax(psd)]

        # Mean power
        mean_power = np.mean(psd)

        features.append([centroid, bandwidth, rolloff, dom_freq, mean_power])

    return np.array(features)
