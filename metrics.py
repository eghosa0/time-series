import numpy as np


def skewness(res):
        centered = res - np.mean(res)
        m2 = np.mean(centered**2)
        m3 = np.mean(centered**3)
        if m2 == 0:
            return 0.0
        return float(m3 / (m2**1.5))


def kurtosis(res):
    centered = res - np.mean(res)
    m2 = np.mean(centered**2)
    m4 = np.mean(centered**4)
    if m2 == 0:
        return 0.0
    return float(m4 / (m2**2))