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


def autocovariance(x, lag):
    """gamma_j = (1/T) * sum_{t=j+1}^{T} x_t x_{t-j}, con x gia' centrata."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if lag == 0:
        return float(x @ x / T)
    return float(x[lag:] @ x[:-lag] / T)


def long_run_variance(x, lags):
    """Varianza di lungo periodo (Newey-West, pesi di Bartlett):
    sigma2_LR = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j
    (stessa formula usata nel KPSS)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    s2 = autocovariance(x, 0)
    for j in range(1, lags + 1):
        s2 += 2.0 * (1.0 - j / (lags + 1.0)) * autocovariance(x, j)
    return float(s2)


def newey_west_lags(T):
    """Regola usata nel progetto: L = floor(4 * (T/100)^(1/4))."""
    return int(np.floor(4.0 * (T / 100.0) ** 0.25))


# ------------------------------------------------------------------
# Criteri informativi (stessa forma "per osservazione" del progetto CAPM)
# k = numero di parametri stimati della media
# ------------------------------------------------------------------
def aic(res, k):
    res = np.asarray(res, dtype=float)
    n = len(res)
    return float(np.log(res @ res / n) + 2.0 * k / n)


def bic(res, k):
    res = np.asarray(res, dtype=float)
    n = len(res)
    return float(np.log(res @ res / n) + k * np.log(n) / n)


# ------------------------------------------------------------------
# Metriche di previsione fuori campione
# ------------------------------------------------------------------
def rmse(errors):
    e = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def mae(errors):
    e = np.asarray(errors, dtype=float)
    return float(np.mean(np.abs(e)))


def coverage(actual, lower, upper):
    """Quota di osservazioni che cadono dentro l'intervallo di previsione."""
    a = np.asarray(actual, dtype=float)
    return float(np.mean((a >= np.asarray(lower)) & (a <= np.asarray(upper))))
