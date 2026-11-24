import matplotlib.pyplot as plt
import numpy as np
from models import *


class DW_test:
    def __init__(self, model, d_L=1.65, d_U=1.69):
        self.model = model
        self.d_L = d_L
        self.d_U = d_U

    def test(self):
        res = self.model.get_stats()["residuals"]
        num = np.sum((res[1:] - res[: len(res) - 1]) ** 2)
        den = res.T @ res
        dw = float(num / den)

        if self.d_U < dw < (4 - self.d_U):
            accepted = True
            verdict = "H0 accettata (no autocorrelazione del 1° ordine): modello accettato."
        elif dw < self.d_L or dw > (4 - self.d_L):
            accepted = False
            verdict = "H0 rifiutata (autocorrelazione del 1° ordine): modello non accettato."
        else:
            accepted = False
            verdict = "Zona di indecisione DW: modello non confermato."

        return dw, accepted, verdict


class ADF_test:  # 'ct'
    def __init__(self, series, maxlag=None):
        self.series = np.asarray(series, dtype=float)
        self.maxlag = maxlag

    def _fit_adf(self, y, dy, lag):
        T = len(dy)
        n_obs = T - lag
        Y = dy[lag:]
        y_lag1 = y[lag:-1]
        trend = np.arange(lag + 1, T + 1, dtype=float)
        cols = [np.ones(n_obs), trend, y_lag1]
        for i in range(1, lag + 1):
            cols.append(dy[lag - i : T - i])
        X = np.column_stack(cols)

        beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        res = Y - X @ beta
        k = X.shape[1]
        s2 = float((res.T @ res) / (n_obs - k))
        cov_beta = s2 * np.linalg.pinv(X.T @ X)
        t_stat = float(beta[2] / np.sqrt(cov_beta[2, 2]))
        aic = np.log((res.T @ res) / n_obs) + 2.0 * k / n_obs
        return t_stat, float(aic)

    def test(self):
        y = self.series
        dy = np.diff(y)
        n = len(y)

        if self.maxlag is None:
            max_p = int(np.floor(12.0 * (n / 100.0) ** 0.25))
            best_aic = np.inf
            best_stat = None
            for lag in range(max_p + 1):
                t_stat, aic = self._fit_adf(y, dy, lag)
                if aic < best_aic:
                    best_aic = aic
                    best_stat = t_stat
            stat = best_stat
        else:
            stat, _ = self._fit_adf(y, dy, self.maxlag)

        # Valori critici asintotici ADF con costante e trend ('ct')
        crit = -3.41
        stationary = bool(stat < crit)

        if stationary:
            verdict = f"H0 rifiutata (ADF={stat:.4f} < {crit}): serie stazionaria intorno a trend ('ct')."
        else:
            verdict = f"H0 non rifiutata (ADF={stat:.4f} >= {crit}): possibile radice unitaria ('ct')."

        return stat, stationary, verdict


class KPSS_test:  # 'ct'
    def __init__(self, series, nlags=None):
        self.series = np.asarray(series, dtype=float)
        self.nlags = nlags

    def test(self):
        y = self.series
        T = len(y)
        trend = np.arange(1, T + 1, dtype=float)
        X = np.column_stack([np.ones(T), trend])

        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        res = y - X @ beta
        S = np.cumsum(res)

        lags = int(np.floor(4.0 * (T / 100.0) ** 0.25)) if self.nlags is None else self.nlags
        gamma_0 = float((res.T @ res) / T)
        s2 = gamma_0
        for l in range(1, lags + 1):
            w = 1.0 - l / (lags + 1.0)
            gamma_l = float((res[l:].T @ res[:-l]) / T)
            s2 += 2.0 * w * gamma_l

        stat = float(np.sum(S ** 2) / (T ** 2 * s2))

        # Valori critici asintotici KPSS con costante e trend ('ct')
        crit = 0.146
        stationary = bool(stat < crit)

        if stationary:
            verdict = f"H0 non rifiutata (KPSS={stat:.4f} < {crit}): serie trend-stazionaria ('ct')."
        else:
            verdict = f"H0 rifiutata (KPSS={stat:.4f} >= {crit}): serie non stazionaria ('ct')."

        return stat, stationary, verdict


class ACF_test:
    def __init__(self, series, nlags=20, save_path="acf_plot.png"):
        self.series = np.asarray(series, dtype=float)
        self.nlags = min(nlags, len(self.series) - 1)
        self.save_path = save_path

    def _compute_acf(self):
        y = self.series - np.mean(self.series)
        denom = float(y.T @ y)
        values = np.ones(self.nlags + 1)
        for k in range(1, self.nlags + 1):
            values[k] = float((y[k:].T @ y[:-k]) / denom)
        return values

    def _optimal_q(self, values):
        bound = 1.96 / np.sqrt(len(self.series))
        significant_lags = [
            lag for lag in range(1, len(values)) if abs(values[lag]) > bound
        ]
        return int(significant_lags[-1]) if significant_lags else 0

    def plot(self, values, save_path=None):
        path = save_path or self.save_path
        bound = 1.96 / np.sqrt(len(self.series))
        lags = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stem(lags, values, basefmt=" ")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(bound, color="red", linestyle="--", linewidth=1.0, label="±1.96/√T")
        ax.axhline(-bound, color="red", linestyle="--", linewidth=1.0)
        ax.fill_between(lags, -bound, bound, color="red", alpha=0.1)
        ax.set_title("Autocorrelation Function (ACF)")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def test(self, save_path=None):
        values = self._compute_acf()
        q_opt = self._optimal_q(values)
        saved_file = self.plot(values, save_path=save_path)
        return q_opt, saved_file


class PACF_test:
    def __init__(self, series, nlags=20, save_path="pacf_plot.png"):
        self.series = np.asarray(series, dtype=float).ravel()
        self.nlags = min(nlags, len(self.series) // 2 - 1)
        self.save_path = save_path

    def _compute_pacf(self):
        y = self.series
        n = len(y)
        values = np.ones(self.nlags + 1)

        for k in range(1, self.nlags + 1):
            Y = y[k:]
            cols = [np.ones(len(Y))]
            for j in range(1, k + 1):
                cols.append(y[k - j : n - j])
            X = np.column_stack(cols)

            beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
            values[k] = float(beta[-1])

        return values

    def _optimal_p(self, values):
        bound = 1.96 / np.sqrt(len(self.series))
        significant_lags = [
            lag for lag in range(1, len(values)) if abs(values[lag]) > bound
        ]
        return int(significant_lags[-1]) if significant_lags else 0

    def plot(self, values, save_path=None):
        path = save_path or self.save_path
        bound = 1.96 / np.sqrt(len(self.series))
        lags = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stem(lags, values, basefmt=" ")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(bound, color="red", linestyle="--", linewidth=1.0, label="±1.96/√T")
        ax.axhline(-bound, color="red", linestyle="--", linewidth=1.0)
        ax.fill_between(lags, -bound, bound, color="red", alpha=0.1)
        ax.set_title("Partial Autocorrelation Function (PACF)")
        ax.set_xlabel("Lag")
        ax.set_ylabel("PACF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def test(self, save_path=None):
        values = self._compute_pacf()
        p_opt = self._optimal_p(values)
        saved_file = self.plot(values, save_path=save_path)
        return p_opt, saved_file