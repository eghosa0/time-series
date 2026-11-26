import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from metrics import kurtosis, long_run_variance, newey_west_lags, skewness


ADF_COEF = {
    "c": (-2.86154, -2.8903, -4.234, -40.04),
    "ct": (-3.41049, -4.3904, -9.036, -45.374),
}

KPSS_CRIT = {
    "c": 0.463,
    "ct": 0.146,
}


def adf_critical_value(regression, nobs):
    b = ADF_COEF[regression]
    return b[0] + b[1] / nobs + b[2] / nobs**2 + b[3] / nobs**3


class ADF_test:
    def __init__(self, series, maxlag=None, regression="ct"):
        self.series = np.asarray(series, dtype=float)
        self.maxlag = maxlag
        self.regression = regression

    def _fit_adf(self, y, dy, lag, start):
        T = len(dy)
        n_obs = T - start
        Y = dy[start:]
        y_lag1 = y[start:-1]
        cols = [np.ones(n_obs)]
        if self.regression == "ct":
            cols.append(np.arange(start + 1, T + 1, dtype=float))
        cols.append(y_lag1)
        for i in range(1, lag + 1):
            cols.append(dy[start - i:T - i])
        X = np.column_stack(cols)
        j = len(cols) - lag - 1

        beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        res = Y - X @ beta
        k = X.shape[1]
        s2 = float((res.T @ res) / (n_obs - k))
        cov_beta = s2 * np.linalg.pinv(X.T @ X)
        t_stat = float(beta[j] / np.sqrt(cov_beta[j, j]))
        aic = np.log((res.T @ res) / n_obs) + 2.0 * k / n_obs
        return t_stat, float(aic), n_obs

    def test(self):
        y = self.series
        dy = np.diff(y)
        n = len(y)

        if self.maxlag is None:
            max_p = int(np.floor(12.0 * (n / 100.0) ** 0.25))
            aics = [self._fit_adf(y, dy, lag, max_p)[1] for lag in range(max_p + 1)]
            self.lag = int(np.argmin(aics))
        else:
            self.lag = self.maxlag

        stat, _, nobs = self._fit_adf(y, dy, self.lag, self.lag)
        self.nobs = nobs
        self.crit = adf_critical_value(self.regression, nobs)
        crit = self.crit
        stationary = bool(stat < crit)

        r = self.regression
        if stationary:
            verdict = f"H0 rifiutata (ADF={stat:.4f} < {crit:.3f}, lag={self.lag}): nessuna radice unitaria ('{r}')."
        else:
            verdict = f"H0 non rifiutata (ADF={stat:.4f} >= {crit:.3f}, lag={self.lag}): radice unitaria ('{r}')."

        return stat, stationary, verdict


class KPSS_test:
    def __init__(self, series, nlags=None, regression="ct"):
        self.series = np.asarray(series, dtype=float)
        self.nlags = nlags
        self.regression = regression

    def test(self):
        y = self.series
        T = len(y)
        if self.regression == "ct":
            X = np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float)])
        else:
            X = np.ones((T, 1))

        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        res = y - X @ beta
        S = np.cumsum(res)

        self.lags = newey_west_lags(T) if self.nlags is None else self.nlags
        s2 = long_run_variance(res, self.lags)

        stat = float(np.sum(S**2) / (T**2 * s2))

        self.crit = KPSS_CRIT[self.regression]
        crit = self.crit
        stationary = bool(stat < crit)

        r = self.regression
        if stationary:
            verdict = f"H0 non rifiutata (KPSS={stat:.4f} < {crit}): serie stazionaria ('{r}')."
        else:
            verdict = f"H0 rifiutata (KPSS={stat:.4f} >= {crit}): serie non stazionaria ('{r}')."

        return stat, stationary, verdict


class Trend_test:
    def __init__(self, series, kind="drift", nlags=None):
        self.series = np.asarray(series, dtype=float)
        self.kind = kind
        self.nlags = nlags

    def test(self):
        x = self.series
        T = len(x)
        L = newey_west_lags(T) if self.nlags is None else self.nlags

        if self.kind == "drift":
            est = float(np.mean(x))
            se = float(np.sqrt(long_run_variance(x, L) / T))
        else:
            X = np.column_stack([np.ones(T), np.arange(1, T + 1, dtype=float)])
            XtX_inv = np.linalg.inv(X.T @ X)
            b = XtX_inv @ X.T @ x
            u = x - X @ b
            g = X * u[:, None]
            S = g.T @ g
            for j in range(1, L + 1):
                w = 1.0 - j / (L + 1.0)
                G = g[j:].T @ g[:-j]
                S += w * (G + G.T)
            V = XtX_inv @ S @ XtX_inv
            est = float(b[1])
            se = float(np.sqrt(V[1, 1]))

        t_stat = est / se
        p_value = float(2 * stats.norm.sf(abs(t_stat)))
        significant = p_value < 0.05
        what = "drift" if self.kind == "drift" else "trend"
        verdict = (f"{what} = {est:.5f} (SE HAC {se:.5f}), t = {t_stat:.3f}, p = {p_value:.3f}: "
                   + (f"{what} significativo." if significant else f"{what} non significativo."))
        return t_stat, p_value, significant, verdict


class ACF_test:
    def __init__(self, series, nlags=20, save_path="acf_plot.png", title="Autocorrelation Function (ACF)"):
        self.series = np.asarray(series, dtype=float)
        self.nlags = min(nlags, len(self.series) - 1)
        self.save_path = save_path
        self.title = title

    def _compute_acf(self):
        y = self.series - np.mean(self.series)
        denom = float(y.T @ y)
        values = np.ones(self.nlags + 1)
        for k in range(1, self.nlags + 1):
            values[k] = float((y[k:].T @ y[:-k]) / denom)
        return values

    def bartlett_bounds(self, values):
        T = len(self.series)
        bounds = np.zeros(len(values))
        for k in range(1, len(values)):
            bounds[k] = 1.96 * np.sqrt((1.0 + 2.0 * np.sum(values[1:k] ** 2)) / T)
        return bounds

    def _optimal_q(self, values):
        bounds = self.bartlett_bounds(values)
        for k in range(1, len(values)):
            if abs(values[k]) <= bounds[k]:
                return k - 1
        return len(values) - 1

    def plot(self, values, save_path=None):
        path = save_path or self.save_path
        bounds = self.bartlett_bounds(values)
        lags = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stem(lags[1:], values[1:], basefmt=" ")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.plot(lags[1:], bounds[1:], color="red", linestyle="--", linewidth=1.0, label="Bande di Bartlett (95%)")
        ax.plot(lags[1:], -bounds[1:], color="red", linestyle="--", linewidth=1.0)
        ax.fill_between(lags[1:], -bounds[1:], bounds[1:], color="red", alpha=0.1)
        ax.set_title(self.title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def test(self, save_path=None):
        values = self._compute_acf()
        self.values = values
        q_opt = self._optimal_q(values)
        saved_file = self.plot(values, save_path=save_path)
        return q_opt, saved_file


class PACF_test:
    def __init__(self, series, nlags=20, save_path="pacf_plot.png", title="Partial Autocorrelation Function (PACF)"):
        self.series = np.asarray(series, dtype=float).ravel()
        self.nlags = min(nlags, len(self.series) // 2 - 1)
        self.save_path = save_path
        self.title = title

    def _compute_pacf(self):
        y = self.series
        n = len(y)
        values = np.ones(self.nlags + 1)

        for k in range(1, self.nlags + 1):
            Y = y[k:]
            cols = [np.ones(len(Y))]
            for j in range(1, k + 1):
                cols.append(y[k - j:n - j])
            X = np.column_stack(cols)

            beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
            values[k] = float(beta[-1])

        return values

    def _optimal_p(self, values):
        bound = 1.96 / np.sqrt(len(self.series))
        for k in range(1, len(values)):
            if abs(values[k]) <= bound:
                return k - 1
        return len(values) - 1

    def plot(self, values, save_path=None):
        path = save_path or self.save_path
        bound = 1.96 / np.sqrt(len(self.series))
        lags = np.arange(len(values))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stem(lags[1:], values[1:], basefmt=" ")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(bound, color="red", linestyle="--", linewidth=1.0, label="±1.96/√T")
        ax.axhline(-bound, color="red", linestyle="--", linewidth=1.0)
        ax.fill_between(lags[1:], -bound, bound, color="red", alpha=0.1)
        ax.set_title(self.title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("PACF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def test(self, save_path=None):
        values = self._compute_pacf()
        self.values = values
        p_opt = self._optimal_p(values)
        saved_file = self.plot(values, save_path=save_path)
        return p_opt, saved_file


class T_test:
    def __init__(self, model):
        self.model = model

    def test(self):
        st = self.model.get_stats()
        beta = self.model.beta
        se = st["std_errors"]
        df = len(self.model.Y) - len(beta)
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), df)
        return t, p, p < 0.05


class LjungBox_test:
    def __init__(self, residuals, lags=10, dof=None):
        self.residuals = np.asarray(residuals, dtype=float)
        self.lags = lags
        self.dof = lags if dof is None else dof

    def test(self):
        T = len(self.residuals)
        c = self.residuals - np.mean(self.residuals)
        denom = np.sum(c**2)
        rhos = np.array([np.sum(c[k:] * c[:-k]) / denom for k in range(1, self.lags + 1)])
        ks = np.arange(1, self.lags + 1, dtype=float)
        q = float(T * (T + 2.0) * np.sum((rhos**2) / (T - ks)))
        self.crit = float(stats.chi2.ppf(0.95, self.dof))
        p_value = float(stats.chi2.sf(q, self.dof))
        ok = q <= self.crit
        return q, p_value, ok


class JarqueBera_test:
    def __init__(self, residuals):
        self.residuals = np.asarray(residuals, dtype=float)

    def test(self):
        T = len(self.residuals)
        s = skewness(self.residuals)
        k = kurtosis(self.residuals)
        jb = (T / 6.0) * (s**2 + ((k - 3.0) ** 2) / 4.0)
        crit = float(stats.chi2.ppf(0.95, 2))
        p_value = float(stats.chi2.sf(jb, 2))
        ok = jb <= crit
        verdict = f"JB={jb:.4f} (crit={crit:.4f}, p={p_value:.4f}, S={s:.3f}, K={k:.3f}) -> normali={ok}"
        return jb, p_value, ok, verdict


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


class Coefficient_test:
    def __init__(self, estimate, se, value):
        self.estimate, self.se, self.value = estimate, se, value

    def test(self):
        z = (self.estimate - self.value) / self.se
        p = float(2 * stats.norm.sf(abs(z)))
        return float(z), p, p >= 0.05


class Stability_test:
    def __init__(self, build_model, series, split):
        self.build_model = build_model
        self.series = np.asarray(series, dtype=float)
        self.split = split

    def test(self):
        m1 = self.build_model()
        m1.train(self.series[: self.split])
        m2 = self.build_model()
        m2.train(self.series[self.split:])
        se1 = m1.get_stats()["std_errors"]
        se2 = m2.get_stats()["std_errors"]
        z = (m1.beta - m2.beta) / np.sqrt(se1**2 + se2**2)
        p = 2 * stats.norm.sf(np.abs(z))
        return m1.beta, se1, m2.beta, se2, z, p, bool(np.all(p >= 0.05))


class DieboldMariano_test:
    def __init__(self, e1, e2, h=1):
        self.e1 = np.asarray(e1, dtype=float)
        self.e2 = np.asarray(e2, dtype=float)
        self.h = h

    def test(self):
        d = self.e1**2 - self.e2**2
        n = len(d)
        lrv = long_run_variance(d, self.h - 1)
        dm = float(np.mean(d) / np.sqrt(lrv / n))
        p_two = float(2 * stats.norm.sf(abs(dm)))
        p_one = float(stats.norm.cdf(dm))
        verdict = (f"DM = {dm:.3f}, p bilaterale = {p_two:.3f}, p unilaterale (1 migliore di 2) = {p_one:.3f}")
        return dm, p_two, p_one, verdict