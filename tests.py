import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from metrics import kurtosis, long_run_variance, newey_west_lags, skewness


# ======================================================================
# Valori critici
# ======================================================================
# ADF: superfici di risposta di MacKinnon (2010), c(T) = b0 + b1/T + b2/T^2 + b3/T^3
ADF_COEF = {
    "c": {
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.05: (-2.86154, -2.8903, -4.234, -40.04),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {
        0.01: (-3.95877, -9.0531, -28.428, -134.155),
        0.05: (-3.41049, -4.3904, -9.036, -45.374),
        0.10: (-3.12705, -2.5856, -3.925, -22.38),
    },
}

# KPSS: valori asintotici (Kwiatkowski et al., 1992)
KPSS_CRIT = {
    "c": {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739},
    "ct": {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216},
}


def adf_critical_value(regression, alpha, nobs):
    b = ADF_COEF[regression][alpha]
    return b[0] + b[1] / nobs + b[2] / nobs**2 + b[3] / nobs**3


# ======================================================================
# Stazionarieta'
# ======================================================================
class ADF_test:
    """Dickey-Fuller aumentato.
    Regressione:  dy_t = alpha (+ beta t) + delta y_{t-1} + sum_j gamma_j dy_{t-j} + e_t
    H0: delta = 0 (radice unitaria)   H1: delta < 0 (stazionaria)   -> coda SINISTRA.
    regression = 'c' (solo costante) oppure 'ct' (costante e trend)."""

    def __init__(self, series, maxlag=None, regression="ct", alpha=0.05):
        self.series = np.asarray(series, dtype=float)
        self.maxlag = maxlag
        self.regression = regression
        self.alpha = alpha

    def _fit_adf(self, y, dy, lag, start):
        # osservazioni usate: dy[start:], con start >= lag
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
        j = len(cols) - lag - 1           # posizione di y_{t-1}

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
            # tutti i modelli confrontati con l'AIC usano le STESSE osservazioni
            max_p = int(np.floor(12.0 * (n / 100.0) ** 0.25))
            aics = [self._fit_adf(y, dy, lag, max_p)[1] for lag in range(max_p + 1)]
            self.lag = int(np.argmin(aics))
        else:
            self.lag = self.maxlag

        # regressione finale con il ritardo scelto, su tutte le osservazioni disponibili
        stat, _, nobs = self._fit_adf(y, dy, self.lag, self.lag)
        self.nobs = nobs
        self.crit = {a: adf_critical_value(self.regression, a, nobs) for a in (0.01, 0.05, 0.10)}
        crit = self.crit[self.alpha]
        stationary = bool(stat < crit)

        r = self.regression
        if stationary:
            verdict = f"H0 rifiutata (ADF={stat:.4f} < {crit:.3f}, lag={self.lag}): nessuna radice unitaria ('{r}')."
        else:
            verdict = f"H0 non rifiutata (ADF={stat:.4f} >= {crit:.3f}, lag={self.lag}): radice unitaria ('{r}')."

        return stat, stationary, verdict


class KPSS_test:
    """KPSS.  H0: serie stazionaria (attorno a una media 'c' o a un trend 'ct').
    eta = sum S_t^2 / (T^2 * sigma2_LR), S_t = somme parziali dei residui -> coda DESTRA."""

    def __init__(self, series, nlags=None, regression="ct", alpha=0.05):
        self.series = np.asarray(series, dtype=float)
        self.nlags = nlags
        self.regression = regression
        self.alpha = alpha

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
        crit = self.crit[self.alpha]
        stationary = bool(stat < crit)

        r = self.regression
        if stationary:
            verdict = f"H0 non rifiutata (KPSS={stat:.4f} < {crit}): serie stazionaria ('{r}')."
        else:
            verdict = f"H0 rifiutata (KPSS={stat:.4f} >= {crit}): serie non stazionaria ('{r}')."

        return stat, stationary, verdict


class Trend_test:
    """t-test con errori standard di Newey-West (HAC).
    kind='drift': H0 media di x_t = 0 (x_t = serie differenziata)  ->  c'e' un drift?
    kind='trend': H0 b = 0 in x_t = a + b t + u_t                  ->  c'e' un trend deterministico?"""

    def __init__(self, series, kind="drift", nlags=None, alpha=0.05):
        self.series = np.asarray(series, dtype=float)
        self.kind = kind
        self.nlags = nlags
        self.alpha = alpha

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
            g = X * u[:, None]                      # x_t * u_t
            S = g.T @ g
            for j in range(1, L + 1):
                w = 1.0 - j / (L + 1.0)
                G = g[j:].T @ g[:-j]
                S += w * (G + G.T)
            V = XtX_inv @ S @ XtX_inv               # sandwich di Newey-West
            est = float(b[1])
            se = float(np.sqrt(V[1, 1]))

        t_stat = est / se
        p_value = float(2 * stats.norm.sf(abs(t_stat)))
        significant = p_value < self.alpha
        what = "drift" if self.kind == "drift" else "trend"
        verdict = (f"{what} = {est:.5f} (SE HAC {se:.5f}), t = {t_stat:.3f}, p = {p_value:.3f}: "
                   + (f"{what} significativo." if significant else f"{what} non significativo."))
        return t_stat, p_value, significant, verdict


# ======================================================================
# Identificazione: ACF e PACF
# ======================================================================
class ACF_test:
    """ACF con bande di Bartlett: al ritardo k, sotto H0 'MA(k-1)',
    SE(rho_k) = sqrt((1 + 2 sum_{j<k} rho_j^2) / T).
    q suggerito = (primo k non significativo) - 1."""

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
    """PACF: phi_kk = ultimo coefficiente dell'AR(k) stimato con OLS.
    Sotto H0 'AR(k-1)', SE(phi_kk) ~ 1/sqrt(T).  p suggerito = (primo k non significativo) - 1."""

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


# ======================================================================
# Diagnostica dei coefficienti e dei residui
# ======================================================================
class T_test:
    """t-test sui coefficienti: t = beta / SE(beta), SE dalla diagonale di sigma^2 (X'X)^{-1}."""

    def __init__(self, model, alpha=0.05):
        self.model = model
        self.alpha = alpha

    def test(self):
        st = self.model.get_stats()
        beta = self.model.beta
        se = st["std_errors"]
        df = len(self.model.Y) - len(beta)
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), df)
        return t, p, p < self.alpha


class LjungBox_test:
    """Q = T(T+2) sum_{k=1}^{m} rho_k^2 / (T-k)  ~  chi2(dof).
    dof = m sulla serie osservata, m - p - q sui residui di un ARMA(p, q)."""

    def __init__(self, residuals, lags=10, dof=None, alpha=0.05):
        self.residuals = np.asarray(residuals, dtype=float)
        self.lags = lags
        self.dof = lags if dof is None else dof
        self.alpha = alpha

    def test(self):
        T = len(self.residuals)
        c = self.residuals - np.mean(self.residuals)
        denom = np.sum(c**2)
        rhos = np.array([np.sum(c[k:] * c[:-k]) / denom for k in range(1, self.lags + 1)])
        ks = np.arange(1, self.lags + 1, dtype=float)
        q = float(T * (T + 2.0) * np.sum((rhos**2) / (T - ks)))
        self.crit = float(stats.chi2.ppf(1 - self.alpha, self.dof))
        p_value = float(stats.chi2.sf(q, self.dof))
        ok = q <= self.crit                     # True = residui compatibili con rumore bianco
        return q, p_value, ok


class JarqueBera_test:
    """JB = T/6 * (S^2 + (K-3)^2 / 4)  ~  chi2(2) sotto normalita'."""

    def __init__(self, residuals, alpha=0.05):
        self.residuals = np.asarray(residuals, dtype=float)
        self.alpha = alpha

    def test(self):
        T = len(self.residuals)
        s = skewness(self.residuals)
        k = kurtosis(self.residuals)
        jb = (T / 6.0) * (s**2 + ((k - 3.0) ** 2) / 4.0)
        crit = float(stats.chi2.ppf(1 - self.alpha, 2))
        p_value = float(stats.chi2.sf(jb, 2))
        ok = jb <= crit
        verdict = f"JB={jb:.4f} (crit={crit:.4f}, p={p_value:.4f}, S={s:.3f}, K={k:.3f}) -> normali={ok}"
        return jb, p_value, ok, verdict


class DW_test:
    """Durbin-Watson: d ~ 2(1 - rho_1). Valido solo per regressioni senza la variabile
    dipendente ritardata tra i regressori: per i modelli ARMA usare il Ljung-Box."""

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
    """z-test su un singolo coefficiente: H0 beta = beta0 (es. theta = 2 - sqrt(3))."""

    def __init__(self, estimate, se, value, alpha=0.05):
        self.estimate, self.se, self.value, self.alpha = estimate, se, value, alpha

    def test(self):
        z = (self.estimate - self.value) / self.se
        p = float(2 * stats.norm.sf(abs(z)))
        return float(z), p, p >= self.alpha


class Stability_test:
    """Stabilita' dei parametri: stima su due sottocampioni e z-test sulla differenza,
    z = (b1 - b2) / sqrt(SE1^2 + SE2^2) per ogni coefficiente."""

    def __init__(self, build_model, series, split, alpha=0.05):
        self.build_model = build_model
        self.series = np.asarray(series, dtype=float)
        self.split = split
        self.alpha = alpha

    def test(self):
        m1 = self.build_model()
        m1.train(self.series[: self.split])
        m2 = self.build_model()
        m2.train(self.series[self.split:])
        se1 = m1.get_stats()["std_errors"]
        se2 = m2.get_stats()["std_errors"]
        z = (m1.beta - m2.beta) / np.sqrt(se1**2 + se2**2)
        p = 2 * stats.norm.sf(np.abs(z))
        return m1.beta, se1, m2.beta, se2, z, p, bool(np.all(p >= self.alpha))


# ======================================================================
# Confronto delle previsioni
# ======================================================================
class DieboldMariano_test:
    """H0: i due modelli hanno lo stesso errore quadratico atteso.
    d_t = e1_t^2 - e2_t^2,  DM = mean(d) / sqrt(sigma2_LR(d) / n)  ~ N(0,1).
    DM < 0: il modello 1 prevede meglio."""

    def __init__(self, e1, e2, h=1, alpha=0.05):
        self.e1 = np.asarray(e1, dtype=float)
        self.e2 = np.asarray(e2, dtype=float)
        self.h = h
        self.alpha = alpha

    def test(self):
        d = self.e1**2 - self.e2**2
        n = len(d)
        lrv = long_run_variance(d, self.h - 1)
        dm = float(np.mean(d) / np.sqrt(lrv / n))
        p_two = float(2 * stats.norm.sf(abs(dm)))
        p_one = float(stats.norm.cdf(dm))          # H1: modello 1 migliore
        verdict = (f"DM = {dm:.3f}, p bilaterale = {p_two:.3f}, p unilaterale (1 migliore di 2) = {p_one:.3f}")
        return dm, p_two, p_one, verdict
