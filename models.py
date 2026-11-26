import numpy as np
from statis import Stats
from metrics import aic, bic


def lag_matrix(x, lags, start):
    """Righe t = start, ..., T-1; colonne x_{t-1}, ..., x_{t-lags}."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if lags == 0:
        return np.empty((T - start, 0))
    return np.column_stack([x[start - j:T - j] for j in range(1, lags + 1)])


def ols(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


class TimeSerie:
    def __init__(self):
        self.beta = None
        self.X = None
        self.Y = None
        self.fitted = None
        self.residuals = None
        self.stats = Stats()

    def next_row(self):
        """Riga dei regressori per il periodo T+1 (di default l'ultima riga)."""
        return self.X[-1]

    def get_stats(self):
        if self.beta is None or self.fitted is None:
            raise ValueError("Model has not been trained yet. Please call 'train' first.")

        stats = self.stats.compute_stats(
            parameters=self.beta.shape[0],
            X=self.X,
            Y=self.Y,
            predictions=self.fitted,
            X_new=self.next_row(),
        )
        return stats

    def train(self):
        pass


class ARMA(TimeSerie):
    """ARMA(p, q) con costante, stimato con il metodo di Hannan-Rissanen (due OLS).

    Passo 1: AR lungo di ordine m  ->  stima degli shock eps_t (non osservabili).
    Passo 2: OLS di y_t su [1, y_{t-1..t-p}, eps_{t-1..t-q}].

    Con q = 0 il passo 1 non serve: e' un AR(p) stimato con OLS.
    Le stime sono consistenti; per un controllo si confrontano con la massima
    verosimiglianza di statsmodels nel notebook.
    """

    def __init__(self, p, q, const=True, initial_ar=None):
        super().__init__()
        self.p = p
        self.q = q
        self.const = const
        self.initial_ar = initial_ar

    # ----- costruzione del dataset (nome originale mantenuto) -----
    def transform_dataset(self, y, lag):
        n = len(y)
        X = np.column_stack([y[lag - j:n - j] for j in range(1, lag + 1)])
        return X, y[lag:]

    transfomr_dataset = transform_dataset

    def _design(self, y, eps, start):
        cols = []
        n = len(y) - start
        if self.const:
            cols.append(np.ones((n, 1)))
        cols.append(lag_matrix(y, self.p, start))
        if self.q > 0:
            cols.append(lag_matrix(eps, self.q, start))
        return np.hstack(cols)

    # ----- stima -----
    def train(self, dataset):
        y = np.asarray(dataset, dtype=float).ravel()
        self.y = y
        T = len(y)
        p, q = self.p, self.q

        eps = None
        if q > 0:
            m = self.initial_ar or max(int(np.floor(np.log(T) ** 2)), 2 * max(p, q))
            X1 = lag_matrix(y, m, m)
            if self.const:
                X1 = np.hstack([np.ones((T - m, 1)), X1])
            b1 = ols(X1, y[m:])
            eps = np.full(T, np.nan)
            eps[m:] = y[m:] - X1 @ b1
            self.m = m
            start = m + q
        else:
            start = p

        X = self._design(y, eps, start)
        Y = y[start:]
        beta = ols(X, Y) if X.shape[1] > 0 else np.empty(0)

        self.start = start
        self.X, self.Y, self.beta = X, Y, beta
        self.fitted = X @ beta if X.shape[1] > 0 else np.zeros_like(Y)
        self.residuals = self.Y - self.fitted

        i = 0
        self.c = float(beta[0]) if self.const else 0.0
        i += int(self.const)
        self.phi = beta[i:i + p]
        self.theta = beta[i + p:i + p + q]

        self.innovations = self._innovations(y)
        valid = self.innovations[p:]
        self.sigma2 = float(valid @ valid / (len(valid) - len(beta)))
        return self.fitted

    def _innovations(self, y):
        """Shock ricostruiti in modo ricorsivo con i coefficienti finali:
        e_t = y_t - c - sum phi_i y_{t-i} - sum theta_j e_{t-j}, con e_t = 0 per t < p."""
        T = len(y)
        e = np.zeros(T)
        for t in range(self.p, T):
            val = y[t] - self.c
            for i in range(1, self.p + 1):
                val -= self.phi[i - 1] * y[t - i]
            for j in range(1, self.q + 1):
                if t - j >= 0:
                    val -= self.theta[j - 1] * e[t - j]
            e[t] = val
        return e

    def next_row(self):
        row = []
        if self.const:
            row.append(1.0)
        row += [self.y[-i] for i in range(1, self.p + 1)]
        row += [self.innovations[-j] for j in range(1, self.q + 1)]
        return np.array(row)

    # ----- proprieta' del modello -----
    def roots(self):
        """Radici di 1 - phi_1 z - ... - phi_p z^p  e  1 + theta_1 z + ... + theta_q z^q."""
        ar = np.roots(np.r_[-self.phi[::-1], 1.0]) if self.p > 0 else np.array([])
        ma = np.roots(np.r_[self.theta[::-1], 1.0]) if self.q > 0 else np.array([])
        return ar, ma

    def is_stationary(self):
        return bool(np.all(np.abs(self.roots()[0]) > 1.0))

    def is_invertible(self):
        return bool(np.all(np.abs(self.roots()[1]) > 1.0))

    def psi_weights(self, n):
        """Pesi della forma MA(inf): psi_0 = 1, psi_j = theta_j + sum_i phi_i psi_{j-i}."""
        psi = np.zeros(n)
        psi[0] = 1.0
        for j in range(1, n):
            val = self.theta[j - 1] if j <= self.q else 0.0
            for i in range(1, min(j, self.p) + 1):
                val += self.phi[i - 1] * psi[j - i]
            psi[j] = val
        return psi

    def information_criteria(self, burn=None):
        """AIC e BIC sugli shock ricostruiti, scartando le prime `burn` osservazioni
        (usare lo stesso burn per tutti i modelli confrontati)."""
        burn = self.p if burn is None else burn
        res = self.innovations[burn:]
        k = len(self.beta)
        return aic(res, k), bic(res, k)

    # ----- previsione -----
    def forecast(self, h):
        """Previsione ricorsiva: shock futuri = 0, shock passati = innovazioni stimate.
        Varianza dell'errore a h passi: sigma^2 * sum_{j<h} psi_j^2."""
        T = len(self.y)
        y_ext = list(self.y)
        e_ext = list(self.innovations)
        for k in range(h):
            t = T + k
            val = self.c
            for i in range(1, self.p + 1):
                val += self.phi[i - 1] * y_ext[t - i]
            for j in range(1, self.q + 1):
                val += self.theta[j - 1] * e_ext[t - j]
            y_ext.append(val)
            e_ext.append(0.0)
        mean = np.array(y_ext[T:])
        psi = self.psi_weights(h)
        var = self.sigma2 * np.cumsum(psi**2)
        return mean, var


class ARIMA(TimeSerie):
    """ARIMA(p, d, q) con d in {0, 1}: ARMA(p, q) sulla serie differenziata d volte.
    Con d = 1 la costante e' il drift (crescita media per periodo)."""

    def __init__(self, p, d, q, const=True, initial_ar=None):
        super().__init__()
        if d not in (0, 1):
            raise ValueError("Sono supportati solo d = 0 e d = 1.")
        self.p, self.d, self.q = p, d, q
        self.arma = ARMA(p, q, const=const, initial_ar=initial_ar)

    def train(self, dataset):
        y = np.asarray(dataset, dtype=float).ravel()
        self.y = y
        x = np.diff(y) if self.d == 1 else y
        self.arma.train(x)
        for attr in ("beta", "X", "Y", "fitted", "residuals", "innovations", "sigma2",
                     "c", "phi", "theta", "start"):
            setattr(self, attr, getattr(self.arma, attr))
        return self.fitted

    def get_stats(self):
        return self.arma.get_stats()

    def roots(self):
        return self.arma.roots()

    def information_criteria(self, burn=None):
        return self.arma.information_criteria(burn)

    def forecast(self, h):
        """Previsione dei livelli: con d = 1 si risommano le differenze previste.
        I pesi psi dei livelli sono le somme cumulate dei pesi della serie differenziata."""
        mean_x, _ = self.arma.forecast(h)
        psi = self.arma.psi_weights(h)
        if self.d == 1:
            mean = self.y[-1] + np.cumsum(mean_x)
            psi = np.cumsum(psi)
        else:
            mean = mean_x
        var = self.sigma2 * np.cumsum(psi**2)
        return mean, var


class RandomWalk(TimeSerie):
    """y_t = y_{t-1} (+ drift) + eps_t. Previsione: ultimo valore (+ h * drift)."""

    def __init__(self, drift=False):
        super().__init__()
        self.y = None
        self.drift = drift
        self.mu = 0.0

    def train(self, dataset):
        y = np.asarray(dataset, dtype=float).ravel()
        self.y = y
        self.mu = float(np.mean(np.diff(y))) if self.drift else 0.0
        self.X = y[:-1].reshape(-1, 1)
        self.Y = y[1:]
        self.fitted = self.X[:, 0] + self.mu
        self.beta = np.array([1.0])
        self.residuals = self.Y - self.fitted
        k = int(self.drift)
        self.sigma2 = float(self.residuals @ self.residuals / (len(self.Y) - k))
        return self.fitted

    def get_stats(self):
        # Il coefficiente di y_{t-1} e' fissato a 1 (non stimato): l'unico parametro
        # eventualmente stimato e' il drift, con varianza sigma^2 / n.
        n = len(self.Y)
        par_var = self.sigma2 / n if self.drift else 0.0
        return {
            "residuals": self.residuals,
            "residual_variance": self.sigma2,
            "parameter_variance": np.array([[par_var]]),
            "forecast_error_variance": self.sigma2 + par_var,
        }

    def forecast(self, h):
        steps = np.arange(1, h + 1)
        return self.y[-1] + self.mu * steps, self.sigma2 * steps


def rolling_one_step(build_model, y, n_test, z=1.96):
    """Previsioni a un passo con origine mobile sulle ultime n_test osservazioni.
    build_model() deve restituire un modello nuovo con train() e forecast()."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    preds, lower, upper = [], [], []
    for i in range(T - n_test, T):
        model = build_model()
        model.train(y[:i])
        mean, var = model.forecast(1)
        preds.append(mean[0])
        lower.append(mean[0] - z * np.sqrt(var[0]))
        upper.append(mean[0] + z * np.sqrt(var[0]))
    actual = y[T - n_test:]
    preds = np.array(preds)
    return actual, preds, actual - preds, np.array(lower), np.array(upper)


if __name__ == "__main__":
    from dataset import load

    y = load().values
    model = ARIMA(0, 1, 1, const=False)
    model.train(y)
    print(model.beta, model.forecast(3))
