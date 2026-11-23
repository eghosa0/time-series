import numpy as np
from statistics import Stats


class TimeSerie:
    def __init__(self):
        self.X = None
        self.Y = None
        self.fitted = None
        self.residuals = None
        self.stats = Stats()

    def get_stats(self):
        if self.beta is None or self.fitted is None:
            raise ValueError("Model has not been trained yet. Please call 'train' first.")

        stats = self.stats.compute_stats(
            parameters=self.beta.shape[0],
            X=self.X,
            Y=self.Y,
            predictions=self.fitted,
            X_new=self.X[-1],
        )
        return stats

    def train(self):
        pass



class ARMA(TimeSerie):
    def __init__(self, p, q):
        super().__init__()
        self.p = p
        self.q = q

    
    def get_stats(self):
        if self.beta is None or self.fitted is None:
            raise ValueError("Model has not been trained yet. Please call 'train' first.")

        stats = self.stats.compute_stats(
            parameters=self.beta.shape[0],
            X=self.X,
            Y=self.Y,
            predictions=self.fitted,
            X_new=self.X[-1],
        )
        return stats

    def transfomr_dataset(self, y, lag):
        n = len(y)
        X = np.column_stack([y[lag-j:n-j] for j in range(1, lag+1)])
        return X, y[lag:]


    def model(self, dataset, lag):
        X, Y = self.transfomr_dataset(dataset, lag)
        beta = np.linalg.inv(X.T@X) @ X.T@Y
        fitted = X @ beta
        return beta, fitted, Y, X


    def train(self, dataset):
        beta_AR, fitted_AR, Y_AR, X_AR = self.model(dataset, self.p)
        residuals = Y_AR - fitted_AR
        beta_MA, fitted_MA, Y_MA, X_MA = self.model(residuals, self.q)

        self.fitted = fitted_AR[self.q:] + fitted_MA
        self.beta = np.hstack([beta_AR, beta_MA])
        self.X = np.hstack([X_AR[self.q:], X_MA])
        self.Y = Y_AR[self.q:]
        return self.fitted



class RandomWalk(TimeSerie):
    def __init__(self):
        super().__init__()
        self.y = None

    def train(self, dataset):
        y = np.asarray(dataset, dtype=float).ravel()
        self.y = y
        self.X = y[:-1].reshape(-1, 1)
        self.Y = y[1:]
        self.fitted = self.X[:, 0]
        self.residuals = self.Y - self.fitted
        return self.fitted

    def get_stats(self):
        if self.fitted is None:
            raise ValueError("Model has not been trained yet. Please call 'train' first.")

        stats = self.stats.compute_stats(
            parameters=1,
            X=self.X,
            Y=self.Y,
            predictions=self.fitted,
            X_new=self.X[-1],
        )

        return stats




if __name__ == "__main__":
    from dataset import load
    from test import DW_test

    dataset = load().reset_index()
    pricies = np.log(dataset.iloc[:, 1].values)
    returns = pricies[1:] - pricies[:-1]
    
    model = ARMA(3, 2)
    model.train(pricies)

    rw = RandomWalk()
    rw.train(pricies)

    test = DW_test(rw)

    print(test.test())

    """import matplotlib.pyplot as plt

    start_idx = model.p + model.q
    actual_aligned = pricies[start_idx:]

    plt.figure(figsize=(10, 4))
    plt.plot(actual_aligned, label="Log-prices osservati", alpha=0.8)
    plt.plot(model.fitted, label=f"Fitted ARMA({model.p}, {model.q})", linestyle="--")
    plt.title("Log-prices osservati vs Fitted ARMA")
    plt.xlabel("Osservazioni (finestra allineata)")
    plt.ylabel("Log-prezzo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()"""
    
    