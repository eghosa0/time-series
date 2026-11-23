import numpy as np
from statistics import Stats
#import matplotlib.pyplot as plt


class ARMA:
    def __init__(self, p, q):
        self.AR_beta:np.ndarray = None
        self.MA_beta:np.ndarray = None
        self.p = p
        self.q = q
        self.X_AR = None
        self.Y_AR = None
        self.X_MA = None
        self.Y_MA = None
        self.fitted_AR = None
        self.fitted_MA = None
        self.fitted = None
        self.beta = None
        self.stats = Stats()

    
    def get_stats(self):
        if self.beta is None or self.fitted is None:
            raise ValueError("Model has not been trained yet. Please call 'train' first.")

        X_aligned = np.hstack([self.X_AR[self.q:], self.X_MA])
        Y_aligned = self.Y_AR[self.q:]

        stats = self.stats.compute_stats(
            parameters=self.beta.shape[0],
            X=X_aligned,
            Y=Y_aligned,
            predictions=self.fitted,
            X_new=X_aligned[-1],
        )
        return stats

    def transfomr_dataset(self, y, lag):
        n = len(y)
        X = np.column_stack([y[lag-j:n-j] for j in range(1, lag+1)])
        return X, y[lag:]


    def train(self, dataset):
        X, Y = self.transfomr_dataset(dataset, self.p)
        self.AR_beta = np.linalg.inv(X.T@X) @ X.T@Y
        self.X_AR = X
        self.Y_AR = Y
        self.fitted_AR = X @ self.AR_beta
        residuals = self.Y_AR - self.fitted_AR
        X, Y = self.transfomr_dataset(residuals, self.q)
        self.MA_beta = np.linalg.inv(X.T@X) @ X.T@Y
        self.X_MA = X
        self.Y_MA = Y
        self.fitted_MA = X @ self.MA_beta
        self.fitted = self.fitted_AR[self.q:] + self.fitted_MA
        self.beta = np.hstack([self.AR_beta, self.MA_beta])
        return self.AR_beta, self.MA_beta



class RandomWalk:
    def __init__(self):
        self.y = None
        self.fitted = None
        self.residuals = None

    def train(self, dataset):
        y = np.asarray(dataset, dtype=float).ravel()
        self.y = y
        self.fitted = y[:-1]
        self.residuals = y[1:] - self.fitted
        return self.fitted



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


    """start_idx = model.p + model.q
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
    
    