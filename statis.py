import numpy as np


class Stats:
    def __init__(self):
        self.stats = {}


    def init_vals(self, parameters, X, Y, predictions, X_new):
        self.parameters = parameters
        self.X = X
        self.Y = Y
        self.predictions = predictions
        self.X_new = X_new


    def compute_stats(self, parameters, X, Y, predictions, X_new):
        self.init_vals(parameters, X, Y, predictions, X_new)
        self.stats['residuals'] = self.get_residuals()
        self.stats['residual_variance'] = self.get_variance()
        self.stats['parameter_variance'] = self.get_parameters_variance()
        self.stats['std_errors'] = np.sqrt(np.diag(self.stats['parameter_variance']))
        self.stats['forecast_error_variance'] = self.forecast_error_variance()
        return self.stats


    def get_residuals(self):
        return self.Y - self.predictions


    def get_variance(self):
        res = self.get_residuals()
        var = res.T @ res / (len(self.Y) - self.parameters)
        return float(var)


    def get_parameters_variance(self):
        return self.get_variance() * np.linalg.inv(self.X.T @ self.X)


    def forecast_error_variance(self):
        # sigma^2 * (1 + x' (X'X)^{-1} x): x deve essere la riga dei regressori
        # del periodo da prevedere (T+1), non l'ultima riga del campione.
        x_new = np.asarray(self.X_new, dtype=float).reshape(1, -1)

        if x_new.shape[0] != 1:
            raise ValueError("Input matrix X_new must have only one row.", x_new.shape)

        var = self.get_variance()
        er_1 = x_new @ self.get_parameters_variance() @ x_new.T + var
        return float(np.squeeze(er_1))


    def __str__(self):
        return f"Metrics: {self.stats}"
