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
        self.stats['forecast_error_variance'] = self.forecast_error_variance()
        self.stats['skewness'] = self.get_skewness()
        self.stats['kurtosis'] = self.get_kurtosis()
        return self.stats


    def get_residuals(self):
            return self.Y - self.predictions

    
    def get_variance(self):
        res = self.get_residuals()
        var = res.T@res/(len(self.Y)-self.parameters)
        return var


    def get_parameters_variance(self):
        return self.get_variance() * np.linalg.inv(self.X.T@self.X)


    def forecast_error_variance(self):
        self.X_new = self.X_new.reshape(1, -1)
        
        if self.X_new.shape[0] != 1:
            raise ValueError("Input matrix X_new must have only one row.", self.X_new.shape)

        var = self.get_variance()
        er_1 = self.X_new@self.get_parameters_variance()@self.X_new.T + var
        
        return er_1

    def get_skewness(self):
        res = np.asarray(self.get_residuals(), dtype=float).ravel()
        centered = res - np.mean(res)
        m2 = np.mean(centered**2)
        m3 = np.mean(centered**3)
        if m2 == 0:
            return 0.0
        return float(m3 / (m2**1.5))

    def get_kurtosis(self):
        res = np.asarray(self.get_residuals(), dtype=float).ravel()
        centered = res - np.mean(res)
        m2 = np.mean(centered**2)
        m4 = np.mean(centered**4)
        if m2 == 0:
            return 0.0
        return float(m4 / (m2**2))

    def __str__(self):
        return f"Metrics: {self.stats}"