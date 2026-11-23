import numpy as np

class DW_test: #DA CONSULTARE I VALORI TABELLATI PER CONFERMARE O MENO H0
    def __init__(self, model):
        self.model = model

    def test(self, alpha=0.05):
        res = self.model.get_stats()['residuals']
        num = np.sum((res[1:] - res[:len(res)-1])**2)
        den = res.T@res
        return num/den