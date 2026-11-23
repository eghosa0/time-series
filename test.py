import numpy as np

class DW_test: #DA CONSULTARE I VALORI TABELLATI PER CONFERMARE O MENO H0
    def __init__(self, model, d_L=1.65, d_U=1.69):
        self.model = model
        self.d_L = d_L
        self.d_U = d_U

    def test(self, alpha=0.05):
        res = self.model.get_stats()['residuals']
        num = np.sum((res[1:] - res[:len(res)-1])**2)
        den = res.T@res
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
