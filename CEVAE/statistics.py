import torch


class Statistics(object):
    def __init__(self):
        self.data = dict(mu0=None, mu1=None, t=None, x=None, yf=None, ycf=None)

    def collect(self, label, value):
        if self.data[label] is None:
            self.data[label] = value
        else:
            self.data[label] = torch.cat((self.data[label], value), 0)

        if self.data['mu0'] is not None and self.data['mu1'] is not None and self.data['mu0'].shape == self.data['mu1'].shape:
            self.true_ITE = self.data['mu1'] - self.data['mu0']

    def _RMSE_ITE(self, y0, y1):
        predicted_ITE = torch.where(
            self.data['t'] == 1, self.data['yf'] - y0, y1 - self.data['yf'])
        error = self.true_ITE - predicted_ITE
        return torch.sqrt(torch.mean(torch.mul(error, error)))

    def _absolute_ATE(self, y0, y1):
        return torch.abs(torch.mean(y1 - y0) - torch.mean(self.true_ITE))

    def _PEHE(self, y0, y1):
        error = self.true_ITE - (y1 - y0)
        return torch.sqrt(torch.mean(torch.mul(error, error)))

    def y_errors(self, y0, y1):
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]
        factual_y = (1. - self.data['t']) * y0 + self.data['t'] * y1
        counterfactual_y = self.data['t'] * y0 + (1. - self.data['t']) * y1

        factual_y_diff = factual_y - self.data['yf']
        counterfactual_y_diff = counterfactual_y - self.data['ycf']

        RMSE_factual = torch.sqrt(torch.mean(
            torch.mul(factual_y_diff, factual_y_diff)))
        RMSE_counterfactual = torch.sqrt(torch.mean(
            torch.mul(counterfactual_y_diff, counterfactual_y_diff)))

        return RMSE_factual, RMSE_counterfactual

    def calculate(self, y0, y1):
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]
        ITE = self._RMSE_ITE(y0, y1)
        ATE = self._absolute_ATE(y0, y1)
        PEHE = self._PEHE(y0, y1)
        return ITE, ATE, PEHE
