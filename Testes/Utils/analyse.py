from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Analyser:
    def __init__(self, data: np.ndarray, **kwargs):
        self.data = data
        self.name = kwargs['name'] if 'name' in kwargs.keys() else 'DATA'
        self.model = kwargs['model'] if 'model' in kwargs.keys() else 'additive'
        self.freq = kwargs['freq'] if 'freq' in kwargs.keys() else 1

    def __str__(self):
        return 'Analizador instanciado!!'

    def decomposition(self) -> DecomposeResult:
        return seasonal_decompose(self.data, self.model, freq=self.freq)

    def plot_decomposition(self, *args, **kwargs):
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else (12, 8)
        valid_args = ['seasonal', 'trend', 'resid', 'observed']
        invalid = [v for v in args if v not in valid_args]
        if len(invalid) > 0:
            messages = ''
            for v in invalid:
                messages += f'Argumento {v} é invalido\n'
            print(messages)
            del invalid, messages
        args = [v for v in args if v in valid_args]
        fig, ax = plt.subplots(nrows=len(args), ncols=1, figsize=figsize)
        dec = self.decomposition()
        for i, v in enumerate(args):
            if len(args) > 1:
                ax[i].set_title(v)
            if v == 'seasonal':
                dec.seasonal.plot(ax=ax[i] if len(args) > 1 else ax)
            elif v == 'trend':
                dec.trend.plot(ax=ax[i] if len(args) > 1 else ax)
            elif v == 'resid':
                dec.resid.plot(ax=ax[i] if len(args) > 1 else ax)
            else:
                dec.observed.plot(ax=ax[i] if len(args) > 1 else ax)
        return fig

    def stationarity(self):
        row = [self.name]
        result = adfuller(self.data)
        row.extend([result[0], result[1]])
        row.append('H0 Accepted' if result[1] > 0.05 else 'H0 Rejected')
        return pd.DataFrame([row], columns=['Dataset', 'ADF Statistic', 'p-value', 'Result'])

    def homokedasticity(self):
        row = [self.name]
        cut = len(self.data) // 2
        ds1, ds2 = self.data[:cut], self.data[cut:]
        result = levene(ds1, ds2)
        row.extend([result.statistic, result.pvalue])
        row.append('H0 Accepted' if result.pvalue > 0.05 else 'H0 Rejected')
        return pd.DataFrame([row], columns=['Dataset', 'Levene Statistic', 'p-value', 'Result'])

    def analyse(self):
        pass



# --------------------------------------------------------------------------------------------------

def decomposition(data, model='additive', freq=1):
    result = seasonal_decompose(data, model=model, freq=freq)
    tmp = result.plot()
    return tmp

def stationarity(data, name):
    row = [name]
    result = adfuller(data)
    row.extend([result[0], result[1]])
    row.append('H0 Accepted' if result[1] > 0.05 else 'H0 Rejected')
    return pd.DataFrame([row], columns=['Dataset', 'ADF Statistic', 'p-value', 'Result'])

def homokedasticity(data, name):
    row = [name]
    cut = len(data) // 2
    ds1, ds2 = data[:cut], data[cut:]
    result = levene(ds1, ds2)
    row.extend([result.statistic, result.pvalue])
    row.append('H0 Accepted' if result.pvalue > 0.05 else 'H0 Rejected')
    return pd.DataFrame([row], columns=['Dataset', 'Levene Statistic', 'p-value', 'Result'])

def analyse(data, name, model='additive', freq=1, show_data=True, show_acf=True, show_dec=True):
    print('='*100)

    if show_data:
        fig, ax = plt.figure(figsize=(15, 5))
        ax.title(name)
        ax.plot(data)
    decomposition(data, model, freq)

    adf = stationarity(data, name)
    s = str(adf) + '\n\n'
    s += '-' * 50 + '\n'
    if 'Rejected' in adf['Result']:
        s += 'A série é estacionária!!\n'
    else:
        s += 'A série é não-estacionária!!\n'
    hmk = homokedasticity(data, name)
    s += '-' * 50 + '\n'
    s += str(hmk) + '\n\n'
    s += '-' * 50 + '\n'
    if 'Accepted' in hmk['Result']:
        s += 'As variâncias das sub-amostras são iguais \n'
    else:
        s += 'As variâncias das sub-amostras não são iguais\n'
    print(s)
    plot_acf(data)
    print('=' * 100)