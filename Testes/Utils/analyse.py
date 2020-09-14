# Autor: Matheus Cascalho dos Santos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.figure import Figure


class Analyser:
    def __init__(self, data: np.ndarray, **kwargs):
        if type(data) != np.ndarray:
            data = np.array(data)
        self.__data = data
        self.name = kwargs['name'] if 'name' in kwargs.keys() else 'DATA'
        self.model = kwargs['model'] if 'model' in kwargs.keys() else 'additive'
        self.freq = kwargs['freq'] if 'freq' in kwargs.keys() else 1
        self.__description = ''

        self.analyse()

    def __str__(self):
        return self.__description

    def __len__(self):
        return len(self.__data)

    @property
    def data(self):
        return self.__data

    def esperanca(self):
        value_counts = pd.Series(self.__data).value_counts()
        total = len(self.__data)
        s = 0
        for d, c in zip(value_counts.index, value_counts.values):
            s += d * (c / total)
        return s

    def stats(self) -> pd.Series:
        st = pd.Series(self.__data).describe()
        st['Esp'] = self.esperanca()
        st['Var'] = self.__data.std() ** 2
        return st

    def decomposition(self) -> DecomposeResult:
        try:
            return seasonal_decompose(self.__data, self.model, period=self.freq)
        except ValueError:
            print(f'Exceção: {ValueError}')
            self.model = 'additive'
            print(f'\nFoi adotado o modelo padrão {self.model}')
            return seasonal_decompose(self.__data, self.model, period=self.freq)

    def plot_decomposition(self, *args, **kwargs) -> Figure:
        """
        Plot decomposition from data

        :param args: 'seasonal', 'trend', 'resid' or 'observed'
        :param kwargs: figsize
        :return:
        """
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
            a = ax[i] if len(args) > 1 else ax
            a.set_title(v)
            if v == 'seasonal':
                a.plot(dec.seasonal)
            elif v == 'trend':
                a.plot(dec.trend)
            elif v == 'resid':
                a.plot(dec.resid)
            else:
                a.plot(dec.observed)
        return fig

    def stationarity(self) -> pd.DataFrame:
        row = [self.name]
        result = adfuller(self.__data)
        row.extend([result[0], result[1]])
        row.append('H0 Accepted' if result[1] > 0.05 else 'H0 Rejected')
        row.append('Não Estacionária'if result[1] > 0.05 else 'Estacionária')
        return pd.DataFrame([row], columns=['Dataset', 'ADF Statistic', 'p-value', 'Result', 'Estacionariedade'])

    def homoscedasticity(self) -> pd.DataFrame:
        row = [self.name]
        cut = len(self.__data) // 2
        ds1, ds2 = self.__data[:cut], self.__data[cut:]
        result = levene(ds1, ds2)
        row.extend([result.statistic, result.pvalue])
        row.append('H0 Accepted' if result.pvalue > 0.05 else 'H0 Rejected')
        row.append('Homocedástica'if result[1] > 0.05 else 'Heterocedástica')
        return pd.DataFrame([row], columns=['Dataset', 'Levene Statistic', 'p-value', 'Result', 'Cedasticidade'])

    def analyse(self) -> None:
        adf: pd.DataFrame = self.stationarity()
        hmk: pd.DataFrame = self.homoscedasticity()
        self.__description = ''
        if 'Rejected' in adf['Result'].values[0]:
            self.__description += 'A série é estacionária!!\n'
        else:
            self.__description += 'A série é não-estacionária!!\n'

        if 'H0 Accepted' in hmk['Result'].values[0]:
            self.__description += 'As variâncias das sub-amostras são iguais \n'
        else:
            self.__description += 'As variâncias das sub-amostras não são iguais\n'

    def reset(self, data, **kwargs) -> None:
        self.__data = data
        self.name = kwargs['name'] if 'name' in kwargs.keys() else 'DATA'
        self.model = kwargs['model'] if 'model' in kwargs.keys() else 'additive'
        self.freq = kwargs['freq'] if 'freq' in kwargs.keys() else 1
        self.__description = ''
        self.analyse()

    def acf(self, lags=10):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        plot_acf(self.__data, lags=lags, ax=ax[0])
        plot_pacf(self.__data, lags=lags, ax=ax[1])
        plt.tight_layout()

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    f = '../AirQuality/AirQualityUCI.csv'
    df = pd.read_csv(f, sep=';')
    data = df['PT08.S1(CO)'].dropna()
    data = data[np.where(data > 0)[0]]
    an = Analyser(data, model='multiplicative')
    print(an.stats())
    l = ['seasonal', 'resid', 'observed']
    g = an.plot_decomposition(*l, figsize=(20, 10))
    g.savefig('data.png')
    print(an)

