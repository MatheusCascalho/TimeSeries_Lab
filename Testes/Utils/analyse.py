from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import matplotlib.pyplot as plt

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