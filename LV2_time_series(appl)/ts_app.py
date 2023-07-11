import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime



path = 'AAPL_2006-01-01_to_2018-01-01.csv'

data = pd.read_csv(path)
data['Date'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))

'''
fig,ax = plt.subplots(figsize = (16,8))
ax.plot('Date','Open',data=data)
ax.plot('Date','High',color='green',data=data)
ax.plot('Date','Low',color='red',data=data)
ax.set_title('AAPL')
plt.xlabel('Date')
plt.ylabel('price')


import statsmodels.api as sm
data1 = data[['Date','Open']].set_index('Date')

resul = sm.tsa.seasonal_decompose(data1['Open'] ,model="multiplicative",extrapolate_trend='freq',period=180)
fig=resul.plot()
plt.show()


# 平穩性檢定(ADF)
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

data1 = data[['Date','Open']].set_index('Date')
adftes = adfuller(data1,autolag='AIC')
dfoutput = pd.Series(adftes[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in adftes[4].items():
    dfoutput[f'Critical Value {key}'] = value
print (dfoutput)

# 移除trend
# 如果週期為1個月   將資料減去1個月前資料 達到去除trend目的

from pandas import Series as Series
data1 = data[['Date','Open']].set_index('Date')

def differ(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        diff.append(dataset.iloc[i,] - dataset.iloc[i - interval,])
    return Series(diff)

new_ts1=differ(data1)
new_ts12=differ(data1,12)
fig,ax = plt.subplots(3,1,figsize= (16,8))
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

ax1.plot(data1.index,'Open',data=data1)
ax1.set_title('origin')
ax2.plot(new_ts1)
ax3.plot(new_ts12)
plt.ylabel('price')
plt.show()

adftes = adfuller(new_ts1,autolag='AIC')
dfoutput = pd.Series(adftes[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in adftes[4].items():
    dfoutput[f'Critical Value {key}'] = value
print (dfoutput)

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

def tsplot(y, lags=None,title=''):
    if not isinstance(y, pd.Series):
        return print('not Series')    
    plt.style.context('ggplot')
    fig = plt.figure(figsize=(10, 8))
    #mpl.rcParams['font.family'] = 'Ubuntu Mono'
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    pp_ax = plt.subplot2grid(layout, (2, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')        
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()
    plt.show()


data1 = data[['Date','Open']].set_index('Date')
data1 =pd.Series(data.Open,index=data.index)

tsplot(data1, lags=12,title='')

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.api as smt


data1 = data[['Date','Open']].set_index('Date')
data1 =pd.Series(data.Open,index=data.index)

best_aic = np.inf 
best_order = None
best_mdl = None

for i in range(5):
    for j in range(5):
        for k in range(5):
            try:
                tmp_mdl = smt.ARIMA(data1, order=(i, j, k)).fit()
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, j, k)
                    best_mdl = tmp_mdl
            except: continue
'''












