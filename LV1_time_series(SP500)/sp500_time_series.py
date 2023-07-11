import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('SP500_weekly.csv')

data1 = data[['Open',"Date"]]
data1 = data1.set_index('Date')
data1['Time'] = np.arange(len(data1))

plt.style.use('seaborn-whitegrid')  # 繪圖模板
plt.rc('axes',labelweight='bold',labelsize=18,labelcolor='black',titleweight='bold',titlesize=20,titlecolor='red')
plt.rc('figure',figsize=(15,5),titlesize=25,titleweight='bold')

fig,ax=plt.subplots()
ax.plot('Time','Open',data=data1,color='0.8')
ax = sns.regplot(x='Time',y='Open',data=data1,scatter_kws={'color':'0.8'})
ax.set_title('TS for sp500 open')



# lag one week
data1['lag_1'] = data1['Open'].shift(1)
fig,ax=plt.subplots()
ax.set_aspect('equal')  # 設置x,y軸相等
ax = sns.regplot(x='lag_1',y='Open',data=data1,scatter_kws={'color':'0.8'})
ax.set_title('lag one week')


# 開盤價線性回歸
from sklearn.linear_model import LinearRegression

plot_params = dict(color='red',style='-',legend=False)

x = data1.loc[:,['Time']]    # dataframe
y = data1.loc[:,'Open']      # series
model = LinearRegression()
model.fit(x,y)
y_pre = pd.Series(model.predict(x),index=x.index)

ax = y.plot(**plot_params)
ax = y_pre.plot(ax=ax,linewidth=3)


# 過去一個月平均趨勢
MAV = data1['Open'].rolling(4,center=True).mean()
ax = data1['Open'].plot(style='--',color='0.8')
MAV.plot(ax = ax,linewidth=2,title='last one month moving average')


# Trend 隨時間累積產生變化
# 建構自定義模型
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

# constant 使用固定常數作為 bias
# 多項式階數為1階
dp = DeterministicProcess(index=data1.index,constant=True,order=1,drop=True)
x1 = dp.in_sample()
y = data1.loc[:,'Open'] 

model = LinearRegression()
model.fit(x1,y)
y_pre = pd.Series(model.predict(x1),index=x1.index)

plot_params = dict(color='red',style='-',legend=False)
ax = y.plot(**plot_params)
ax = y_pre.plot(ax=ax,linewidth=3,color='blue', label="Trend")

x2 = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(x2),index=x2.index)
ax = y_fore.plot(ax=ax,linewidth=3,color = 'green', label="Trend Forecast")
ax.legend()



# 季節性
from statsmodels.tsa.seasonal import seasonal_decompose
# 乘法
# resul_mul =seasonal_decompose(data1['Open'],model='multiplicative',extrapolate_trend='freq',period=48) # 一年48週
# 加法
resul_mul =seasonal_decompose(data1['Open'],model='additive',extrapolate_trend='freq',period=48)
resul_mul.plot()




