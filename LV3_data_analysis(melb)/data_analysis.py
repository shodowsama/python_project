import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = './melb_data.csv'
data = pd.read_csv(path)

'''

print(data.columns)
print(data.info())
# 'Landsize', 'BuildingArea', 'YearBuilt' 有遺失值
# 依變項為 'Price'

'''

# lattitude longtitude 原始資料在經緯的分布差異不大 
data[['day','month','year']] = data.Date.str.split('/',expand=True)
y = data.Price
dataa = data.drop(['Price','Lattitude','Longtitude'],axis=1)
# data1 = dataa.select_dtypes(exclude='object') # 類別以外(int ,float)
# data2 = dataa.select_dtypes(include='object') # 僅取類別


def plot_data(dat,y_dat,colnam):
    plt.style.use('classic')
    dat = dat.join(y_dat)
    if len(set(dat[colnam])) >50:
        return print(colnam + 'over')
    # 處理遺失值
    if dat[[colnam]].isnull().any().values == True:
        dat = dat.dropna(axis=0,subset=colnam)

    # 處理浮點數
    if dat[[colnam]].dtypes.values == 'float64' :
        dat[[colnam]] = dat[[colnam]].round().astype(int) 

    dat1 = dat.groupby(colnam)[y_dat.name].agg(['median','count'])    
    fig,ax = plt.subplots(1,2,figsize=(12,8))

    ax1 = ax[0]
    ax2 = ax[1]
    sns.barplot(ax=ax1,x=dat1.index,y=dat1['count'])
    ax1.set_title(colnam +'_count')
    ax1.set_xlabel(colnam)
    ax1.set_ylabel('count')

    sns.barplot(ax=ax2,x=dat1.index,y=dat1['median'])
    ax2.set_title(colnam +'_median_price')
    ax2.set_xlabel(colnam)
    ax2.set_ylabel('median_price')

    plt.show()

"""
# 'Rooms' 'Bedroom2' 'Bathroom' 
# 房間存在正相關且集中
# 刪除 'Rooms'和'Bedroom2':7以下 'Bathroom':6以下

check_room1 = pd.crosstab(dataa.Rooms,dataa.Bedroom2,margins=True)
check_room2 = pd.crosstab(dataa.Rooms,dataa.Bathroom,margins=True)
check_room3 = pd.crosstab(dataa.Bedroom2,dataa.Bathroom,margins=True)

fig,ax=plt.subplots(1,2,figsize=(12,8))
sns.countplot('Rooms',hue='Bathroom',data=dataa1,ax=ax[0])
sns.countplot('Bedroom2',hue='Bathroom',data=dataa1,ax=ax[1])


# type U和T車位在5以下 
# 房間數在5以上為type U
# Car 6以上刪除
check_type = pd.crosstab([dataa.Rooms,dataa.Car],dataa.Type,margins=True)
fig,ax=plt.subplots(figsize=(12,8))
sns.factorplot('Car','Rooms',hue='Type',data=dataa,ax=ax)

check_Car = pd.crosstab(dataa.Car,dataa.Type,margins=True)
"""
# yearbuilt 區間合併
# 使用 Regionname 進行插補
# type H主要在 1800年之後建造

dataa['house_year'] = dataa['YearBuilt'].apply(lambda x: (x//10)*10 if x!=None else None )

"""
dataa1 = dataa[dataa.house_year !=None]
fig,ax=plt.subplots(1,3,figsize=(18,8))
sns.distplot(dataa1[dataa1['Type']=='u'].house_year,ax=ax[0] )
ax[0].set_title('Type U')
sns.distplot(dataa1[dataa1['Type']=='t'].house_year,ax=ax[1] )
ax[1].set_title('Type T')
sns.distplot(dataa1[dataa1['Type']=='h'].house_year,ax=ax[2] )
ax[2].set_title('Type H')
"""

dataa['house_year'] = dataa['house_year'].apply(lambda x: 1870 if x<1870 else x )
dataa['house_year'].fillna(value=dataa.groupby('Regionname')['house_year'].transform('median'),inplace=True)

# distance 區間合併 27,32,35,38
dataa['Distance'] = dataa['Distance'].apply(lambda x: (x//3)*3 if x>=27 else x )

"""
# 'BuildingArea'  'Landsize'  刪除
ddta = dataa.dropna(axis=0,subset=('BuildingArea','Landsize'))
fig,ax=plt.subplots(figsize=(12,8))
sns.lineplot(x='BuildingArea',y='Landsize',ax=ax,data=ddta)

# print(dataa.groupby('Regionname')['Regionname'].agg('count'))

fig,ax=plt.subplots(figsize=(18,10))
sns.relplot(x= 'Distance',y= 'CouncilArea',hue='month' ,col= 'Method',row='year',ax=ax ,data=dataa)
plt.show()
# 'CouncilArea'中'Hume'之後僅在2017年出現，且'Distance'在20之內
# 'CouncilArea'與'Distance'似乎存在正相關

fig,ax=plt.subplots(figsize=(18,10))
sns.barplot('CouncilArea','Distance',data=dataa)
plt.show()

check = pd.crosstab(dataa.CouncilArea,dataa.Regionname,margins=True)
sns.heatmap(check.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 留'CouncilArea' 刪除 'Distance' 'Regionname'

fig,ax=plt.subplots(figsize=(18,10))
sns.relplot(x= 'Distance',y= 'month',hue='year',row='Method',ax=ax ,data=dataa)
plt.show()
# type 沒有差異
# 2017年在10 11 12 月沒有銷售
# 刪除年  年差異或許存在問題
""" 
dataa = pd.merge(dataa,y,right_index=True,left_index=True)
dataa = dataa[(dataa.Rooms<7)]
dataa.Price = dataa.Price /10000
data_last = dataa[['Rooms', 'Type', 'Method', 'SellerG', 'Car', 'CouncilArea', 'month', 'house_year','Price']]

# type 使用 one hotEncoder
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown='ignore',sparse=False).fit_transform(data_last[['Type']])
colnames = ['Type_' + s for s in sorted(list(set(data_last['Type'])))]
onehotdata = pd.DataFrame(onehot,columns=colnames,index=data_last.index)

# SellerG 使用 Frequency Encoding
FEcod = data_last.SellerG.value_counts()
FEcheck = { FEcod.index[i] :FEcod.values[i]  for i in range(len(FEcod))  }
SellerG = pd.DataFrame(data_last['SellerG'].map(FEcheck) )
SellerG.columns = ['seller_FE']

# CouncilArea 使用 Target Encoding
TEcod = data_last.groupby('CouncilArea')['Price'].agg('median')
TEcheck = { TEcod.index[i] :TEcod.values[i]  for i in range(len(TEcod))  }
CouncilArea = pd.DataFrame(data_last['CouncilArea'].map(TEcheck) )
CouncilArea.columns = ['CouncilArea_TE']

# Method 使用 LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelencod = LabelEncoder().fit_transform(data_last['Method'])
labelcod = pd.DataFrame(labelencod,index=data_last.index)
labelcod.columns = ['Method_recode']

data_last = pd.concat([data_last,onehotdata,SellerG,CouncilArea,labelcod],axis=1)
data_last1 = data_last.dropna(axis=0)
y = data_last1.Price.astype('int')
x = data_last1.drop(['Price'],axis=1)
x.month = x.month.astype('int')
x1 = x.select_dtypes(exclude='object')


from sklearn.model_selection import train_test_split
from sklearn import metrics

train_x, val_x, train_y, val_y = train_test_split(x1, y,test_size=0.3, random_state = 1064)

'''
# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_x,train_y)
pred_logis = model.predict(val_x)
print(metrics.accuracy_score(pred_logis,val_y))


# Support Vector Machines(Linear and radial)
from sklearn import svm
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_x,train_y)
svm_rad=model.predict(val_x)
print(metrics.accuracy_score(svm_rad,val_y))

from sklearn import svm
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_x,train_y)
svm_lin=model.predict(val_x)
print(metrics.accuracy_score(svm_lin,val_y))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y)
predRF=model.predict(val_x)
print(metrics.accuracy_score(predRF,val_y))

# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier() 
model.fit(train_x,train_y)
predKN=model.predict(val_x)
print(metrics.accuracy_score(predKN,val_y))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_x,train_y)
predNB=model.predict(val_x)
print(metrics.accuracy_score(predNB,val_y))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
predtree=model.predict(val_x)
print(metrics.accuracy_score(predtree,val_y))

fig, ax = plt.subplots(figsize=(12,8))
pd.Series(model.feature_importances_,train_x.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax)
ax.set_title('Feature Importance')
plt.show()


# cross validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold = KFold(n_splits=5)

classfer =['Support Vector Machines(radial)','Random Forest','K-Nearest Neighbours','Naive Bayes','Decision Tree']
models = [svm.SVC(kernel='rbf'),RandomForestClassifier(n_estimators=100),KNeighborsClassifier(),GaussianNB(),DecisionTreeClassifier()]

cv_mean = []
cv_std = []
for i in models:
    model = i 
    cv_result = cross_val_score(model,x1,y,cv=kfold,scoring='accuracy')
    cv_mean.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_resul = pd.DataFrame({'cv_mean':cv_mean,'cv_std':cv_std},index=classfer)


from sklearn.metrics import confusion_matrix

fig,ax=plt.subplots(3,2,figsize=(18,12))
y_pre = cross_val_predict(svm.SVC(kernel='rbf'),x1,y,cv=5)
sns.heatmap(confusion_matrix(y,y_pre),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for SVM')
plt.show()

# Voting Classifier
from sklearn.ensemble import VotingClassifier

ensem = VotingClassifier(estimators=[('K-Nearest Neighbours',KNeighborsClassifier()),('Decision Tree',DecisionTreeClassifier()),('Naive Bayes',GaussianNB())],voting='soft').fit(train_x,train_y)
cross = cross_val_score(ensem,x1,y, cv = 10,scoring = "accuracy")


# Bagging(KNN & Tree)
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5),random_state=1024,n_estimators=300)
model.fit(train_x,train_y)
predBKN=model.predict(val_x)
print(metrics.accuracy_score(predBKN,val_y))

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=1024,n_estimators=50)
model.fit(train_x,train_y)
predBT=model.predict(val_x)
print(metrics.accuracy_score(predBT,val_y))


# Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xg

ada = AdaBoostClassifier(n_estimators=100,random_state=1024,learning_rate=0.1)
cross = cross_val_score(ada,x1,y, cv = 10,scoring = "accuracy")
print('The cv score for AdaBoost is:',cross.mean())

grad = GradientBoostingClassifier(n_estimators=100,random_state=1024,learning_rate=0.1)
cross = cross_val_score(grad,x1,y, cv = 10,scoring = "accuracy")
print('The cv score for AdaBoost is:',cross.mean())

xgboost = xg.XGBClassifier(n_estimators=100,random_state=1024,learning_rate=0.1)
cross = cross_val_score(xgboost,x1,y, cv = 10,scoring = "accuracy")
print('The cv score for XGBoost is:',cross.mean())
'''