'''
HR Analytics: Job Change of Data Scientists

Note:

多數特稱為類別，且部分為特徵有高比例為唯一值

特徵 :

enrollee_id : 唯一ID
city: 所在城市
city_ development _index : 城市發展指數     (連續)
gender: 性別
relevent_experience: 相關經驗
enrolled_university: 大學課程類型
education_level: 教育程度
major_discipline :畢業學科
experience: 工作經驗年數
company_size: 當前公司員工數
company_type : 當前雇主類型
last_new_job: 上份工作與當前工作的年數差異
training_hours: 培訓時數                    (連續)
target: 0 不尋找新工作, 1 尋找工作

定義問題 :
了解特徵與 '學員願意到新公司上班而進行培訓' 之間的關係，意即具有何者特徵的學員，較有可能是為了找到新工作

'''

from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn import model_selection
from sklearn import feature_selection
from xgboost import XGBClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('aug_train.csv')

# print(train.info())
'''
# 檢視資料

for col in train.columns:
    if train[col].dtypes == 'object':
        print(list(set(train[col])))
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        train[col].value_counts().plot(kind='bar', ax=ax[0])
        plt.xticks(rotation=90)
        sns.countplot(x=col, hue='target', ax=ax[1], data=train)
        plt.show()
    else:
        continue

# city: 過度集中，'city_21'較容易找新工作

# experience: 大於20較多 (間隔5?)
# ['10', '17', nan, '14', '11', '18', '16', '4', '1', '<1', '9', '15',
#    '12', '5', '2', '>20', '8', '7', '3', '6', '20', '13', '19']

# company_size: 較分散  (小公司跳槽大公司?)
# ['100-500', '10/49', '1000-4999', nan, '<10', '5000-9999', '50-99', '10000+', '500-999']

# education_level: Graduate較多，且較容易找新工作
# ['High School', 'Masters', 'Phd', nan, 'Graduate', 'Primary School']

# major_discipline: 過度集中STEM，且較容易找新工作
# ['Humanities', nan, 'Other', 'Arts', 'Business Degree', 'No Major', 'STEM']

# company_type: 過度集中Pvt Ltd
# ['Pvt Ltd', 'NGO', 'Early Stage Startup', nan, 'Public Sector', 'Funded Startup', 'Other']

# last_new_job: 1年較多，never較容易找新工作
# ['1', '2', nan, '3', '4', '>4', 'never']

# gender: 多為男性，男女在target部分較無差異
# ['Other', 'Male', nan, 'Female']

# relevent_experience: 多具有相關經驗，且較不容易找新工作
# ['Has relevent experience', 'No relevent experience']

# enrolled_university: 多不修讀相關課程，part time 較不容易找新工作
# ['Part time course', 'Full time course', nan, 'no_enrollment']

# train['city_development_index'].value_counts().plot(kind='bar')
# city_development_index: 和 city 分布相似

# train['training_hours'].value_counts().plot(kind='bar')
# training_hours: 較分散(跟城市有關?)

# train['target'].value_counts().plot(kind='bar')
# target: 0約為1的3倍(不尋找工作? 找不到工作?)


檢查特徵之遺失值比例

print(train.columns[train.isnull().any().values == True])
# 具有遺失值 : gender, enrolled_university, education_level, major_discipline, experience, company_size, company_type, last_new_job
for col in ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']:
    print(pd.crosstab(train[col].fillna('missing'),
                      train['target'].fillna('missing'), normalize=True, dropna=True, margins=True))

'''
# 資料清理
# 具有順序的編號
clean_nums = {
    'enrolled_university': {'Full time course': 3, 'Part time course': 2, 'no_enrollment': 1},
    'relevent_experience': {'Has relevent experience': 1, 'No relevent experience': 0},
    'education_level': {'Masters': 4, 'Phd': 5, 'High School': 2, 'Primary School': 1, 'Graduate': 3},
    'company_size': {'<10': 0, '10/49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7},
    'last_new_job': {'>4': 5, 'never': 0},
    'experience': {'>20': '25', '<1': '0'}
}

train = train.replace(clean_nums)

# 將city 編碼
label = LabelEncoder()
train['city'] = label.fit_transform(train['city'])

# major_discipline 與 company_type 刪除
train = train.drop(['major_discipline', 'company_type'], axis=1)

# NA 處理
# print(train['gender'].unique())
train['gender'] = train['gender'].fillna('Other')

# 假設NA代表在職中，且無經驗，教育程度為未受過教育，因此為0
train[['last_new_job', 'experience', 'education_level']] = train[[
    'last_new_job', 'experience', 'education_level']].fillna(0)

# 假設NA代表輟學，因此同屬於'no_enrollment'
train['enrolled_university'] = train['enrolled_university'].fillna(1)

# company_size 則依據 'education_level' 進行插補
train['company_size'] = train['company_size'].fillna(train.groupby(
    'education_level')['company_size'].transform('mean').round())

# gender 用 one-hot，照字母順序處理列名稱
onehot = OneHotEncoder()
train[['Female', 'Male', 'Other']] = onehot.fit_transform(
    train[['gender']]).toarray()

# training_hours 和 experience 分為5個級距
train['training_hours_cut'] = pd.qcut(
    train['training_hours'], 5, labels=[1, 2, 3, 4, 5])
train['experience_cut'] = pd.qcut(
    train['experience'].astype(int), 5, labels=[1, 2, 3, 4, 5])

for col in train.columns:
    if col not in ['gender', 'city_development_index']:
        train[col] = train[col].astype('int64')
'''
# 探索性分析

for col in train.columns:
    if (len(set(train[col])) < 20) & (train[col].dtype == 'object'):
        print('correlation by :', col)
        print(pd.crosstab(train[col], train['target'],
              normalize=True, dropna=True, margins=True))
        print('-'*20, '\n')

# 連續 city_development_index, experience, last_new_job, training_hours
# 類別 gender ,enrolled_university ,education_level ,relevent_experience ,company_size

fig, ax = plt.subplots(2, 4, figsize=(16, 12))
sns.boxplot(x='city_development_index', hue='target', data=train,
              ax=ax[0, 0]).set(title='city_development_index')
sns.boxplot(x='experience', hue='target', data=train,
              ax=ax[0, 1]).set(title='experience')
sns.boxplot(x='last_new_job', hue='target', data=train,
              ax=ax[0, 2]).set(title='last_new_job')
sns.boxplot(x='training_hours', hue='target', data=train,
              ax=ax[0, 3]).set(title='training_hours')
sns.histplot(x='city_development_index', hue='target', data=train,
             ax=ax[1, 0]).set(title='city_development_index')
sns.histplot(x='experience', hue='target', data=train,
             ax=ax[1, 1]).set(title='experience')
sns.histplot(x='last_new_job', hue='target', data=train,
             ax=ax[1, 2]).set(title='last_new_job')
sns.histplot(x='training_hours', hue='target', data=train,
             ax=ax[1, 3]).set(title='training_hours')
plt.show()
# print(train.query("city_development_index < 0.45"))
'''
y_re = train[['target']]
x_test = train.drop('target', axis=1)

# 將資料標準化
num_col = ['experience', 'company_size', 'last_new_job', 'training_hours_cut',
           'experience_cut', 'education_level', 'enrolled_university']
skew_col = ['city_development_index', 'training_hours']


norm_num = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)
skew_num = Pipeline(
    steps=[
        ('Quantile', QuantileTransformer(output_distribution='normal')),
        ('scaler', StandardScaler())
    ]
)
preprocess = ColumnTransformer(
    [
        ('norm', norm_num, num_col),
        ('skew', skew_num, skew_col)
    ]
)

X = preprocess.fit_transform(x_test)
xtest = pd.DataFrame(X, columns=num_col +
                     skew_col).join(x_test.drop(num_col+skew_col, axis=1))


x_re = xtest[['city_development_index', 'relevent_experience', 'enrolled_university', 'education_level',
              'experience', 'company_size', 'last_new_job', 'training_hours',
              'Female', 'Male', 'Other']]


# Undersampling
undersampler = RandomUnderSampler(random_state=42)
x, y = RandomUnderSampler(random_state=42).fit_resample(x_re, y_re)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=432)

'''
model = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # GLM
    linear_model.SGDClassifier(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost
    XGBClassifier()
]

model_result_org = pd.DataFrame(
    columns=['model', 'parameters', 'cv_accuracy', 'mae', 'mse', 'f1'])
row_index = 0

for ml in model:
    print(ml.__class__.__name__)
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    model_result_org.loc[row_index, 'model'] = ml.__class__.__name__
    model_result_org.loc[row_index, 'parameters'] = str(ml.get_params())
    model_result_org.loc[row_index, 'mae'] = mean_absolute_error(y_test, y_pred)
    model_result_org.loc[row_index, 'mse'] = mean_squared_error(y_test, y_pred)
    model_result_org.loc[row_index, 'f1'] = f1_score(y_test, y_pred)
    model_result_org.loc[row_index, 'cv_accuracy'] = cross_val_score(
        ml, x_train, y_train, cv=10, scoring='accuracy').mean()
    row_index += 1

# 第一次試做 : f1_score最好為 GaussianNB  之 0.513875
# 第二次試做 : f1_score最好為 GradientBoosting 為 0.751387  (部分資料標準化 + 欠採樣 + Quantile Transform)
# 第三次試做 : f1_score最好為 GradientBoosting  (加入cross_val做確認)

# Hyper-Parameters

model = ensemble.GradientBoostingClassifier()
model.fit(x_train, y_train)

result_cv = cross_val_score(model, x_train, y_train,
                            cv=10, scoring='accuracy').mean()

param = {'n_estimators': [40, 50, 60], 'learning_rate':[0.5, 0.1, 0.05], 'max_features': ['sqrt', 'log2', 8, None], 'max_depth':[3, 6, 9]}
new_model = GridSearchCV(model, param, cv=10, scoring='accuracy')
new_model.fit(x_train, y_train)

# feature selection

model = ensemble.GradientBoostingClassifier()
model_rfecv = RFECV(estimator=model, step=1, cv=10)
model_rfecv.fit(x_train, y_train)
column_rfe = x_train.columns.values[model_rfecv.get_support()]

result_cv = cross_val_score(model, x_train[column_rfe], y_train,
                            cv=10, scoring='accuracy').mean()

param = {'n_estimators': [40, 50, 60], 'learning_rate': [
    0.5, 0.1, 0.05], 'max_features': ['sqrt', 'log2', 8, None], 'max_depth': [3, 6, 9]}
new_model = GridSearchCV(model, param, cv=10, scoring='accuracy')
new_model.fit(x_train[column_rfe], y_train)

print(result_cv)
print(column_rfe)
print(new_model.score(x_train[column_rfe], y_train))
print(new_model.best_params_)

# ['city_development_index' 'relevent_experience' 'enrolled_university' 'education_level' 'experience' 'company_size' 'last_new_job' 'training_hours' 'Male']
# {'learning_rate': 0.1, 'max_depth': 3, 'max_features': None, 'n_estimators': 50}
'''

test = pd.read_csv('aug_test.csv')

test = test.replace(clean_nums)
test['city'] = label.fit_transform(test['city'])
test = test.drop(['major_discipline', 'company_type'], axis=1)
test['gender'] = test['gender'].fillna('Other')
test[['last_new_job', 'experience', 'education_level']] = test[[
    'last_new_job', 'experience', 'education_level']].fillna(0)
test['enrolled_university'] = test['enrolled_university'].fillna(1)
test['company_size'] = test['company_size'].fillna(test.groupby(
    'education_level')['company_size'].transform('mean').round())
test[['Female', 'Male', 'Other']] = onehot.fit_transform(
    test[['gender']]).toarray()
test['training_hours_cut'] = pd.qcut(
    test['training_hours'], 5, labels=[1, 2, 3, 4, 5])
test['experience_cut'] = pd.qcut(
    test['experience'].astype(int), 5, labels=[1, 2, 3, 4, 5])

for col in test.columns:
    if col not in ['gender', 'city_development_index']:
        test[col] = test[col].astype('int64')

X = preprocess.fit_transform(test)

xtest = pd.DataFrame(X, columns=num_col +
                     skew_col).join(test.drop(num_col+skew_col, axis=1))
column_new = ['city_development_index', 'relevent_experience', 'enrolled_university',
              'education_level', 'experience', 'company_size', 'last_new_job', 'training_hours', 'Male']
x_re = xtest[column_new]

model = ensemble.GradientBoostingClassifier(
    n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(x_train[column_new], y_train)
test["target"] = model.predict(x_re)
