# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

data=pd.read_csv('bf3_data_2022_01_07.csv')
data.head()

data["average"]=data["CO"]/data["CO2"]

data.head()

data.shape

# select numerical columns
df_numeric = data.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
# select non-numeric columns
df_non_numeric = data.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values

df_numeric.head()

df_non_numeric.head()

data.describe()

data.isnull().sum()

data.isnull()

data.isnull().sum().sum()

data=data.drop("DATE_TIME",axis=1)

cols=data.columns
cols.size

# null_cols=[]
# for i in cols:
#     if data[i].isnull().sum()>0:
#         null_cols.append(i)
# data=data.drop(null_cols,axis=1,inplace=True)

data.shape

df=data.drop("SKIN_TEMP_AVG",axis=1)

df.corr()

df1=df.drop(["CB_FLOW","CB_PRESS","O2_FLOW","PCI","HB_TEMP","HB_PRESS","TOP_PRESS"],axis=1)

data.dtypes

df1.info()

df1.describe()

df1.isnull().sum()

df1['CB_TEMP'] = df1['CB_TEMP'].fillna(df1['CB_TEMP'].mean())

df1['STEAM_FLOW'] = df1['STEAM_FLOW'].fillna(df1['STEAM_FLOW'].mean())

df1['STEAM_TEMP'] = df1['STEAM_TEMP'].fillna(df1['STEAM_TEMP'].mean())

df1['STEAM_PRESS'] = df1['STEAM_PRESS'].fillna(df1['STEAM_PRESS'].mean())

df1['O2_PRESS'] = df1['O2_PRESS'].fillna(df1['O2_PRESS'].mean())

df1['O2_PER'] = df1['O2_PER'].fillna(df1['O2_PER'].mean())

df1['ATM_HUMID'] = df1['ATM_HUMID'].fillna(df1['ATM_HUMID'].std())

df1['TOP_TEMP1'] = df1['TOP_TEMP1'].fillna(df1['TOP_TEMP1'].mean())

df1['TOP_TEMP2'] = df1['TOP_TEMP2'].fillna(df1['TOP_TEMP2'].mean())

df1['TOP_TEMP3'] = df1['TOP_TEMP3'].fillna(df1['TOP_TEMP3'].median())

df1['TOP_TEMP4'] = df1['TOP_TEMP4'].fillna(df1['TOP_TEMP4'].mean())

df1['TOP_SPRAY'] = df1['TOP_SPRAY'].fillna(df1['TOP_SPRAY'].std())

df1['TOP_TEMP'] = df1['TOP_TEMP'].fillna(df1['TOP_TEMP'].mean())

df1['TOP_PRESS_1'] = df1['TOP_PRESS_1'].fillna(df1['TOP_PRESS_1'].mean())

df1['CO'] = df1['CO'].fillna(df1['CO'].mean())

df1['CO2'] = df1['CO2'].fillna(df1['CO2'].mean())

df1['H2'] = df1['H2'].fillna(df1['H2'].mean())

df1['average'] = df1['average'].fillna(df1['average'].mean())

data.info()

df1.isnull().sum()

df1.shape

import sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

X = df1.iloc[:,1:18].values  #features
y = df1.iloc[:,17:18].values  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape

rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
prediction = rf.predict(X_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print("mean square error",mse)
print("root mean square error", rmse)

prediction

mape = np.mean(np.abs((y_test - prediction) / np.abs(y_test)))
print('Mean Absolute Percentage Error:', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))