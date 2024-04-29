# %% [markdown]
# 
# # 因子实战 第二集 如何根据历史数据预测股价...?
# ## 从调包侠到因子学徒...?
# 
# ---
# 
# ##### “历史是一面镜子，你可以从中看到未来” - 埃德蒙·伯克
# 
# ### [@大导演哈罗德](mailto:zhongfangyuan@link.cuhk.edu.cn)
# ### 香港中文大学(深圳) 
# ### 金融工程 
# ### 大四
# ### [Bilibili](https://space.bilibili.com/629573485)
# 
# ---
# 
# ## 注意：本篇内容适合具有一定的机器学习基础的同学学习
# 
# ---
# 
# ## 作为一个人工智能互联网时代，知识的获取已经变得非常容易，但是知识的应用却是一个非常难的事情，因为知识的应用需要大量的实践。
# ## 本篇内容将会带领大家从零开始，一步一步的完成一个因子的构建，希望大家能够在这个过程中学到一些东西。
# 
# ![image.png](attachment:image.png)
# 
# 

# %% [markdown]
# 

# %% [markdown]
# ## 先当一个 调包侠，再当一个 调包侠，最后当一个... 调包侠。
# 
# #### enough is enough
# 

# %%
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import os
import re
from sklearn.model_selection import train_test_split #随机划分训练集和测试集
from sklearn.model_selection import TimeSeriesSplit #根据时间序列划分训练集和测试集
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_hastie_10_2

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN

from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt

# %% [markdown]
# ## 读取市场的宏观数据
# 
# 原因：市场的宏观数据，是影响股票的重要因素，所以需要读取市场的宏观数据。并且中国市场的宏观数据由于政策的影响，以及文化属性，有很多的特殊性，所以需要对市场的宏观数据进行分析。很多时候宏观因子对股票的整体影响，远远大于股票的个体因子。
# 
# 另外值得注意的是，市场的宏观数据，是影响股票的重要因素，但是市场的宏观数据，也是受到股票的影响的。所以市场的宏观数据，和股票的个体因子，是相互影响的。并且市场的宏观数据，和股票的个体因子，是相互影响的，这个影响是双向的。所以市场的宏观数据，和股票的个体因子，是相互影响的，这个影响是双向的，这个影响是双向的，这个影响是双向的，重要的事情说三遍。
# 

# %%
df_new_data = pd.read_excel('宏观数据.xlsx')
df_new_data = df_new_data.drop(df_new_data.index[0:2])

# %%
df_new_data

# %%
df_x = pd.read_csv('X.csv')
df_x

# %%
df_y = pd.read_csv('y.csv')
df_y

# %% [markdown]
# ## 加上宏观因子之后
# 
# ### // 步骤:
# #### // 1. 读取市场的宏观数据
# #### // 2. 根据日期merge到因子数据中
# #### // （课后大家可以尝试一下，这里的考验的地方是对于merge的掌握，而且很多时候日常数据很容易出现二次读取，要有耐心，要细心，注意对齐。）
# #### 提示 
# 
# ```python 
# df_merge = pd.merge(df, df_new_data, on='Trddt', how='left')
# ```
# 
# 
# 
# 
# 

# %%
df = pd.read_csv('X_new.csv')
df

# %%
X = df[['ILLIQ', 'PE', 'PB', 'PS', 'CirculatedMarketValue', 'ChangeRatio', 'Liquidility','Cnshrtrdtl', 'Cnvaltrdtl', 'Cdretwdos', "Cdretwdtl"]]


# %%
df = pd.read_csv('y_new.csv')
y = df['Dretwd']

# %%
y_shift_one = y.shift(-1)

# %%
y_shift_one.fillna(method='ffill', inplace=True)

# %% [markdown]
# ## 再看看因子的数据

# %%
X

# %%
y_shift_one

# %% [markdown]
# # 时间序列模型：
# 
# - ARIMA (Autoregressive Integrated Moving Average) 模型
# - GARCH (Generalized Autoregressive Conditional Heteroskedasticity) 模型
# - LSTM (Long Short-Term Memory) 神经网络
# 
# # 机器学习模型：
# 
# - 线性回归
# - 随机森林
# - 支持向量机 (SVM)
# - XGBoost 或 LightGBM 梯度提升树模型
# 
# # 深度学习模型：
# 
# - 卷积神经网络 (CNN) 用于处理股票图表数据
# - 循环神经网络 (RNN) 或 LSTM 用于序列数据预测
# - 注意力机制 (Attention) 用于处理序列数据
# 

# %%
! pip3.9 install xgboost

# %% [markdown]
# ![image.png](attachment:image.png)

# %%


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings


# %% [markdown]
# ### 关于时间序列数据处理的一些思考

# %%
# 没有人是先知
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, ？？？random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y_shift_one, test_size=0.2, shuffle=False)


# 高级的方法：Time Series Cross Validation: 
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, test_index in tscv.split(y):
    # y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # arima_model = ARIMA(y_train, order=(5,1,0))
    # arima_model_fit = arima_model.fit()
    # y_pred_arima = arima_model_fit.forecast(steps=len(y_test))
    # rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
    # rmse_arima_scores.append(rmse_arima)
    # average_rmse_arima = np.mean(rmse_arima_scores)


# %% [markdown]
# ### 线性回归模型

# %%
# Linear Regression Model
# Train and evaluate a simple linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
win_ratio_lr = np.mean(np.sign(y_pred_lr) == np.sign(y_test.values))

print('Linear Regression RMSE: ', rmse_lr)
print('Linear Regression Win Ratio: ', win_ratio_lr)

# %%
# 随机森林
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

# %%
y_rf_pred = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf_pred))
win_ratio_rf = np.mean(np.sign(y_rf_pred) == np.sign(y_test.values))
print('Random Forest RMSE: ', rmse_rf)
print('Random Forest Win Ratio: ', win_ratio_rf)

# %%
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

feature_importances = rf_model.feature_importances_

# Plotting feature importances for visualization
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('因子重要性')
plt.ylabel('因子')
plt.title('随机森林模型因子重要性')
plt.show()

# %%
parameters = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(rf_model, parameters, cv=3)
rf_grid.fit(X_train, y_train.values.ravel())


# %%
y_rf_grid_pred = rf_grid.predict(X_test)

# %%
# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(rf_grid.best_params_))
print("Best score is {}".format(rf_grid.best_score_))
print("Random Forest GRID RMSE: ", np.sqrt(mean_squared_error(y_test, rf_grid.predict(X_test))))
print("Random Forest GRID Win Ratio: ", np.mean(np.sign(rf_grid.predict(X_test)) == np.sign(y_test.values)))

# %% [markdown]
# it is actually getting worse

# %%
import pickle
filename = 'rf_grid_search_best_estimator_.sav'
pickle.dump(rf_grid.best_estimator_, open(filename, 'wb'))

# %%
# XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train.values.ravel())
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
win_ratio_xgb = np.mean(np.sign(y_pred_xgb) == np.sign(y_test.values))

# %%


print("Linear Regression - RMSE:", rmse_lr, "Win Ratio:", win_ratio_lr)
print("Random Forest - RMSE:", rmse_rf, "Win Ratio:", win_ratio_rf)
print("XGBoost - RMSE:", rmse_xgb, "Win Ratio:", win_ratio_xgb)


# %%
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_lr, label='Linear Regression')
plt.plot(y_pred_xgb, label='XGBoost')
plt.plot(y_rf_grid_pred, label='Random Forest')
plt.legend()
plt.show()


# %% [markdown]
# ## 下一步？
# 
# ### 应用：只要胜率大于50%，就可以赚钱....吗？
# ### 炼丹？

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %%
X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Fit model
lstm_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=72, validation_data=(X_test_reshaped, y_test), verbose=2, shuffle=False)

# Predict and evaluate
y_lstm_pred = lstm_model.predict(X_test_reshaped)
lstm_mse = mean_squared_error(y_test, y_lstm_pred)
lstm_rmse = mean_squared_error(y_test, y_lstm_pred, squared=False)
print(f'LSTM - Mean Squared Error: {lstm_mse}')
print(f'LSTM - Root Mean Squared Error: {lstm_rmse}')
print(f'LSTM - R^2 Score: {r2_score(y_test, y_lstm_pred)}')


# %%
# plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_lstm_pred, label='LSTM')



