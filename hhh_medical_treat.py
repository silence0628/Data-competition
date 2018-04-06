# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:18:38 2018

@author: Administrator
"""

# Parameters
FUDGE_FACTOR = 0.985  # Multiply forecasts by this
XGB_WEIGHT = 0.3200
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.0620
NN_WEIGHT = 0.0800
CAT_WEIGHT=0.4000
XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models
BASELINE_PRED = 5.631925   # Baseline based on mean of training data, per Oleg
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout, BatchNormalization
#from keras.layers.advanced_activations import PReLU
#from keras.layers.noise import GaussianDropout
#from keras.optimizers import Adam
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Imputer
#from catboost import CatBoostRegressor
#from tqdm import tqdm
#from dateutil.parser import parse
###### READ IN RAW DATA
#import xgboost as xgb
#from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LinearRegression
#import datetime as dt
#%%
import numpy as np
import pandas as pd
import lightgbm as lgb
#import gc  # 内存垃圾回收机制模块
import random
from dateutil.parser import parse  #  统一转换日期格式

#%%
print( "\n读取文件数据······")
data_path = 'E:\The_most_powerful_laboratory\Tianchi_bigData\AI_diabetes_prediction\The_program_and_Result\Tianchi_data\\'

train = pd.read_csv(data_path+'d_train_20180102.csv')  
test = pd.read_csv(data_path+'d_test_A_20180102.csv') 

#%%
print( "\n为 LightGBM 进行 数据处理工作······" )
def make_feature_lightgbm(train,test):
    """将train 和 test中的id列的值给复制出来，赋值给 train_id,test_id"""
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    """将train 和test 两个dataframe进行拼接起来"""
    data = pd.concat([train,test])
    """将原始数据中的性别“男”替换为0，“女” 替换为1，缺失替换为0，数据文件在“dataset”文件夹下"""
    data['性别'] = data['性别'].map({'男':1,'女':0})
    """将日期转换为 天数，方便后续计算，或者是删除 """
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days  
    """按列提取均值，并填入NAN位置，inplace=True表示不显示处理结果，False为显示处理结果"""
    data.fillna(data.median(axis=0),inplace=True)
    """改变格式，float64-->> float32"""
    for c, dtype in zip(data.columns, data.dtypes):
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)
    """
    上面的操作都是为了对数据进行处理，之前合并数据是为了简化操作
    重新切割 data，分为 训练特征和测试特征
    """
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    
    return train_feat,test_feat

df_train,df_test= make_feature_lightgbm(train,test)

#%%
"""将训练数据中的id 和 血糖 两列删除掉，赋值给 x_train"""
x_train = df_train.drop(['id', '血糖'], axis=1)  
"""y_train 训练标签"""
y_train = df_train['血糖'].values  # array([6.06, 5.39, 5.59, ..., 5.24, 6.37, 6.  ], dtype=float32)
#print(x_train.shape, y_train.shape)  #(5642, 40) (5642,)
"""列出表格中的所有标题栏"""
train_columns = x_train.columns  

"""
   x_train.dtypes=float32
   x_train.dtypes == object   false
   下面这个循环只是用来检查是否存在columns中存在非object的名称
"""
for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)
    
"""Python垃圾回收机制:gc模块  解决内存泄露问题"""
#del df_train; gc.collect()

"""change x_train 的type 由 DataFrame 变为 float32 """
x_train = x_train.values.astype(np.float32, copy=False) 
"""将数据集导入lgb函数当中，包括训练数据，以及对应的标签"""
d_train = lgb.Dataset(x_train, label=y_train)
#%%
"""RUN LIGHTGBM"""
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

"""np.random.seed(0)的作用：作用：使得随机数据可预测"""
np.random.seed(0)
random.seed(0)
print("\nFitting LightGBM model ...")

clf = lgb.train(params, d_train, 1000)

"""内存泄露，垃圾回收"""
#del d_train; gc.collect()
#del x_train; gc.collect()

print("\n准备基于LightGBM的预测 ...")
print("  准备测试数据 x_test...")
x_test = df_test.drop(['id','血糖'], axis=1)
print("\n开始进行 LightGBM 预测...")
p_test = clf.predict(x_test)
#%%
"""内存泄露，垃圾回收"""
#del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test))
result_path = 'E:\The_most_powerful_laboratory\Tianchi_bigData\AI_diabetes_prediction\The_program_and_Result\\'
pd.DataFrame(p_test).to_csv(result_path+'other_treat2_pm_lgb.csv',header=None,index=False)