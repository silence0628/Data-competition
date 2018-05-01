# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:46:49 2018

@author: Administrator
"""

import pandas as pd

titanic = pd.read_csv('E:\The_most_powerful_laboratory\Machine_learning04\\titanic\\train.csv')

X = titanic[['Pclass','Age','Sex']]
y = titanic['Survived']
X['Age'].fillna(X['Age'].mean(),inplace=True)
# 数据分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state = 33)

# 使用特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient='record'))
#%%
"""s使用单一决策树模型进行模型训练以及预测分析"""
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)


"""使用随机森林分类器进行模型训练以及预测分析"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)


"""使用梯度提升决策树进行集成模型的训练以及预测分析"""
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)


#%%
from sklearn.metrics import classification_report

print('The accuracy of 单一决策树模型 is :',dtc.score(X_test,y_test))
print(classification_report(dtc_y_pred,y_test,target_names=['died','survived']))

print('The accuracy of 随机森林分类器 is :',rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test,target_names=['died','survived']))

print('The accuracy of 集成模型之梯度提升决策树 is :',gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test,target_names=['died','survived']))
