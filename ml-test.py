# -*- coding: cp936 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn import svm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model
from sys import argv
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def mkdir(path_write):
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        print('目录已存在')


base_statistics = np.delete(pd.read_csv('I:/Conventional-triaxial-test-ppp/Python data/statistics-information.csv').values, 0, axis=1)
micro_structure = np.delete(pd.read_csv('I:/Conventional-triaxial-test-ppp/Python data/micro-structure-statistics-information.csv').values, 0, axis=1)
volume_fraction = np.delete(pd.read_csv('I:/Conventional-triaxial-test-ppp/Python data/volume-fraction.csv').values, 0, axis=1)
# statistics_array
# 1~12列：微观结构的宏观统计量
# 13列：体积分数
# 14~114列：不同级配的质量分布
# 115、116、118、119列：IG、摩擦系数、颗粒杨氏模量、围压
# 120、121、122、123列：G_0、G_1、G_2、G_3
statistics_array = np.hstack((micro_structure, volume_fraction, base_statistics))
for i in range(3):
    print(i)
    seed_value = 10000 + i
    np.random.seed(seed_value)
    np.random.shuffle(statistics_array)
    macroscopic_statistics_of_microstructure = statistics_array[:, range(0, 12)]
    DEM_parameter = statistics_array[:, [12, 114, 115, 117, 118]]
    # 将微观结构统计量与围压、摩擦系数、颗粒杨氏模量合并
    macroscopic_statistics_of_microstructure_and_some_DEM_parameter = statistics_array[:, list(range(0, 12))+[115, 117, 118]]
    G_0 = statistics_array[:, 119]
    G_1 = statistics_array[:, 120]
    G_2 = statistics_array[:, 121]
    G_3 = statistics_array[:, 122]
    feature_list = [macroscopic_statistics_of_microstructure,
                    DEM_parameter,
                    macroscopic_statistics_of_microstructure_and_some_DEM_parameter]
    G_list = [G_0, G_1, G_2, G_3]
    for j, feature in enumerate(feature_list):
        print(j)
        for k, G in enumerate(G_list):
            print(k)
            for train_size in range(100, 4100, 100):
                print(train_size)
                model = xgb.Booster(model_file='I:/Conventional-triaxial-test-ppp/Python data/model/xgb-'
                                    + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(train_size) + '.model')
                train_dataset_predict = model.predict(xgb.DMatrix(feature[0:train_size]))
                test_dataset_predict = model.predict(xgb.DMatrix(feature[train_size:]))
                train_dataset_score = r2_score(G[0:train_size], train_dataset_predict)
                test_dataset_score = r2_score(G[train_size:], test_dataset_predict)
                atomfile = open('I:/Conventional-triaxial-test-ppp/Python data/ml-score/score-'
                                + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(train_size) + '.txt', 'w')
                atomfile.write(str(train_dataset_score))
                atomfile.write(' ')
                atomfile.write(str(test_dataset_score))
                atomfile.close()
