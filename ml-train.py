# -*- coding: cp936 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sys import argv
import joblib


def mkdir(path_write):
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        print('目录已存在')


def xgboost_train(x_train, y_train):
    parameters = {'max_depth': [2, 4, 6, 8],
                  'n_estimators': [50, 150, 250, 350],
                  'min_child_weight': [2, 4]}
    xgboost_model = XGBClassifier(learning_rate=0.01,
                                  n_estimators=1000,
                                  max_depth=5,
                                  min_child_weight=1,
                                  objective='binary:logistic',
                                  gamma=0.1,
                                  subsample=0.75,
                                  colsample_bytree=0.75,
                                  reg_alpha=0.01,
                                  reg_lambda=2,
                                  scale_pos_weight=1,
                                  seed=520,
                                  missing=None)
    gsearch = GridSearchCV(xgboost_model, param_grid=parameters, scoring='roc_auc', cv=5, verbose=100)
    gsearch.fit(x_train, y_train)
    best = gsearch.best_estimator_.get_params()
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'max_depth': int(best['max_depth']),  # 最大深度
              'eta': 0.01,  # 学习率
              'subsample': 0.75,  # 采样数
              'colsample_bytree': 0.75,  # 样本列采样
              'min_child_weight': int(best['min_child_weight']),  # 终点节点最小样本占比的和
              'silent': 0,  # 是否显示
              'gamma': 0.1,  # 是否后剪枝
              'alpha': 0.01,  # L1 正则化
              'lambda': 2,  # L2 正则化
              'seed': 520  # 随机种子
              }
    data_train = xgb.DMatrix(x_train, y_train)
    model = xgb.train(params, data_train, num_boost_round=int(best['n_estimators']))
    return model


def svc_train(x_train, y_train):
    clf_search = svm.LinearSVC()
    parameters = {'C': [0.001, 0.01, 0.1, 10, 100, 1000]}
    gsearch = GridSearchCV(clf_search, param_grid=parameters, scoring='roc_auc', cv=5, verbose=100)
    gsearch.fit(x_train, y_train)
    best = gsearch.best_estimator_.get_params()
    clf = svm.LinearSVC(C=float(best['C']))
    clf.fit(x_train, y_train)
    return clf


def MLP_train(x_train, y_train):
    clf_search = MLPClassifier()
    parameters = {'hidden_layer_sizes': [(50), (50, 50), (100), (100, 100), (150), (150, 150)],
                  'alpha': [0.0001, 0.001, 0.01],
                  'batch_size': [100, 300, 500],
                  'learning_rate_init': [0.0001, 0.001, 0.01],
                  'tol': [1e-5, 1e-4, 1e-3],
                  }
    gsearch = GridSearchCV(clf_search, param_grid=parameters, scoring='roc_auc', cv=5, verbose=100)
    gsearch.fit(x_train, y_train)
    best = gsearch.best_estimator_.get_params()
    clf = MLPClassifier(hidden_layer_sizes=best['hidden_layer_sizes'],
                        alpha=best['alpha'],
                        batch_size=best['batch_size'],
                        learning_rate_init=best['learning_rate_init'],
                        tol=best['tol'])
    clf.fit(x_train, y_train)
    return clf


def RF_train(x_train, y_train):
    clf_search = RandomForestClassifier()
    parameters = {'n_estimators': [50, 100, 150, 200],
                  'max_depth': [1, 2, 3, 4, 5],
                  'min_samples_split': [10, 30, 50, 70, 90],
                  'min_samples_leaf': [10, 30, 50]
                  }
    gsearch = GridSearchCV(clf_search, param_grid=parameters, scoring='roc_auc', cv=5, verbose=100)
    gsearch.fit(x_train, y_train)
    best = gsearch.best_estimator_.get_params()
    clf = RandomForestClassifier(n_estimators=best['n_estimators'],
                                 max_depth=best['max_depth'],
                                 min_samples_split=best['min_samples_split'],
                                 min_samples_leaf=best['min_samples_leaf'])
    clf.fit(x_train, y_train)
    return clf


system_list = ['PVC-initialfric01-loadfric001',
               'PVC-initialfric01-loadfric002',
               'PVC-initialfric01-loadfric015',
               'PVC-initialfric01-loadfric05',
               'PVC-initialfric01-loadfric1']
system = system_list[int(argv[1])]
time_windows_list = [1]
threshold_1_list = [99.5]
percentile = threshold_1_list[0]

mkdir('/scratch/ybzhang/Conventional-triaxial-test/' + system + '/' + 'model/' + str(threshold_1_list[0]))

for threshold_1 in threshold_1_list:
    for a in range(len(time_windows_list)):
        time_windows = time_windows_list[a]
        dataset = 'dataset_' + str(time_windows) + '.txt'
        feature_number_use = 90
        feature_number_all = 90
        feature_delete = []
        file = open('/scratch/ybzhang/Conventional-triaxial-test/' + system + '/' +
                    'dataset/' + str(percentile) + '/' + dataset, 'r')
        lines = file.readlines()
        file.close()

        x_data = np.zeros(shape=[len(lines), feature_number_use])
        for i in range(len(lines)):
            m = 0
            for j in range(feature_number_all):
                if j in feature_delete:
                    continue
                x_data[i][m] = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[i])[j])
                m += 1

        y_data = np.zeros(shape=[len(lines), ])
        for i in range(len(lines)):
            y_data[i] = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[i])[feature_number_all])

        plastic_judge = y_data == 1
        elastic_judge = y_data == 0

        plastic_data_x = x_data[plastic_judge]
        plastic_data_y = y_data[plastic_judge]

        elastic_data_x = x_data[elastic_judge]
        elastic_data_y = y_data[elastic_judge]

        for j in range(10):
            random_array = np.arange(plastic_data_x.shape[0])
            np.random.shuffle(random_array)
            plastic_data_x = plastic_data_x[random_array[0:5000]]
            plastic_data_y = plastic_data_y[random_array[0:5000]]
            elastic_data_x = elastic_data_x[random_array[0:5000]]
            elastic_data_y = elastic_data_y[random_array[0:5000]]
            x_data = np.vstack((plastic_data_x, elastic_data_x))
            y_data = np.concatenate((plastic_data_y, elastic_data_y))
            x_train, y_train = x_data, y_data
            xgboost_model = xgboost_train(x_train, y_train)
            svc_model = svc_train(x_train, y_train)
            MLP_model = MLP_train(x_train, y_train)
            RF_model = RF_train(x_train, y_train)
            xgboost_model.save_model('/scratch/ybzhang/Conventional-triaxial-test/' + system + '/model/' + str(threshold_1) + '/' + 'model' + '-' + str(time_windows) + '-' + str(j) + '.model')
            joblib.dump(svc_model, 'svc.model')
            joblib.dump(MLP_model, 'mlp.model')
            joblib.dump(RF_model, 'rf.model')