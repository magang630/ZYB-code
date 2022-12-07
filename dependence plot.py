# -*- coding: cp936 -*-
import re
import numpy as np
import shap
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

columns_dict1 = ['coordination_number', 'coordination_number_min', 'coordination_number_max',
                 'coordination_number_mean', 'coordination_number_std',  # 5
                 'cellfraction', 'cellfraction_max', 'cellfraction_mean', 'anisotropic_coefficient',
                 'anisotropic_coefficient_min',  # 10
                 'anisotropic_coefficient_max', 'q2_std', 'q4', 'q4_min', 'q4_max',  # 15
                 'q4_mean', 'q4_std',
                 'q6', 'q6_min', 'q6_max',  # 20
                 'q6_mean', 'q6_std',
                 'q8', 'q8_min', 'q8_max',  # 25
                 'q8_mean', 'q8_std',
                 'q10_std',
                 'w2', 'w2_min',  # 30
                 'w2_max',
                 'w2_mean', 'w2_std',
                 'w4', 'w4_min',  # 35
                 'w4_max',
                 'w4_mean', 'w4_std',
                 'w6', 'w6_min',  # 40
                 'w6_max',
                 'w6_mean',
                 'w8', 'w8_min', 'w8_max',  # 45
                 'w10', 'w10_min', 'w10_max',
                 'w10_mean', 'w10_std']  # 50

columns_dict2 = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8',
                 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8',
                 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8',
                 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8',
                 ]  # 50

columns_dict3 = [r'$CN$', r'$CN_{\rmmin}$', r'$CN_{\rmmax}$',
                 r'$CN_{\rmmean}$', r'$CN_{\rmstd}$',
                 r'$\phi_{\rmlocal}$', r'$\phi_{\rmlocal,min}$', r'$\phi_{\rmlocal,max}$',
                 r'$\phi_{\rmlocal,mean}$', r'$\phi_{\rmlocal,std}$',
                 r'$\beta_{\rm1}^{\rm0,2}$', r'$\beta_{\rm1,mim}^{\rm0,2}$', r'$\beta_{\rm1,max}^{\rm0,2}$',
                 r'$\beta_{\rm1,mean}^{\rm0,2}$', r'$\beta_{\rm1,std}^{\rm0,2}$',
                 r'$q_{\rm4}$', r'$q_{\rm4,min}$', r'$q_{\rm4,max}$',
                 r'$q_{\rm4,mean}$', r'$q_{\rm4,std}$',
                 r'$q_{\rm6}$', r'$q_{\rm6,min}$', r'$q_{\rm6,max}$',
                 r'$q_{\rm6,mean}$', r'$q_{\rm6,std}$',
                 r'$w_{\rm4}$', r'$w_{\rm4,min}$', r'$w_{\rm4,max}$',
                 r'$w_{\rm4,mean}$', r'$w_{\rm4,std}$',
                 r'$w_{\rm6}$', r'$w_{\rm6,min}$', r'$w_{\rm6,max}$',
                 r'$w_{\rm6,mean}$', r'$w_{\rm6,std}$']

columns_dict = ['coordination_number', 'cellfraction',
                'anisotropic_coefficient',
                'q4',
                'q6',
                'w4',
                'w6']

ini_number = 0
time_windows = 100
try_list = range(0, 5)

for i in range(len(try_list)):
    print(i)
    try_now = try_list[i]

    model_name = 'model' + '-' + str(time_windows) + '-' + str(try_now) + '.model'
    model = xgb.Booster(
        model_file='J:/DATA/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/20210331/model/cluster40' + '/' + model_name)

    dataset = 'precursor' + '_' + str(time_windows) + '.txt'
    feature_number = 65
    feature_number_origin = 35
    feature_delete = [15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                      44, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    feature_use = [0, 5, 10, 20, 25, 45, 50]

    file = open('J:/DATA/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/20210331/dataset/cluster40/' + dataset, 'r')
    liness = file.readlines()
    file.close()

    if len(liness) >= 25000:
        lines = liness[0:1000]
    else:
        lines = liness

    x_data = np.zeros(shape=[len(lines), feature_number_origin])
    for x in range(len(lines)):
        m = 0
        for j in range(feature_number):
            if j in feature_delete:
                continue
            x_data[x][m] = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[x])[j])
            m += 1

    y_data = np.zeros(shape=[len(lines), ])
    for x in range(len(lines)):
        y_data[x] = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[x])[feature_number])

    x_train, y_train = x_data, y_data

    # c_dict = [columns_dict[inde[x]] for x in range(len(inde))]

    explainer = shap.TreeExplainer(model)

    # y_base = explainer.expected_value

    shap_values = explainer.shap_values(x_train)

    if try_now == 0:
        shap_values_sort = np.zeros_like(shap_values)
        for x in range(shap_values.shape[1]):
            for y in range(shap_values.shape[0]):
                shap_values_sort[y][x] = shap_values[y][x]

    else:
        shap_values_sort_single = np.zeros_like(shap_values)
        for x in range(shap_values.shape[1]):
            for y in range(shap_values.shape[0]):
                shap_values_sort_single[y][x] = shap_values[y][x]

        shap_values_sort = np.vstack((shap_values_sort, shap_values_sort_single))

    if try_now == 0:
        x_feature = np.zeros_like(x_train)
        for x in range(x_train.shape[1]):
            for y in range(x_train.shape[0]):
                x_feature[y][x] = x_train[y][x]

    else:
        x_feature_single = np.zeros_like(x_train)
        for x in range(x_train.shape[1]):
            for y in range(x_train.shape[0]):
                x_feature_single[y][x] = x_train[y][x]

        x_feature = np.vstack((x_feature, x_feature_single))

    '''
    feature_values = np.mean(np.abs(shap_values), axis=0)

    sort_id = range(0, 50)

    feature_values_sort = np.zeros_like(feature_values)

    for x in range(len(feature_values_sort)):
        feature_values_sort[inde[x]] = feature_values[x]

    feature_values_all.append(feature_values)
    '''

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=[2.7559055118110236, 2.3622047244094486], dpi=600)
legend_font = {"family": 'Arial', "size": 10}

# ax.set_yticks([-0.04, -0.02, 0.00, 0.02, 0.04, 0.06])
# ax.set_yticklabels(labels=[-0.04, -0.02, 0.00, 0.02, 0.04, 0.06], fontdict=legend_font)
# ax.set_ylabel(ylabel='afa',fontproperties='Arial', fontsize=8)
color = ['#88D3F3', '#DB6031', '#5FB459', '#784995', '#E3C365', '#1B78BC', '#E5302D']
# shap.dependence_plot(5, shap_values_sort, x_feature, interaction_index = None, feature_names=columns_dict3, show=False, ax=ax, x_jitter=1, dot_size=2, cmap='RdBu')
shap.dependence_plot(5, shap_values_sort, x_feature, interaction_index=None, feature_names=columns_dict3, show=False,
                     ax=ax, x_jitter=1, dot_size=2, color='#3274A1')

plt.xlabel(r'$\phi_{\rmlocal}$', fontdict=legend_font)
plt.ylabel(r'SHAP value for $\phi_{\rmlocal}$', fontdict=legend_font)
plt.xticks([0.55, 0.60, 0.65, 0.70, 0.75], fontproperties='Arial', size=10)
plt.yticks([-2.7, -1.8, -0.9, 0, 0.9, 1.8], fontproperties='Arial', size=10)

# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=8)
font = {'family': 'sans-serif',
        'color': 'k',
        'weight': 'normal',
        'size': 8}
# cbar = ax.collections[0].colorbar
# cbar.set_label(r'$q_{\rm6}$',fontdict=font)
plt.xlim(0.55, 0.75)
plt.ylim(-2.7, 1.8)

yminorLocator = MultipleLocator(0.45)
ax.yaxis.set_minor_locator(yminorLocator)
xminorLocator = MultipleLocator(0.025)
ax.xaxis.set_minor_locator(xminorLocator)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
plt.plot([0.55, 0.75], [0.0, 0.0], c='k', linestyle='--', linewidth=0.5)
# plt.show()
plt.savefig('J:/文件/张一博/小论文/第一篇/单个Fig/' + 'dependence-cluster40.tif', bbox_inches='tight')
