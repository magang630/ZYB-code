# -*- coding: cp936 -*-

from __future__ import division
import re
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from scipy import optimize


def mkdir(path_write):
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        print('目录已存在')


def numba_correlation(Par_metrics_t0, Par_metrics_t1, type=None):
    if type == 'pearson':
        corr, p_value = pearsonr(Par_metrics_t0, Par_metrics_t1)
    elif type == 'spearman':
        corr, p_value = spearmanr(Par_metrics_t0, Par_metrics_t1)
    else:
        mean_metric_t0 = np.mean(Par_metrics_t0)
        mean_metric_t1 = np.mean(Par_metrics_t1)
        Chi_square = np.mean((Par_metrics_t0 - mean_metric_t0) ** 2.0)
        corr = np.mean((Par_metrics_t1 - mean_metric_t1) * (Par_metrics_t0 - mean_metric_t0)) / Chi_square
    metrics_corr = corr
    return metrics_corr


def fit_power_law(x_data, y_data):
    log10x = np.log10(x_data)
    log10y = np.log10(y_data)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    p_init = np.array([10.0, -10.0])
    out = optimize.leastsq(errfunc, p_init, args=(log10x, log10y), full_output=True)
    p_final = out[0]
    return p_final


def read_dem_position(step):
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01polysize/sort position/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    lines = lines[9:]
    particle_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    position_x = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    position_y = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    position_z = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    radius = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    points = np.array(list(zip(position_x, position_y, position_z)))
    return particle_id, points, radius


def read_local_deviatoric_strain_position(index, delta_time):
    filename = 'noffine-measure' + str(index) + '.txt'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01polysize/noffine-measure/' + str(delta_time) + '/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    local_deviatoric_strain = list(map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[1]
                                                          for line in lines])))
    return local_deviatoric_strain


def TemporalCorrelation(initial_step, interval_step, cyc_number, ini_number, time_windows, delta_time):
    frame_time_cr_shear_strain = []
    for i in range(cyc_number):
        if i < delta_time: continue
        print(i)
        # Particle non affine measures at γ
        Par_shear_strain_0 = read_local_deviatoric_strain_position(i, delta_time)
        # 读取i时刻的颗粒体系位置信息
        particle_id, points, radius = read_dem_position(initial_step + interval_step * i)
        # Find the particles in the simulation domain
        Par_xcor, Par_ycor, Par_zcor = points[:, 0], points[:, 1], points[:, 2]
        x_min = np.min([(Par_xcor[x] - radius[x]) for x in range(len(radius))])
        y_min = np.min([(Par_ycor[x] - radius[x]) for x in range(len(radius))])
        z_min = np.min([(Par_zcor[x] - radius[x]) for x in range(len(radius))])
        x_max = np.max([(Par_xcor[x] + radius[x]) for x in range(len(radius))])
        y_max = np.max([(Par_ycor[x] + radius[x]) for x in range(len(radius))])
        z_max = np.max([(Par_zcor[x] + radius[x]) for x in range(len(radius))])
        select_boundary = [[x_min + 10 * np.mean(radius), x_max - 10 * np.mean(radius)],
                           [y_min + 10 * np.mean(radius), y_max - 10 * np.mean(radius)],
                           [z_min + 10 * np.mean(radius), z_max - 10 * np.mean(radius)]]
        select_par_index = []
        for j in range(len(particle_id)):
            if select_boundary[0][0] <= Par_xcor[j] <= select_boundary[0][1]:
                if select_boundary[1][0] <= Par_ycor[j] <= select_boundary[1][1]:
                    if select_boundary[2][0] <= Par_zcor[j] <= select_boundary[2][1]:
                        select_par_index.append(j)
        Par_shear_strain_0 = [Par_shear_strain_0[index] for index in select_par_index]
        Par_metrics_t0 = Par_shear_strain_0
        time_corr = np.zeros(len(time_windows))
        for j, frame_shift in enumerate(time_windows):
            if i + frame_shift > cyc_number - 1: continue
            # Particle non affine measures at γ+δγ
            Par_shear_strain_1 = read_local_deviatoric_strain_position(i + frame_shift, delta_time)
            # Find the particles in the simulation domain
            Par_shear_strain_1 = [Par_shear_strain_1[index] for index in select_par_index]
            Par_metrics_t1 = Par_shear_strain_1
            metrics_corr = numba_correlation(Par_metrics_t0, Par_metrics_t1, type='pearson')
            time_corr[j] = metrics_corr
        time_corr = np.array(time_corr)
        frame_time_cr_shear_strain.append(time_corr)
    pd.DataFrame(np.array(frame_time_cr_shear_strain)).to_csv('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01polysize/temporal correlation' + 'local_strain' + str(delta_time) + '.csv')


def main_function():
    initial_step = 238000000
    interval_step = 100000
    cyc_number = 11
    ini_number = 0
    time_windows = range(1, 100, 1)
    delta_time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for delta_time in delta_time_list:
        TemporalCorrelation(initial_step, interval_step, cyc_number, ini_number, time_windows, delta_time)


main_function()
