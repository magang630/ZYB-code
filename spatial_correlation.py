# -*- coding: cp936 -*-

from __future__ import division
import re
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
from numba import jit
import pandas as pd
from scipy import optimize


def mkdir(path_write):
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        print('目录已存在')

@jit(nopython=True)
def numba_correlation(Par_coord, Par_inside_region, d50, r_step, r_list,
                      Par_D2min, Par_shear_strain, Par_D2min_in_region, Par_shear_strain_in_region):

    r_num = len(r_list)
    cr_count = np.zeros(r_num)
    cr_D2min = np.zeros(r_num)
    cr_shear_strain = np.zeros(r_num)

    normalized_cr_D2min = np.zeros(r_num)
    normalized_cr_shear_strain = np.zeros(r_num)

    square_mean_D2min = np.mean(Par_D2min_in_region*Par_D2min_in_region)
    mean_square_D2min = (np.mean(Par_D2min_in_region))**2.0
    square_mean_shear_strain = np.mean(Par_shear_strain_in_region*Par_shear_strain_in_region)
    mean_square_shear_strain = (np.mean(Par_shear_strain_in_region))**2.0

    Par_num = Par_coord.shape[0]
    for i in range(Par_num):
        if not Par_inside_region[i]: continue
        for j in range(i, Par_num):
            if not Par_inside_region[j]: continue
            if i == j: continue
            # distij = vertex_distance(Par_coord[i], Par_coord[j])/d50
            distij = np.sqrt((Par_coord[i][0] - Par_coord[j][0])**2.0 + (Par_coord[i][1] - Par_coord[j][1])**2.0 + (Par_coord[i][2] - Par_coord[j][2])**2.0)
            distij /= d50
            k = int(np.ceil(distij/r_step)) - 1
            if (k >= r_num): continue
            if (k < 0): k = 0
            cr_count[k] += 1.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  scalar product correlation function
            cr_D2min[k] += Par_D2min[i]*Par_D2min[j]
            cr_shear_strain[k] += Par_shear_strain[i]*Par_shear_strain[j]

    for k in range(r_num):
        if cr_count[k]:
            cr_D2min[k] /= cr_count[k]
            cr_shear_strain[k] /= cr_count[k]
        else:
            cr_D2min[k] = 0.0
            cr_shear_strain[k] = 0.0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.2 normalized spatial correlation function
        #
        normalized_cr_D2min[k] = (cr_D2min[k] - mean_square_D2min)/(square_mean_D2min - mean_square_D2min)
        normalized_cr_shear_strain[k] = (cr_shear_strain[k] - mean_square_shear_strain)/(square_mean_shear_strain - mean_square_shear_strain)

    range_correlation = {}
    range_correlation['D2min'] = normalized_cr_D2min
    range_correlation['shear_strain'] = normalized_cr_shear_strain

    return range_correlation


def fit_power_law(x_data, y_data):
    log10x = np.log10(x_data)
    log10y = np.log10(y_data)
    fitfunc = lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    p_init = np.array([10.0, -10.0])
    out = optimize.leastsq(errfunc, p_init, args=(log10x, log10y), full_output=True)
    p_final = out[0]
    return p_final


def read_dem_position(step):
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric01shearrate01polysize/sort position/' + filename, 'r')
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


def read_local_deviatoric_strain_position(index, time_windows):
    filename = 'noffine-measure' + str(index) + '.txt'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric01shearrate01polysize/noffine-measure/' + str(time_windows) + '/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    d2min = list(map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0]
                                                        for line in lines])))
    local_deviatoric_strain = list(map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[1]
                                                        for line in lines])))
    return np.array(d2min), np.array(local_deviatoric_strain)


def SpatialCorrelation(initial_step, interval_step, cyc_number, ini_number, time_windows, d50):
    frame_cr_D2min = []
    frame_cr_shear_strain = []
    for i in range(cyc_number):
        if i + time_windows >= cyc_number: continue
        print(i)
        # Particle non affine measures at γ
        Par_D2min, Par_shear_strain = read_local_deviatoric_strain_position(i + time_windows, time_windows)
        # 读取i时刻的颗粒体系位置信息
        particle_id, points, radius = read_dem_position(initial_step + interval_step * i)
        # Find the particles in the simulation domain
        Par_xcor, Par_ycor, Par_zcor = points[:, 0], points[:, 1], points[:, 2]
        Par_coord = np.stack((Par_xcor, Par_ycor, Par_zcor), axis=1)
        Par_num = len(Par_coord)
        x_min = np.min([(Par_xcor[i] - radius[i]) for i in range(len(radius))])
        y_min = np.min([(Par_ycor[i] - radius[i]) for i in range(len(radius))])
        z_min = np.min([(Par_zcor[i] - radius[i]) for i in range(len(radius))])
        x_max = np.max([(Par_xcor[i] + radius[i]) for i in range(len(radius))])
        y_max = np.max([(Par_ycor[i] + radius[i]) for i in range(len(radius))])
        z_max = np.max([(Par_zcor[i] + radius[i]) for i in range(len(radius))])
        select_boundary = [[x_min + 5 * np.mean(radius), x_max - 5 * np.mean(radius)],
                           [y_min + 5 * np.mean(radius), y_max - 5 * np.mean(radius)],
                           [z_min + 5 * np.mean(radius), z_max - 5 * np.mean(radius)]]
        Par_inside_region = np.zeros(Par_num).astype(bool)
        select_par_index = []
        for j in range(len(particle_id)):
            if select_boundary[0][0] <= Par_xcor[j] <= select_boundary[0][1]:
                if select_boundary[1][0] <= Par_ycor[j] <= select_boundary[1][1]:
                    if select_boundary[2][0] <= Par_zcor[j] <= select_boundary[2][1]:
                        select_par_index.append(j)
                        Par_inside_region[j] = True
                    else:
                        Par_inside_region[j] = False
                else:
                    Par_inside_region[j] = False
            else:
                Par_inside_region[j] = False
        Par_shear_strain_in_region = np.array([Par_shear_strain[index] for index in select_par_index])
        Par_D2min_in_region = np.array([Par_D2min[index] for index in select_par_index])

        r_step = 0.2
        r_list = np.arange(1.0, 20, r_step)
        range_correlation = numba_correlation(Par_coord, Par_inside_region, d50, r_step, r_list,
                                              Par_D2min, Par_shear_strain, Par_D2min_in_region, Par_shear_strain_in_region)
        frame_cr_D2min.append(range_correlation['D2min'])
        frame_cr_shear_strain.append(range_correlation['shear_strain'])
    pd.DataFrame(np.array(frame_cr_D2min)).to_csv(path_output + '/' + '02spatial_correlation_d2min' + str(time_windows) + '.csv')
    pd.DataFrame(np.array(frame_cr_shear_strain)).to_csv(path_output + '/' + '02spatial_correlation_local_strain' + str(time_windows) + '.csv')


initial_step = 275800000
interval_step = 100000
path_output = 'D:/循环剪切试验和机器学习/cyc16000fric01shearrate01polysize/spatial correlation'
mkdir(path_output)
cyc_number = 1001
ini_number = 0
d50 = 0.02
time_windows_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for time_windows in time_windows_list:
    SpatialCorrelation(initial_step, interval_step, cyc_number, ini_number, time_windows, d50)
