# -*- coding: cp936 -*-
# 选择某一体系某一时刻做边界效应分析
# 使用方法：归一化局部密度函数
# 参考文献：Granular materials flow like complex fluids
from __future__ import division
import re
import os
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from math import floor


def mkdir(path_write):
    # 判断目录是否存在
    # 存在：True
    # 不存在：False
    folder = os.path.exists(path_write)
    # 判断结果
    if not folder:
        # 如果不存在，则创建新目录
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        # 如果目录已存在，则不创建，提示目录已存在
        print('目录已存在')


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(figsize=[2.7559055118110236, 2.3622047244094486], dpi=600)
system_list = ['0.01', '0.05', '0.1', '0.2', '0.3', '0.4']
color = ['#D75126', '#6BB4E2', '#70BB5E', '#904A95', '#DDB018', '#3A8AC5', '#E5302D']
for i,system in enumerate(system_list):
    path = 'K:/Conventional-triaxial-test/New/' + system + '/python data/test-1/sort position/'
    list_dir = os.listdir(path)
    dump_frame = []
    file_prefix = 'dump-'
    file_suffix = '.sample'
    prefix_len = len(file_prefix)
    suffix_len = len(file_suffix)
    for file in list_dir:
        dump_frame.append(int(file[prefix_len:][:-suffix_len]))
    dump_frame = np.array(sorted(dump_frame))
    initial_step = np.min(dump_frame)
    filename = 'dump-' + str(initial_step) + '.sample'
    atomfile = open(path + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    lines = lines[9:]
    particle_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    particle_id_read = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    position_x_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    position_y_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    position_z_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    radius_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    particle_id.sort()
    position_x = [position_x_read[particle_id_read.index(particle_id[x])] for x in range(len(particle_id))]
    position_y = [position_y_read[particle_id_read.index(particle_id[x])] for x in range(len(particle_id))]
    position_z = [position_z_read[particle_id_read.index(particle_id[x])] for x in range(len(particle_id))]
    radius = [radius_read[particle_id_read.index(particle_id[x])] for x in range(len(particle_id))]
    points = np.array(list(zip(position_x, position_y, position_z)))
    x_min = float('%.4f' % (np.min(position_x) - np.max(radius)))
    x_max = float('%.4f' % (np.max(position_x) + np.max(radius)))
    y_min = float('%.4f' % (np.min(position_y) - np.max(radius)))
    y_max = float('%.4f' % (np.max(position_y) + np.max(radius)))
    z_min = float('%.4f' % (np.min(position_z) - np.max(radius)))
    z_max = np.max(position_z) + np.max(radius)
    d50 = np.mean(radius)
    delta_distance_x = 0.1 * d50
    delta_distance_y = 0.1 * d50
    delta_distance_z = 0.1 * d50
    delta_number_x = int((x_max - x_min) / delta_distance_x)
    delta_number_y = int((y_max - y_min) / delta_distance_y)
    delta_number_z = int((z_max - z_min) / delta_distance_z)
    delta_distance_z = (z_max - z_min) / delta_number_z
    number_density_x = np.zeros(shape=[delta_number_x, ])
    number_density_y = np.zeros(shape=[delta_number_y, ])
    number_density_z = np.zeros(shape=[delta_number_z, ])
    x_axis = [(delta_distance_x * x + delta_distance_x / 2) / d50  for x in range(delta_number_x)]
    print(np.max(x_axis))
    y_axis = [(delta_distance_y * x + delta_distance_y / 2) / d50  for x in range(delta_number_y)]
    print(np.max(y_axis))
    z_axis = [(delta_distance_z * x + delta_distance_z / 2) / d50  for x in range(delta_number_z)]
    print(np.max(z_axis))
    for x in range(len(points)):
        index_x = int(floor((position_x[x] - x_min) / delta_distance_x))
        index_y = int(floor((position_y[x] - y_min) / delta_distance_y))
        index_z = int(floor((position_z[x] - z_min) / delta_distance_z))
        number_density_x[index_x] += 1
        number_density_y[index_y] += 1
        number_density_z[index_z] += 1
    number_density = len(points) / ((x_max - x_min) * (y_max - y_min) * (z_max - z_min))
    for x in range(len(number_density_x)):
        number_density_x[x] /= (number_density * delta_distance_x * (y_max - y_min) * (z_max - z_min))
    for x in range(len(number_density_y)):
        number_density_y[x] /= (number_density * delta_distance_y * (z_max - z_min) * (x_max - x_min))
    for x in range(len(number_density_z)):
        number_density_z[x] /= (number_density * delta_distance_z * (x_max - x_min) * (y_max - y_min))
    # plt.plot(x_axis, number_density_x, linewidth=0.5, color=color[i], label=r'$\mu=$'+system)
    # plt.plot(y_axis, number_density_y, linewidth=0.5, color=color[i], label=r'$\mu=$'+system)
    plt.plot(z_axis, number_density_z, linewidth=0.5, color=color[i], label=r'$\mu=$'+system)

plt.axvline(5, linewidth=1.0, color='k', linestyle='--')
plt.axvline(43, linewidth=1.0, color='k', linestyle='--')
legend_font = {"family": 'Arial', "size": 10}
plt.xlabel(r'$z/d_{\rm{50}}$', fontdict=legend_font)
plt.ylabel(r'$\rho_{\rmz}/\rho$', fontdict=legend_font)
plt.xlim(0, 48)
plt.ylim(0, 8)
plt.xticks(fontproperties='Arial', size=10)
plt.yticks(fontproperties='Arial', size=10)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
legend_font = {"family": 'Arial', "size": 8}
plt.legend(edgecolor='w', prop=legend_font, loc='upper center', ncol=1)
plt.savefig('J:/个人文件/小论文/第二篇/单个Fig/number-density-z.tif', bbox_inches='tight')
