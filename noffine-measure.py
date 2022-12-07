# -*- coding: cp936 -*-

from __future__ import division
import re
import numpy as np
import os
from sys import argv


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


def read_dem_position(step):
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open('I:/Conventional-triaxial-test/' + system + '/python-data/test-' + str(test_index) + '/sort position/' + filename, 'r')
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


def read_voronoi_neighbours(index):
    voronoi_neighbour_id = 'voronoi_neighbour_id' + str(int(index)) + '.txt'
    atomfile = open('I:/Conventional-triaxial-test/' + system + '/python-data/test-' + str(test_index) + '/voronoi neighbour/' + voronoi_neighbour_id, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    voronoi_neighbour_string = [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line) for line in lines]
    voronoi_neighbour = []
    for x in range(len(voronoi_neighbour_string)):
        voronoi_neighbour.append(list(map(int, voronoi_neighbour_string[x])))
    bonds = []
    for x in range(len(voronoi_neighbour)):
        # 使用这种方法剔除了邻域不互相对称的颗粒，由剔除面积小于平均面积百分之五的邻域点所造成的不对称
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds.append([x, voronoi_neighbour[x][y]])
    bonds = np.array(bonds)
    voronoi_neighbour_use = []
    for x in range(len(voronoi_neighbour_string)):
        voronoi_neighbour_use.append([])
    for x in range(len(bonds)):
        voronoi_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voronoi_neighbour_use[bonds[x][1]].append(bonds[x][0])
    return voronoi_neighbour_use


time_windows_list = [2]
test_index = 40
system_list = ['PVC-initialfric01-loadfric04']
system = system_list[0]
for time_windows in time_windows_list:
    dump_path = 'I:/Conventional-triaxial-test/' + system + '/' + 'python-data/test-' + str(test_index) + '/sort position'
    list_dir = os.listdir(dump_path)
    dump_frame = []
    file_prefix = 'dump-'
    file_suffix = '.sample'
    prefix_len = len(file_prefix)
    suffix_len = len(file_suffix)
    for file in list_dir:
        dump_frame.append(int(file[prefix_len:][:-suffix_len]))
    dump_frame = np.array(sorted(dump_frame))
    initial_step = np.min(dump_frame)
    interval_step = 20000
    path_output = 'I:/Conventional-triaxial-test/' + system + '/python-data/noffine-measure/test-' + \
                  str(test_index) + '' + '/' + str(time_windows)
    mkdir(path_output)
    cyc_number = 1
    ini_number = 72
    for ii in range(cyc_number):
        if ii + ini_number - time_windows < 0:
            continue
        i = ii + ini_number
        particle_id_e, points_e, radius_e = read_dem_position(initial_step + interval_step * i)
        particle_id_s, points_s, radius_s = read_dem_position(initial_step + interval_step * (i - time_windows))
        Par_xcor_e, Par_ycor_e, Par_zcor_e = points_e[:, 0], points_e[:, 1], points_e[:, 2]
        Par_xcor_s, Par_ycor_s, Par_zcor_s = points_s[:, 0], points_s[:, 1], points_s[:, 2]
        Par_volume = [4. / 3.0 * np.pi * radius_s[j] ** 3.0 for j in range(len(radius_s))]
        Par_num = len(particle_id_s)
        Par_shear_strain = np.zeros(Par_num)
        Par_D2min = np.zeros(Par_num)
        voronoi_neighbour = read_voronoi_neighbours(i)
        for j in range(Par_num):
            CG_volume = 0
            matrix_X = np.zeros([3, 3])
            matrix_Y = np.zeros([3, 3])
            matrix_affine = np.zeros([3, 3])
            for k in voronoi_neighbour[j]:
                CG_volume += Par_volume[k]
                pos_relative_s = np.array([Par_xcor_s[k], Par_ycor_s[k], Par_zcor_s[k]]) - np.array(
                    [Par_xcor_s[j], Par_ycor_s[j], Par_zcor_s[j]])
                pos_relative_e = np.array([Par_xcor_e[k], Par_ycor_e[k], Par_zcor_e[k]]) - np.array(
                    [Par_xcor_e[j], Par_ycor_e[j], Par_zcor_e[j]])
                matrix_X += np.outer(pos_relative_e, pos_relative_s) * Par_volume[k]
                matrix_Y += np.outer(pos_relative_s, pos_relative_s) * Par_volume[k]
            matrix_X = matrix_X / CG_volume
            matrix_Y = matrix_Y / CG_volume
            matrix_eye = np.eye(3)
            try:
                matrix_Y_inv = np.linalg.inv(matrix_Y)
            except:
                continue
            else:
                matrix_affine = np.dot(matrix_X, matrix_Y_inv) - matrix_eye
            local_strain_tensor = -(matrix_affine + matrix_affine.T) / 2
            evals, evecs = np.linalg.eig(local_strain_tensor)
            Par_shear_strain[j] = np.sqrt(2.0 / 9.0 * ((evals[0] - evals[1]) ** 2.0 + (evals[1] - evals[2]) ** 2.0
                                                       + (evals[0] - evals[2]) ** 2.0))
            for k in voronoi_neighbour[j]:
                pos_relative_s = np.array([Par_xcor_s[k], Par_ycor_s[k], Par_zcor_s[k]]) - np.array(
                    [Par_xcor_s[j], Par_ycor_s[j], Par_zcor_s[j]])
                pos_relative_e = np.array([Par_xcor_e[k], Par_ycor_e[k], Par_zcor_e[k]]) - np.array(
                    [Par_xcor_e[j], Par_ycor_e[j], Par_zcor_e[j]])
                matrix_temp = np.dot((matrix_affine + matrix_eye), pos_relative_s.reshape(-1, 1))
                vector_temp = np.squeeze(np.asarray(matrix_temp))
                vector = pos_relative_e - vector_temp
                Par_D2min[j] += np.sum(vector * vector) * Par_volume[k]
            Par_D2min[j] /= CG_volume
        file_name = 'noffine-measure' + str(i) + '.txt'
        atomfile = open(path_output + '/' + file_name, 'w')
        for x in range(len(Par_shear_strain)):
            atomfile.write(str(Par_D2min[x]))
            atomfile.write(' ')
            atomfile.write(str(Par_shear_strain[x]))
            atomfile.write('\n')
        atomfile.close()
