# -*- coding: cp936 -*-

from __future__ import division
import re
import numpy as np
import os
import math
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


def read_dem_position(step, path):
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open(path + '/sort position/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    x_min = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[5])[0])
    x_max = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[5])[1])
    y_min = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[6])[0])
    y_max = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[6])[1])
    z_min = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[7])[0])
    z_max = float(re.findall(r'-?\d+\.?\d*e?[-+]?\d*', lines[7])[1])
    lines = lines[9:]
    particle_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    position_x = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    position_y = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    position_z = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    particle_original_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[6] for line in lines])))
    transfer_type = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[7] for line in lines])))
    radius = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    points = np.array(list(zip(position_x, position_y, position_z)))
    return particle_id, points, radius, particle_original_id, transfer_type, x_min, x_max, y_min, y_max, z_min, z_max


def read_voronoi_neighbours(index, path):
    voronoi_neighbour_id = 'voronoi_neighbour_id' + str(index) + '.txt'
    atomfile = open(path + '/voronoi neighbour/' + voronoi_neighbour_id, 'r')
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


def compute_local_deviatoric_strain(initial_step, path, path_output, step, initial_index, actual_particle_number):
    particle_id_e, points_e, radius_e, particle_original_id_e, transfer_type_e, x_min_e, x_max_e, y_min_e, y_max_e, \
    z_min_e, z_max_e = read_dem_position(initial_step + step, path)
    particle_id_s, points_s, radius_s, particle_original_id_s, transfer_type_s, x_min_s, x_max_s, y_min_s, y_max_s, \
    z_min_s, z_max_s = read_dem_position(initial_step, path)
    Par_xcor_e, Par_ycor_e, Par_zcor_e = points_e[:, 0], points_e[:, 1], points_e[:, 2]
    Par_xcor_s, Par_ycor_s, Par_zcor_s = points_s[:, 0], points_s[:, 1], points_s[:, 2]
    Par_volume = [4. / 3. * np.pi * radius_s[j] ** 3. for j in range(len(radius_s))]
    Par_num = len(particle_id_s)
    Par_shear_strain = np.zeros(actual_particle_number)
    Par_D2min = np.zeros(actual_particle_number)
    voronoi_neighbour = read_voronoi_neighbours(initial_index, path)
    for j in range(actual_particle_number):
        CG_volume = 0
        matrix_X = np.zeros([3, 3])
        matrix_Y = np.zeros([3, 3])
        matrix_affine = np.zeros([3, 3])
        for k in voronoi_neighbour[j]:
            point_to_particle = list(set(np.where(np.array(particle_original_id_e) == particle_original_id_s[k])[0]) &
                                     set(np.where(np.array(transfer_type_e) == transfer_type_s[k])[0]))
            if len(point_to_particle) > 0:
                l = point_to_particle[0]
            else:
                l = particle_original_id_s[k]
            CG_volume += Par_volume[k]
            pos_relative_s = np.array([Par_xcor_s[k], Par_ycor_s[k], Par_zcor_s[k]]) - np.array(
                [Par_xcor_s[j], Par_ycor_s[j], Par_zcor_s[j]])
            pos_relative_e = np.array([Par_xcor_e[l], Par_ycor_e[l], Par_zcor_e[l]]) - np.array(
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
            point_to_particle = list(set(np.where(np.array(particle_original_id_e) == particle_original_id_s[k])[0]) &
                                     set(np.where(np.array(transfer_type_e) == transfer_type_s[k])[0]))
            if len(point_to_particle) > 0:
                l = point_to_particle[0]
            else:
                l = particle_original_id_s[k]
            pos_relative_s = np.array([Par_xcor_s[k], Par_ycor_s[k], Par_zcor_s[k]]) - np.array([Par_xcor_s[j], Par_ycor_s[j], Par_zcor_s[j]])
            pos_relative_e = np.array([Par_xcor_e[l], Par_ycor_e[l], Par_zcor_e[l]]) - np.array([Par_xcor_e[j], Par_ycor_e[j], Par_zcor_e[j]])
            matrix_temp = np.dot((matrix_affine + matrix_eye), pos_relative_s.reshape(-1, 1))
            vector_temp = np.squeeze(np.asarray(matrix_temp))
            vector = pos_relative_e - vector_temp
            Par_D2min[j] += np.sum(vector*vector)*Par_volume[k]
        Par_D2min[j] /= CG_volume
    Par_displacement = np.zeros(Par_num)
    Par_displacement_xy = np.zeros(Par_num)
    for j in range(actual_particle_number):
        Par_displacement[j] = math.sqrt((Par_xcor_e[j] - Par_xcor_s[j]) ** 2 + (Par_ycor_e[j] - Par_ycor_s[j]) ** 2 + (
                    Par_zcor_e[j] - Par_zcor_s[j]) ** 2)
        Par_displacement_xy[j] = math.sqrt((Par_xcor_e[j] - Par_xcor_s[j]) ** 2 + (Par_ycor_e[j] - Par_ycor_s[j]) ** 2)
    file_name = 'dynamic-measure-' + str(step) + '.txt'
    atomfile = open(path_output + '/' + file_name, 'w')
    for x in range(actual_particle_number):
        atomfile.write(str(Par_D2min[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_shear_strain[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_displacement[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_displacement_xy[x]))
        atomfile.write(' ')
        atomfile.write('\n')
    atomfile.close()
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open(path_output + '/' + filename, 'w')
    atomfile.write('ITEM: TIMESTEP')
    atomfile.write('\n')
    atomfile.write(str(step))
    atomfile.write('\n')
    atomfile.write('ITEM: NUMBER OF ATOMS')
    atomfile.write('\n')
    atomfile.write(str(actual_particle_number))
    atomfile.write('\n')
    atomfile.write('ITEM: BOX BOUNDS mm mm mm')
    atomfile.write('\n')
    atomfile.write(str(x_min_e))
    atomfile.write(' ')
    atomfile.write(str(x_max_e))
    atomfile.write('\n')
    atomfile.write(str(y_min_e))
    atomfile.write(' ')
    atomfile.write(str(y_max_e))
    atomfile.write('\n')
    atomfile.write(str(z_min_e))
    atomfile.write(' ')
    atomfile.write(str(z_max_e))
    atomfile.write('\n')
    atomfile.write('ITEM: ATOMS id type radius x y z D2min Local_Strain Dis Dis_xy')
    atomfile.write('\n')
    for x in range(actual_particle_number):
        atomfile.write(str(particle_id_e[x]))
        atomfile.write(' ')
        atomfile.write(str(particle_id_e[x]))
        atomfile.write(' ')
        atomfile.write(str(radius_e[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_xcor_e[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_ycor_e[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_zcor_e[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_D2min[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_shear_strain[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_displacement[x]))
        atomfile.write(' ')
        atomfile.write(str(Par_displacement_xy[x]))
        atomfile.write(' ')
        atomfile.write('\n')
    atomfile.close()


actual_particle_number = 5000
initial_step = 0
initial_index = 0
step_list = [1000, 10000, 100000, 1000000]
strain_list = [1e-6, 1e-5, 1e-4, 1e-3]

grading_list = ['1.0-1.0', '0.9-1.1', '0.8-1.2']
friction_list = ['0.01', '0.03', '0.05', '0.07', '0.09', '0.1', '0.3', '0.5']
test_id_list = [1, 2, 3, 4, 5]
for grading in grading_list:
    for friction in friction_list:
        for test_id in test_id_list:
            print(grading+'~'+friction+'~'+str(test_id))
            path = 'I:/Conventional-triaxial-test-grading-change-ppp-2/python-data/' + \
                   grading + '/fric-' + friction + '/test-' + str(test_id)
            path_output = path + '/dynamic measure'
            mkdir(path_output)
            for index, step in enumerate(step_list):
                compute_local_deviatoric_strain(initial_step, path, path_output, step, initial_index, actual_particle_number)
