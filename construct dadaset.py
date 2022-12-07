# -*- coding: cp936 -*-
from __future__ import division
import re
import numpy as np
import pandas as pd

path = 'cyc5300fric01shearrate025'
initial_step = 15400000
radius = 0.01
interval_step = 800000
distance_from_boundary_list = [5, 6, 7, 8, 9]
feature_category = ['A', 'B', 'C']
cyc_number = 2499
ini_number = 0

for i in range(cyc_number):
    i = i + ini_number
    print(i)
    step = initial_step + i * interval_step
    tem = '%d' % step
    tem1 = '%d' % i
    tem2 = '%d' % (i + 1)

    pos_file = 'dump-' + tem + '.sample'

    boo_index1 = 'boo_index' + tem1 + '.txt'
    boo_index2 = 'boo_index' + tem2 + '.txt'

    cellfraction1_file = 'cellfraction' + tem1 + '.txt'
    cellfraction2_file = 'cellfraction' + tem2 + '.txt'

    radial_file1 = 'radial_value-' + tem1 + '.txt'

    aguler_file1 = 'aguler_value-' + tem1 + '.txt'

    atomfile = open('../' + path + '/sort position/' + pos_file, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    lines = lines[9:]
    particle_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    position_x = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    position_y = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    position_z = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    points = np.array(list(zip(position_x, position_y, position_z)))
    xmin = np.min(position_x) - np.max(radius)
    xmax = np.max(position_x) + np.max(radius)
    ymin = np.min(position_y) - np.max(radius)
    ymax = np.max(position_y) + np.max(radius)
    zmin = np.min(position_z) - np.max(radius)
    zmax = np.max(position_z) + np.max(radius)

    atomfile = open('../' + path + '/boo index/' + boo_index1, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    q6_1 = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))

    atomfile = open('../' + path + '/boo index/' + boo_index2, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    q6_2 = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))

    atomfile = open('../' + path + '/cellfraction/' + cellfraction1_file, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    cellfraction_1 = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines]))

    atomfile = open('../' + path + '/cellfraction/' + cellfraction2_file, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    cellfraction_2 = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines]))

    atomfile = open('../' + path + '/Symmetry functions/' + radial_file1, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    radial_1 = [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line) for line in lines]

    atomfile = open('../' + path + '/Symmetry functions/' + aguler_file1, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    aguler_1 = [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line) for line in lines]

    sym_feature_array = np.zeros(shape=[len(radial_1), len(radial_1[0]) + len(aguler_1[0])])
    for x in range(len(radial_1)):
        for y in range(len(radial_1[x])):
            sym_feature_array[x][y] = radial_1[x][y]
        for y in range(len(aguler_1[x])):
            sym_feature_array[x][y + len(radial_1[x])] = aguler_1[x][y]

    old_feature_MRO_file = 'old_feature_MRO' + tem1 + '.csv'
    old_feature_MRO = pd.read_csv('../' + path + '/some old feature MRO/' + old_feature_MRO_file)
    old_feature_MRO_array = old_feature_MRO.values
    old_feature_MRO_array = np.delete(old_feature_MRO_array, 0, axis=1)

    interstice_distribution_MRO_file = 'interstice_distribution_MRO' + tem1 + '.csv'
    interstice_distribution_MRO = pd.read_csv('../' + path + '/interstice distribution MRO/' +
                                              interstice_distribution_MRO_file)
    interstice_distribution_MRO_array = interstice_distribution_MRO.values
    interstice_distribution_MRO_array = np.delete(interstice_distribution_MRO_array, 0, axis=1)

    feature_all = [sym_feature_array, old_feature_MRO_array, interstice_distribution_MRO_array]
    for a in range(len(distance_from_boundary_list)):
        distance_from_boundary = distance_from_boundary_list[a]
        select_boundary = [[xmin + distance_from_boundary * radius, xmax - distance_from_boundary * radius],
                           [ymin + distance_from_boundary * radius, ymax - distance_from_boundary * radius],
                           [zmin + distance_from_boundary * radius, zmax - distance_from_boundary * radius]]
        for b in range(len(feature_category)):
            dataset = 'dataset_' + 'dfb_' + '%d' % distance_from_boundary + 'cat_' + feature_category[b] + '.txt'
            atomfile = open('../' + path + '/dataset/' + dataset, 'a+')
            feature_number = feature_all[b].shape[1]
            for x in range(len(q6_1)):
                point = points[x]
                if select_boundary[0][0] <= point[0] <= select_boundary[0][1] \
                        and select_boundary[1][0] <= point[1] <= select_boundary[1][1] \
                        and select_boundary[2][0] <= point[2] <= select_boundary[2][1]:
                    if q6_1[x] >= 0.555 and q6_1[x] <= 0.595 and cellfraction_1[x] > 0.72:
                        if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                            print(1)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 1
                            atomfile.write(str(label))
                            atomfile.write('\n')
                        if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                            print(1)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 1
                            atomfile.write(str(label))
                            atomfile.write('\n')
                        else:
                            print(0)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 0
                            atomfile.write(str(label))
                            atomfile.write('\n')
                    if q6_1[x] >= 0.465 and q6_1[x] <= 0.505 and cellfraction_1[x] > 0.72:
                        if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                            print(1)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 1
                            atomfile.write(str(label))
                            atomfile.write('\n')
                        if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                            print(1)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 1
                            atomfile.write(str(label))
                            atomfile.write('\n')
                        else:
                            print(0)
                            for y in range(feature_number):
                                atomfile.write(str(feature_all[b][x][y]))
                                atomfile.write(' ')
                            label = 0
                            atomfile.write(str(label))
                            atomfile.write('\n')
            atomfile.close()
        for b in range(len(feature_category)):
            for c in range(len(feature_category)):
                if c > b:
                    dataset = 'dataset_' + 'dfb_' + '%d' % distance_from_boundary + 'cat_' + feature_category[b] + \
                              feature_category[c] + '.txt'
                    atomfile = open('../' + path + '/dataset/' + dataset, 'a+')
                    feature_number1 = feature_all[b].shape[1]
                    feature_number2 = feature_all[c].shape[1]
                    for x in range(len(q6_1)):
                        point = points[x]
                        if select_boundary[0][0] <= point[0] <= select_boundary[0][1] \
                                and select_boundary[1][0] <= point[1] <= select_boundary[1][1] \
                                and select_boundary[2][0] <= point[2] <= select_boundary[2][1]:
                            if q6_1[x] >= 0.555 and q6_1[x] <= 0.595 and cellfraction_1[x] > 0.72:
                                if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                                    print(1)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 1
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                                if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                                    print(1)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 1
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                                else:
                                    print(0)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 0
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                            if q6_1[x] >= 0.465 and q6_1[x] <= 0.505 and cellfraction_1[x] > 0.72:
                                if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                                    print(1)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 1
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                                if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                                    print(1)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 1
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                                else:
                                    print(0)
                                    for y in range(feature_number1):
                                        atomfile.write(str(feature_all[b][x][y]))
                                        atomfile.write(' ')
                                    for y in range(feature_number2):
                                        atomfile.write(str(feature_all[c][x][y]))
                                        atomfile.write(' ')
                                    label = 0
                                    atomfile.write(str(label))
                                    atomfile.write('\n')
                    atomfile.close()
        for b in range(len(feature_category)):
            for c in range(len(feature_category)):
                for d in range(len(feature_category)):
                    if c > b and d > c:
                        dataset = 'dataset_' + 'dfb_' + '%d' % distance_from_boundary + 'cat_' + \
                                  feature_category[b] + feature_category[c] + feature_category[d] + '.txt'
                        atomfile = open('../' + path + '/dataset/' + dataset, 'a+')
                        feature_number1 = feature_all[b].shape[1]
                        feature_number2 = feature_all[c].shape[1]
                        feature_number3 = feature_all[d].shape[1]
                        for x in range(len(q6_1)):
                            point = points[x]
                            if select_boundary[0][0] <= point[0] <= select_boundary[0][1] \
                                    and select_boundary[1][0] <= point[1] <= select_boundary[1][1] \
                                    and select_boundary[2][0] <= point[2] <= select_boundary[2][1]:
                                if q6_1[x] >= 0.555 and q6_1[x] <= 0.595 and cellfraction_1[x] > 0.72:
                                    if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                                        print(1)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 1
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                                    if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                                        print(1)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 1
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                                    else:
                                        print(0)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 0
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                                if q6_1[x] >= 0.465 and q6_1[x] <= 0.505 and cellfraction_1[x] > 0.72:
                                    if q6_2[x] >= 0.555 and q6_2[x] <= 0.595 and cellfraction_2[x] > 0.72:
                                        print(1)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 1
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                                    if q6_2[x] >= 0.465 and q6_2[x] <= 0.505 and cellfraction_2[x] > 0.72:
                                        print(1)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 1
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                                    else:
                                        print(0)
                                        for y in range(feature_number1):
                                            atomfile.write(str(feature_all[b][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number2):
                                            atomfile.write(str(feature_all[c][x][y]))
                                            atomfile.write(' ')
                                        for y in range(feature_number3):
                                            atomfile.write(str(feature_all[d][x][y]))
                                            atomfile.write(' ')
                                        label = 0
                                        atomfile.write(str(label))
                                        atomfile.write('\n')
                        atomfile.close()
