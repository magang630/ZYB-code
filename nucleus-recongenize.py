# -*- coding: cp936 -*-
from __future__ import division
import numpy as np
import re
import os
from sys import argv
from tkinter import _flatten


def mkdir(path_write):
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        print('目录已存在')


def merge_list(L):
    length = len(L)
    for i in range(1, length):
        for j in range(i):
            if L[i] == {0} or L[j] == {0}:
                continue
            x = L[i].union(L[j])
            y = len(L[i]) + len(L[j])
            if len(x) < y:
                L[i] = x
                L[j] = {0}
    return [i for i in L if i != {0}]


def compute_cluster_all(voroni_neighbour, stable_state, density=3):
    mark = np.zeros(len(voroni_neighbour))
    for i in range(len(voroni_neighbour)):
        connection_number = 0
        if stable_state[i] == 1:
            for j in range(len(voroni_neighbour[i])):
                if stable_state[voroni_neighbour[i][j]] == 1:
                    connection_number += 1
            if connection_number >= density:
                mark[i] = 1
    mark_done = np.zeros(len(voroni_neighbour))
    cluster_all_initial = []
    for i in range(len(voroni_neighbour)):
        if stable_state[i] == 1 and mark[i] == 1:
            cluster = [i]
            for j in range(len(voroni_neighbour[i])):
                if stable_state[voroni_neighbour[i][j]] == 1 \
                        and mark[voroni_neighbour[i][j]] == 1 \
                        and mark_done[voroni_neighbour[i][j]] == 0:
                    cluster.append(voroni_neighbour[i][j])
                    # mark_done[voroni_neighbour[i][j]] = 1
            cluster_all_initial.append(set(cluster))
    return merge_list(cluster_all_initial)


radius = 0.01
initial_step = 30800000
interval_step = 100000
ini_number = 0
cyc_number = 5001
path_output = 'D:/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/cluster/' + str(int(argv[1])) + '/'
mkdir(path_output)

for i in range(cyc_number):
    print(i)
    stable_cry_file_name = 'stable_cry_state' + str(i) + '.txt'
    stable_cry_file = open('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/stable state/' +
                           stable_cry_file_name, 'r')
    lines = stable_cry_file.readlines()
    stable_cry_file.close()
    stable_cry = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines]))

    step = initial_step + i * interval_step
    pos_file = 'dump-' + str(step) + '.sample'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/sort position/' + pos_file, 'r')
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
    select_boundary = [[xmin + 10 * radius, xmax - 10 * radius],
                       [ymin + 10 * radius, ymax - 10 * radius],
                       [zmin + 10 * radius, zmax - 10 * radius]]

    voroni_neighbour_file = 'voronoi_neighbour_id' + str(i) + '.txt'
    atomfile = open('D:/循环剪切试验和机器学习/cyc16000fric002shearrate01press5/voronoi neighbour/'
                    + voroni_neighbour_file, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    voroni_neighbour_string = [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line) for line in lines]
    voroni_neighbour = []
    for x in range(len(voroni_neighbour_string)):
        voroni_neighbour.append(list(map(int, voroni_neighbour_string[x])))
    bonds = []
    for x in range(len(voroni_neighbour)):
        for y in range(len(voroni_neighbour[x])):
            if voroni_neighbour[x][y] > x:
                bonds.append([x, voroni_neighbour[x][y]])
    bonds = np.array(bonds)
    voroni_neighbour_use = []
    for x in range(len(lines)):
        voroni_neighbour_use.append([])
    for x in range(len(bonds)):
        voroni_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voroni_neighbour_use[bonds[x][1]].append(bonds[x][0])

    cluster = compute_cluster_all(voroni_neighbour_use, stable_cry)
    cluster_big_than_100 = []
    for x in range(len(cluster)):
        if len(cluster[x]) >= int(argv[1]):
            cluster_big_than_100.append(list(cluster[x]))

    particle_in_cluster_big_than_100 = list(_flatten(cluster_big_than_100))

    cluster_file = 'cluster' + str(i) + '.txt'
    atomfile = open(path_output + cluster_file, 'w')
    for x in range(len(points)):
        if x in particle_in_cluster_big_than_100:
            atomfile.write(str(1))
        else:
            atomfile.write(str(0))
        atomfile.write('\n')
    atomfile.close()
