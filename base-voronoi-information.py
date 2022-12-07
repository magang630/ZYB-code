# -*- coding: cp936 -*-
from __future__ import division
import re
import os
import math
import pyvoro
import numpy as np
from sys import argv
from scipy.spatial import ConvexHull

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


def triangle_area(va, vb, vc):
    """
    Calculate the volume of a triangle, given the three vertices of va, vb, vc.
    Args:
        va/vb/vc (array-like): coordinates of vertex 1, 2, 3.
    Returns:
        (float): area of the triangle.
    """
    triangle_area = 0.5 * np.linalg.norm(np.cross(np.array(va) - np.array(vc),
                                                  np.array(vb) - np.array(vc)))
    return triangle_area


def compute_simplice_area(vertice1, vertice2, vertice3):
    # problem1: compute error -> eij = eik causes error.
    # neglect
    # 进行面积计算的时候尽量避免使用除法，由此而导致误差
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    h = (np.dot(eij, eij) - (np.dot(eij, eik) / (np.linalg.norm(eik))) ** 2) ** 0.5
    return (np.linalg.norm(eik)) * h / 2


def compute_area(vertices_input, adjacent_cell_input, vertices_id_input, simplice_input):
    area_judge_in = np.zeros(shape=[len(adjacent_cell_input), ], dtype=int)
    area = np.zeros(shape=[len(adjacent_cell_input), ])
    sing_area = np.zeros(shape=[len(simplice_input), ])
    for a in range(len(simplice_input)):
        sing_area[a] = compute_simplice_area(vertices_input[simplice_input[a][0]],
                                             vertices_input[simplice_input[a][1]],
                                             vertices_input[simplice_input[a][2]])
    for a in range(len(simplice_input)):
        for b in range(len(adjacent_cell_input)):
            if simplice_input[a][0] in vertices_id_input[b]:
                if simplice_input[a][1] in vertices_id_input[b]:
                    if simplice_input[a][2] in vertices_id_input[b]:
                        area[b] += sing_area[a]
    average_area = np.mean(area)
    for a in range(len(adjacent_cell_input)):
        if area[a] >= 0.05 * average_area:
            area_judge_in[a] = 1
    return area_judge_in


test_index = 1
path_output1 = 'python-code-data/test-' + str(test_index) + '/sort position'
path_output2 = 'python-code-data/test-' + str(test_index) + '/cellfraction'
path_output3 = 'python-code-data/test-' + str(test_index) + '/voronoi neighbour'
mkdir(path_output1)
mkdir(path_output2)
mkdir(path_output3)
initial_step = 9380000
interval_step = 10000
cyc_number = 25
ini_number = int(argv[1])

for i in range(cyc_number):
    i = i + ini_number
    print(i)
    step = initial_step + i * interval_step
    tem = '%d' % step
    tem1 = '%d' % i
    filename = 'dump-' + tem + '.sample'

    atomfile = open('test-' + str(test_index) + '/load/post/' + filename, 'r')
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
    atomfile = open('python-code-data/test-' + str(test_index) + '/sort position/' + filename, 'w')
    atomfile.write('ITEM: TIMESTEP')
    atomfile.write('\n')
    atomfile.write('ITEM: TIMESTEP')
    atomfile.write('\n')
    atomfile.write('ITEM: NUMBER OF ATOMS')
    atomfile.write('\n')
    atomfile.write('ITEM: NUMBER OF ATOMS')
    atomfile.write('\n')
    atomfile.write('ITEM: BOX BOUNDS mm mm mm')
    atomfile.write('\n')
    atomfile.write('ITEM: BOX BOUNDS mm mm mm')
    atomfile.write('\n')
    atomfile.write('ITEM: BOX BOUNDS mm mm mm')
    atomfile.write('\n')
    atomfile.write('ITEM: BOX BOUNDS mm mm mm')
    atomfile.write('\n')
    atomfile.write('ITEM: ATOMS id type radius x y z vx vy vz fx fy fz omegax omegay omegaz')
    atomfile.write('\n')
    for x in range(len(particle_id)):
        atomfile.write(str(particle_id[x]))
        atomfile.write(' ')
        atomfile.write(str(particle_id[x]))
        atomfile.write(' ')
        atomfile.write(str(radius[x]))
        atomfile.write(' ')
        atomfile.write(str(position_x[x]))
        atomfile.write(' ')
        atomfile.write(str(position_y[x]))
        atomfile.write(' ')
        atomfile.write(str(position_z[x]))
        atomfile.write(' ')
        atomfile.write('\n')
    atomfile.close()
    x_min = np.min([(position_x[i] - radius[i]) for i in range(len(radius))])
    y_min = np.min([(position_y[i] - radius[i]) for i in range(len(radius))])
    z_min = np.min([(position_z[i] - radius[i]) for i in range(len(radius))])
    x_max = np.max([(position_x[i] + radius[i]) for i in range(len(radius))])
    y_max = np.max([(position_y[i] + radius[i]) for i in range(len(radius))])
    z_max = np.max([(position_z[i] + radius[i]) for i in range(len(radius))])
    limits = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    dispersion = 5.0 * np.mean(radius)
    voro = pyvoro.compute_voronoi(points, limits, dispersion, radius, periodic=[False] * 3)
    volume = []
    for x in range(len(voro)):
        volume.append(voro[x]['volume'])
    ball_volume = [(radius[x] ** 3) * 4 * math.pi / 3 for x in range(len(radius))]
    cellfraction = [ball_volume[x] / volume[x] for x in range(len(voro))]
    atomfile = open('python-code-data/test-' + str(test_index) + '/cellfraction/' + 'cellfraction' + str(i) + '.txt', 'w')
    for x in range(len(cellfraction)):
        atomfile.write(str(cellfraction[x]))
        atomfile.write('\n')
    atomfile.close()
    adjacent_cell_all = []
    for x in range(len(voro)):
        vertices = voro[x]['vertices']
        ch = ConvexHull(vertices)
        simplice = np.array(ch.simplices)
        faces = voro[x]['faces']
        adjacent_cell = []
        for y in range(len(faces)):
            adjacent_cell.append(faces[y]['adjacent_cell'])
        vert_id = []
        for y in range(len(faces)):
            vert_id.append(faces[y]['vertices'])
        area_judge = compute_area(vertices, adjacent_cell, vert_id, simplice)
        adjacent_cell_use = []
        for y in range(len(adjacent_cell)):
            if area_judge[y] == 1:
                adjacent_cell_use.append(adjacent_cell[y])
        adjacent_cell_all.append(adjacent_cell_use)
    voroni_neighbour_id = 'voronoi_neighbour_id' + str(i) + '.txt'
    atomfile = open('python-code-data/test-' + str(test_index) + '/voronoi neighbour/' + voroni_neighbour_id, 'w')
    for x in range(len(adjacent_cell_all)):
        for y in range(len(adjacent_cell_all[x])):
            atomfile.write(str(adjacent_cell_all[x][y]))
            atomfile.write(' ')
        atomfile.write('\n')
    atomfile.close()
