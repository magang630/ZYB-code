# -*- coding: UTF-8 -*-
from __future__ import division
import re
import math
import os
import pyvoro
import boo
import requests
import openpyxl
import pandas as pd
import numpy as np
from numba import jit
from sys import argv, exit
from scipy import special
from matplotlib import pyplot as plt
from scipy.spatial import KDTree, ConvexHull


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@jit(nopython=True)
def compute_cos_ijk(posi, posj, posk):
    # 计算向量ij与ik的夹角的cos值
    eij = np.array([posj[0] - posi[0], posj[1] - posi[1], posj[2] - posi[2]])
    eik = np.array([posk[0] - posi[0], posk[1] - posi[1], posk[2] - posi[2]])
    cos = np.dot(eij, eik) / (np.linalg.norm(eij) * np.linalg.norm(eik))
    return cos


@jit(nopython=True)
def compute_dis(posj, posk):
    # 计算三维空间中两点的距离
    ejk = np.array([posk[0] - posj[0], posk[1] - posj[1], posk[2] - posj[2]])
    dis = np.linalg.norm(ejk)
    return dis


@jit(nopython=True)
def compute_tetrahedron_volume(vertice1, vertice2, vertice3, vertice4):
    # 计算四面体的体积，通过给定四面体的四个顶点
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    return abs(np.dot(eil, np.cross(eij, eik))) / 6


@jit(nopython=True)
def compute_solide_angle(vertice1, vertice2, vertice3, vertice4):
    # 计算固体角
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    len_eij = np.linalg.norm(eij)
    len_eik = np.linalg.norm(eik)
    len_eil = np.linalg.norm(eil)
    return 2 * math.atan2(abs(np.dot(eij, np.cross(eik, eil))),
                          (len_eij * len_eik * len_eil + np.dot(eij, eik) * len_eil
                           + np.dot(eij, eil) * len_eik + np.dot(eik, eil) * len_eij))


@jit(nopython=True)
def compute_simplice_area(vertice1, vertice2, vertice3):
    # 计算三角形的面积，通过给定的三个顶点
    # problem1: compute error -> eij = eik causes error.
    # neglect
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    h = (np.dot(eij, eij) - (np.dot(eij, eik) / (np.linalg.norm(eik))) ** 2) ** 0.5
    return (np.linalg.norm(eik)) * h / 2


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


def calc_beta_rad(pvec):
    """
    polar angle [0, pi]
    """
    return np.arccos(pvec[2])  # arccos:[0, pi]


def calc_gamma_rad(pvec):
    """
    azimuth angle [0, 2pi]
    """
    gamma = np.arctan2(pvec[1], pvec[0])
    if gamma < 0.0:
        gamma += 2 * np.pi
    return gamma


def vertex_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2.0 + (a[1] - b[1]) ** 2.0 + (a[2] - b[2]) ** 2.0)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compute_cellfraction(voro_input, radius_input):
    volume = []
    for a in range(len(voro_input)):
        volume.append(voro_input[a]['volume'])
    ball_volume = [(radius_input[a]**3)*4*math.pi/3 for a in range(len(radius_input))]
    cellfraction_in = [ball_volume[a] / volume[a] for a in range(len(voro_input))]
    return cellfraction_in


def compute_voronoi_idx(voro_input):
    particle_number = len(voro_input)
    voronoi_idx = {}
    voronoi_idx3_in = np.zeros(shape=[particle_number, ])
    voronoi_idx4_in = np.zeros(shape=[particle_number, ])
    voronoi_idx5_in = np.zeros(shape=[particle_number, ])
    voronoi_idx6_in = np.zeros(shape=[particle_number, ])
    voronoi_idx7_in = np.zeros(shape=[particle_number, ])
    for a in range(len(voro_input)):
        faces_in = voro_input[a]['faces']
        vertices_id_length = []
        for b in range(len(faces_in)):
            vertices_id_length.append(len(faces_in[b]['vertices']))
        voronoi_idx3_in[a] = vertices_id_length.count(3)
        voronoi_idx4_in[a] = vertices_id_length.count(4)
        voronoi_idx5_in[a] = vertices_id_length.count(5)
        voronoi_idx6_in[a] = vertices_id_length.count(6)
        voronoi_idx7_in[a] = vertices_id_length.count(7)
    voronoi_idx['3'] = voronoi_idx3_in
    voronoi_idx['4'] = voronoi_idx4_in
    voronoi_idx['5'] = voronoi_idx5_in
    voronoi_idx['6'] = voronoi_idx6_in
    voronoi_idx['7'] = voronoi_idx7_in
    return voronoi_idx


def compute_boop(voronoi_neighbour, points):
    bonds1 = []
    for x in range(len(voronoi_neighbour)):
        # 使用这种方法剔除了邻域不互相对称的颗粒，由剔除面积小于平均面积百分之五的邻域点所造成的不对称
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds1.append([x, voronoi_neighbour[x][y]])
    bonds1 = np.array(bonds1)

    q2m_1 = boo.bonds2qlm(points, bonds1, l=2)
    q4m_1 = boo.bonds2qlm(points, bonds1, l=4)
    q6m_1 = boo.bonds2qlm(points, bonds1, l=6)
    q8m_1 = boo.bonds2qlm(points, bonds1, l=8)
    q10m_1 = boo.bonds2qlm(points, bonds1, l=10)

    q2_1 = boo.ql(q2m_1)
    q4_1 = boo.ql(q4m_1)
    q6_1 = boo.ql(q6m_1)
    q8_1 = boo.ql(q8m_1)
    q10_1 = boo.ql(q10m_1)

    w2_1 = boo.wl(q2m_1)
    w4_1 = boo.wl(q4m_1)
    w6_1 = boo.wl(q6m_1)
    w8_1 = boo.wl(q8m_1)
    w10_1 = boo.wl(q10m_1)

    scale_2 = (5 / (4 * math.pi)) ** 1.5
    scale_4 = (9 / (4 * math.pi)) ** 1.5
    scale_6 = (13 / (4 * math.pi)) ** 1.5
    scale_8 = (17 / (4 * math.pi)) ** 1.5
    scale_10 = (21 / (4 * math.pi)) ** 1.5

    w2_hat = [w2_1[x] / (q2_1[x] ** 3 * scale_2) for x in range(len(w2_1))]
    w4_hat = [w4_1[x] / (q4_1[x] ** 3 * scale_4) for x in range(len(w4_1))]
    w6_hat = [w6_1[x] / (q6_1[x] ** 3 * scale_6) for x in range(len(w6_1))]
    w8_hat = [w8_1[x] / (q8_1[x] ** 3 * scale_8) for x in range(len(w8_1))]
    w10_hat = [w10_1[x] / (q10_1[x] ** 3 * scale_10) for x in range(len(w10_1))]

    boop_all = np.array(list(zip(q2_1, q4_1, q6_1, q8_1, q10_1,
                                 w2_hat, w4_hat, w6_hat, w8_hat, w10_hat,
                                 )))
    return boop_all


@jit(nopython=True)
def minkowski_tensor_w1_02(area, face_normal_vector, w1_02_array_input):
    # compute anisotropic of the voronoi by Minkowski tensor W1_02.
    # Reference: G. E. Schr¨oder-Turk1(a),W.Mickel1,M.Schr¨oter2. Disordered spherical bead packs are anisotropic. doi: 10.1209/0295-5075/90/34001 May.
    w1_02_array = w1_02_array_input
    for a in range(len(area)):
        w1_02_array += np.outer(face_normal_vector[a], face_normal_vector[a]) * area[a]
    return w1_02_array


def compute_anisotropy_w1_02(area_all, face_normal_vector):
    anisotropic_coefficient = np.zeros(shape=[len(area_all), ])
    for x in range(len(area_all)):
        w1_02_array = np.zeros(shape=[3, 3])
        W1_02 = minkowski_tensor_w1_02(area_all[x], face_normal_vector[x], w1_02_array)
        eig = np.linalg.eig(np.mat(W1_02))[0]
        anisotropic_coefficient[x] = np.min(eig) / np.max(eig)
    return anisotropic_coefficient


@jit(nopython=True)
def MRO(old_feature_SRO_array_input, boop_SRO_array_input, anisotropic,
        MRO_array_input,
        f_use_array_input, voronoi_neighbour_input, neigh_id_length_index_input):
    # medium range order
    feature_MRO = np.empty_like(MRO_array_input)
    for aa in range(7):
        a = 5 * aa
        feature_now = old_feature_SRO_array_input[:, aa]
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(1):
        a = (7 + aa) * 5
        feature_now = anisotropic
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    for aa in range(10):
        a = (8 + aa) * 5
        feature_now = boop_SRO_array_input[:, aa]
        for b in range(len(voronoi_neighbour_input)):
            f_use_not = np.zeros_like(f_use_array_input)
            for c in range(neigh_id_length_index_input[b]):
                f_use_not[c] = feature_now[voronoi_neighbour_input[b][c]]
            f_use = f_use_not[0: neigh_id_length_index_input[b]]
            feature_MRO[a][b] = feature_now[b]
            feature_MRO[a + 1][b] = np.min(f_use)
            feature_MRO[a + 2][b] = np.max(f_use)
            feature_MRO[a + 3][b] = np.mean(f_use)
            mean = np.mean(f_use)
            square = 0.0
            for c in range(len(f_use)):
                square += (f_use[c] - mean) ** 2
            sqrt = math.sqrt((square / len(f_use)))
            feature_MRO[a + 4][b] = sqrt
    return feature_MRO


def zip_feature(Coordination_number_by_Voronoi_tessellation, cellfraction, a, b, c, d, e):
    feature_all = np.array(list(zip(Coordination_number_by_Voronoi_tessellation,
                                    cellfraction, a, b, c, d, e
                                    )))
    return feature_all


def compute_conventional_feature(points, area_all, face_normal_vector, neighbour, voronoi, radius):
    # step1. set constant
    particle_number = len(points)
    MRO_array = np.empty(shape=[90, particle_number])
    f_use_array = np.empty(shape=[particle_number, ])
    # step1. modify voronoi neighbour information
    voronoi_neighbour = []
    for x in range(len(neighbour)):
        voronoi_neighbour_now = []
        for value in neighbour[x]:
            if value >= 0:
                voronoi_neighbour_now.append(value)
        voronoi_neighbour.append(voronoi_neighbour_now)
    bonds = []
    for x in range(len(voronoi_neighbour)):
        for y in range(len(voronoi_neighbour[x])):
            if voronoi_neighbour[x][y] > x:
                bonds.append([x, voronoi_neighbour[x][y]])
    bonds = np.array(bonds)
    voronoi_neighbour_use = []
    for x in range(len(neighbour)):
        voronoi_neighbour_use.append([])
    for x in range(len(bonds)):
        voronoi_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voronoi_neighbour_use[bonds[x][1]].append(bonds[x][0])
    neigh_id_length_index = []
    for x in range(len(voronoi_neighbour_use)):
        neigh_id_length_index.append(len(voronoi_neighbour_use[x]))
    # 2020.10.24 在这里将中心颗粒也取为MR计算的一部分
    neigh_id = np.zeros(shape=[particle_number, max(neigh_id_length_index)+1], dtype=int)
    for x in range(len(voronoi_neighbour_use)):
        for y in range(len(voronoi_neighbour_use[x])):
            neigh_id[x][y] = int(voronoi_neighbour_use[x][y])
        neigh_id[x][len(voronoi_neighbour_use[x])] = int(x)
    neigh_id_length_index = np.array(neigh_id_length_index)
    # step2. compute
    # 2.1 coordination number by voronoi tessellation
    coordination_number_voronoi_tessellation = np.zeros(shape=[len(points), ])
    for x in range(len(voronoi_neighbour)):
        coordination_number_voronoi_tessellation[x] = len(voronoi_neighbour[x])
    # 2.4 cell fraction
    cellfraction = compute_cellfraction(voronoi, radius)
    # 2.5 voronoi index and i-fold symm
    voronoi_idx = compute_voronoi_idx(voronoi)
    # 2.6 zip feature above
    feature_all = zip_feature(coordination_number_voronoi_tessellation, cellfraction,
                              voronoi_idx['3'], voronoi_idx['4'], voronoi_idx['5'], voronoi_idx['6'], voronoi_idx['7'])
    # 2.7.1 boop
    boop_all = compute_boop(voronoi_neighbour, points)
    # 2.9 anisotropic of voronoi cell by Minkowski tensor W1_02
    anisotropic = compute_anisotropy_w1_02(area_all, face_normal_vector)
    # 2.10 select MRO
    old_feature_SRO_array = feature_all
    boop_SRO_array = boop_all
    feature_out = MRO(old_feature_SRO_array, boop_SRO_array, anisotropic, MRO_array, f_use_array, neigh_id,
                      neigh_id_length_index).T
    return feature_out


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_position_information(dump_path, frame):
    # 读取颗粒位置信息
    particle_info = open('../shear/post/dump-' + str(frame) + '.sample', 'r')
    lines = particle_info.readlines()
    particle_info.close()
    lines = lines[9:]
    Par_id = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    Par_id_read = list(map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
    Par_xcor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    Par_ycor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    Par_zcor_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    Par_radius_read = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    Par_id.sort()
    Par_xcor = [Par_xcor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_ycor = [Par_ycor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_zcor = [Par_zcor_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_radius = [Par_radius_read[Par_id_read.index(Par_id[x])] for x in range(len(Par_id))]
    Par_coord = np.array(list(zip(Par_xcor, Par_ycor, Par_zcor)))
    x_min = np.min([(Par_xcor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    y_min = np.min([(Par_ycor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    z_min = np.min([(Par_zcor[i] - Par_radius[i]) for i in range(len(Par_radius))])
    x_max = np.max([(Par_xcor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    y_max = np.max([(Par_ycor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    z_max = np.max([(Par_zcor[i] + Par_radius[i]) for i in range(len(Par_radius))])
    boundary = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    return Par_coord, Par_radius, boundary


def compute_area(vertices_input, adjacent_cell_input, vertices_id_input, simplice_input):
    area_judge_in = np.zeros(shape=[len(adjacent_cell_input), ], dtype=int)
    area_in = np.zeros(shape=[len(adjacent_cell_input), ])
    sing_area = np.zeros(shape=[len(simplice_input), ])
    for a in range(len(simplice_input)):
        sing_area[a] = triangle_area(vertices_input[simplice_input[a][0]],
                                             vertices_input[simplice_input[a][1]],
                                             vertices_input[simplice_input[a][2]])
    for a in range(len(simplice_input)):
        for b in range(len(adjacent_cell_input)):
            if simplice_input[a][0] in vertices_id_input[b]:
                if simplice_input[a][1] in vertices_id_input[b]:
                    if simplice_input[a][2] in vertices_id_input[b]:
                        area_in[b] += sing_area[a]
    average_area = np.mean(area_in)
    for a in range(len(adjacent_cell_input)):
        if area_in[a] >= 0.05 * average_area:
            area_judge_in[a] = 1
    return area_judge_in, area_in


def compute_face_normal_vector(vertices_id, vertices, origin):
    face_normal_vector = np.zeros(shape=[len(vertices_id), 3])
    for i in range(len(vertices_id)):
        normal = np.cross(vertices[int(vertices_id[i][0])] - vertices[int(vertices_id[i][1])],
                          vertices[int(vertices_id[i][0])] - vertices[int(vertices_id[i][2])])
        normal /= np.linalg.norm(normal)
        # if np.dot(vertices[int(vertices_id[i][0])] - origin, normal) < 0:
        # normal *= -1
        for j in range(3):
            face_normal_vector[i][j] = normal[j]
    return face_normal_vector


def eliminate_useless_adjacent_cell(voronoi, points):
    # 剔除面积小于平均面积百分之五的邻域点,这可能会造成互为邻域颗粒之间的不对称，后面的程序需要逐一处理
    adjacent_cell_all = []
    area_all_particle = []
    face_normal_vector_all = []
    for x in range(len(voronoi)):
        vertices = voronoi[x]['vertices']
        ch = ConvexHull(vertices)
        simplice = np.array(ch.simplices)
        faces = voronoi[x]['faces']
        adjacent_cell = []
        for y in range(len(faces)):
            adjacent_cell.append(faces[y]['adjacent_cell'])
        vert_id = []
        for y in range(len(faces)):
            vert_id.append(faces[y]['vertices'])
        area_judge, area = compute_area(vertices, adjacent_cell, vert_id, simplice)
        adjacent_cell_use = []
        for y in range(len(adjacent_cell)):
            if area_judge[y] == 1:
                adjacent_cell_use.append(adjacent_cell[y])
        adjacent_cell_all.append(adjacent_cell_use)
        area_all_particle.append(area)
        face_normal_vector = compute_face_normal_vector(vert_id, vertices, points[x])
        face_normal_vector_all.append(face_normal_vector)
    return adjacent_cell_all, area_all_particle, face_normal_vector_all


def compute_voronoi_neighbour(points, radius, limits, d50):
    dispersion = 5 * d50 / 2
    voronoi = pyvoro.compute_voronoi(points, limits, dispersion, radius, periodic=[False] * 3)
    neighbour, area, face_normal_vector = eliminate_useless_adjacent_cell(voronoi, points)
    return voronoi, neighbour, area, face_normal_vector


path = '../cyc16000fric002shearrate01press5'
path_output = '../python-code-data/property'
mkdir(path_output)
initial_step = 28000000
interval_step = 100000
cyc_number = 100
ini_number = int(argv[1])
d50 = 0.02
par_number = 16000

for i in range(cyc_number):
    i = i + ini_number
    #print(i)
    step = initial_step + i * interval_step

    Par_coord, Par_radius, boundary = read_position_information(path, step)
    voronoi, voronoi_neighbour, area_all, face_normal_vector = compute_voronoi_neighbour(Par_coord, Par_radius,
                                                                                             boundary, d50)
    print('The %d th frame' % step)

    conventional_feature = compute_conventional_feature(points=Par_coord, area_all=area_all,
                                                        face_normal_vector=face_normal_vector,
                                                        neighbour=voronoi_neighbour, voronoi=voronoi,
                                                        radius=Par_radius)

    columns_dict = ['coordination_number', 'coordination_number_min', 'coordination_number_max', 'coordination_number_mean', 'coordination_number_std',
                    'cellfraction', 'cellfraction_min', 'cellfraction_max',
                    'cellfraction_mean', 'cellfraction_std',
                    'anisotropic_coefficient','anisotropic_coefficient_min', 'anisotropic_coefficient_max',
                    'anisotropic_coefficient_mean', 'anisotropic_coefficient_std',
                    'q2', 'q2_min', 'q2_max',
                    'q2_mean', 'q2_std',
                    'q4', 'q4_min', 'q4_max',
                    'q4_mean', 'q4_std',
                    'q6', 'q6_min', 'q6_max',
                    'q6_mean', 'q6_std',
                    'q8', 'q8_min', 'q8_max',
                    'q8_mean', 'q8_std',
                    'q10', 'q10_min', 'q10_max',
                    'q10_mean', 'q10_std',
                    'w2', 'w2_min', 'w2_max',
                    'w2_mean', 'w2_std',
                    'w4', 'w4_min', 'w4_max',
                    'w4_mean', 'w4_std',
                    'w6', 'w6_min', 'w6_max',
                    'w6_mean', 'w6_std',
                    'w8', 'w8_min', 'w8_max',
                    'w8_mean', 'w8_std',
                    'w10', 'w10_min', 'w10_max',
                    'w10_mean', 'w10_std']

    file_name = 'SIP' + str(i) + '.txt'
    atomfile = open('../python-code-data/property/' + file_name, 'w')
    for i in range(par_number):
        for j in range(90):
            atomfile.write(str(conventional_feature[i][j]))
            atomfile.write(' ')
        atomfile.write('\n')
    atomfile.close()
