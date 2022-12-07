# -*- coding: cp936 -*-
from __future__ import division
import re
import os


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


path = 'J:/Cylic-shear-monosize/Different friction/fric-005/liggghts-data/shear/post/'
path_output = 'J:/Cylic-shear-monosize/Different friction/fric-005/' + '/sort position/'
mkdir(path_output)
initial_step = 27400000
interval_step = 100000
cyc_number = 5001
ini_number = 0

for i in range(cyc_number):
    i = i + ini_number
    print(i)
    step = initial_step + i * interval_step
    tem = '%d' % step
    tem1 = '%d' % i
    filename = 'dump-' + tem + '.sample'

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

    atomfile = open(path_output + filename, 'w')
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
