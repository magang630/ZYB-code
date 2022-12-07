# -*- coding: cp936 -*-
from __future__ import division
import re
import numpy as np
import random
import pandas as pd

path = 'cyc16000fric01shearrate01SA01'
initial_step = 25600000
interval_step = 200000
radius = 0.01
par_number = 16000
cyc_number = 3001
ini_number = 0
time_windows = 5

crystalline_all = np.zeros(shape=[par_number, cyc_number], dtype=int)
q6_all = np.zeros(shape=[par_number, cyc_number], dtype=float)
cellfraction_all = np.zeros(shape=[par_number, cyc_number], dtype=float)

for i in range(cyc_number):
    i = i + ini_number
    print(i)

    boo_index = 'boo_index' + str(i) + '.txt'
    q6_file = open('../' + path + '/boo index/' + boo_index, 'r')
    lines = q6_file.readlines()
    q6_file.close()
    q6 = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))

    cellfraction_file = 'cellfraction' + str(i) + '.txt'
    fraction_file = open('../' + path + '/cellfraction/' + cellfraction_file, 'r')
    lines = fraction_file.readlines()
    fraction_file.close()
    cellfraction = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines]))

    for x in range(len(cellfraction)):
        q6_all[x][i] = q6[x]
        cellfraction_all[x][i] = cellfraction[x]
        if 0.555 <= q6[x] <= 0.595 and cellfraction[x] > 0.72:
            crystalline_all[x][i] = 1
        if 0.465 <= q6[x] <= 0.505 and cellfraction[x] > 0.72:
            crystalline_all[x][i] = 1

for i in range(cyc_number):
    dataset = 'stable_cry_state' + str(i) + '.txt'
    atomfile = open('../' + path + '/stable state/' + dataset, 'w')
    i = i + ini_number
    crystalline_now = crystalline_all[:, i]
    print(i)
    if i == 0:
        for x in range(len(crystalline_now)):
            q6_now = [q6_all[x][0], q6_all[x][1], q6_all[x][2]]
            cellfraction_now = [cellfraction_all[x][0], cellfraction_all[x][1], cellfraction_all[x][2]]
            q6_ave = np.mean(q6_now)
            cellfraction_ave = np.mean(cellfraction_now)
            if 0.555 <= q6_ave <= 0.595 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            elif 0.465 <= q6_ave <= 0.505 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            else:
                atomfile.write(str(0))
                atomfile.write('\n')
    elif i == 1:
        for x in range(len(crystalline_now)):
            q6_now = [q6_all[x][0], q6_all[x][1], q6_all[x][2], q6_all[x][3]]
            cellfraction_now = [cellfraction_all[x][0], cellfraction_all[x][1], cellfraction_all[x][2], cellfraction_all[x][3]]
            q6_ave = np.mean(q6_now)
            cellfraction_ave = np.mean(cellfraction_now)
            if 0.555 <= q6_ave <= 0.595 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            elif 0.465 <= q6_ave <= 0.505 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            else:
                atomfile.write(str(0))
                atomfile.write('\n')
    elif i == 4999:
        for x in range(len(crystalline_now)):
            q6_now = [q6_all[x][5000], q6_all[x][4999], q6_all[x][4998], q6_all[x][4997]]
            cellfraction_now = [cellfraction_all[x][5000], cellfraction_all[x][4999], cellfraction_all[x][4998],
                                cellfraction_all[x][4997]]
            q6_ave = np.mean(q6_now)
            cellfraction_ave = np.mean(cellfraction_now)
            if 0.555 <= q6_ave <= 0.595 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            elif 0.465 <= q6_ave <= 0.505 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            else:
                atomfile.write(str(0))
                atomfile.write('\n')
    elif i == 5000:
        for x in range(len(crystalline_now)):
            q6_now = [q6_all[x][5000], q6_all[x][4999], q6_all[x][4998]]
            cellfraction_now = [cellfraction_all[x][5000], cellfraction_all[x][4999], cellfraction_all[x][4998]]
            q6_ave = np.mean(q6_now)
            cellfraction_ave = np.mean(cellfraction_now)
            if 0.555 <= q6_ave <= 0.595 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            elif 0.465 <= q6_ave <= 0.505 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            else:
                atomfile.write(str(0))
                atomfile.write('\n')
    else:
        for x in range(len(crystalline_now)):
            q6_now = [q6_all[x][i - a - 2] for a in range(time_windows)]
            cellfraction_now = [cellfraction_all[x][i - a - 2] for a in range(time_windows)]
            q6_ave = np.mean(q6_now)
            cellfraction_ave = np.mean(cellfraction_now)
            if 0.555 <= q6_ave <= 0.595 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            elif 0.465 <= q6_ave <= 0.505 and cellfraction_ave > 0.72:
                atomfile.write(str(1))
                atomfile.write('\n')
            else:
                atomfile.write(str(0))
                atomfile.write('\n')
