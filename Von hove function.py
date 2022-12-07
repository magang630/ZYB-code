# -*- coding: cp936 -*-
from __future__ import division
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 0. DEM parameter
d50 = 0.02

# 1. Load data
path = 'cyc16000fric002shearrate01polysize'
initial_step = 28000000
interval_step = 100000
initial_number = 500
cycle_number = 1001

step = initial_step + initial_number * interval_step
filename = 'dump-' + str(step) + '.sample'
atomfile = open('../' + path + '/sort position/' + filename, 'r')
lines = atomfile.readlines()
atomfile.close()
lines = lines[9:]
position_x = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
position_y = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
position_z = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
radius = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
points = np.array(list(zip(position_x, position_y, position_z)))
x_min = np.min([(position_x[i] - radius[i]) for i in range(len(radius))])
x_max = np.max([(position_x[i] + radius[i]) for i in range(len(radius))])
y_min = np.min([(position_y[i] - radius[i]) for i in range(len(radius))])
y_max = np.max([(position_y[i] + radius[i]) for i in range(len(radius))])
z_min = np.min([(position_z[i] - radius[i]) for i in range(len(radius))])
z_max = np.max([(position_z[i] + radius[i]) for i in range(len(radius))])
select_boundary = [[x_min + 5.0 * d50, x_max - 5.0 * d50],
                   [y_min + 5.0 * d50, y_max - 5.0 * d50],
                   [z_min + 5.0 * d50, z_max - 5.0 * d50]]
par_number = len(points)

pos = []
rad = []
for i in range(cycle_number):
    step = initial_step + (initial_number + i) * interval_step
    filename = 'dump-' + str(step) + '.sample'
    atomfile = open('../' + path + '/sort position/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    lines = lines[9:]
    position_x = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines]))
    position_y = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines]))
    position_z = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines]))
    radius = list(map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines]))
    points = np.array(list(zip(position_x, position_y, position_z)))
    pos.append(points)
    rad.append(radius)

# 3、Gs(d,t)
length = d50 * 0.4
d_distance = 0.05 * d50 / 4
number = int(length / d_distance) + 1

length_1 = d50 * 0.4
d_distance_1 = 0.05 * d50 / 8
number_1 = int(length_1 / d_distance_1) + 1

length_2 = d50 * 0.4
d_distance_2 = 0.05 * d50 / 8
number_2 = int(length_2 / d_distance_2) + 1

pos_initial = pos[0]
select_number = [1, 10, 100, 400, 700, 1000]
von_hove = [[]] * (len(select_number))
for m in range(len(select_number)):
    if m == 0:
        rad_now = rad[m]
        now_von_hove = []
        for j in range(number_1):
            sum = 0.0
            d = j * d_distance_1
            d1 = d - d_distance_1 / 2
            d2 = d + d_distance_1 / 2
            n = 0
            for i in range(par_number):
                if select_boundary[0][0] <= pos_initial[i][0] <= select_boundary[0][1] \
                        and select_boundary[1][0] <= pos_initial[i][1] <= select_boundary[1][1] \
                        and select_boundary[2][0] <= pos_initial[i][2] <= select_boundary[2][1]:
                    # if rad_now[i] * 2 < d50:
                        n += 1
                        dis_x = abs(pos[select_number[m]][i][0] - pos_initial[i][0])
                        dis_y = abs(pos[select_number[m]][i][1] - pos_initial[i][1])
                        dis_z = abs(pos[select_number[m]][i][2] - pos_initial[i][2])
                        move = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                        if d1 <= move < d2:
                            # sum += 1
                            sum += 1 / d_distance_1
            sum = 4 * math.pi * d50 ** 2 * sum / n
            # sum = sum/4 * math.pi * d ** 2
            now_von_hove.append(sum)
        von_hove[m] = now_von_hove
    if 1 <= m <= 2:
        rad_now = rad[m]
        now_von_hove = []
        for j in range(number_2):
            sum = 0.0
            d = j * d_distance_2
            d1 = d - d_distance_2 / 2
            d2 = d + d_distance_2 / 2
            n = 0
            for i in range(par_number):
                if select_boundary[0][0] <= pos_initial[i][0] <= select_boundary[0][1] \
                        and select_boundary[1][0] <= pos_initial[i][1] <= select_boundary[1][1] \
                        and select_boundary[2][0] <= pos_initial[i][2] <= select_boundary[2][1]:
                    # if rad_now[i] * 2 < d50:
                        n += 1
                        dis_x = abs(pos[select_number[m]][i][0] - pos_initial[i][0])
                        dis_y = abs(pos[select_number[m]][i][1] - pos_initial[i][1])
                        dis_z = abs(pos[select_number[m]][i][2] - pos_initial[i][2])
                        move = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                        if d1 <= move < d2:
                            # sum += 1
                            sum += 1 / d_distance_2
            sum = 4 * math.pi * d50 ** 2 * sum / n
            # sum = sum/4 * math.pi * d ** 2
            now_von_hove.append(sum)
        von_hove[m] = now_von_hove
    if m >= 3:
        rad_now = rad[m]
        now_von_hove = []
        for j in range(number):
            sum = 0.0
            d = j * d_distance
            d1 = d - d_distance / 2
            d2 = d + d_distance / 2
            n = 0
            for i in range(par_number):
                if select_boundary[0][0] <= pos_initial[i][0] <= select_boundary[0][1] \
                        and select_boundary[1][0] <= pos_initial[i][1] <= select_boundary[1][1] \
                        and select_boundary[2][0] <= pos_initial[i][2] <= select_boundary[2][1]:
                    # if rad_now[i] * 2 < d50:
                        n += 1
                        dis_x = abs(pos[select_number[m]][i][0] - pos_initial[i][0])
                        dis_y = abs(pos[select_number[m]][i][1] - pos_initial[i][1])
                        dis_z = abs(pos[select_number[m]][i][2] - pos_initial[i][2])
                        move = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                        if  d1 <= move < d2:
                            # sum += 1
                            sum += 1 / d_distance
            sum = 4 * math.pi * d50 ** 2 * sum / n
            # sum = sum/4 * math.pi * d ** 2
            now_von_hove.append(sum)
        von_hove[m] = now_von_hove

# step3、Plot
xaxis = []
for i in range(number):
    d = i * d_distance
    d = d / d50
    xaxis.append(d)

xaxis_1 = []
for i in range(number_1):
    d = i * d_distance_1
    d = d / d50
    xaxis_1.append(d)

xaxis_2 = []
for i in range(number_2):
    d = i * d_distance_2
    d = d / d50
    xaxis_2.append(d)

plot = []
legend = []
for i in range(len(select_number)):
    plot.append(von_hove[i])
    legend_now = r'$T$ = ' + '%d' % (select_number[i])
    legend.append(legend_now)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
legend_font = {"family": 'Arial', "size": 8}
fig, ax = plt.subplots(figsize=[3.5, 3], dpi=600)

color = ['#D75126', '#6BB4E2', '#70BB5E', '#904A95', '#DDB018', '#3A8AC5', '#E5302D']
shape = ['o', 's', '^', 'v', 'D', 'p']

plt.plot(xaxis_1, plot[0], 's-', label=legend[0], markerfacecolor='w', markersize=2, linewidth=1, color=color[0])
plt.plot(xaxis_2, plot[1], '^-', label=legend[1], markerfacecolor='w',  markersize=2, linewidth=1, color=color[1])
plt.plot(xaxis_2, plot[2], 'v-', label=legend[2], markerfacecolor='w', markersize=2, linewidth=1, color=color[2])
plt.plot(xaxis, plot[3], '>-', label=legend[3], markerfacecolor='w', markersize=2, linewidth=1, color=color[3])
plt.plot(xaxis, plot[4], '<-', label=legend[4], markerfacecolor='w', markersize=2, linewidth=1, color=color[4])
plt.plot(xaxis, plot[5], 'o-', label=legend[5], markerfacecolor='w', markersize=2, linewidth=1, color=color[5])
# plt.plot(xaxis, plot[6], 'k-p', label=legend[6], markerfacecolor='w', markersize=2, linewidth=1)

plt.xlabel(r'Displacement,$d$($d$$\rm_{50}$)', fontdict=legend_font)
plt.ylabel('4$\pi$$d^2$Gs($d$,$t$)', fontdict=legend_font)
plt.legend(edgecolor='w', prop=legend_font)
plt.xticks(fontproperties='Arial', size=8)
plt.yticks(fontproperties='Arial', size=8)
plt.xlim(0, 0.4)
plt.ylim(0, 10)
xminorLocator = MultipleLocator(0.05)
ax.xaxis.set_minor_locator(xminorLocator)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
yminorLocator = MultipleLocator(1)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_minor_formatter(plt.NullFormatter())
# path = os.path.dirname(os.path.abspath(__file__))
# figurename = 'Gs(d,t)5' + '.png'
# plt.savefig(path + '/' + figurename, dpi=600)
plt.show()
