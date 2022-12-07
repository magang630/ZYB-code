# -*- coding: cp936 -*-
from __future__ import division
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib
from scipy import optimize
from PIL import Image
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from scipy.optimize import fsolve
from sympy import *
from sympy.abc import x
from scipy import integrate
import re
from scipy.stats import describe, skew, kurtosis
from sklearn.manifold import TSNE

statistics_array = np.zeros(shape=[6250, 13])
micro_structure_array = np.zeros(shape=[6250, 12])

grading_list = ['D-2.9', 'D-2.5', 'D-2.0', 'D-1.0', 'D--2.0', 'D--5.0', 'Bell', 'Monosize', 'Binary', 'Uniform']
friction_list = [0.1, 0.2, 0.3, 0.4, 0.5]
initial_friction_list = [0.2, 0.4, 0.6, 0.8, 1.0]
ptc_youngsModulus_list = ['1e9', '5e9', '10e9', '15e9', '20e9']
ptc_youngsModulus_value_list = [1, 5, 10, 15, 20]
cofining_press_list = ['0.01e6', '0.05e6', '0.1e6', '0.5e6', '1e6']
cofining_press_value_list = [10, 50, 100, 500, 1000]
n = 0
for aa, grading in enumerate(grading_list):
    for bb, friction in enumerate(friction_list):
        for cc, initial_friction in enumerate(initial_friction_list):
            for dd, ptc_youngsModulus in enumerate(ptc_youngsModulus_list):
                for ee, cofining_press in enumerate(cofining_press_list):
                    path = 'I:/Conventional-triaxial-test-ppp/Python data/cellfraction'
                    atomfile = open(path + '/' + grading + '-' + str(friction) + '-' +
                                    str(initial_friction) + '-' +  ptc_youngsModulus + '-' + cofining_press + '-cellfraction.txt', 'r')
                    lines = atomfile.readlines()
                    atomfile.close()
                    cellfraction = list(map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
                    statistics_array[n][0] = describe(cellfraction)[2]
                    statistics_array[n][1] = describe(cellfraction)[3]
                    statistics_array[n][2] = describe(cellfraction)[4]
                    statistics_array[n][3] = describe(cellfraction)[5]
                    micro_structure_array[n][0] = describe(cellfraction)[2]
                    micro_structure_array[n][1] = describe(cellfraction)[3]
                    micro_structure_array[n][2] = describe(cellfraction)[4]
                    micro_structure_array[n][3] = describe(cellfraction)[5]
                    path = 'I:/Conventional-triaxial-test-ppp/Python data/contact-number'
                    atomfile = open(path + '/' + grading + '-' + str(friction) + '-' + str(initial_friction)
                                    + '-' + ptc_youngsModulus + '-' + cofining_press + '-contact-number.txt', 'r')
                    lines = atomfile.readlines()
                    atomfile.close()
                    contact_number = list(map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
                    statistics_array[n][4] = describe(contact_number)[2]
                    statistics_array[n][5] = describe(contact_number)[3]
                    statistics_array[n][6] = describe(contact_number)[4]
                    statistics_array[n][7] = describe(contact_number)[5]
                    micro_structure_array[n][4] = describe(contact_number)[2]
                    micro_structure_array[n][5] = describe(contact_number)[3]
                    micro_structure_array[n][6] = describe(contact_number)[4]
                    micro_structure_array[n][7] = describe(contact_number)[5]
                    path = 'I:/Conventional-triaxial-test-ppp/Python data/coordination-number'
                    atomfile = open(path + '/' + grading + '-' + str(friction) + '-' + str(initial_friction)
                                    + '-' + ptc_youngsModulus + '-' + cofining_press + '-coordination-number.txt', 'r')
                    lines = atomfile.readlines()
                    atomfile.close()
                    coordination_number = list(
                        map(float, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines])))
                    statistics_array[n][8] = describe(coordination_number)[2]
                    statistics_array[n][9] = describe(coordination_number)[3]
                    statistics_array[n][10] = describe(coordination_number)[4]
                    statistics_array[n][11] = describe(coordination_number)[5]
                    micro_structure_array[n][8] = describe(coordination_number)[2]
                    micro_structure_array[n][9] = describe(coordination_number)[3]
                    micro_structure_array[n][10] = describe(coordination_number)[4]
                    micro_structure_array[n][11] = describe(coordination_number)[5]
                    n += 1

reduction = TSNE(n_components=1).fit_transform(micro_structure_array)
for i in range(len(statistics_array)):
    statistics_array[i][12] = reduction[i]

pd.DataFrame(statistics_array).to_csv('I:/Conventional-triaxial-test-ppp/Python data/micro-structure-statistics-information.csv')
