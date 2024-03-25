##########################################################
# Phase-cycle simulation parameters
import numpy as np

T2s_f = 6.75
T2s_l = 30.07
T1 = 38

tau1 = 10e-3
# 2 cycles
Xi = np.array([0, 90])

omega = 2 * np.pi * 0 # offset in rad/s

phi = np.array([60, 120, 180, 240, 300, 360])

# Xi0  Xi90
# [(1)  (2)]
RF_1_phi = [90+phi, 90+phi]
RF_2_phi = [90+phi+Xi[0], 90+phi+Xi[1]]
RF_3_phi = np.array([0, 0]) * phi
sADC_phi = np.array([0, 0]) * phi