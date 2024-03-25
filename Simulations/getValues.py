import numpy as np
import math
from scipy.optimize import fmin

Na_gamma = 11.262e6  # 1/(T*s), gyromagnetic ratio for 23Na



def get_bruker_groupDelay(acqp):
    return np.floor(np.double(acqp['ACQ_RxFilterInfo'][1:-1].split(',')[0]))
def get_randomAB(a, b, N):
    r = a + (b - a) * np.random.rand(N)
    return r


def get_wPnorm(vec, w, p):
    wNMRS = np.linalg.norm(vec * w, p)
    return wNMRS

def get_w0(B0):
    w0 = Na_gamma * B0
    return w0

def get_ptauC(tauCs, a, b, tcm):
    Norm = 2 * a * b / (np.sqrt(np.pi) * (a + b) * tauCs)
    ptauCs = Norm * np.exp(-a ** 2 * (np.log(tauCs / tcm)) ** 2)
    ptauCs[tauCs > tcm] = Norm[tauCs > tcm] * np.exp(-b ** 2 * (np.log(tauCs[tauCs > tcm] / tcm)) ** 2)
    return ptauCs


def get_wQs(tauCs, tauC0, wQ0, wQ1):
    wQs = np.ones((1, len(tauCs))) * wQ0
    wQs[tauCs > tauC0] = wQ1
    return wQs


def get_logRange(startExp, endExp, stepSize):
    r_tmp = np.arange(startExp, endExp)
    t_tmp = 10 ** r_tmp
    logRange_tmp = np.arange(1, 10, stepSize).reshape(-1, 1) * t_tmp
    logRange = logRange_tmp.reshape(1, -1).squeeze()
    logRange = np.concatenate((logRange, [10 ** endExp]))
    return logRange
def get_J(m, tauC, wQ, w0):
    x = (w0 * tauC) ** 2  # s
    Jm = (wQ ** 2) / 5 * tauC / (1 + m ** 2 * x)
    Km = m * w0 * tauC * Jm
    return Jm, Km
# Jen Model

def get_J_Jen(m, tauC, wQ, Jen, w0):
    x = (w0 * tauC) ** 2  # s
    Jm = (wQ ** 2) / 5 * tauC / (1 + m ** 2 * x) + Jen
    Km = m * w0 * tauC * Jm
    return Jm, Km

def get_JenModel(T1f, T1s, T2f, T2s, w0):
    Ts_mess = np.array([T1f, T1s, T2f, T2s])
    Rs_mess = 1 / Ts_mess

    wQbar = 0
    Jen0 = 10
    tauC0 = 5e-8
    wQ0 = 1e5
    wShift_RMS0 = 7
    initialVec = [Jen0, tauC0, wQ0, wShift_RMS0]

    weights = Ts_mess
    p = 2  # p-Norm value

    def MinFun(JenVal):
        return get_wPnorm((get_Rs_ZQSQ_Jen(JenVal[0], JenVal[1], JenVal[2], wQbar, JenVal[3], w0) - Rs_mess), weights,p)

    JenVal = fmin(MinFun, initialVec)

    Jen, tauC, wQ, wShift_RMS = JenVal[0], JenVal[1], JenVal[2], JenVal[3]
    return  Jen, tauC, wQ, wShift_RMS

def get_tauOpt(Tif, Tis):
    return np.log(Tis/Tif)/(1/Tif-1/(Tis))
def get_relaxationTimes_Jen(tauC, wQ, wQbar, Jen, wShift_RMS, w0):
    R0s = get_Rs_Jen(0, tauC, wQ, wQbar, Jen, wShift_RMS, w0)
    R1s = get_Rs_Jen(1, tauC, wQ, wQbar, Jen, wShift_RMS, w0)
    T1s = 1 / R0s[1]
    T1f = 1 / R0s[0]
    T2s = 1 / R1s[1]
    T2f = 1 / R1s[0]
    return T1s, T1f, T2s, T2f

def get_relaxationTimes(tauC, wQ, w0):
    sizeTauC = len(tauC)
    sizeWQ = len(wQ)
    tauCs = np.reshape(np.repeat(tauC, sizeWQ), (sizeWQ, sizeTauC))  # s
    wQs = np.reshape(np.repeat(wQ, sizeTauC), (sizeTauC, sizeWQ)).T  # Hz

    x = (w0 * tauCs) ** 2  # s
    a0 = (1 + 4 * x) / (1 + x)
    b0 = (6 / 5 * x * tauCs * wQs ** 2) / (4 * x ** 2 + 5 * x + 1)
    a1 = (2 + 9 * x + 4 * x ** 2) / (2 + 5 * x)
    b1 = 1 / 5 * tauCs * wQs ** 2 * (4 * x) / (1 + 4 * x)

    T1f = (a0 - 1) / (a0 * b0)
    T1s = (a0 - 1) / (b0)
    T2f = (a1 - 1) / (a1 * b1)
    T2s = (a1 - 1) / (b1)

    return T1s, T1f, T2s, T2f


def get_Rs(q, tauC, wQ, w0):
    J0, _ = get_J(0, tauC, wQ, w0)
    J1, _ = get_J(1, tauC, wQ, w0)
    J2, _ = get_J(2, tauC, wQ, w0)

    if q == 0:
        R01 = 2 * J1
        R02 = 2 * J2
        R03 = 2 * J1 + 2 * J2
        Rs = [R01, R02, R03]
    elif q == 1 or q == -1:
        R11 = J0 + J1
        R12 = J1 + J2
        R13 = J0 + J1 + 2 * J2
        Rs = [R11, R12, R13]
    elif q == 2 or q == -2:
        R21 = J0 + 2 * J1 + J2
        R22 = J0 + J2
        Rs = [R21, R22]
    elif q == 3 or q == -3:
        R31 = J1 + J2
        Rs = [R31]

    return Rs


def get_Rs_Jen(q, tauC, wQ, wQbar, Jen, wShift_RMS, w0):
    J0, _ = get_J_Jen(0, tauC, wQ, Jen, w0)
    J1, _ = get_J_Jen(1, tauC, wQ, Jen, w0)
    J2, _ = get_J_Jen(2, tauC, wQ, Jen, w0)

    if q == 0:
        R01 = 2 * J1
        R02 = 2 * J2
        R03 = 2 * J1 + 2 * J2
        Rs = [R01, R02, R03]
    elif q == 1 or q == -1:
        R11 = J0 + J1 + J2 - np.sqrt(J2 ** 2 - wQbar ** 2) + abs(q) * wShift_RMS
        R12 = J1 + J2 + abs(q) * wShift_RMS
        R13 = J0 + J1 + J2 + np.sqrt(J2 ** 2 - wQbar ** 2) + abs(q) * wShift_RMS
        Rs = [R11, R12, R13]
    elif q == 2 or q == -2:
        R21 = J0 + J1 + J2 + np.sqrt(J1 ** 2 - wQbar ** 2) + abs(q) * wShift_RMS
        R22 = J0 + J1 + J2 - np.sqrt(J1 ** 2 - wQbar ** 2) + abs(q) * wShift_RMS
        Rs = [R21, R22]
    elif q == 3 or q == -3:
        R31 = J1 + J2 + abs(q) * wShift_RMS
        Rs = [R31]

    return Rs
def get_Rs_aniso_ptauC_wQ(q, a, b, tcm, wQ0, wQ1, tauC0, wQbar, w0):
    tauCs = get_logRange(-14, -6, 0.1)
    ptauCs = get_ptauC(tauCs, a, b, tcm)
    deltaTauC = np.array([tauCs[1] - tauCs[0]] + list(np.diff(tauCs)))
    wQs = get_wQs(tauCs, tauC0, wQ0, wQ1)

    J0s, _ = get_J(0, tauCs, wQs, w0)
    J1s, _ = get_J(1, tauCs, wQs, w0)
    J2s, _ = get_J(2, tauCs, wQs, w0)

    J0 = np.sum(J0s * ptauCs * deltaTauC)
    J1 = np.sum(J1s * ptauCs * deltaTauC)
    J2 = np.sum(J2s * ptauCs * deltaTauC)

    if q == 0:
        R01 = 2 * J1
        R02 = 2 * J2
        R03 = 2 * J1 + 2 * J2
        Rs = [R01, R02, R03]
    elif q == 1 or q == -1:
        R11 = (J0 + J1 + J2 - np.sqrt(J2 ** 2 - wQbar ** 2))
        R12 = (J1 + J2)
        R13 = (J0 + J1 + J2 + np.sqrt(J2 ** 2 - wQbar ** 2))
        Rs = [R11, R12, R13]
    elif q == 2 or q == -2:
        R21 = (J0 + J1 + J2 + np.sqrt(J1 ** 2 - wQbar ** 2))
        R22 = (J0 + J1 + J2 - np.sqrt(J1 ** 2 - wQbar ** 2))
        Rs = [R21, R22]
    elif q == 3 or q == -3:
        R31 = (J1 + J2)
        Rs = [R31]

    return Rs

def get_Rs_ZQSQ_Jen(Jen, tauC, wQ, wQbar, wShift_RMS, w0):
    R0s = get_Rs_Jen(0, tauC, wQ, wQbar, Jen, wShift_RMS, w0)
    R1s = get_Rs_Jen(1, tauC, wQ, wQbar, Jen, wShift_RMS, w0)
    Rs = np.concatenate([R0s[:2], R1s[:2]])
    return Rs

def get_f(q, t, tauC, wQ, w0):
    J0, _ = get_J(0, tauC, wQ, w0)
    J1, _ = get_J(1, tauC, wQ, w0)
    J2, _ = get_J(2, tauC, wQ, w0)

    if q == 0:
        R01 = 2 * J1
        R02 = 2 * J2
        R03 = 2 * J1 + 2 * J2
        f011 = 1 / 5 * (np.exp(-R01 * t) + 4 * np.exp(-R02 * t))
        f013 = 2 / 5 * (np.exp(-R01 * t) - np.exp(-R02 * t))
        f033 = 1 / 5 * (4 * np.exp(-R01 * t) + np.exp(-R02 * t))
        f022 = np.exp(-R03 * t)
        fqkks = [f011, f013, f033, f022]
    elif q == 1 or q == -1:
        R11 = J0 + J1
        R12 = J1 + J2
        R13 = J0 + J1 + 2 * J2
        f111 = 1 / 5 * (3 * np.exp(-R11 * t) + 2 * np.exp(-R12 * t))
        f113 = np.sqrt(6) / 5 * (np.exp(-R11 * t) - np.exp(-R12 * t))
        f133 = 1 / 5 * (2 * np.exp(-R11 * t) + 3 * np.exp(-R12 * t))
        f122 = np.exp(-R13 * t)
        fqkks = [f111, f113, f133, f122]
    elif q == 2 or q == -2:
        R21 = J0 + 2 * J1 + J2
        R22 = J0 + J2
        f222 = np.exp(-R21 * t)
        f233 = np.exp(-R22 * t)
        fqkks = [f222, f233]
    elif q == 3 or q == -3:
        R31 = J1 + J2
        f333 = np.exp(-R31 * t)
        fqkks = [f333]

    return fqkks

#  Relaxation functions f(t) --> Km's Beiträge erstmal vernachlässigt
#  anisotropic environment, B0 inhomgeneities
# Jen Model
def get_f_Jen(q, t, tauC, wQ, wQbar, Jen, wShift_RMS, w0):
    J0, K0 = get_J_Jen(0, tauC, wQ, Jen, w0)
    J1, K1 = get_J_Jen(1, tauC, wQ, Jen, w0)
    J2, K2 = get_J_Jen(2, tauC, wQ, Jen, w0)
    Rs = get_Rs_Jen(q, tauC, wQ, wQbar, Jen, wShift_RMS, w0)

    if q == 0:
        R01, R02, R03 = Rs
        f011 = 1/5 * (np.exp(-R01 * t) + 4 * np.exp(-R02 * t))
        f013 = 2/5 * (np.exp(-R01 * t) - np.exp(-R02 * t))
        f033 = 1/5 * (4 * np.exp(-R01 * t) + np.exp(-R02 * t))
        f022 = np.exp(-R03 * t)
        fqkks = np.vstack((f011, f013, f033, f022))
    elif q == 1 or q == -1:
        R11, R12, R13 = Rs
        mu = J2 / np.sqrt(J2**2 - wQbar**2)
        v = wQbar / np.sqrt(J2**2 - wQbar**2)

        f111 = 1/5 * (3/2 * (1 + mu) * np.exp(-R11 * t) + 2 * np.exp(-R12 * t) + 3/2 * (1 - mu) * np.exp(-R13 * t))
        f122 = 1/2 * ((1 - mu) * np.exp(-R11 * t) + (1 + mu) * np.exp(-R13 * t))
        f133 = 1/5 * ((1 + mu) * np.exp(-R11 * t) + 3 * np.exp(-R12 * t) + (1 - mu) * np.exp(-R13 * t))
        f113 = np.sqrt(6)/5 * (1/2 * (1 + mu) * np.exp(-R11 * t) - np.exp(-R12 * t) + 1/2 * (1 - mu) * np.exp(-R13 * t))
        f112 = 1j/2 * np.sqrt(3)/5 * v * (np.sign(q) * np.exp(-R11 * t) - np.sign(q) * np.exp(-R13 * t))
        f123 = 1j/np.sqrt(10) * v * (np.sign(q) * np.exp(-R11 * t) - np.sign(q) * np.exp(-R13 * t))

        fqkks = np.vstack((f111, f113, f133, f112, f123, f122))  # * 1/2 * (q + 1)
    elif q == 2 or q == -2:
        R21, R22 = Rs
        mu = J1 / np.sqrt(J1**2 - wQbar**2)
        v = wQbar / np.sqrt(J1**2 - wQbar**2)

        f222 = 1/2 * ((1 + mu) * np.exp(-R21 * t) + (1 - mu) * np.exp(-R22 * t))
        f233 = 1/2 * ((1 + mu) * np.exp(-R21 * t) + (1 - mu) * np.exp(-R22 * t))
        f223 = -1j/2 * v * (np.sign(q) * np.exp(-R21 * t) - np.sign(q) * np.exp(-R22 * t))
        fqkks = np.vstack((f222, f233, f223))
    elif q == 3 or q == -3:
        R31 = Rs[0]
        f333 = np.exp(-R31 * t)
        fqkks = np.vstack((f333))

    return fqkks

# #
def get_WignerD(j, mbar, m, theta):
    """
    get the nxn pulse superoperator for a specific order m
    :param j:
    :param mbar:
    :param m:
    :param theta:
    :return:
    """
    smax = max([m - mbar, j - mbar, j + m])  # find max s value, such that all factorials are non-negative
    smax = max(smax, 0)
    djmm = 0

    for s in range(smax + 1):
        if mbar - m + s < 0 or j + m - s < 0 or j - mbar - s < 0:
            continue

        term = ((-1) ** (mbar - m + s) /
                (math.factorial(j + m - s) * math.factorial(s) * math.factorial(mbar - m + s) * math.factorial(
                    j - mbar - s)) *
                (math.cos(math.radians(theta / 2)) ** (2 * j + m - mbar - 2 * s)) *
                (math.sin(math.radians(theta / 2)) ** (mbar - m + 2 * s)))

        djmm += term

    djmm *= math.sqrt(
        math.factorial(j + mbar) * math.factorial(j - mbar) * math.factorial(j + m) * math.factorial(j - m))
    return djmm


def get_pulseOperator(flipAngle, phase, m):
    ns = np.arange(-3, 4)
    pulseOperator = np.zeros((len(ns), len(ns)), dtype=complex)

    for nIndex, n in enumerate(ns):
        for nBarIndex, nbar in enumerate(ns):
            if abs(n) <= m and abs(nbar) <= m:
                pulseOperator[nIndex, nBarIndex] = np.exp(1j * (n - nbar) * np.deg2rad(phase)) * get_WignerD(
                    m, n, nbar, flipAngle)

    return pulseOperator