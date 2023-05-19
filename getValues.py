import numpy as np


class getValues:
    Na_gamma = 11.262e6  # 1/(T*s), gyromagnetic ratio for 23Na

    @staticmethod
    def get_randomAB(a, b, N):
        r = a + (b - a) * np.random.rand(N)
        return r

    @staticmethod
    def get_wPnorm(vec, w, p):
        wNMRS = np.linalg.norm(vec * w, p)
        return wNMRS

    @staticmethod
    def get_w0(B0):
        w0 = getValues.Na_gamma * B0
        return w0

    @staticmethod
    def get_ptauC(tauCs, a, b, tcm):
        Norm = 2 * a * b / (np.sqrt(np.pi) * (a + b) * tauCs)
        ptauCs = Norm * np.exp(-a ** 2 * (np.log(tauCs / tcm)) ** 2)
        ptauCs[tauCs > tcm] = Norm[tauCs > tcm] * np.exp(-b ** 2 * (np.log(tauCs[tauCs > tcm] / tcm)) ** 2)
        return ptauCs

    @staticmethod
    def get_wQs(tauCs, tauC0, wQ0, wQ1):
        wQs = np.ones((1, len(tauCs))) * wQ0
        wQs[tauCs > tauC0] = wQ1
        return wQs

    @staticmethod
    def get_logRange(startExp, endExp, stepSize):
        r_tmp = np.arange(startExp, endExp)
        t_tmp = 10 ** r_tmp
        logRange_tmp = np.arange(1, 10, stepSize).reshape(-1, 1) * t_tmp
        logRange = logRange_tmp.reshape(1, -1).squeeze()
        logRange = np.concatenate((logRange, [10 ** endExp]))
        return logRange

    # Jen Model
    @staticmethod
    def get_J_Jen(m, tauC, wQ, Jen, w0):
        x = (w0 * tauC) ** 2  # s
        Jm = (wQ ** 2) / 5 * tauC / (1 + m ** 2 * x) + Jen
        Km = m * w0 * tauC * Jm
        return Jm, Km

    @staticmethod
    def get_JenModel(T1f, T1s, T2f, T2s, w0):
        Ts_mess = np.array([T1f, T1s, T2f, T2s])
        Rs_mess = 1 / Ts_mess

        wQbar = 0
        Jen0 = 10;
        tauC0 = 5e-8;
        wQ0 = 1e5;
        wShift_RMS0 = 0
        intialVec = [Jen0, tauC0, wQ0, wShift_RMS0]

        weights = Ts_mess
        p = 2  # p-Norm value

        def MinFun(JenVal):
            Rs_ZQSQ_Jen = getValues.get_Rs_ZQSQ_Jen(JenVal[0], JenVal[1], JenVal[2], wQbar, JenVal[3], w0)
            diff = Rs_ZQSQ_Jen - Rs_mess
            return getValues.get_wPnorm(diff, weights, p)

        JenVal

