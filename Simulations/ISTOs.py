import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import getValues
##################################################################
# Hard Pulses and relaxation without macroscopic anisotropy
# Tmn matrix contains the fraction of each Tmn basis element
# and a potential B0shift/inhomogeneity


class TmnEvo:
    """
    coherence order m = 1, 2, 3 -> changes for pulses
    !!!! BUT !!! PYTHON so 1,2,3 cohernce orders -> 0,1,2
    I added (-1) wherever there was indexed with m
    rank n = -3, -2, -1, 0, 1, 2, 3 -> changes for realaxation
    T3m3 := T3-3 , and T33 := T3+3
    """
    Teq = np.array([[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])  # Thermal equilibrium = T10

    def __init__(self, B0, tauC, wQ, wQbar, Jen, wShift, wShift_RMS, wShift_FID, Tmn_value=None):
        if Tmn_value is None:
            self.Tmn = self.Teq
        else:
            self.Tmn = np.array(Tmn_value)

        self.B0 = B0
        self.w0 = getValues.get_w0(B0)

        self.tauC = tauC # correlation time
        self.wQ = wQ # quadrupole coupling constant
        self.wQbar = wQbar # mean(wQ) -> anisotropy
        self.Jen = Jen # summarizes wQ_en and tauC_en for extreme narrowing

        self.wShift = wShift
        self.wShift_RMS = wShift_RMS
        self.wShift_FID = wShift_FID

        self.tsDataPoints = 2048  # Default value, change as needed

    @staticmethod
    def getTmn_matrix(self, Tmn, m, n):
        return Tmn[m-1][(n-1) + 4]

    def TmnPM2TmnAS(self, Tmn_m, Tmn_p):
        # convert Tmn with plus/minus to symmetric/asymmetric Tmn
        Tmn_a = 1 / np.sqrt(2) * (Tmn_m - Tmn_p)
        Tmn_s = 1 / np.sqrt(2) * (Tmn_m + Tmn_p)

        return Tmn_a, Tmn_s

    def TmnAS2TmnPM(self, Tmn_a, Tmn_s):
        # convert Tmn with plus/minus to symmetric/asymmetric
        Tmn_p = 1 / np.sqrt(2) * (Tmn_s - Tmn_a)
        Tmn_m = 1 / np.sqrt(2) * (Tmn_s + Tmn_a)
        return Tmn_m, Tmn_p

    def M2Tmn(self, M):
        # converts M vector to Matrix Tmn
        T10 = M[0]
        T1m1, T11 = self.TmnAS2TmnPM(M[1], M[2])

        T20 = M[3]
        T2m1, T21 = self.TmnAS2TmnPM(M[4], M[5])
        T2m2, T22 = self.TmnAS2TmnPM(M[6], M[7])

        T30 = M[8]
        T3m1, T31 = self.TmnAS2TmnPM(M[9], M[10])
        T3m2, T32 = self.TmnAS2TmnPM(M[11], M[12])
        T3m3, T33 = self.TmnAS2TmnPM(M[13], M[14])

        Tmn = [
            [0, 0, T1m1, T10, T11, 0, 0],
            [0, 0, T2m2, T2m1, T20, T21, T22],
            [T3m3, T3m2, T3m1, T30, T31, T32, T33]
            ]
        return Tmn

    def Tmn2M(self,Tmn):
        M = np.array([0] * 15)

        M[0] = self.getTmn_matrix(Tmn, 1, 0)
        M[1], M[2] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 1, -1), self.getTmn_matrix(Tmn, 1, 1))

        M[3] = self.getTmn_matrix(Tmn, 2, 0)
        M[4], M[5] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 2, -1), self.getTmn_matrix(Tmn, 2, 1))
        M[6], M[7] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 2, -2), self.getTmn_matrix(Tmn, 2, 2))

        M[8] = TmnEvo.getTmn_matrix(Tmn, 3, 0)
        M[9], M[10] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 3, -1), self.getTmn_matrix(Tmn, 3, 1))
        M[11], M[12] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 3, -2), self.getTmn_matrix(Tmn, 3, 2))
        M[13], M[14] = self.TmnPM2TmnAS(self.getTmn_matrix(Tmn, 3, -3), self.getTmn_matrix(Tmn, 3, 3))

        return M

    def Ms2Tmns(self, M):
        # converts M vector timeseries of form M(time dim, Tmn dim) to Tmn matrix time series
        sizeM = np.shape(M)
        Tmn_non = np.zeros([1, sizeM[0]])  # nonexisting Tmns

        T10 = M[:, 0]
        T1m1, T11 = self.TmnAS2TmnPM(M[:, 1], M[:, 2])

        T20 = M[:, 3]
        T2m1, T21 = self.TmnAS2TmnPM(M[:, 4], M[:, 5])
        T2m2, T22 = self.TmnAS2TmnPM(M[:, 6], M[:, 7])

        T30 = M[:, 8]
        T3m1, T31 = self.TmnAS2TmnPM(M[:, 9], M[:, 10])
        T3m2, T32 = self.TmnAS2TmnPM(M[:, 11], M[:, 12])
        T3m3, T33 = self.TmnAS2TmnPM(M[:, 13], M[:, 14])

        Tmns = np.zeros([3, 7, sizeM[0]])
        Tmns[0, :, :] = np.vstack([Tmn_non, Tmn_non, T1m1.T, T10.T, T11.T, Tmn_non, Tmn_non])
        Tmns[1, :, :] = np.vstack([Tmn_non, T2m2.T, T2m1.T, T20.T, T21.T, T22.T, Tmn_non])
        Tmns[2, :, :] = np.vstack([T3m3.T, T3m2.T, T3m1.T, T30.T, T31.T, T32.T, T33.T])

        return np.array(Tmns)

    def getTmn(self, m, n):
        return self.Tmn[m-1][(n-1) + 4]

    def setTmn(self, m, n, Tmn_value):
        self.Tmn[m-1][(n-1) + 4] = Tmn_value

    def copy(self):
        Tmn = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS, self.wShift_FID,
                     self.Tmn)
        Tmn.tsDataPoints = self.tsDataPoints
        return Tmn

    def pulse(self, flipAngle, phase):
        Tmn_prePulse = np.array(self.Tmn)

        # Perform operations on each row separately
        Tmn_pulse1 = np.array((getValues.get_pulseOperator(flipAngle, phase, 1) @ Tmn_prePulse[0].transpose())).transpose()
        Tmn_pulse2 = np.array((getValues.get_pulseOperator(flipAngle, phase, 2) @ Tmn_prePulse[1].transpose())).transpose()
        Tmn_pulse3 = np.array((getValues.get_pulseOperator(flipAngle, phase, 3) @ Tmn_prePulse[2].transpose())).transpose()

        # Combine the results into a matrix
        Tmn_pulse_matrix = [Tmn_pulse1, Tmn_pulse2, Tmn_pulse3]

        # Create a new TmnEvo object with the updated matrix
        Tmn_pulse = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS, self.wShift_FID,
                           Tmn_pulse_matrix)
        return Tmn_pulse

    def getFIDphase(self, ts, phaseRX):
        Tmn_preRelax = self.Tmn.copy()  # for T10 Relax, Teq must be subtracted and added back at the end
        fm1kks = getValues.get_f_Jen(-1, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)

        # single quantum relaxation
        fm111, fm113, fm112 = fm1kks[0, :], fm1kks[1, :], -fm1kks[3, :]

        T1m1_pre = self.getTmn(1, -1)
        T3m1_pre = self.getTmn(3, -1)
        T2m1_pre = self.getTmn(2, -1)

        T1m1_relax = fm111 * T1m1_pre + fm112 * T2m1_pre + fm113 * T3m1_pre

        FID = T1m1_relax * np.exp(1j * np.deg2rad(phaseRX))

        return FID

    def relaxation_Jen(self, tevo):
        Tmn_preRelax = self.copy()
        Tmn_preRelax.Tmn = Tmn_preRelax.Tmn - Tmn_preRelax.Teq  # recover thermal equilibrium

        if tevo == 0:
            ts = np.array([0])
        else:
            ts = np.arange(0, tevo, tevo / self.tsDataPoints)

        Tmn_non = np.zeros((len(ts)))  # nonexisting Tmns
        T10_ts = np.ones((len(ts)))

        fm333 = getValues.get_f_Jen(-3, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        fm2kks = getValues.get_f_Jen(-2, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        fm1kks = getValues.get_f_Jen(-1, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        f0kks = getValues.get_f_Jen(0, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        f1kks = getValues.get_f_Jen(1, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        f2kks = getValues.get_f_Jen(2, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)
        f333 = getValues.get_f_Jen(3, ts, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift_RMS, self.w0)

        # zero quantum relaxation
        f011, f013, f033, f022 = f0kks[:4, :]
        T10_pre = Tmn_preRelax.getTmn(1, 0)
        T30_pre = Tmn_preRelax.getTmn(3, 0)
        T20_pre = Tmn_preRelax.getTmn(2, 0)
        T10_relax = f011 * T10_pre + f013 * T30_pre + T10_ts  # add T10_ts to recover thermal equilibrium
        T30_relax = f013 * T10_pre + f033 * T30_pre
        T20_relax = f022 * T20_pre

        # single quantum relaxation
        fm111, fm113, fm133, fm122 = fm1kks[:4, :]
        fm112, fm123 = -fm1kks[4:, :]
        f111, f113, f133, f122 = f1kks[:4, :]
        f112, f123 = f1kks[4:, :]
        T1m1_pre = Tmn_preRelax.getTmn(1, -1)
        T11_pre = Tmn_preRelax.getTmn(1, 1)
        T3m1_pre = Tmn_preRelax.getTmn(3, -1)
        T31_pre = Tmn_preRelax.getTmn(3, 1)
        T2m1_pre = Tmn_preRelax.getTmn(2, -1)
        T21_pre = Tmn_preRelax.getTmn(2, 1)
        T1m1_relax = (fm111 * T1m1_pre + fm112 * T2m1_pre + fm113 * T3m1_pre) * np.exp(
            -1 * (1j * self.wShift) * ts)
        T3m1_relax = (fm113 * T1m1_pre + fm123 * T2m1_pre + fm133 * T3m1_pre) * np.exp(
            -1 * (1j * self.wShift) * ts)
        T2m1_relax = (fm112 * T1m1_pre + fm122 * T2m1_pre + fm123 * T3m1_pre) * np.exp(
            -1 * (1j * self.wShift) * ts)
        T11_relax = (f111 * T11_pre + f112 * T21_pre + f113 * T31_pre) * np.exp(1 * (1j * self.wShift) * ts)
        T31_relax = (f113 * T11_pre + f123 * T21_pre + f133 * T31_pre) * np.exp(1 * (1j * self.wShift) * ts)
        T21_relax = (f112 * T11_pre + f122 * T21_pre + f123 * T31_pre) * np.exp(1 * (1j * self.wShift) * ts)

        # MQCs
        f222, f233, f223 = f2kks[:3, :]
        fm222, fm233, fm223 = fm2kks[:3, :]
        T2m2_pre = Tmn_preRelax.getTmn(2, -2)
        T22_pre = Tmn_preRelax.getTmn(2, 2)
        T3m2_pre = Tmn_preRelax.getTmn(3, -2)
        T32_pre = Tmn_preRelax.getTmn(3, 2)
        # n=-2
        T2m2_relax = (fm222 * T2m2_pre + fm223 * T3m2_pre) * np.exp(-2 * (1j * self.wShift) * ts)
        T3m2_relax = (fm233 * T3m2_pre + fm223 * T2m2_pre) * np.exp(-2 * (1j * self.wShift) * ts)
        # n=+2
        T22_relax = (f222 * T22_pre + f223 * T32_pre) * np.exp(2 * (1j * self.wShift) * ts)
        T32_relax = (f233 * T32_pre + f223 * T22_pre) * np.exp(2 * (1j * self.wShift) * ts)
        # m = 3
        T3m3_pre = Tmn_preRelax.getTmn(3, -3)
        T33_pre = Tmn_preRelax.getTmn(3, 3)
        T3m3_relax = fm333.squeeze() * T3m3_pre * np.exp(-3 * (1j * self.wShift) * ts)
        T33_relax = f333.squeeze() * T33_pre * np.exp(3 * (1j * self.wShift) * ts)

        Tmn_timeSeries = np.zeros((3, 7, len(ts)), dtype=np.complex128)
        Tmn_timeSeries[0, :, :] = [Tmn_non, Tmn_non, T1m1_relax, T10_relax, T11_relax, Tmn_non, Tmn_non]
        Tmn_timeSeries[1, :, :] = [Tmn_non, T2m2_relax, T2m1_relax, T20_relax, T21_relax, T22_relax, Tmn_non]
        Tmn_timeSeries[2, :, :] = [T3m3_relax, T3m2_relax, T3m1_relax, T30_relax, T31_relax, T32_relax, T33_relax]

        Tmn_relax = Tmn_timeSeries[:, :, -1]
        Tmn_relaxation = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS,
                                self.wShift_FID, Tmn_relax)
        return Tmn_relaxation, Tmn_timeSeries