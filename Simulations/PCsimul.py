
import scipy.optimize as opt
from scipy import fmin
import numpy as np
import getValues
from ISTOs import TmnEvo

#TODO: - MADQF Sequence
#      - TQTPPI w/o 180

class PhaseCycle:
    def __init__(self, B0, tauC, wQ, wQbar, Jen, wShift, wShift_RMS):
        self.B0 = 9.4  # T
        self.w0 = getValues.get_w0(B0)
        self.tauC = tauC
        self.wQ = wQ
        self.wQbar = wQbar
        self.Jen = Jen
        self.wShift = wShift # B-field inhom. -> mean(Omega)
        self.wShift_RMS = wShift_RMS # B-field inhom. -> ~std(Omega) --> intrinsical, doesnt get refocussed by 180°
        self.wShift_FID = 0 # B-field inhom. for 1st dim FID --> refocussing possible
        self.tevoStep = 0.1000 * 1e-3  # s
        self.tevo0 = 0.1305 * 1e-3  # s
        self.tmix = 0.1500 * 1e-3  # s
        self.startPhase = 90  # degrees
        self.alphas = None
        self.NumPhaseCycles = None
        self.PulseAngles = []
        self.TR = 600e-3  # s  repetition time
        self.dataPoints = 2048  # 2048
        self.deadtimeFID = 100e-6  # s
        self.dwelltimeFID = 50e-6  # s
        self.noisePower = 0  # noise Power in Watts, linear, currently only for T3i sequence
        self.flip90 = 90  # can be set to different values to simulate B1+
        self.flip180 = 180  # inhomogeneities
        self.TEstep = 1e-3
        self.TEmax = 400e-3


    @staticmethod
    def general3Pulse(TmnStart, alpha, beta, tevo, tmix, flip1, flip2, flip3):
        """
        general 3 pulse sequence step. Phase of 3 pulse is always 0.
        :param TmnStart:
        :param alpha:
        :param beta:
        :param tevo:
        :param tmix:
        :param flip1:
        :param flip2:
        :param flip3:
        :return:
        """
        Tmn_pulse90_1 = TmnStart.pulse(flip1, alpha)
        Tmn_evo, _ = Tmn_pulse90_1.relaxation_Jen(tevo)

        Tmn_pulse90_2 = Tmn_evo.pulse(flip2, beta)
        Tmn_evo_2, _ = Tmn_pulse90_2.relaxation_Jen(tmix)

        Tmn_pulse90_3 = Tmn_evo_2.pulse(flip3, 0)
        TmnEnd = Tmn_pulse90_3

        # Calculate TQ signal (T3-3, T33 after the mixing period)
        TQp = Tmn_evo_2.getTmn(3, +3) * getValues.get_WignerD(3, -1, +3, flip3)
        TQm = Tmn_evo_2.getTmn(3, -3) * getValues.get_WignerD(3, -1, -3, flip3)
        TQ = TQp + TQm

        # Calculate SQ signal (T1-1, T11, T3-1, T31 after the mixing period)
        SQp = (Tmn_evo_2.getTmn(1, +1) * getValues.get_WignerD(1, -1, +1, flip3) +
               Tmn_evo_2.getTmn(3, +1) * getValues.get_WignerD(3, -1, +1, flip3))
        SQm = (Tmn_evo_2.getTmn(1, -1) * getValues.get_WignerD(1, -1, -1, flip3) +
               Tmn_evo_2.getTmn(3, -1) * getValues.get_WignerD(3, -1, -1, flip3))
        SQ = SQp + SQm

        return TmnEnd, TQ, SQ

    @staticmethod
    def general3Pulse3phase(TmnStart, alpha, beta, phase3, tevo, tmix, flip1, flip2, flip3):
        Tmn_pulse90_1 = TmnStart.pulse(flip1, alpha)
        Tmn_evo, _ = Tmn_pulse90_1.relaxation_Jen(tevo)
        Tmn_pulse90_2 = Tmn_evo.pulse(flip2, beta)
        Tmn_evo_2, _ = Tmn_pulse90_2.relaxation_Jen(tmix)
        Tmn_pulse90_3 = Tmn_evo_2.pulse(flip3, phase3)

        TmnEnd = Tmn_pulse90_3

        # calculate TQ signal (T3-3, T33 after mixing period)
        TQp = Tmn_evo_2.getTmn(3, +3) * getValues.get_WignerD(3, -1, +3, flip3)
        TQm = Tmn_evo_2.getTmn(3, -3) * getValues.get_WignerD(3, -1, -3, flip3)
        TQ = TQp + TQm

        # calculate SQ signal (T1-1, T11, T3-1, T31 after mixing period)
        SQp = (Tmn_evo_2.getTmn(1, +1) * getValues.get_WignerD(1, -1, +1, flip3) +
               Tmn_evo_2.getTmn(3, +1) * getValues.get_WignerD(3, -1, +1, flip3))
        SQm = (Tmn_evo_2.getTmn(1, -1) * getValues.get_WignerD(1, -1, -1, flip3) +
               Tmn_evo_2.getTmn(3, -1) * getValues.get_WignerD(3, -1, -1, flip3))
        SQ = SQp + SQm

        return TmnEnd, TQ, SQ
    @staticmethod
    def general3Pulse_softPulse(TmnStart, alpha, beta, tevo, tmix, pulseStrength, flip2, flip3):
        # Insert your code here for the general3Pulse_softPulse method
        pass
    def phase_cor_FID(self, fid):
        # Calculate the phase correction factor using optimization
        fid = np.array(fid)
        fidi = fid[0, :]  # Assuming 1st row of FID contains data
        def banana(x):
            return -np.mean(np.real(fidi * np.exp(1j * x)))
        #banana = lambda x: -np.mean(np.real(fidi * np.exp(1j * x)))
        x_optimized = opt.fmin(banana, [0])

        # Apply the phase correction to the FID
        if np.max(np.real(fid[0, :] * np.exp(1j * x_optimized[0]))) < 0:
            FID = fid * np.exp(-1j * x_optimized[0])
        else:
            FID = fid * np.exp(1j * x_optimized[0])

        return FID

    @staticmethod
    def general3Pulse_w180(TmnStart, alpha, beta, tevo, tmix, flip1, flip2, flip3, flipRefocus):
        Tmn_pulse90_1 = TmnStart.pulse(flip1, alpha)
        Tmn_evo_1, Tmn_timeSeries_evo_1 = Tmn_pulse90_1.relaxation_Jen(tevo / 2)

        Tmn_pulse180 = Tmn_evo_1.pulse(flipRefocus, alpha + 90)
        Tmn_evo_2, Tmn_timeSeries_evo_2 = Tmn_pulse180.relaxation_Jen(tevo / 2)

        Tmn_pulse90_2 = Tmn_evo_2.pulse(flip2, beta)
        Tmn_evo_3, Tmn_timeSeries_mix = Tmn_pulse90_2.relaxation_Jen(tmix)

        Tmn_pulse90_3 = Tmn_evo_3.pulse(flip3, 0)
        TmnEnd = Tmn_pulse90_3

        # Calculate TQ signal (T3-3, T33 after mixing period)
        TQp = Tmn_evo_3.getTmn(3, +3) * getValues.get_WignerD(3, -1, +3, flip3)
        TQm = Tmn_evo_3.getTmn(3, -3) * getValues.get_WignerD(3, -1, -3, flip3)
        TQ = TQp + TQm

        # Calculate SQ signal (T1-1, T11, T3-1, T31 after mixing period)
        SQp = (Tmn_evo_3.getTmn(1, +1) * getValues.get_WignerD(1, -1, +1, flip3) +
               Tmn_evo_3.getTmn(3, +1) * getValues.get_WignerD(3, -1, +1, flip3))
        SQm = (Tmn_evo_3.getTmn(1, -1) * getValues.get_WignerD(1, -1, -1, flip3) +
               Tmn_evo_3.getTmn(3, -1) * getValues.get_WignerD(3, -1, -1, flip3))
        SQ = SQp + SQm

        return TmnEnd, TQ, SQ

    @staticmethod
    def threePulse_mixing180(TmnStart, alpha, beta, tevo, tmix, flip1, flip2, flip3, flipRefocus):
        """
        Idea, apply refocus inbetween 2-3
        :param TmnStart:
        :param alpha:
        :param beta:
        :param tevo:
        :param tmix:
        :param flip1:
        :param flip2:
        :param flip3:
        :param flipRefocus:
        :return:
        """

        Tmn_pulse90_1 = TmnStart.pulse(flip1, alpha)
        Tmn_evo_1, Tmn_timeSeries_evo_1 = Tmn_pulse90_1.relaxation_Jen(tevo / 2)

        #Tmn_pulse180 = Tmn_evo_1.pulse(flipRefocus, alpha + 90)
        #Tmn_evo_2, Tmn_timeSeries_evo_2 = Tmn_pulse180.relaxation_Jen(tevo / 2)

        Tmn_pulse90_2 = Tmn_evo_1.pulse(flip2, beta)
        Tmn_evo_2, Tmn_timeSeries_mix = Tmn_pulse90_2.relaxation_Jen(tmix)


        Tmn_pulse180 = Tmn_evo_2.pulse(flipRefocus, alpha + 90)
        Tmn_evo_3, Tmn_timeSeries_evo_3 = Tmn_pulse180.relaxation_Jen(tmix / 2)
        Tmn_pulse90_3 = Tmn_pulse180.pulse(flip3, 0)
        TmnEnd = Tmn_pulse90_3



        # Calculate TQ signal (T3-3, T33 after mixing period)
        TQp = Tmn_evo_3.getTmn(3, +3) * getValues.get_WignerD(3, -1, +3, flip3)
        TQm = Tmn_evo_3.getTmn(3, -3) * getValues.get_WignerD(3, -1, -3, flip3)
        TQ = TQp + TQm

        # Calculate SQ signal (T1-1, T11, T3-1, T31 after mixing period)
        SQp = (Tmn_evo_3.getTmn(1, +1) * getValues.get_WignerD(1, -1, +1, flip3) +
               Tmn_evo_3.getTmn(3, +1) * getValues.get_WignerD(3, -1, +1, flip3))
        SQm = (Tmn_evo_3.getTmn(1, -1) * getValues.get_WignerD(1, -1, -1, flip3) +
               Tmn_evo_3.getTmn(3, -1) * getValues.get_WignerD(3, -1, -1, flip3))
        SQ = SQp + SQm

        return TmnEnd, TQ, SQ

    def SpinEcho(self):
        times = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)
        FIDs = []
        TEs = np.arange(0, self.TEmax + self.TEstep, self.TEstep)

        for TE in TEs:
            wShift = 0
            wShift_RMS = 0
            wShiftFID = 0
            Tmn_i = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, wShift, wShift_RMS, wShiftFID)
            Tmn_pulse1 = Tmn_i.pulse(self.flip90, 0)
            Tmn_evo, Tmn_timeSeries_evo_1 = Tmn_pulse1.relaxation_Jen(TE / 2)
            Tmn_pulse2 = Tmn_evo.pulse(self.flip180, 90)
            Tmn_evo2, Tmn_timeSeries_evo_2 = Tmn_pulse2.relaxation_Jen(0)
            FID = Tmn_evo2.getFIDphase(times, 0)
            FIDs.append(FID)

        return np.array(FIDs), TEs, times

    def CL(self, tevo):
        TmnStart = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS, self.wShift_FID)
        self.alphas = [150, 210, 270, 330, 390, 450]  # Set your alphas
        phis1_a = self.alphas
        phis1_b = phis1_a
        phis2_a = phis1_a
        phis2_b = [phi + 90 for phi in phis2_a]
        phiRX = 0
        phi3 = 0

        lengthPhis = len(phis1_a)
        times = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)

        Tmn_a = TmnStart
        Tmn_b = TmnStart
        FIDs_a = np.zeros((lengthPhis, len(times)), dtype=np.complex)
        FIDs_b = np.zeros((lengthPhis, len(times)), dtype=np.complex)

        for iPhi in range(lengthPhis):
            phi1_a = phis1_a[iPhi]
            phi1_b = phis1_b[iPhi]
            phi2_a = phis2_a[iPhi]
            phi2_b = phis2_b[iPhi]

            Tmn_a, _, _ = PhaseCycle.general3Pulse3phase(Tmn_a, phi1_a, phi2_a, phi3, tevo, self.tmix, self.flip90,
                                                         self.flip90, self.flip90)
            Tmn_b, _, _ = PhaseCycle.general3Pulse3phase(Tmn_b, phi1_b, phi2_b, phi3, tevo, self.tmix, self.flip90,
                                                         self.flip90, self.flip90)

            FIDs_a[iPhi, :] = Tmn_a.getFIDphase(times, phiRX)
            FIDs_b[iPhi, :] = Tmn_b.getFIDphase(times, phiRX)

            Tmn_a, _ = Tmn_a.relaxation_Jen(self.TR)
            Tmn_b, _ = Tmn_b.relaxation_Jen(self.TR)

        return FIDs_a, FIDs_b, times


    def TQTPPI_w180(self, mixing180=False):
        """complete TQTPPI sequence with 180° refocussing pulse and DQ filter: Run sequence twice with phase shift of 180° and add FIDs
        returns:
        - FIDs_p1: FID with beta = alpha + 90°
        - FIDs_p2: FID with beta = alpha - 90°
        - tevos: vector of evolution times
        - TQs_p1, 2: TQ signal(T33 + T3 - 3 -->obs pulse --> T3 - 1)"""
        TmnStart = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS,
                               self.wShift_FID)
        lengthAlpha = len(self.alphas)
        times = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)
        Tmn_p1 = TmnStart.copy()
        Tmn_p2 = TmnStart.copy()
        Tmns_p1 = []
        Tmns_p2 = []
        FIDs_p1 = []
        FIDs_p2 = []
        tevos = np.zeros(self.NumPhaseCycles * lengthAlpha)
        TQs_p1 = np.zeros(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        TQs_p2 = np.zeros(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        SQs_p1 = np.zeros(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        SQs_p2 = np.zeros(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)

        if not(mixing180):
            for j in range(1, self.NumPhaseCycles + 1):
                for alphaIndex in range(1, lengthAlpha + 1):
                    tevo = self.tevo0 + ((j - 1) * lengthAlpha + alphaIndex - 1) *self.tevoStep
                    tevos[(j - 1) * lengthAlpha + alphaIndex - 1] = tevo
                    alpha = self.alphas[alphaIndex - 1]
                    beta_p1 = alpha + 90
                    beta_p2 = alpha - 90

                    Tmn_p1, TQ_p1, SQ_p1 = self.general3Pulse_w180(Tmn_p1, alpha, beta_p1, tevo, self.tmix, self.flip90,
                                                                   self.flip90, self.flip90, self.flip180)
                    Tmn_p2, TQ_p2, SQ_p2 = self.general3Pulse_w180(Tmn_p2, alpha, beta_p2, tevo, self.tmix, self.flip90,
                                                                   self.flip90, self.flip90, self.flip180)

                    Tmns_p1.append(Tmn_p1.copy())
                    Tmns_p2.append(Tmn_p2.copy())

                    FIDs_p1.append(Tmn_p1.getFIDphase(times, 0)) # gets FID as acquired by ADC
                    FIDs_p2.append(Tmn_p2.getFIDphase(times, 0)) # -----''--------------------

                    TQs_p1[(j - 1) * lengthAlpha + alphaIndex - 1] = TQ_p1
                    TQs_p2[(j - 1) * lengthAlpha + alphaIndex - 1] = TQ_p2

                    SQs_p1[(j - 1) * lengthAlpha + alphaIndex - 1] = SQ_p1
                    SQs_p2[(j - 1) * lengthAlpha + alphaIndex - 1] = SQ_p2

                    Tmn_p1, _ = Tmn_p1.relaxation_Jen(self.TR)
                    Tmn_p2, _ = Tmn_p2.relaxation_Jen(self.TR)
        else:
            for j in range(1, self.NumPhaseCycles + 1):
                for alphaIndex in range(1, lengthAlpha + 1):
                    tevo = self.tevo0 + ((j - 1) * lengthAlpha + alphaIndex - 1) * self.tevoStep
                    tevos[(j - 1) * lengthAlpha + alphaIndex - 1] = tevo
                    alpha = self.alphas[alphaIndex - 1]
                    beta_p1 = alpha + 90
                    beta_p2 = alpha - 90

                    Tmn_p1, TQ_p1, SQ_p1 = self.threePulse_mixing180(Tmn_p1, alpha, beta_p1, tevo, self.tmix, self.flip90,
                                                                   self.flip90, self.flip90, self.flip180)
                    Tmn_p2, TQ_p2, SQ_p2 = self.threePulse_mixing180(Tmn_p2, alpha, beta_p2, tevo, self.tmix, self.flip90,
                                                                   self.flip90, self.flip90, self.flip180)

                    Tmns_p1.append(Tmn_p1.copy())
                    Tmns_p2.append(Tmn_p2.copy())

                    FIDs_p1.append(Tmn_p1.getFIDphase(times, 0))  # gets FID as acquired by ADC
                    FIDs_p2.append(Tmn_p2.getFIDphase(times, 0))  # -----''--------------------

                    TQs_p1[(j - 1) * lengthAlpha + alphaIndex - 1] = TQ_p1
                    TQs_p2[(j - 1) * lengthAlpha + alphaIndex - 1] = TQ_p2

                    SQs_p1[(j - 1) * lengthAlpha + alphaIndex - 1] = SQ_p1
                    SQs_p2[(j - 1) * lengthAlpha + alphaIndex - 1] = SQ_p2

                    Tmn_p1, _ = Tmn_p1.relaxation_Jen(self.TR)
                    Tmn_p2, _ = Tmn_p2.relaxation_Jen(self.TR)
        return np.array(FIDs_p1), np.array(FIDs_p2), np.array(tevos), np.array(TQs_p1), np.array(TQs_p2), np.array(SQs_p1), np.array(SQs_p2)

    def TQTPPI_wo180(self):
        TmnStart = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS, self.wShift_FID)
        lengthAlpha = len(self.alphas)
        times = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)
        Tmn_p1 = TmnStart.copy()
        Tmn_p2 = TmnStart.copy()
        Tmns_p1 = []
        Tmns_p2 = []
        FIDs_p1 = []
        FIDs_p2 = []
        TQs_p1, TQs_p2 = [], []
        SQs_p1, SQs_p2 = [], []
        tevos = np.zeros(self.NumPhaseCycles * lengthAlpha)

        for j in range(1, self.NumPhaseCycles + 1):
            for alphaIndex in range(1, lengthAlpha + 1):
                tevo = self.tevo0
                #tevo = self.tevo0 + (j - 1) * lengthAlpha + alphaIndex - 1 * self.tevoStep
                #tevos[(j - 1) * lengthAlpha + alphaIndex - 1] = tevo
                alpha = self.alphas[alphaIndex - 1]
                beta_p1 = alpha + 90
                beta_p2 = alpha - 90

                Tmn_p1, TQ_p1, SQ_p1 = PhaseCycle.general3Pulse(Tmn_p1, alpha, beta_p1, tevo, self.tmix, self.flip90,
                                                                  self.flip90, self.flip90)
                Tmn_p2, TQ_p2, SQ_p2 = PhaseCycle.general3Pulse(Tmn_p2, alpha, beta_p2, tevo, self.tmix, self.flip90,
                                                                  self.flip90, self.flip90)

                Tmns_p1.append(Tmn_p1.copy())
                Tmns_p2.append(Tmn_p2.copy())

                TQs_p1.append(TQ_p1)
                TQs_p2.append(TQ_p2)
                SQs_p1.append(SQ_p1)
                SQs_p2.append(SQ_p2)

                FIDs_p1.append(Tmn_p1.getFIDphase(times, 0))
                FIDs_p2.append(Tmn_p2.getFIDphase(times, 0))

                Tmn_p1, _ = Tmn_p1.relaxation_Jen(self.TR)
                Tmn_p2, _ = Tmn_p2.relaxation_Jen(self.TR)

        return np.array(FIDs_p1), np.array(FIDs_p2), tevos, np.array(TQs_p1), np.array(TQs_p2), np.array(SQs_p1), np.array(SQs_p2)

    def TQTPPI_Fleysher(self):
        TmnStart = TmnEvo(self.B0, self.tauC, self.wQ, self.wQbar, self.Jen, self.wShift, self.wShift_RMS, self.wShift_FID)

        times = self.deadtimeFID + self.dwelltimeFID * np.arange(self.dataPoints)
        Tmn_p1 = TmnStart
        Tmn_p2 = TmnStart
        Tmns_p1 = []
        Tmns_p2 = []
        FIDs_p1 = []
        FIDs_p2 = []
        N = 6
        alpha1 = 30
        alpha2 = 120
        alphas = np.arange(0, np.pi / 3 * N, np.pi / 3) + alpha1
        betas = np.arange(0, np.pi / 3 * N, np.pi / 3) + alpha2
        # phi_RX = np.arange(0, np.pi * N, np.pi)
        phi_RX = [0] * len(alphas)
        lengthAlpha = len(alphas)

        tevos = np.zeros(self.NumPhaseCycles * lengthAlpha)
        TQs_p1 = np.empty(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        TQs_p2 = np.empty(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        SQs_p1 = np.empty(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)
        SQs_p2 = np.empty(self.NumPhaseCycles * lengthAlpha, dtype=np.complex)


        for j in range(self.NumPhaseCycles):
            for alphaIndex in range(lengthAlpha):
                tevo = self.tevo0 + ((j - 1) * lengthAlpha + alphaIndex - 1) * self.tevoStep
                tevos[(j - 1) * lengthAlpha + alphaIndex] = tevo
                alpha = alphas[alphaIndex]
                beta_p1 = betas[alphaIndex]
                beta_p2 = betas[alphaIndex] + 90

                Tmn_p1, TQ_p1, SQ_p1 = PhaseCycle.general3Pulse(Tmn_p1, alpha, beta_p1, tevo, self.tmix, self.flip90,
                                                                self.flip90, self.flip90)
                Tmn_p2, TQ_p2, SQ_p2 = PhaseCycle.general3Pulse(Tmn_p2, alpha, beta_p2, tevo, self.tmix, self.flip90,
                                                                self.flip90, self.flip90)

                Tmns_p1.append(Tmn_p1)
                Tmns_p2.append(Tmn_p2)

                FIDs_p1.append(Tmn_p1.getFIDphase(times, phi_RX[alphaIndex]))
                FIDs_p2.append(Tmn_p2.getFIDphase(times, phi_RX[alphaIndex]))

                TQs_p1[(j - 1) * lengthAlpha + alphaIndex] = TQ_p1
                TQs_p2[(j - 1) * lengthAlpha + alphaIndex] = TQ_p2

                SQs_p1[(j - 1) * lengthAlpha + alphaIndex] = SQ_p1
                SQs_p2[(j - 1) * lengthAlpha + alphaIndex] = SQ_p2

                Tmn_p1, _ = Tmn_p1.relaxation_Jen(self.TR)
                Tmn_p2, _ = Tmn_p2.relaxation_Jen(self.TR)

        return np.array(FIDs_p1), np.array(FIDs_p2), np.array(tevos), np.array(TQs_p1), np.array(TQs_p2), np.array(SQs_p1), np.array(SQs_p2)
    
    def get_TQTPPI_FID(self,tevos, FIDs_p1, FIDs_p2):
        # Combine FIDs from p1 and p2
        FIDs = FIDs_p1 + FIDs_p2

        # Apply phase correction to the combined FID
        FID_corrected = self.phase_cor_FID(FIDs)

        # Sum FIDs along the first dimension
        TQTPPI_FID = np.sum(FID_corrected, axis=1)

        return tevos, TQTPPI_FID, FID_corrected




