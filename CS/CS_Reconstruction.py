import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift
from pywt import threshold as thresh


class RecoCS:

    def __init__(self, DataFull, samplingMethod, accelerationFactor, CS_Algorithm='IST-D',simul=True):
        initialData = DataFull
        self.DataFull = initialData
        self.samplingMethod = samplingMethod
        self.accelerationFactor = accelerationFactor
        self.lenDataFull = len(DataFull)
        self.lenDataCS = int(len(DataFull) / accelerationFactor)
        self.generateCSdata()
        self.NIter = 5000
        self.mode = 'soft'  # Thresholding mode: either 'soft' or 'hard'
        self.threshold = 0.75
        self.CS_Algorithm = CS_Algorithm

    @staticmethod
    def weightedSoftTH(y, thresholdMatrix):
        s = np.abs(y) - thresholdMatrix
        s = (s + np.abs(s)) / 2
        return np.sign(y) * s

    def generateCSdata(self):

        if self.samplingMethod == 'PoissonGap':
            pass
        else:
            self.samplingMask = self.generateCSMaskUniform()
        CS_data = self.DataFull.copy()
        CS_data[np.invert(self.samplingMask)] = 0
        self.CS_data = CS_data



    def generateCSMaskUniform(self):
        """
        Generates a uniform random undersampling mask with a fixed undersampling factor.

        Args:
        obj: Object containing the following properties:
            - dataPointsFull: Integer representing the total number of data points.
            - dataPointsCS: Integer representing the number of data points to be randomly undersampled.

        Returns:
        samplingMask: A boolean array with length equal to dataPointsFull, where True indicates that the corresponding
                      data point is to be retained and False indicates that the data point is to be discarded.
        """
        CS_samplingMask_indices = np.random.permutation(np.arange(self.lenDataFull))[:self.lenDataCS]
        CS_samplingMask = np.zeros(self.lenDataFull)
        CS_samplingMask[CS_samplingMask_indices] = 1
        samplingMask = CS_samplingMask == 1
        return samplingMask

    def csReconstruction(self):
        cs_input, cs_mask = self.CS_data, self.samplingMask

        if self.CS_Algorithm == 'IST_D':
            cs_output = self.IST_D(cs_input, cs_mask)
        elif self.CS_Algorithm == 'IST_S':
            cs_output = self.IST_S(cs_input, cs_mask)
        elif self.CS_Algorithm == 'CLEAN':
            pass
            #cs_output = CLEAN(obj, cs_input, cs_mask)

        else:
            cs_output = self.IST_D(cs_input, cs_mask)
        return cs_output

    def IST_D(self, cs_input, cs_mask):
        """
        IST_D Algorithm. Takes FT spectrum as input.
        :param cs_input: FT Spectrum of Data
        :param cs_mask:
        :return: Reconstructed Spectrum
        """
        N_Iter = self.NIter
        t = self.threshold
        cs_input_init = cs_input.copy()
        cs_output_ft = np.zeros(len(cs_input), dtype=np.complex128)
        #norm_TH = 0.99 * abs(self.TH_spectrum / max(self.TH_spectrum))
        #norm_TH_it = norm_TH.copy()
        for i in range(N_Iter):
            ft_cs = fft(cs_input_init)
            th = t * max(abs(ft_cs)) # th is first a relative threshold
            th_ft_cs = thresh(ft_cs, th, self.mode)

            """ if self.UseTHexp and not self.UseVE and not self.UseZeroAdd:
                rand_T2s = np.random.uniform(low=self.rangeT2s[0], high=self.rangeT2s[1])
                rand_T2f = np.random.uniform(low=self.rangeT2f[0], high=self.rangeT2f[1])
                rand_A = np.random.uniform(low=self.rangeA[0], high=self.rangeA[1])
                rand_C = 1 - rand_A
                th_EXP = ifft(th_ft_cs) * np.exp(-self.xValues / rand_T2s)
                th_ft_cs = fft(th_EXP)"""

            cs_output_ft += th_ft_cs

            th_cs = ifft(np.array(th_ft_cs))
            th_cs[~cs_mask] = 0
            cs_input_init = np.real(cs_input_init) -th_cs

        cs_output = ifft(cs_output_ft)
        return cs_output

    def IST_S(self, cs_input, cs_mask):
        """
        Returns spectrum- NOT FID !!!
        :param cs_input:
        :param cs_mask:
        :return: Spectrum
        """
        N_Iter = self.NIter
        t = self.threshold
        cs_input_init = cs_input.copy()

        cs_output = np.zeros(len(cs_input), dtype=np.complex128)

        for i in range(N_Iter):
            ft_cs = fft(cs_input_init)
            th = t * max(abs(ft_cs)) * (N_Iter - i)/N_Iter# th is first a relative threshold
            th_ft_cs = thresh(ft_cs, th, self.mode)
            th_cs = ifft(th_ft_cs);
            cs_input_init[~cs_mask] = th_cs[~cs_mask]
        cs_output = cs_input_init
        return cs_output




