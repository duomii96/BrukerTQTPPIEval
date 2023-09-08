import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift
import numpy.random as rnd
from pywt import threshold as thresh
from scipy.optimize import curve_fit


class RecoCS:

    def __init__(self, DataFull, samplingMethod, accelerationFactor, CS_Algorithm='IST-D',simul=True, multipleFIDs=False):
        initialData = DataFull
        self.multipleFIDs = multipleFIDs
        if DataFull.ndim > 1:
            self.multipleFIDs = True
        self.DataFull = initialData
        self.samplingMethod = samplingMethod
        self.accelerationFactor = accelerationFactor
        self.lenDataFull = DataFull.shape[-1]
        self.lenDataCS = int(DataFull.shape[-1] / accelerationFactor)
        self.generateCSdata() # generates CSDAta
        self.NIter = 5000
        self.mode = 'soft'  # Thresholding mode: either 'soft' or 'hard'
        self.threshold = 0.7
        self.CS_Algorithm = CS_Algorithm
        print(f"CS Alg.: {self.CS_Algorithm}")
        self.x = np.arange(self.lenDataFull) * 2* np.pi / 360 # Thats default- but must be set from outside using x from GenClass

    @staticmethod
    def weightedSoftTH(y, thresholdMatrix):
        s = np.abs(y) - thresholdMatrix
        s = (s + np.abs(s)) / 2
        return np.sign(y) * s

    @staticmethod
    def fixedTQTPPI(x, a, b, c, d, e):

        return a * np.sin(x + b) + c * np.sin(3 * x + d) + e

    def generateCSdata(self):

        if self.samplingMethod == 'PoissonGap':
            self.samplingMask = self.generate_poisson_gap()
        else:
            self.samplingMask = self.generateCSMaskUniform()
        CS_data = self.DataFull.copy()

        if self.multipleFIDs:
            CS_data[:,np.invert(self.samplingMask)]= 0
        else:
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

    def generate_poisson_gap(self):
        p = self.lenDataCS
        z = self.lenDataFull
        ld = self.accelerationFactor
        adj = 2.0 * (ld - 1)  # initial guess of adjustment
        samplingMask = np.zeros(z)
        v = np.zeros(z)
        n = 0
        while n != p:
            n = 0
            i = 0
            while i < z:
                v[n] = i
                i += 1
                k = rnd.poisson(adj * np.sin((i + 0.5) / (z + 1) * np.pi / 2))
                i += k
                n += 1
            if n > p:
                adj = adj * 1.02  # too many points created
            if n < p:
                adj = adj / 1.02  # too few points created
        for p, s in enumerate(v):
            if s != 0:
                samplingMask[int(s)] = 1
            else:
                pass
        samplingMask = np.array(samplingMask, dtype='uint') == 1
        return samplingMask

    def csReconstruction(self):
        cs_input, cs_mask = self.CS_data, self.samplingMask

        if self.CS_Algorithm == 'IST_D':
            cs_output = self.IST_D(cs_input, cs_mask)
        elif self.CS_Algorithm == 'IST_S':
            cs_output = self.IST_S(cs_input, cs_mask)
        elif self.CS_Algorithm == 'NUSF':
            print("X data adjusted correctly ?")
            cs_output =self.nonUniformFit(self.x)
            #cs_output = CLEAN(obj, cs_input, cs_mask)

        else:
            cs_output = self.IST_D(cs_input, cs_mask)
        return cs_output

    def IST_D(self, cs_input, cs_mask):
        """
        Keeps balance between sparsity and measured data.
        IST_D Algorithm. Takes FT spectrum as input.
        :param cs_input: FT Spectrum of Data
        :param cs_mask:
        :return: Reconstructed Spectrum
        """
        N_Iter = self.NIter
        t = self.threshold
        cs_input_init = cs_input.copy()
        cs_output_ft = np.zeros_like(cs_input, dtype=np.complex128) # initialize x as zero vector

        if self.multipleFIDs:
            THplot = []

            for i in range(N_Iter):
                ft_cs = fft(cs_input_init, axis=1)
                th = t * np.max(np.abs(ft_cs), axis=1) # get threshold

                # where() selects from two arrays depending on the condition. If true 1, else 2
                th_ft_cs = np.where(np.abs(ft_cs) >= th[:, np.newaxis], ft_cs, 0)  # Apply thresholding


                cs_output_ft += th_ft_cs # data that is wanted, only peaks above threshold

                th_cs = np.fft.ifft(th_ft_cs, axis=1)
                th_cs[:, ~cs_mask] = 0  # Set elements that haven#t been measured to 0 along the second dimension
                cs_input_init = cs_input_init - th_cs

            cs_output = ifft(cs_output_ft, axis=1)
        #norm_TH = 0.99 * abs(self.TH_spectrum / max(self.TH_spectrum))
        #norm_TH_it = norm_TH.copy()

        else:
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
        Strict accordance with measured data at each iteration.
        :param cs_input:
        :param cs_mask:
        :return: Spectrum
        """
        N_Iter = self.NIter
        t = self.threshold
        cs_input_init = cs_input.copy()

        cs_output_ft = np.zeros(len(cs_input), dtype=np.complex128)
        if self.multipleFIDs:

            for i in range(N_Iter):
                ft_cs = fft(cs_input_init, axis=1)
                th = t * np.max(np.abs(ft_cs), axis=1) * (N_Iter - i)/N_Iter
                th_ft_cs = np.where(np.abs(ft_cs) >= th[:, np.newaxis], ft_cs, 0)  # Apply thresholding

                cs_output_ft += th_ft_cs

                th_cs = np.fft.ifft(th_ft_cs, axis=1)
                th_cs[:, ~cs_mask] = 0  # Set masked elements to 0 along the second dimension
                cs_input_init = np.real(cs_input_init) - th_cs


            cs_output = ifft(cs_output_ft, axis=1)
        else:
            for i in range(N_Iter):
                ft_cs = fft(cs_input_init)
                th = t * max(abs(ft_cs)) * (N_Iter - i)/N_Iter# th is first a relative threshold
                th_ft_cs = thresh(ft_cs, th, self.mode)
                th_cs = ifft(th_ft_cs);
                cs_input_init[~cs_mask] = th_cs[~cs_mask]
            cs_output = cs_input_init
        return cs_output


    def nonUniformFFT(self):
        """
        Variables: (om, Nd, Kd, Jd).
        - om = (M,1), with being number of non cartesian points i.e. mask in our case
        - Nd = (Nfid,1), Number of points in fixedFID
        - Kd = (Nkspace, 1) Spectra
        - Jd = (6,1), size of Interpolator
        Takes already undersampled (non-cartesian) data (1D or 2D) as input.
        Needs:  - Mask
                - not Zero filled CS Data
        :return:
        """
        # https://jyhmiinlin.github.io/pynufft/index.html#document-tutor/init

        X_notZeroFilled = np.arange(self.lenDataFull)[self.samplingMask]
        Nd = (self.lenDataFull, 1) # maybe half it ?
        Kd = (self.lenDataFull, 1)
        Jd = (6, 1) # default

        # create NUFFT object
        pass

    def nonUniformFit(self,x):
        """
        Fit the undersampled FID withput Reconstruction of the Missing data points.
        :return:
        """
        #fitVecX = np.tile(np.arange(self.lenDataFull)[self.samplingMask], self.DataFull.shape[0]).reshape(self.DataFull.shape[0],-1)
        #fitVecX = np.arange(self.lenDataFull, dtype=np.float)[self.samplingMask]
        #fitVecX *= 2 * np.pi / 360
        p_initial = np.array([0.7754, -0.7195, 0.2081, 0.8009, -0.002237])  # 2% Agarose default
        #pOptAll, pCovAll = [], []
        try:
            pall = [curve_fit(self.fixedTQTPPI, x[self.samplingMask], self.CS_data[i ,self.samplingMask], p0=p_initial) for i in range(self.CS_data.shape[0])]
            pAll = pall[0]
        except:
            # If only one FID is fitted
            pAll = curve_fit(self.fixedTQTPPI, x[self.samplingMask], self.CS_data[self.samplingMask], p0=p_initial)
         # curve_fit(self.fixedTQTPPI, fitVecX, self.CS_data[:,self.samplingMask], p0=p_initial)
        return pAll







