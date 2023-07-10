import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import fft, fftshift, ifft
import numpy as np
import plotting as ptg
from getValues import getValues


class FitTQTPPI():
    """

    """
    def __init__(self, mqFID, method):
        self.method = method
        self.mqFID = mqFID.copy()
        self.fitVariant = 'SQTQ'
        self.alphas = method['PhaseList'][::2]
        self.fitVectorX = self.getX()
        self.dataSaveDir = '/Users/duomii/Documents/PhD/MeasurementEval'

        if np.amax(np.real(mqFID)) > np.amax(np.imag(mqFID)):
            self.mqFID = np.real(self.mqFID)/np.max(np.real(self.mqFID))
            self.fitVectorY = np.real(self.mqFID)

        else:
            self.mqFID = np.imag(self.mqFID) / np.amax(np.imag(self.mqFID))
            self.fitVectorY = np.real(self.mqFID)
        self.preFitSpectrum = fftshift(fft(self.mqFID))
        self.preFitSpectrumNorm = self.preFitSpectrum.copy()/np.amax(self.preFitSpectrum)





    @staticmethod
    def SQTQ(x, a, b, c, d, e, f, g, h, q):
        """
        SQ and TQ Fit for normal TQTPPI FID.
        """
        return (a*np.sin(2*np.pi*q * x+f)*np.exp(-x/b) + c*np.sin(2*np.pi* q *x+f)*np.exp(-x/d)
        + e*np.sin(3*2*np.pi*q*x+g)*(-np.exp(-x/b)+np.exp(-x/d))+ h)

    @staticmethod
    def SQTQDQ():
        pass

    def getX(self):
        tevoStep = self.method['EvoTimeStep'] * 10**(-6) # conversion to seconds
        tevo0 = self.method['EvoTime'] * 10**(-3) # conversion to seconds
        self.f = 1/(len(self.alphas)* tevoStep)
        x = np.arange(self.method['NumPhaseCycles']*len(self.alphas), dtype=np.float64)
        x[0] = tevo0
        #xScaled[1:] = x[1:] * 2
        x[1:] = x[1:]  * tevoStep + tevo0
        return x



    def getBoundsSQTQ(self):
        p1_initial = getValues.get_randomAB(3., 15., 1)
        p2_initial = getValues.get_randomAB(0.0001, 0.02, 1)
        p3_initial = getValues.get_randomAB(3., 15., 1)
        p4_initial = getValues.get_randomAB(0.03, 0.1, 1)
        p5_initial = getValues.get_randomAB(0.0, 0.30, 1)
        p6_initial = 0;
        p7_initial = np.pi #-2 * pi
        p8_initial = .01
        p9_initial = self.f

        p_initialMono = [p1_initial, p2_initial, p3_initial, p4_initial, p5_initial, p6_initial, p7_initial, p8_initial, p9_initial];
        # STIMMEN DIE EINHEITEN ?
        p_lower_boundsMono = [0, 0.0001,   0,  0.0001,   0, - 2 * np.pi, - 2 * np.pi, - .1,  p9_initial - 1];
        p_upper_boundsMono = [15., 0.1000,   15.,  0.1000,   1.5,   3 * np.pi,   3 * np.pi,   .1,  p9_initial + 1];
        return  (p_lower_boundsMono,p_upper_boundsMono), p_initialMono


    def fit(self):

        if self.fitVariant == 'SQTQ':
            SQTQbounds, p_initial = self.getBoundsSQTQ()
            # Least squares fit
            self.params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY, p0=p_initial,bounds=SQTQbounds)
            #self.params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY)
            self.stds = np.sqrt(np.diag(pcov)) # pcov diagonal is variance result parameters

            self.fitFID = self.SQTQ(self.fitVectorX, *self.params)


            # FFT to get spectrum
            self.fitSpectrum = fftshift(fft(self.fitFID))
            self.fitSpectrumNorm = self.fitSpectrum/np.amax(np.abs(self.fitSpectrum))

        else:
            pass



    def get_figure(self,type='spectrum'):

        if type == 'fid':
            # Here .reshape() instead transpose() ?

            #plt.plot(self.fitVectorX, np.real(self.fitFID))
            figure =  ptg.myplot(np.array([self.fitVectorY, np.real(self.fitFID)]).transpose(),
            xdata=self.fitVectorX)
        else:
            """figure = plt.figure()
            plt.plot(np.real(self.preFitSpectrum))
            plt.plot(np.real(self.fitSpectrum))"""
            figure =  ptg.myplot(np.array([np.real(self.preFitSpectrumNorm), np.real(self.fitSpectrumNorm)]).transpose())
        return figure

    def get_fitParams(self):
        # return Params as dictionary ?
        # only for TQTPPI
        params = {}
        paramkeys = ["ASQs", "T2s", "ASQf", "T2f", "ATQ"]
        errs = ["ASQs_std", "T2s_std", "ASQf_std", "T2f_std", "ATQ_std"]

        for idx,key in enumerate(paramkeys):
            params[key] = self.params[idx]
            params[errs[idx]] = self.stds[idx]
        params["ratio"]= params["ATQ"]/(params["ASQs"]+params["ASQf"])
        params["ratio_std"] = np.sqrt((params["ATQ_std"] / (params["ASQs"]+params["ASQf"])) ** 2 + (params["ATQ"] * params["ASQs_std"] / (params["ASQs"]+params["ASQf"]) ** 2) ** 2
                            + (params["ATQ"] * params["ASQf_std"] / (params["ASQs"]+params["ASQf"]) ** 2) ** 2)
        params["T2s"] *= 1000
        params["T2f"] *= 1000
        params["T2f_std"] *= 1000
        params["T2s_std"] *= 1000

        if params["T2f"] > params["T2s"]:
            # swap T2 values if fast and slow component turn out to switched
            # Change Amplitudes as well ????
            temp, tempStd = params["T2f"], params["T2f_std"]
            params["T2f"], params["T2f_std"] = params["T2s"], params["T2s_std"]
            params["T2s"], params["T2s_std"] = temp, tempStd
            del temp, tempStd
        return params



class fixedFitTQTPPI(FitTQTPPI):

    def __init__(self, mqFIDs, evoTimes, secondDimFit=False):
        self.mqFIDs = mqFIDs.copy()
        self.numFIDs, self.numPhaseInc = mqFIDs.shape[0], mqFIDs.shape[1]
        self.evoTimes = evoTimes
        # FT along phase increment. Axes swaps zero frequency only for given axis dimension
        self.spectralDataAcq = fftshift(fft(self.mqFIDs, axis=1), axes=(1,))

        # self.sqVal  = np.amax(np.abs(mqFIDs[(int(mqFIDs.shape[0] / 4)):(int(mqFIDs.shape[0] / 2)), 0, 0]))
        # find index of max SQ signal, along pahse increment dimension and all spectr. Assumes max is SQ value.
        _,self.posSq,_ = np.unravel_index(np.argmax(self.spectralDataAcq[1,:int(self.numPhaseInc/2),:], axis=None), self.spectralDataAcq[:,:int(self.numPhaseInc/2),:].shape)
        #self.posSq = posSq + (int(self.numPhaseInc / 4)) - 1
        self.posTq = self.posSq - 32

        if secondDimFit:
            pass
        else:
            self.SQs, self.TQs = self.get_SQTQpeaks()



    def get_SQTQpeaks(self):

        SQpeaks = self.spectralDataAcq[:,self.posSq,:]
        TQpeaks = np.squeeze(self.spectralDataAcq[:,self.posTq,:])
        return SQpeaks, TQpeaks

    def get_NoiseEstimate(self, spec):
        # Noise in spectrum taken for every spectral point
        # Remove SQ and TQ peaks from spectrum as well as offsett
        mask = np.zeros_like(spec)
        mask[:, [self.posTq, self.posSq, self.posSq + 16,self.posSq + 32, self.posTq + 96], :] = 1
        # mask spectra
        specForNoise = self.spectralDataAcq.copy()
        specForNoise = specForNoise[np.invert(mask)]
        # calculate noise
        noiseSpec = np.var(specForNoise, axis=1)





























