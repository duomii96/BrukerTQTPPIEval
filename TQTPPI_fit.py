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
        Implements SQ and TQ Fit for normal TQTPPI FID.
        """
        return (a*np.sin(2*np.pi*q * x+f)*np.exp(-x/b) + c*np.sin(2*np.pi* q *x+f)*np.exp(-x/d)
        + e*np.sin(3*2*np.pi*q*x+g)*(-np.exp(-x/b)+np.exp(-x/d))+ h)

    @staticmethod
    def SQTQDQ():
        pass

    def getX(self):
        tevoStep = self.method['EvoTimeStep'] * 10**(-6)
        tevo0 = self.method['EvoTime'] * 10**(-3)
        self.f = 1/(len(self.alphas)* tevoStep)
        x = np.arange(self.method['NumPhaseCycles']*len(self.alphas), dtype=np.float64)
        x[0] = tevo0
        #xScaled[1:] = x[1:] * 2
        x[1:] = x[1:]  * tevoStep + tevo0
        return x

    def getBoundsSQTQ(self):
        p1_initial = getValues.get_randomAB(3., 15., 1);
        p2_initial = getValues.get_randomAB(0.0001, 0.02, 1);
        p3_initial = getValues.get_randomAB(3., 15., 1);
        p4_initial = getValues.get_randomAB(0.03, 0.1, 1);
        p5_initial = getValues.get_randomAB(0.0, 0.30, 1);
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
            stds = np.sqrt(np.diag(pcov)) # pcov diagonal is viariance result parameters

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
        return self.params







