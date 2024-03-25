from commonImports import *
from scipy.optimize import curve_fit
import plotting as ptg
from Simulations import getValues
import lmfit

class FitTQTPPI():
    """

    """
    def __init__(self, mqFID, method, multipleInit=False):
        self.method = method
        self.mqFID = mqFID.copy()
        self.fitVariant = 'SQTQ'
        self.alphas = method['PhaseList'][::2]
        self.fitVectorX = self.getX()
        self.dataSaveDir = '/Users/duomii/Desktop/PhD/MeasurementEval'

        if np.amax(np.real(mqFID)) > np.amax(np.imag(mqFID)):
            self.mqFID = np.real(self.mqFID)/np.amax(np.real(self.mqFID))
            self.fitVectorY = np.real(self.mqFID)

        else:
            self.mqFID = np.imag(self.mqFID) / np.amax(np.imag(self.mqFID))
            self.fitVectorY = np.real(self.mqFID)
        self.preFitSpectrum = fftshift(fft(self.mqFID))
        self.preFitSpectrumNorm = self.preFitSpectrum.copy()/np.amax(self.preFitSpectrum)
        self.multipleInit = multipleInit





    @staticmethod
    def SQTQ(x, a, b, c, d, e, f, g, h, q):
        """
        SQ and TQ Fit for normal TQTPPI FID.
        """
        return np.sin(2*np.pi*q * x+f)*(a*np.exp(-x/b) + c*np.exp(-x/d))+ e*np.sin(3*2*np.pi*q*x+g)*(-np.exp(-x/b)+np.exp(-x/d))+ h

    @staticmethod
    def f_31(x, a, b, c, d):
        """
        TQ Transferfunction f_31: t/Tfast .. - t/Tslow
        """
        return a * (np.exp(-x/b)-np.exp(-x/c)) + d
    @staticmethod
    def f_11(x, a, b, c, d, e):
        # transferfunction for SQ
        return a * np.exp(-x/b) + d * np.exp(-x/c) + e

    @staticmethod
    def f_11_31(x, Asqf, Asqs, Atq, T2s, T2f, dc):
        sqRes = FitTQTPPI.f_11(x, Asqf, T2f, Asqs, T2s, dc)
        tqRes = FitTQTPPI.f_31(x, Atq, T2f, T2s, dc)
        return np.array([sqRes, tqRes]).transpose()


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
        p1_initial = getValues.get_randomAB(0.4, 0.9, 1)# Asqf
        p2_initial = getValues.get_randomAB(0.0002, 0.02, 1) #T2f - seconds
        p3_initial = getValues.get_randomAB(0.2, 0.8, 1)# Asqs
        p4_initial = getValues.get_randomAB(0.03, 0.1, 1)#T2s - seconds
        p5_initial = getValues.get_randomAB(0.0, 0.3, 1) #TQ
        p6_initial = 0
        p7_initial = np.pi #-2 * pi
        p8_initial = .01
        p9_initial = self.f

        p_initialMono = [p1_initial, p2_initial, p3_initial, p4_initial, p5_initial, p6_initial, p7_initial, p8_initial, p9_initial]
        # STIMMEN DIE EINHEITEN ?
        p_lower_boundsMono = [0., 0.0001,   0,  0.005,   0.001, - 2 * np.pi, - 2 * np.pi, - .1,  p9_initial - 1]
        p_upper_boundsMono = [1., 0.03,   1.0,  0.1,   0.6,   3 * np.pi,   3 * np.pi,   .1,  p9_initial + 1]
        return  (p_lower_boundsMono,p_upper_boundsMono), p_initialMono


    def fit(self):

        if self.fitVariant == 'SQTQ':
            if self.multipleInit:
                allPars, allStds = [], []
                for i in range(100):
                    SQTQbounds, p_initial = self.getBoundsSQTQ()
                    # Least squares fit
                    params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY, p0=p_initial,
                                                  bounds=SQTQbounds, method='trf')
                    allPars.append(params)
                    allStds.append(np.sqrt(np.diag(pcov)))
                    # self.params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY)
                      # pcov diagonal is variance result parameters
                    del SQTQbounds, p_initial, params, pcov
                allPars, allStds = np.array(allPars), np.array(allStds)
                ratioAll = np.expand_dims(allPars[:,4] /(allPars[:,0] + allPars[:,2]),axis=1)

                plt.subplot(221)
                plt.plot(allPars[:,1], label='$T_{2f}$')
                plt.legend(loc="best")
                plt.subplot(222)
                plt.plot(allPars[:,3], label='$T_{2s}$')
                plt.legend(loc="best")
                plt.subplot(223)
                plt.plot(allPars[:,4], 'v-',label="TQ")
                plt.legend(loc="best")
                plt.subplot(224)
                plt.plot(allPars[:,0], 'v-', label="$A_{SQf}$")
                plt.plot(allPars[:,2], '<-', label="$A_{SQs}$")
                plt.legend(loc="best")
                allPars = np.concatenate((allPars, ratioAll), axis=1)
                self.params = np.mean(allPars, axis=0)
                self.stds = np.std(allPars, axis=0)

            else:


                SQTQbounds, p_initial = self.getBoundsSQTQ()
                # Least squares fit
                self.params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY, p0=p_initial,bounds=SQTQbounds)
                #self.params, pcov = curve_fit(self.SQTQ, self.fitVectorX, self.fitVectorY)
                self.stds = np.sqrt(np.diag(pcov)) # pcov diagonal is variance result parameters

            self.fitFID = self.SQTQ(self.fitVectorX, *self.params[:-1]) # last param is ratio


            # FFT to get spectrum
            self.fitSpectrum = fftshift(fft(self.fitFID))
            self.fitSpectrumNorm = self.fitSpectrum/np.amax(np.abs(self.fitSpectrum))

        else:
            pass

    def fitLMfit(self):
        if self.fitVariant == 'SQTQ':
            if self.multipleInit:
                allPars, allStds = [], []
                for i in range(100):
                    bounds, p_initial = self.getBoundsSQTQ()
                    # Setting up lmfit Parameters object
                    params = lmfit.Parameters()
                    model = lmfit.Model(self.SQTQ)
                    for name, bLow, bUp in zip(model.param_names, bounds[0], bounds[1]):
                        model.set_param_hint(name, min=bLow, max=bUp)
                    params = model.make_params(a=p_initial[0], b=p_initial[1], c=p_initial[2], d=p_initial[3],
                                               e=p_initial[4], f=p_initial[5], g=p_initial[6], h=p_initial[7],
                                               q=p_initial[8])
                    # Perform fitting
                    result = model.fit(self.fitVectorY, params, x=self.fitVectorX)

                    allPars.append([result.best_values[param] for param in result.best_values])
                    allStds.append([result.params[param].stderr for param in result.params])

                allPars, allStds = np.array(allPars), np.array(allStds)
                ratioAll = np.expand_dims(allPars[:, 4] / (allPars[:, 0] + allPars[:, 2]), axis=1)

                plt.subplot(221)
                plt.plot(allPars[:, 1], label='$T_{2f}$')
                plt.legend(loc="best")
                plt.subplot(222)
                plt.plot(allPars[:, 3], label='$T_{2s}$')
                plt.legend(loc="best")
                plt.subplot(223)
                plt.plot(allPars[:, 4], 'v-', label="TQ")
                plt.legend(loc="best")
                plt.subplot(224)
                plt.plot(allPars[:, 0], 'v-', label="$A_{SQf}$")
                plt.plot(allPars[:, 2], '<-', label="$A_{SQs}$")
                plt.legend(loc="best")
                allPars = np.concatenate((allPars, ratioAll), axis=1)
                self.params = np.mean(allPars, axis=0)
                self.stds = np.std(allPars, axis=0)

            else:
                SQTQbounds, p_initial = self.getBoundsSQTQ()

                # Create the lmfit Model
                model = lmfit.Model(self.SQTQ)
                params = model.make_params(a=p_initial[0], b=p_initial[1], c=p_initial[2], d=p_initial[3],
                                           e=p_initial[4], f=p_initial[5], g=p_initial[6], h=p_initial[7],
                                           q=p_initial[8])

                # Perform fitting
                result = model.fit(self.fitVectorY, params, x=self.fitVectorX, bounds=SQTQbounds)

                self.params = [result.best_values[param] for param in result.best_values]
                self.stds = [result.params[param].stderr for param in result.params]

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
        paramStore = {}
        paramkeys = ["ASQs", "T2s", "ASQf", "T2f", "ATQ", "ratio"]
        errs = ["ASQs_std", "T2s_std", "ASQf_std", "T2f_std", "ATQ_std", "ratio_std"]

        for idx,key in enumerate(paramkeys):
            if key == "ratio":
                paramStore[key] = self.params[-1]
                paramStore[errs[idx]] = self.stds[-1]
            else:
                paramStore[key] = self.params[idx]
                paramStore[errs[idx]] = self.stds[idx]

        paramStore["T2s"] *= 1000
        paramStore["T2f"] *= 1000
        paramStore["T2f_std"] *= 1000
        paramStore["T2s_std"] *= 1000

        if paramStore["T2f"] > paramStore["T2s"]:
            # swap T2 values if fast and slow component turn out to switched
            # Change Amplitudes as well ????
            temp, tempStd = paramStore["T2f"], paramStore["T2f_std"]
            tempAmplitude, tempAmplitude_std = paramStore["ASQf"], paramStore["ASQf_std"]
            paramStore["T2f"], paramStore["T2f_std"] = paramStore["T2s"], paramStore["T2s_std"]
            paramStore["T2s"], paramStore["T2s_std"] = temp, tempStd
            paramStore["ASQf"], paramStore["ASQf_std"] = paramStore["ASQs"], paramStore["ASQs_std"]
            paramStore["ASQs"], paramStore["ASQs_std"] = tempAmplitude, tempAmplitude_std

            del temp, tempStd, tempAmplitude, tempAmplitude_std
        return paramStore



class fixedFitTQTPPI(FitTQTPPI):

    def __init__(self, mqFIDs, evoTimes, numPhaseCycles,secondDimFit=False):
        if mqFIDs.ndim <= 1:
            mqFIDs = np.empty((10,10))

        self.mqFIDs = mqFIDs.copy()
        self.numPhaseCycles = numPhaseCycles
        self.numFIDs, self.numPhaseInc = mqFIDs.shape[0], mqFIDs.shape[1]
        self.evoTimes = evoTimes.flatten()
        # FT along phase increment. Axes swaps zero frequency only for given axis dimension
        self.spectralDataAcq = fftshift(fft(self.mqFIDs, axis=1), axes=(1,))

        # self.sqVal  = np.amax(np.abs(mqFIDs[(int(mqFIDs.shape[0] / 4)):(int(mqFIDs.shape[0] / 2)), 0, 0]))
        # find index of max SQ signal, along pahse increment dimension and all spectr. Assumes max is SQ value.
        try:
            _,self.posSq,_ = np.unravel_index(np.argmax(self.spectralDataAcq[1,:int(self.numPhaseInc/2),:], axis=None), self.spectralDataAcq[:,:int(self.numPhaseInc/2),:].shape)
        except:
            _, self.posSq = np.unravel_index(np.argmax(self.spectralDataAcq[1, :int(self.numPhaseInc / 2)], axis=None),
                self.spectralDataAcq[:, :int(self.numPhaseInc / 2)].shape)

        #self.posSq = posSq + (int(self.numPhaseInc / 4)) - 1
        if np.any(mqFIDs):
            self.posTq = self.posSq - 2*self.numPhaseCycles

        if secondDimFit:
            pass
        elif not(np.any(mqFIDs)):
            self.SQs, self.TQs = None, None
        else:
            self.SQs, self.TQs = self.get_SQTQpeaks()



    def get_SQTQpeaks(self):

        try:
            SQpeaks = self.spectralDataAcq[:,self.posSq,:]
            TQpeaks = np.squeeze(self.spectralDataAcq[:,self.posTq,:])
        except:
            SQpeaks = self.spectralDataAcq[:, self.posSq]
            TQpeaks = np.squeeze(self.spectralDataAcq[:, self.posTq])

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

    def fitTQsecondDim(self):

        if not(np.any(self.mqFIDs)):
            print("Using loaded SQ and TQ peaks!")
            print(f'Number of TQs and SQs: {self.SQs.shape}')
            self.SQs, self.TQs = np.real(self.SQs) / np.amax(np.real(self.SQs)), np.real(self.TQs) / np.amax(
                np.real(self.TQs))
        else:

            self.SQs, self.TQs = self.get_SQTQpeaks()

        # normalize Peaks to Max SQ Signal in real part or TQ -> check that
            self.SQs, self.TQs = np.real(self.SQs) /np.amax(np.real(self.SQs)), np.real(self.TQs)/np.amax(np.real(self.TQs))

        pSQ = curve_fit(self.f_11, xdata=np.array(self.evoTimes), ydata=np.squeeze(self.SQs), p0 = [0.6, 40, 1, 0.4, 0.005])
        pTQ = curve_fit(self.f_31, xdata=self.evoTimes, ydata=self.TQs, p0=[0.5, 25, 3, 0.005])
        return pSQ, pTQ

    @staticmethod
    def combinedFit(comboX, Asqs, Asqf, Atq, T2s, T2f, SqOff, TqOff):
        # function that combines both f11 and f31 transfer functions
        half = int(len(comboX)/2)
        sqX = comboX[:half]
        tqX = comboX[half:]
        resSQ = fixedFitTQTPPI.f_11(sqX, Asqs, T2s, Asqf, T2f, SqOff)
        resTQ = fixedFitTQTPPI.f_31(tqX, Atq, T2s, T2f, TqOff)

        return np.append(resSQ, resTQ)
    def fitBoth(self):

        if not(np.any(self.mqFIDs)):
            print("Using loaded SQ and TQ peaks!")
            print(f'Number of TQs and SQs: {self.SQs.shape}')
            self.SQs, self.TQs = np.real(self.SQs) / np.amax(np.real(self.SQs)), np.real(self.TQs) / np.amax(
                np.real(self.TQs))
        else:

            self.SQs, self.TQs = self.get_SQTQpeaks()

        # normalize Peaks to Max SQ Signal in real part or TQ -> check that
            self.SQs, self.TQs = np.real(self.SQs) /np.amax(np.real(self.SQs)), np.real(self.TQs)/np.amax(np.real(self.TQs))

        comboY = np.append(self.SQs, self.TQs)
        comboX = np.append(self.evoTimes, self.evoTimes)
        #pAll, pCovAll = curve_fit(self.f_11_31, self.evoTimes, z_vec, p0=[0.6, 0.4, 0.5, 3, 20, 0.005])
        pInit = np.array([0.6, 0.4, 0.5, 25, 5, 0.005, 0.001])

        (Asqs, Asqf, Atq, T2s, T2f, SqOff, TqOff), parsCov = curve_fit(fixedFitTQTPPI.combinedFit, comboX, comboY, p0=pInit)
        std_Asqs, std_Asqf, std_Atq, std_T2s, std_T2f, std_SqOff, std_TqOff = np.diag(parsCov)

        parsTq = np.array([[Atq, T2s, T2f, TqOff], [std_Atq, std_T2s, std_T2f, std_TqOff]])
        parsSq = np.array([[Asqs, T2s, Asqf, T2f, SqOff], [std_Asqs, std_T2s,std_Asqf, std_T2f, std_SqOff]])

        return parsSq, parsTq
class FitSinglePulse():
    """
    USE:
    dataPath = '...'
    cData, complex_fft, method = NEOreadIn(dataPath)
    SingleP = FitSinglePulse(cData, np.arange(cData.shape[-1]))
    OR:
    Jupyter NOtebook
    """

    def __init__(self, signal, timeData, startIdx = 0, monoExp = False):
        self.sigData = real(np.squeeze(signal)) / np.max(real(signal))
        self.timeData = timeData
        self.mono = monoExp
        if startIdx != 0:
            self.fitData = self.sigData.copy()[startIdx:]
            self.fitTimeData = self.timeData.copy()[startIdx:]
        else:
            startIdx = np.argmax(self.sigData)
            print(f'{startIdx}')
            self.fitData = self.sigData.copy()[startIdx:]
            self.fitTimeData = self.timeData.copy()[startIdx:]

    @staticmethod
    def f11(x, a, b, c, d, e):
        return a * np.exp(-x / b) + c * np.exp(-x / d) + e
    @staticmethod
    def fitWithB0InHom(x, a, b, c, d,db0, e):
        # fit the f11 transfer function with additional B0 inhom factor.
        return np.exp(-db0 * x) * (a * np.exp(-x / b) + c * np.exp(-x / d)) + e
    def fit(self, startIdx=0):
        """
        Fits a single pulse to extract T2* relaxations times. Starts at maximum signal amplitude.
        :param data:
        :return:
        """
        #self.fitDataNorm = self.fitData / np.max(self.fitData)

        pInit = [0.6, 1, 0.4, 40, 0.005]
        self.popt, self.pcov = curve_fit(self.f11, self.fitTimeData, self.fitData, p0=pInit)

        self.T2fast, self.T2slow, self.Afast, self.Aslow = self.popt[1], self.popt[3], self.popt[0], self.popt[2]

    def plotFit(self):
        plt.figure(figsize=(16,9), dpi=120)
        plt.plot(self.fitTimeData, self.fitData, alpha=0.6, linewidth=1.5, label='Data')
        plt.plot(self.fitTimeData, self.f11(self.fitTimeData, *self.popt), linewidth=0.7,label='Fit')
        plt.xlabel('time [ms]')
        plt.ylabel('norm. Signal')
        plt.legend(loc="best")


    def get_tauOpt(self):
        return np.log(self.T2slow / self.T2fast) / (1/self.T2fast-1/self.T2slow)




def phaseCorrGRPDL(fid, acqp):
    s = fid.shape[-1] # length of FID

    phaseCorr = getValues.get_bruker_groupDelay(acqp) # get the group delay value from the acqp file

    # Fourier transform the data
    ftInit = fftshift(fft(fid))
    # apply phase correction
    pdata = ftInit * np.exp(2.j * np.pi * phaseCorr * np.arange(s) / s)

    return np.abs(ifft(pdata))

