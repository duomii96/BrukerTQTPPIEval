from NEOimportTQTPPIspectro import importTQTPPIspec
from fixedEvoTiEval import fixedEvoTimesEval
import numpy as np
import matplotlib.pyplot as plt
from TQTPPI_fit import FitTQTPPI, fixedFitTQTPPI
from fixedEvoTiEval import fixedEvoTimesEval
import stats
from dataToFroFile import toFile
from numpy import real as real
from numpy.fft import fftshift as shift
from numpy.fft import fft

IsFixed = 0  # Default

basePath = '/Users/duomii/Desktop/PhD/Data'

#studyPath = basePath + '/DZ_MedSup_ismrm24_1_37_20231104_115236/'
#startFolder, stopFolder, IsFixed = 94, 172, 1

studyPath = basePath + '/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/'  # BSA
startFolder, stopFolder, IsFixed = 97, 98, 1

skipIndices = []
skipIndices = np.unique(skipIndices)
skipIndices = skipIndices[skipIndices > startFolder]
skipIndices = skipIndices[skipIndices < stopFolder]

mqFIDall, mqSpectraAll, tau = [], [], []
j = 0  # index for multiple repetitions
evoTimes = np.zeros((stopFolder - startFolder - len(skipIndices) + 1, 1))
for k in np.arange(startFolder, stopFolder + 1):
    if k in skipIndices:
        continue
    dataPath = studyPath + f'{k}/'

    preFilter = 0  # Cos²-Fkt um Rauschen im FID entlang omega abzuschneiden (1)
    filterFacPre = 1  # 1 ist genau die Länge von FID. 2 -> halbe Länge

    filter2ndDim = 0  # Cos²-Fkt entlang der EvoTime-Richtung vor der FFT (1). Max Gewichtung bei kleinen EvoTime
    filterFac2ndDim = 1  # 1 ist genau die Länge von EvoTime. 2 -> halbe Länge

    filterFID = 0  # Cos²-Fkt um Rauschen im mqFID entlang EvoTime abzuschneiden nach der FFT (2)
    filterFacPost = 1  #

    preDCcomp = 1  #
    spikeComp = 1  # mega Spike ist weil 5 µs (?) Delay zwischen RF und ADC nicht eingebaut
    postDCcomp = 1
    phaseCorr = 1  # Phasenkorrektur, ausschalten für IRTQTPPI fixed
    w0corr = 1
    zeroFill = 1

    onlyReal = 1;  # FFT of real components or of real and imag. 1 = sym, 0 = asym

    freqDriftVal = [0, 0, 0]

    (method, rawComplexData, complexDataAllPhase, complexDataUnw, realDataAllPhase,
     realDataAllPhaseUnw, mqFID, mqSpectra, mixTime, evoTime) = importTQTPPIspec(dataPath, spikeComp, phaseCorr,
                                                                                 preDCcomp, filter2ndDim,
                                                                                 filterFac2ndDim,
                                                                                 preFilter, filterFacPre, w0corr,
                                                                                 freqDriftVal, postDCcomp, filterFID,
                                                                                 filterFacPost, onlyReal)

    NR = method['Repetitions']
    print(method['MixTime'])

    if method['Method'] == '<User:sr_DQF1>' or method['Method'] == '<User:sr_TQF1>':
        pass
    else:

        sqVal = np.max(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0] / 4)):int(np.fix(mqSpectra.shape[0] / 2)), 0]),
                       axis=0)
        posSq = np.argmax(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0] / 4)):int(np.fix(mqSpectra.shape[0] / 2)), 0]),
                          axis=0)
        # Korrektur nötig, weil zuvor in 8:16 gesucht, d.h. posSq bezieht sich
        # auf diese paar Schichten
        posSq += int(np.fix(np.size(mqSpectra, 0) / 4)) - 1
        #     tqVal, posTq = np.max(np.abs(mqSpectra[int(np.fix(posSq/3))-2:int(np.fix(posSq/3))+2,0,0])), axis=0
        #     posTq = np.fix(posSq/3) - 2 + posTq - 1
        posTq = int(np.ceil(posSq / 3))
    if method['Method'] == '<User:dk_Tqtppi_fix1>' or IsFixed:
        evoTimeVec = np.arange(1, np.size(mqFID, 0) + 1)
    else:
        evoTimeVec = np.arange(evoTime, method['EvoTimeStep'] * 0.001 * (
                    method['NumPhaseSteps'] * method['NumPhaseCycles'] - 1) / (np.size(mqFID, 0) - 1) + evoTime,
                               method['EvoTimeStep'] * 0.001 * (
                                           method['NumPhaseSteps'] * method['NumPhaseCycles'] - 1) / (
                                           np.size(mqFID, 0) - 1))

    if NR == 1:
        mqFIDall.append(mqFID)

        mqSpectraAll.append(mqSpectra)
        tau.append(evoTime)
        evoTimes[j] = method['EvoTime']
        j += 1

    else:

        mqFID_tmp = np.mean(mqFID, axis=1)
        complexData_temp = np.mean(complexDataAllPhase, axis=0)  # needed for new reconstruction
        mqSpectra_tmp = np.mean(mqSpectra, axis=1)
        #         mqFID1[:,j:j+NR-1] = mqFID
        mqFIDall.append(mqFID_tmp)
        complexDataFixedAll.append(complexData_temp)
        mqSpectraAll.append(mqSpectra_tmp)
        tau.append(evoTime)
        evoTimes[j] = method['EvoTime']
        j += 1

    specDataShaped = shift(fft(np.squeeze(real(complexDataAllPhase.transpose()))).transpose(), axes=0)

    spacing = int(method["NumPhaseCycles"]) * 2

    tqPeaks, sqPeaks, ratio, noise_var = fixedEvoTimesEval(specDataShaped, evoTimes, spacing, secondDimFit=False)

    # create time vector, with correction for initial adc deadtime and spike.
    timeVec = np.arange(complexDataAllPhase.shape[-1]) / 10.
    plt.plot(timeVec, sqPeaks)
    plt.show()

