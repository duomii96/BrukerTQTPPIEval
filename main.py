from brukerapi.dataset import Dataset
from brukerapi.folders import Study
from NEOimportTQTPPIspectro import importTQTPPIspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from TQTPPI_fit import FitTQTPPI
from pathlib import Path

#data_path = '/Users/duomii/Documents/PhD/Data/DZ_BR_immersed2_1_5_20221219_092719/9/rawdata.job0'
#studyPath = '/Users/duomii/Documents/PhD/Data/DZ_BR_immersed2_1_5_20221219_092719/'
#studyPath = '/Users/duomii/Documents/PhD/Data/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/' # BSA
studyPath = '/Users/duomii/Documents/PhD/Data/AgarTmDOTP/DZ_DZ_TmDOTP_ISMRM22_1_1_20221016_104839/'
startFolder, stopFolder = 7, 8
skipIndices = []
IsFixed = 0
IsTauMixed=0

skipIndices = np.unique(skipIndices)
skipIndices = skipIndices[skipIndices>startFolder]
skipIndices = skipIndices[skipIndices<stopFolder]

# Need to initialize lists before loop as repetition information ist first accessed in loop
mqFIDall, mqSpectraAll, tau = [], [], []
j = 0 # index for multiple repetitions
evoTimes = np.zeros((stopFolder - startFolder - len(skipIndices)+1, 1))
for k in np.arange(startFolder,stopFolder+1):
    if k in skipIndices:
        continue
    dataPath = studyPath + f'{k}/'

    preFilter = 0 # Cos²-Fkt um Rauschen im FID entlang omega abzuschneiden (1)
    filterFacPre = 1 # 1 ist genau die Länge von FID. 2 -> halbe Länge

    filter2ndDim = 0 # Cos²-Fkt entlang der EvoTime-Richtung vor der FFT (1). Max Gewichtung bei kleinen EvoTime
    filterFac2ndDim = 1 # 1 ist genau die Länge von EvoTime. 2 -> halbe Länge

    filterFID = 0 # Cos²-Fkt um Rauschen im mqFID entlang EvoTime abzuschneiden nach der FFT (2)
    filterFacPost = 1 #

    preDCcomp = 1 #
    spikeComp = 1 # mega Spike ist weil 5 µs (?) Delay zwischen RF und ADC nicht eingebaut
    postDCcomp = 1
    phaseCorr = 1 # Phasenkorrektur, ausschalten für IRTQTPPI fixed
    w0corr = 1
    zeroFill = 1

    onlyReal = 1; # FFT of real components or of real and imag. 1 = sym, 0 = asym

    freqDriftVal = [0, 0, 0]

    (method, rawComplexData, complexDataAllPhase, complexDataUnw, realDataAllPhase,
     realDataAllPhaseUnw, mqFID, mqSpectra, mixTime, evoTime) = importTQTPPIspec(dataPath, spikeComp, phaseCorr, preDCcomp, filter2ndDim, filterFac2ndDim,
                         preFilter, filterFacPre, w0corr, freqDriftVal, postDCcomp, filterFID, filterFacPost, onlyReal)

    NR = method['Repetitions']

    if method['Method'] == '<User:sr_DQF1>' or method['Method'] == '<User:sr_TQF1>':
        pass
    else:

        sqVal = np.max(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0]/4)):int(np.fix(mqSpectra.shape[0]/2)),0]), axis=0)
        posSq = np.argmax(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0]/4)):int(np.fix(mqSpectra.shape[0]/2)),0]), axis=0)
        # Korrektur nötig, weil zuvor in 8:16 gesucht, d.h. posSq bezieht sich
        # auf diese paar Schichten
        posSq += int(np.fix(np.size(mqSpectra,0)/4)) - 1
    #     tqVal, posTq = np.max(np.abs(mqSpectra[int(np.fix(posSq/3))-2:int(np.fix(posSq/3))+2,0,0])), axis=0
    #     posTq = np.fix(posSq/3) - 2 + posTq - 1
        posTq = int(np.ceil(posSq/3))
    if method['Method'] == '<User:dk_Tqtppi_fix1>':
        evoTimeVec = np.arange(1, np.size(mqFID,0)+1)
    else:
        evoTimeVec = np.arange(evoTime, method['EvoTimeStep']*0.001*(method['NumPhaseSteps']*method['NumPhaseCycles']-1)/(np.size(mqFID,0)-1)+evoTime, method['EvoTimeStep']*0.001*(method['NumPhaseSteps']*method['NumPhaseCycles']-1)/(np.size(mqFID,0)-1))





    if NR == 1:
        if IsTauMixed:
            mqFIDall[:,j] = mqFID
            mqSpectraAll[:, j] = mqSpectra
            tau[j] = evoTime
            evoTimes[j] = method.MixTime*1e-3 # us->ms
            j += 1
        else:
            mqFIDall.append(mqFID)
            mqSpectraAll.append(mqSpectra)
            tau.append(evoTime)
            evoTimes[j] = method.EvoTime
            j += 1
    else:
        if IsFixed: #and method.Method == '<User:sr_IRTQTPPI_0180supr>':
            mqFID_tmp = np.mean(mqFID, axis=1)
            mqSpectra_tmp = np.mean(mqSpectra, axis=1)
    #         mqFID1[:,j:j+NR-1] = mqFID
            mqFIDall.append(mqFID_tmp)
            mqSpectraAll.append(mqSpectra_tmp)
            tau.append(evoTime)
            evoTimes[j] = method['EvoTime']
            j += 1
        elif IsTauMixed:
            mqFID_tmp = np.mean(mqFID, axis=1)
            mqSpectra_tmp = np.mean(mqSpectra, axis=1)
            mqFIDall.append(mqFID_tmp)
            mqSpectraAll.append(mqSpectra_tmp)
            tau.append(evoTime)
            evoTimes[j] = method['MixTime'] * 1e-3  # us->ms
            j = j + 1
        else:
            mqFIDall.append(mqFID)
            mqSpectraAll.append(mqSpectra)
            tau.append(evoTime)
            evoTi = int(j / NR)
            evoTimes[evoTi] = method['EvoTime']
            j = j + NR

# convert from list to np.array() and compute mean
mqFID1 = np.mean(np.hstack(mqFIDall), axis=-1)
mqSpectra1 = np.mean(np.hstack(mqSpectraAll), axis=-1)

fit1 = FitTQTPPI(mqFID1, method, studyPath)
fit1.fit()
figure = fit1.get_figure()
plt.show()
print(fit1.get_fitParams())
"""plt.plot(np.real(mqSpectra), linewidth=0.4)
plt.show()"""