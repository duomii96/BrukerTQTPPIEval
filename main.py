from NEOimportTQTPPIspectro import importTQTPPIspec
from NEOimportFleysherTQFspectro import importFleysherTQF
from fixedEvoTiEval import fixedEvoTimesEval
import numpy as np
import matplotlib.pyplot as plt
from TQTPPI_fit import FitTQTPPI, fixedFitTQTPPI
import stats
from dataToFroFile import toFile
from numpy import real as real
from numpy.fft import fftshift as shift
from numpy.fft import fft

# TODO: - Define StudyPath
#       - Define StartStop folder
#       - Define Type of Measurement - w/o 180, fix etc. ...


IsFixed = 0 #Default

basePath = '/Users/duomii/Desktop/PhD/Data'

studyPath = basePath + '/DZ_FleysherTQF_test_1_38_20231114_145611/'
startFolder, stopFolder, IsFixed = 319, 319, 1
#startFolder, stopFolder =94, 172; IsFixed = 1; #fixedWo180, tmix 150
#startFolder, stopFolder =10, 88; IsFixed = 1; # MedSup fixedwo180
#startFolder, stopFolder =176, 176;
#startFolder, stopFolder =202,  230; IsFixed=1; #TQTPPI wo180 fixed
#studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_BR_NaCl_2_1_24_20231021_154741/' # BR in NaCL
# Cell medium only
#studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_CellMedium_1_26_20231022_124338/'
#studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_BrukerTriple_Test2_1_32_20231024_100740/'
#startFolder, stopFolder = 7, 55; IsFixed =1
#startFolder, stopFolder = 7, 7 ; #IsFixed =1
#studyPath = '/Users/duomii/Documents/PhD/Data/DZ_BR_immersed2_1_5_20221219_092719/'

#fixed
#startFolder, stopFolder =58,  61; IsFixed=1; #TQTPPI wo180 fix deltaAlpha = 5°, tevo = 5ms
#startFolder, stopFolder = 135, 137 #129, 152
#startFolder =   58; stopFolder  =   61; IsFixed=1;% TQTPPI wo180 fix deltaAlpha = 5°, tevo = 5ms
# startFolder =   98; stopFolder  =   98; IsFixed=1;% TQTPPI wo180 fix deltaAlpha = 5°, tevo =7.5ms
#startFolder =   98; stopFolder  =   98; IsFixed=1;
# startFolder =   102; stopFolder  =   105; % TQTPPI wo180 fix deltaAlpha = 5°, tevo =2.5ms
# startFolder =   110; stopFolder  =   113; % TQTPPI wo180 fix deltaAlpha = 5°, tevo = 20ms

##########################################################################
##### ---------- ISMRM 24 ------------------------------------------------
#studyPath = basePath + '/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/' # BSA
#studyPath = basePath + '/DZ_Agar3_ismrm24_1_35_20231103_085319/'
#startFolder, stopFolder, IsFixed = 19, 19, 1
#startFolder =   110; stopFolder  =   113; IsFixed = 1; # TQTPPI wo180 fix deltaAlpha = 5°, tevo = 20ms
skipIndices = []
IsTauMixed=0

skipIndices = np.unique(skipIndices)
skipIndices = skipIndices[skipIndices>startFolder]
skipIndices = skipIndices[skipIndices<stopFolder]

# Need to initialize lists before loop as repetition information ist first accessed in loop
mqFIDall, mqSpectraAll, tau = [], [], []
j = 0 # index for multiple repetitions
evoTimes = np.zeros((stopFolder - startFolder - len(skipIndices)+1, 1))
if IsFixed:
    fixedFIDall, complexDataFixedAll = [], []
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

    """(method, rawComplexData, complexDataAllPhase, complexDataUnw, realDataAllPhase,
    realDataAllPhaseUnw, mqFID, mqSpectra, mixTime, evoTime) = importTQTPPIspec(dataPath, spikeComp, phaseCorr, preDCcomp, filter2ndDim, filterFac2ndDim,
                     preFilter, filterFacPre, w0corr, freqDriftVal, postDCcomp, filterFID, filterFacPost, onlyReal)
    """
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

        sqVal = np.max(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0]/4)):int(np.fix(mqSpectra.shape[0]/2)),0]), axis=0)
        posSq = np.argmax(np.abs(mqSpectra[int(np.fix(mqSpectra.shape[0]/4)):int(np.fix(mqSpectra.shape[0]/2)),0]), axis=0)
        # Korrektur nötig, weil zuvor in 8:16 gesucht, d.h. posSq bezieht sich
        # auf diese paar Schichten
        posSq += int(np.fix(np.size(mqSpectra,0)/4)) - 1
    #     tqVal, posTq = np.max(np.abs(mqSpectra[int(np.fix(posSq/3))-2:int(np.fix(posSq/3))+2,0,0])), axis=0
    #     posTq = np.fix(posSq/3) - 2 + posTq - 1
        posTq = int(np.ceil(posSq/3))
    if method['Method'] == '<User:dk_Tqtppi_fix1>' or IsFixed:
        evoTimeVec = np.arange(1, np.size(mqFID,0)+1)
    else:
        evoTimeVec = np.arange(evoTime, method['EvoTimeStep']*0.001*(method['NumPhaseSteps']*method['NumPhaseCycles']-1)/(np.size(mqFID,0)-1)+evoTime, method['EvoTimeStep']*0.001*(method['NumPhaseSteps']*method['NumPhaseCycles']-1)/(np.size(mqFID,0)-1))


    if NR == 1:

        if IsFixed:
            fixedFIDall.append(mqFID)

            mqSpectraAll.append(mqSpectra)
            tau.append(evoTime)
            evoTimes[j] = method['EvoTime']
            j += 1
        else:
            mqFIDall.append(mqFID)
            mqSpectraAll.append(mqSpectra)
            tau.append(evoTime)
            evoTimes[j] = method["EvoTime"]
            j += 1
    else:
        if IsFixed: #and method.Method == '<User:sr_IRTQTPPI_0180supr>':
            mqFID_tmp = np.mean(mqFID, axis=1)
            complexData_temp = np.mean(complexDataAllPhase, axis=0) # needed for new reconstruction
            mqSpectra_tmp = np.mean(mqSpectra, axis=1)
    #         mqFID1[:,j:j+NR-1] = mqFID
            fixedFIDall.append(mqFID_tmp)
            complexDataFixedAll.append(complexData_temp)
            mqSpectraAll.append(mqSpectra_tmp)
            tau.append(evoTime)
            evoTimes[j] = method['EvoTime']
            j += 1
        else:
            mqFIDall.append(mqFID)
            mqSpectraAll.append(mqSpectra)
            tau.append(evoTime)
            evoTi = int(j / NR)
            evoTimes[evoTi] = method['EvoTime']
            j = j + NR


print(f'SHAPE MQ FID: {mqFID.shape}')
if IsFixed:
    mqFIDs = np.array(fixedFIDall.copy())
    complexFixedFIDs = np.array(complexDataFixedAll.copy())
    fit1 = fixedFitTQTPPI(mqFIDs, evoTimes, method['NumPhaseCycles'])
    #fit2 = fixedFitTQTPPI(complexFixedFIDs, evoTimes)
    print(str(fit1.posSq) + str(fit1.posTq) )
    TQs, SQs = fit1.TQs, fit1.SQs
    fig, ax = plt.subplots(2,1)
    ax[0].plot(evoTimes, np.real(TQs))
    ax[0].set_title("TQ")
    ax[0].set_xlabel('Evolution Time [ms]')
    ax[1].plot(evoTimes, np.real(SQs))
    ax[1].set_title("SQ")
    plt.show()

else:
    # convert from list to np.array() and compute mean
    mqFID1 = np.mean(np.hstack(mqFIDall), axis=-1)
    mqSpectra1 = np.mean(np.hstack(mqSpectraAll), axis=-1)
    fit1 = FitTQTPPI(mqFID1, method)
    fit1.fit()

    params = fit1.get_fitParams()
    print(params)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.real(mqFID1), 'o-')
    ax[0].set_title("TQTPPI FID")
    #ax[0].set_xlabel('Evolution Time [ms]')
    ax[1].plot(np.real(mqSpectra1))
    ax[1].set_title("TQTPPI Spectrum")
    plt.show()

#    noise = stats.get_NoiseEstimate(np.hstack(mqSpectraAll))
#    print(f"Noise estimate: {noise}")

#fit1.fit()
#figure = fit1.get_figure()
#plt.show()
#toFile(fit1.get_fitParams(), studyPath, type='TQTPPI')
#plt.plot(np.real(mqSpectra1), linewidth=0.4)
