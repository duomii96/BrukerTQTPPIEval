from commonImports import *
from NEOimportTQTPPIspectro import importTQTPPIspec
from NEOimportFleysherTQFspectro import importFleysherTQF
from fixedEvoTiEval import fixedEvoTimesEval
from sigFit import FitTQTPPI, fixedFitTQTPPI



# TODO: - Define StudyPath
#       - Define StartStop folder
#       - Define Type of Measurement - w/o 180, fix etc. ...




basePath = '/Users/duomii/Desktop/PhD/Data'

# whether to save SQ and TQ peaks, such that data does not have to be reloaded every time
savePeaks = 0
loadTQSQfromSomewhere = 0
studyPath = basePath + '/DZ_ph3_Agar2_1_67_20240315_140342/'
#studyPath = basePath + '/SR_SR_ISMRM22_1_BrukerLin_1_4_20221014_183118/'
startFolder, stopFolder, IsFixed = 7, 7, 0
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

skipIndices = []
IsTauMixed=0

skipIndices = np.unique(skipIndices)
skipIndices = skipIndices[skipIndices>startFolder]
skipIndices = skipIndices[skipIndices<stopFolder]
MediumInversionTimes = []
# Need to initialize lists before loop as repetition information ist first accessed in loop
mqFIDall, mqSpectraAll, tau, complexDataAllPhaseAll = [], [], [], []
j = 0 # index for multiple repetitions
evoTimes = np.zeros((stopFolder - startFolder - len(skipIndices)+1, 1))
if IsFixed:
    fixedFIDall, complexDataFixedAll = [], []
if loadTQSQfromSomewhere:
    prefix = '/Users/duomii/Desktop/PhD/Python/BrukerINx/dataTemp/'
    suffix = studyPath.split('/')[-2] + '.npy'
    del evoTimes
    pathSQ = prefix + 'SQs_fixwo180_'+ suffix
    pathTQ = prefix + 'TQs_fixwo180_'+ suffix
    pathEvo = prefix + 'evoTis_fixwo180_'+ suffix
    SQs = np.load(pathSQ)
    TQs = np.load(pathTQ)
    evoTimes = np.load(pathEvo)
else:
    for k in np.arange(startFolder,stopFolder+1):
        if k in skipIndices:
            continue
        dataPath = studyPath + f'{k}/'
        print(f'Folder: {k}')

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



        try:
            MediumInversionTimes.append(method['InversionTimeMedium'])


        except:
            print('No Medium InvTime found!!')
            pass

        NR = method['Repetitions']
        print(f'Mixing Time: {method["MixTime"]} \n EvoTime: {method["EvoTime"]}')
        #print(method['MedInvTime'])
        if method['Method'] == '<User:sr_DQF>' or method['Method'] == '<User:sr_TQF1>':
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
        if method['Method'] == '<User:dk_Tqtppi_fix1>' or IsFixed or method['Method']== '<User:sr_DQF>':
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
            elif method['Method'] == '<User:sr_DQF>':
                mqFIDall.append(mqFID)
                mqSpectraAll.append(mqSpectra)
                complexDataAllPhaseAll.append(complexDataAllPhase)
                tau.append(evoTime)
                evoTimes[j] = method['EvoTime']
                j += 1
            else:
                mqFIDall.append(mqFID)
                mqSpectraAll.append(mqSpectra)
                tau.append(evoTime)
                evoTimes[j] = method["EvoTime"]
                #j += 1
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
            elif method['Method'] == '<User:sr_DQF>':
                complexDataAllPhaseAll.append(complexDataAllPhase)
                tau.append(evoTime)
                evoTi = int(j / NR)
                evoTimes[evoTi] = method['EvoTime']
                j = j + NR

            else:
                mqFID_temp = np.mean(mqFID, axis=1)
                mqFIDall.append(mqFID_temp)
                mqSpectra_tmp = np.mean(mqSpectra, axis=1)
                mqSpectraAll.append(mqSpectra_tmp)
                tau.append(evoTime)
                evoTi = int(j / NR)
                evoTimes[evoTi] = method['EvoTime']
                j = j + NR
                print(j)

if len(evoTimes) == 1:
    # adjust shape of arrays for later postprocessing
    mqFIDall = np.array(mqFIDall).squeeze()
    mqFIDall = np.expand_dims(mqFIDall, axis=0)

if IsFixed:
    if loadTQSQfromSomewhere:
        fit1 = fixedFitTQTPPI(np.array([]), evoTimes, numPhaseCycles=None)
        fit1.SQs = SQs
        fit1.TQs = TQs
        fit1.evoTimes = evoTimes
        pfitSQ, pfitTQ = fit1.fitTQsecondDim()


    else:
        mqFIDs = np.array(fixedFIDall.copy())
        complexFixedFIDs = np.array(complexDataFixedAll.copy())
        fit1 = fixedFitTQTPPI(mqFIDs, evoTimes, method['NumPhaseCycles'])
        #pSQ, pTQ = fit1.fitTQsecondDim()
        pSQ, pTQ = fit1.fitBoth()
        #fit2 = fixedFitTQTPPI(complexFixedFIDs, evoTimes)
        #print(str(fit1.posSq) + str(fit1.posTq) )
        TQs, SQs = fit1.TQs, fit1.SQs
        print(f'SQ fit parameters for fixed TQTPPI: \n {pSQ[0]}')
        print(f'TQ fit parameters for fixed TQTPPI: \n {pTQ[0]}')

        if savePeaks:
            suffix = studyPath.split('/')[-2]
            np.save('dataTemp/SQs_fixwo180_'+suffix+'.npy', SQs)
            np.save('dataTemp/TQs_fixwo180_' + suffix + '.npy', TQs)
            np.save('dataTemp/evoTis_fixwo180_'+suffix+'.npy', evoTimes.flatten())

    fig, ax = plt.subplots(2,1)
    ax[0].plot(evoTimes, np.real(TQs)/np.amax(np.real(TQs)))
    ax[0].plot(evoTimes, fit1.f_31(fit1.evoTimes, *pTQ[0]), label="NL Fit")
    ax[0].legend(loc="best")
    ax[0].set_title(r"TQ, $T_{{2f}}$={} ms, $T_{{2s}}$={} ms".format(np.round(pTQ[0][1],3), np.round(pTQ[0][2],3)))
    ax[0].set_xlabel('Evolution Time [ms]')
    ax[1].plot(evoTimes, np.real(SQs)/np.amax(np.real(SQs)))
    ax[1].plot(fit1.evoTimes, fit1.f_11(fit1.evoTimes, *pSQ[0]))
    ax[1].set_title(r"SQ, $T_{{2f}}$={} ms, $T_{{2s}}$={}".format(np.round(pSQ[0][1],3), np.round(pSQ[0][3],3)))
    plt.show()

elif method['Method']== '<User:sr_DQF>':
    mqFID1 = np.mean(np.hstack(mqFIDall), axis=-1)
    mqSpectra1 = np.mean(np.hstack(mqSpectraAll), axis=-1)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.real(mqFID1), 'o-')

    ax[0].set_title("TQTPPI FID")
    # ax[0].set_xlabel('Evolution Time [ms]')
    ax[1].plot(np.real(mqSpectra1), linewidth=0.8)
    ax[1].set_xticks([method['NumPhaseCycles'], 2 * method['NumPhaseCycles'], 3 * method['NumPhaseCycles']],
                     ['TQ', 'DQ', 'SQ'])
    ax[1].set_title("TQTPPI Spectrum")
    plt.show()

elif MediumInversionTimes:
    mqSpectraAll = np.array(mqSpectraAll).squeeze()
    SQs = mqSpectraAll[:, 48]
    TQs = mqSpectraAll[:, 16]
    ratio = np.real(TQs / SQs * 100)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(MediumInversionTimes, real(SQs))
    ax[0].plot(MediumInversionTimes, np.abs(SQs), label="Abs. value")
    ax[1].plot(MediumInversionTimes, (TQs))
    ax[1].plot(MediumInversionTimes, np.abs(TQs), label="Abs.value")
    ax[0].grid()
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    ax[0].set_title("SQ")
    ax[1].set_title("TQ")
    ax[1].grid()
    # ax[0].set_xlabel('Evolution Time [ms]')
    ax[2].plot(MediumInversionTimes, ratio)
    ax[2].set_title("ratio")
    ax[2].grid()
    plt.show()
else:
    # convert from list to np.array() and compute mean
    #mqSpectraAll = np.array(mqSpectraAll).squeeze()
    #mqFIDall = np.array(mqFIDall).squeeze()
    mqFID1 = np.mean(np.array(mqFIDall), axis=0)
    mqSpectra1 = np.mean(np.array(mqSpectraAll), axis=0)
    fit1 = FitTQTPPI(mqFID1, method, multipleInit=True)
    fit1.fitLMfit()

    params = fit1.get_fitParams()
    for el in params:
        print(el+f': {params[el]}')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(fit1.fitVectorY, 'o-', markersize=3)
    #ax[0].plot(fit1.fitFID, linewidth=0.8, label="NL Fit")
    ax[0].legend(loc="best")
    ax[0].grid()
    ax[0].set_title("TQTPPI FID")
    #ax[0].set_xlabel('Evolution Time [ms]')
    ax[1].plot(np.real(mqSpectra1)[:int(mqSpectra1.shape[0]/2)], linewidth=0.8)
    ax[1].set_xticks([method['NumPhaseCycles'], 2*method['NumPhaseCycles'], 3*method['NumPhaseCycles']], ['TQ', 'DQ', 'SQ'])
    ax[1].set_title("TQTPPI Spectrum")
    ax[1].grid()
    plt.show()

    print(f'TQ Sig. abs.: {np.real(mqSpectra1)[method["NumPhaseCycles"]]} \n SQ Sig. abs.: {np.real(mqSpectra1)[3*method["NumPhaseCycles"]]}')

#    noise = stats.get_NoiseEstimate(np.hstack(mqSpectraAll))
#    print(f"Noise estimate: {noise}")

#fit1.fit()
#figure = fit1.get_figure()
#plt.show()
#toFile(fit1.get_fitParams(), studyPath, type='TQTPPI')
#plt.plot(np.real(mqSpectra1), linewidth=0.4)
