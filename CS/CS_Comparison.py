import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from CS.CS_Reconstruction import RecoCS
import plotting as plo
from CS.GenerateFixedFID import GenerateFIDfixed
import numpy as np
from CS.ComparisonMetrics import hamming, RMSE, trapzoid, simpson, get_RecoTQSQ
from NEOimportTQTPPIspectro import importTQTPPIspec
from numpy.fft import fft, ifft, fftshift
from stats import getTQSQ
from scipy.optimize import curve_fit
from numpy import real
import os

def fixedTQTPPI(x, a, b, c, d, e):
    return a * np.sin(x + b) + c * np.sin(3 * x + d) + e
def singleQCsignal(x, a ,phase,dc, TQ=False):
    if TQ:
        return a * np.sin(3 * x + phase)+dc
    else:
        return a * np.sin(x + phase) + dc
def fit_FullySampled(x, fids):
    # check normalization
    if np.amax(fids) > 1.2 and np.ndim(fids) >1:
        for idx in range(fids.shape[0]):
            fids[idx,] = fids[idx,] / np.amax(fids[idx,])
    else:
        fids = fids / np.amax(fids)

    p_initial = np.array([0.7754, -0.7195, 0.2081, 0.8009, -0.002237])
    fids = np.squeeze(fids)
    if np.ndim(fids) > 1:
        params_full = []
        for fid in fids:
            fit_temp = curve_fit(fixedTQTPPI, x, fid, p0=p_initial)
            params_full.append(fit_temp)
            del fit_temp
        res = params_full
    else:
        print("Evaluating one FID only!")
        res = curve_fit(fixedTQTPPI, x, fids, p_initial)

    return res



def csToFile(data, type, alg, USF, std, snr, numPhaseCycles, basePath='/Users/duomii/Desktop/PhD/Scripte/CS/'):
    #TODO: - for simulation: write data correctly to file, what data to writw to file ?
    """
    Write cs data to file. Writes SQ, TQ, Ratio and RMSE (for IST-Algs only) to File for each USF.
    In case of IST Reco, also RMSE is written to File. Type is either MEasurement (BSA, Agar) or Simul
    :param data: Reco TQ, SQ and TQ/SQ ratio - depends on Reconstruction - if Simul then only Deviation and RMSE
    :param type: BSA or Simulation
    :param alg: IST_D or NUSF
    :param usf: [..] array like Undersampling factors
    :param gt: GT reconstructed TQ/SQ ratio - when Simulation this is a tuple consisting of the number of different ratios
                used for the simulation and the length of the FID
    :param basePath: Path to store
    :return:
    """

    savePath = basePath + '/' + type + f'_{csAlg}'

    if data.shape[-1] % 7 == 0:
        dataSplit = int(data.shape[-1] / 7)

    else:
        dataSplit = int(data.shape[-1] / 2)

    if 'imul' in type:



        # check if already exists
        if os.path.exists(savePath):

            print("Path exists, data is appended \n")
            # overwrite ? or add/append new data
            with open(savePath + '.txt', 'a') as f:
                #f.write('FID-length: {}\n'.format(gt[-1]))
                f.write('SNRs: {}\n'.format(snr))
                f.write('Phase Cycles: {} \n'.format(numPhaseCycles))
                f.write('Algorithm: ' + alg + '\n')
                for idx, el in enumerate(data):
                    f.write("USF: {}".format(USF[idx])+'\n')
                    for dev, rm in zip(el[:dataSplit], el[dataSplit:]):
                        f.write("{:<20} {:<10} {:<10}".format('', dev, rm))
                        f.write('\n')
        else:
            # For NUSF: data [fsf, usf, rmse]
            print("Create new folder \n")
            with open(savePath + '.txt', 'w') as f:
                #f.write('FID-length: {}\n'.format(gt[-1]))
                f.write('SNRs: {}\n'.format(snr))
                f.write('Phase Cycles: {} \n'.format(numPhaseCycles))
                f.write('Algorithm: ' + alg + '\n')
                f.write("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format('','TQ/SQ FS','TQ/SQ US', 'TQ/SQ Std.[%]','RMSE (FID)', 'SQ', 'SQstd', 'TQ', 'TQstd'))
                for idx, el in enumerate(data):

                    f.write("USF: {}".format(USF[idx])+'\n')
                    for fsf, usf, rm, dv, sq, sqstd, tq, tqstd  in zip(el[:dataSplit], el[dataSplit:2*dataSplit], el[2*dataSplit:3*dataSplit],std[idx], el[3*dataSplit:4*dataSplit], el[4*dataSplit:5*dataSplit], el[5*dataSplit:6*dataSplit], el[6*dataSplit:]):
                        f.write("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format('',fsf , usf, dv, rm, sq, sqstd, tq, tqstd))
                        f.write('\n')
    else:
        # Measurement
        if os.path.exists(savePath):

            print("Path exists, data is appended \n")
            # overwrite ? or add/append new data
            with open(savePath + '.txt', 'a') as f:
                f.write(alg + ':\n')
                for idx, el in enumerate(data):
                    f.write("USF: {}".format(USF[idx]) + '\n')
                    f.write("{:<20} {:<12} {:<12} {:<12}".format('', *el))
                    f.write('\n')
        else:
            # split measured data
            fsf, usf, rm, sq, sqstd, tq, tqstd = data[:dataSplit], data[dataSplit:2 * dataSplit], data[2 * dataSplit:3*dataSplit], \
                data[3 * dataSplit:4*dataSplit], data[4*dataSplit:5*dataSplit], data[5*dataSplit:6*dataSplit], data[6*dataSplit:]
            print("Create new folder \n")
            with open(savePath + '.txt', 'w') as f:
                f.write("Type: {} % \n".format(type))
                f.write(alg + ':\n')
                f.write("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format('', 'TQ/SQ FS', 'TQ/SQ US', 'TQ/SQ Std.[%]',
                                                                      'RMSE (FID)', 'SQ', 'SQstd', 'TQ', 'TQstd'))
                for idx in range(len(USF)):
                    f.write("USF: {}".format(USF[idx]) + '\n')

                    f.write("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format('', fsf[idx], usf[idx], std[idx], rm[idx], sq[idx], sqstd[idx], tq[idx], tqstd[idx]))
                    f.write('\n')

#tests.test_init()
simulation = False
preSampleData = True

####################### ------ Prepare Data -------------------------------------------------------------
if simulation:
    # Generate FIDs
    csGen = GenerateFIDfixed(PhasecycleAngle=60,NumPhaseCycles=16,StartPhase=90)
    params, TQs = csGen.generateParamsFixed_TQ()
    SNRs = csGen.varySNR(start=40, stop=200)
    FIDs, x , params, SNRs = csGen.generateFIDsFixed(params, SNRs) # shape FIDs: (numFIDS, numphaseSteps)
    numOfTQSQgenerated = params.shape[0] # num SNR = Total num / numOfTQSQgen
    targetAtqsq_woNoise = np.repeat(np.transpose(params[:, 2] / params[:, 0]) * 100, len(SNRs))
    # num FIDs = #parameterCombinations * #SNRvariations
else:
    studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/' # BSA
    #studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_Agar2_ismrm5phCycle_1_33_20231030_085218/' # Agar 2%
    #folderNum, isFixed, meas = 24, 1, 'Agar2'
    folderNum, isFixed, meas = 110, 1, 'BSA' # BSA tevo = 20
    # folderNum, isFixed, meas = 112, 1, 'Agar4'  Agar 4% topt= 12
    dataPath = studyPath + f'{folderNum}/'
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
    onlyReal = 1  # FFT of real components or of real and imag. 1 = sym, 0 = asym

    freqDriftVal = [0, 0, 0]

    (method, rawComplexData, complexDataAllPhase, complexDataUnw, realDataAllPhase,
     realDataAllPhaseUnw, mqFID, mqSpectra, mixTime, evoTime) = importTQTPPIspec(dataPath, spikeComp, phaseCorr,
                                                                                 preDCcomp, filter2ndDim,
                                                                                 filterFac2ndDim,
                                                                                 preFilter, filterFacPre, w0corr,
                                                                                 freqDriftVal, postDCcomp, filterFID,
                                                                                 filterFacPost, onlyReal)
    startPhase = method['PhaseList'][0]
    numPhaseSteps = method['NumPhaseSteps']
    phaseStep = 360. / numPhaseSteps
    x_degree = np.arange(90, phaseStep * len(mqFID) + 90, phaseStep)
    x_rad = x_degree * 2 * np.pi / 360.

    (SQpeaks, TQpeaks, ratio) = getTQSQ(mqSpectra)
    truthTQSQRatio = np.array(TQpeaks/SQpeaks *100)
    #print(f"Truth Ratio: {truthTQSQRatio} \%")



### -----------------------------------------------------------------------------------------------------------------

ALG_ALL = ['IST-D', 'IST-S', 'NUSF']
#ALG_ALL = ['NUSF']
#USF = [2, 4, 6, 8, 10, 12, 14, 16]
USF = [2, 4, 6, 8, 10, 12, 14, 16]

nusfParAll, TQSQ_full = [], []


if simulation:


    for csAlg in ALG_ALL:
        print(f"Alg. choosen: {csAlg} \n")
        recoRatios, pcov_reco, rmse, outAll, allMasked, TQSQ_std, \
            TQSQ_full, SQall, SQall_std, TQall, TQall_std = [], [], [], [], [], [], [], [], [], [], []
        std_full, popt_full = [], []
        ###################################################################################################
        ###### ---------------- Fit Full Data As Ground Truth ----- #######################################
        fsfFIDs, usfFIDs = [], []
        fullFID_fit = fit_FullySampled(x, FIDs)
        for fit in fullFID_fit:
            popt, pcov = fit

            fsfFIDs.append(fixedTQTPPI(x, *popt))
            TQSQ_full.append(np.abs(popt[2] / popt[0] * 100))
            std_full.append(np.sqrt(np.diag(pcov)))
            popt_full.append(popt)
            del popt, pcov
        numPhaseCycles = FIDs.shape[-1]
        print(f'Number of Phasecycles Simulated: {numPhaseCycles}')
        # Save results as reference for later
        if os.path.exists(f'CS_FullDataFitStd_{numPhaseCycles}.npy'):
            print('YASsssssssssssssssssssss')
            os.remove(f'CS_FullDataFitStd_{numPhaseCycles}.npy')
            os.remove(f'CS_fullFit_popt_{numPhaseCycles}.npy')
        np.save(f'CS_FullDataFitStd_{numPhaseCycles}.npy', std_full)
        np.save(f'CS_fullFit_popt_{numPhaseCycles}.npy', popt_full)
        del std_full, popt_full
        TQSQ_full_adapt = np.reshape(np.tile(TQSQ_full, len(USF)), (len(USF), -1))
        ###### ---------------------------------------------------- #########################################


        fidInput = FIDs


        numSNRs = int(FIDs.shape[0] / 17)  # since 17 TQ signal values

        ftInput = csGen.mqSpectra
        for f in USF:
            print(f'USF: {f}')
            ReCo = RecoCS(DataFull=fidInput, CS_Algorithm=csAlg, samplingMethod='Uniform', accelerationFactor=f)
            ReCo.x = x
            recoRatios_temp, tqsq_std_temp, TQs_temp, TQs_std_temp, SQs_temp, SQs_std_temp = [], [], [], [], [], []
            if csAlg == 'NUSF':
                # NUSF Reco returns set of parameters for given FIT. Shape is [1 X Nfit]

                a = ReCo.csReconstruction()
                plotSelection = np.random.choice(np.arange(len(FIDs)), 3)
                for idx, reco_el in enumerate(a):
                    popt, pcov = reco_el
                    usfFIDs.append(fixedTQTPPI(x, *popt))
                    TQs_temp.append(np.abs(popt[2]))
                    SQs_temp.append(np.abs(popt[0]))
                    TQs_std_temp.append(np.sqrt(pcov[2, 2]))
                    SQs_std_temp.append(np.sqrt(pcov[0,0]))
                    tqsq_temp = np.abs(popt[2] / popt[0] * 100)
                    recoRatios_temp.append(tqsq_temp)
                    pcov = np.sqrt(pcov)
                    tqsq_std_temp.append(tqsq_temp * np.sqrt((pcov[0,0]/popt[0])**2 + (pcov[2, 2]/popt[2])**2))

                    """if idx in plotSelection:
                        plt.figure()
                        plt.title(f"USF: {f} - Simulation")
                        plt.plot(np.real(fixedTQTPPI(x, *popt)), label="Reco")
                        plt.plot(fidInput[idx], label="GT")
                        plt.legend()
                        plt.show()"""
                    del popt, pcov
                #deviationFromGT_temp = np.abs(1-(recoRatios_temp / targetAtqsq_woNoise))  #percentage deviation
                recoRatios.append(recoRatios_temp)
                SQall.append(SQs_temp)
                TQall.append(TQs_temp)
                SQall_std.append(SQs_std_temp)
                TQall_std.append(TQs_std_temp)
                TQSQ_std.append(tqsq_std_temp)
                #deviationFromGT.append(deviationFromGT_temp)
                del recoRatios_temp, a, tqsq_std_temp, tqsq_temp
            else:
                out = ReCo.csReconstruction()
                outAll.append(out)
                # fit reconstructed spectrum as well using the normal
                fits_temp = fit_FullySampled(x, out)
                for fit in fits_temp:
                    popt, pcov = fit
                    #fsfFIDs.append(fixedTQTPPI(x, *popt))
                    TQs_temp.append(np.abs(popt[2]))
                    SQs_temp.append(np.abs(popt[0]))
                    TQs_std_temp.append(np.sqrt(pcov[2, 2]))
                    SQs_std_temp.append(np.sqrt(pcov[0, 0]))
                    tqsq_temp = np.abs(popt[2] / popt[0])
                    recoRatios_temp.append(tqsq_temp * 100)
                    # Std from fit, factor 10000 from
                    pcov = np.sqrt(pcov)
                    tqsq_std_temp.append(tqsq_temp * np.sqrt((pcov[0, 0] / popt[0]) ** 2 + (pcov[2, 2] / popt[2]) ** 2) * 100)

                    pcov_reco.append(pcov)
                    del popt, pcov
                #devTQSQ.append(np.abs(1 - (np.array(recoRatios_temp) / TQSQ_full)) * 100)
                #ft_out = fftshift(fft(out))
                # get ground truth value here for estimation of reco error
                #recoTQSQ = get_RecoTQSQ(np.real(ft_out)) * 100 # to make it a percentage
                recoRatios.append(recoRatios_temp)
                TQSQ_std.append(tqsq_std_temp)
                SQall.append(SQs_temp)
                TQall.append(TQs_temp)
                SQall_std.append(SQs_std_temp)
                TQall_std.append(TQs_std_temp)
                rmse.append(RMSE(FIDs, np.real(out)))

                del recoRatios_temp, ReCo

        if csAlg == 'IST-S':
            print(np.round(recoRatios[:100],5))

        if csAlg == 'NUSF':
            fsfFIDs, usfFIDs = np.array(fsfFIDs), np.array(usfFIDs)
#            devTQSQ = np.abs(1-(np.array(recoRatios) / np.array(TQSQ_full))) * 100 # percentage deviation
            rmse = RMSE(np.tile(fsfFIDs,[len(USF),1]), usfFIDs)

            csToFile(np.hstack((np.round(TQSQ_full_adapt, 5), np.round(recoRatios, 5), np.round(rmse, 5).reshape((8, -1)), np.round(SQall, 5), np.round(SQall_std, 5), np.round(TQall, 5), np.round(TQall_std, 5))),
                     f'Simulation_{numPhaseCycles}', csAlg, USF, std=np.round(TQSQ_std, 4), snr=numSNRs, numPhaseCycles=numPhaseCycles)
        else:

            csToFile(np.hstack((np.round(TQSQ_full_adapt, 5), np.round(recoRatios, 5), np.round(rmse, 5).reshape((8, -1)), np.round(SQall, 5), np.round(SQall_std, 5), np.round(TQall, 5), np.round(TQall_std, 5))),
                     f'Simulation_{numPhaseCycles}', csAlg, USF, std=np.round(TQSQ_std,5) , snr=numSNRs, numPhaseCycles=numPhaseCycles)
        del recoRatios, pcov_reco, rmse, outAll, allMasked, TQSQ_std, fsfFIDs, usfFIDs, TQSQ_full, TQs_temp, SQs_temp

else:

    # Measurement is used

    numSNRs = 0
    for csAlg in ALG_ALL:
        fidInput = np.squeeze(np.real(mqFID))
        fidInput = fidInput / np.amax(fidInput)
        fullFID_fit = fit_FullySampled(x_rad, fidInput)
        recoRatios, pcov_reco, rmse, outAll, allMasked, TQSQ_std, fsfFIDs, usfFIDs, TQSQ_full = [], [], [], [], [], [], [], [], []
        SQall, TQall, SQall_std, TQall_std, std_full, popt_full = [], [], [], [], [], []



        if fidInput.ndim > 1:
            for fit in fullFID_fit:
                # loop for FUlly Sampled data fit
                popt, pcov = fit
                fsfFIDs.append(fixedTQTPPI(x_rad, *popt))
                TQSQ_full.append(np.abs(popt[2] / popt[0] * 100))
                std_full.append(np.sqrt(np.diag(pcov)))
                popt_full.append(popt)
                del popt, pcov

        else:
            popt, pcov = fullFID_fit
            fsfFIDs = fixedTQTPPI(x_rad, *popt)
            TQSQ_full = np.tile(np.abs(popt[2] / popt[0] * 100), len(USF))
            popt_full.append(popt)
            std_full.append(np.sqrt(np.diag(pcov)))
            del popt, pcov
        numPhaseCycles = fidInput.shape[-1]


        if os.path.exists('CS_FullDataFitStd_meas.npy'):
            print('YASsssssssssssssssssssss')
            os.remove('CS_FullDataFitStd_meas.npy')
            os.remove('CS_fullFit_popt_meas.npy')
        np.save('CS_FullDataFitStd_meas.npy', std_full)
        np.save('CS_fullFit_popt_meas.npy', popt_full)
        del std_full, popt_full



        for f in USF:
            if preSampleData:
                tqFID = np.real(np.squeeze(complexDataAllPhase))
                fidInput = tqFID

            # generate US FID first by undersampling along phase dimension # complexDataAllPhase has shape (1, 1152, spectralDim)
            ReCo = RecoCS(DataFull=fidInput, CS_Algorithm=csAlg, samplingMethod='None', accelerationFactor=f, preSampled=preSampleData)
            ReCo.x = x_rad
            #allMasked.append(ReCo.CS_data)

            if csAlg == 'NUSF':
                # NUSF Reco returns set of parameters for given FIT. Shape is [1 X Nfit]
                popt,pcov = ReCo.csReconstruction()
                usf_reco = fixedTQTPPI(x_rad, *popt)
                usfFIDs.append(usf_reco)

                SQall.append(np.abs(popt[0]))
                TQall.append(np.abs(popt[2]))
                SQall_std.append(np.sqrt(pcov[0,0]))
                TQall_std.append(np.sqrt(pcov[2,2]))
                ratio = np.abs(popt[2] / popt[0] * 100)
                pcov = np.sqrt(pcov)
                TQSQ_std.append(ratio * np.sqrt((pcov[0, 0] / popt[0]) ** 2 + (pcov[2, 2] / popt[2]) ** 2))
                recoRatios.append(ratio)
                rmse.append(np.float(1.0)) # no rmse for USFit

                #usfData.append(np.abs([ sq, tq, np.abs(1-ratio/truthTQSQRatio)*100, RMSE(fidInput, fixedTQTPPI(ReCo.x, a))]))

            else:
                out = ReCo.csReconstruction()

                popt, pcov = fit_FullySampled(x_rad, out)

                # fsfFIDs.append(fixedTQTPPI(x, *popt))
                SQall.append(np.abs(popt[0]))
                TQall.append(np.abs(popt[2]))
                SQall_std.append(np.sqrt(pcov[0, 0]))
                TQall_std.append(np.sqrt(pcov[2, 2]))
                tqsq = np.abs(popt[2] / popt[0] * 100)
                recoRatios.append(tqsq)
                # Std from fit
                pcov = np.sqrt(pcov)
                TQSQ_std.append(tqsq * np.sqrt((pcov[0, 0] / popt[0]) ** 2 + (pcov[2, 2] / popt[2]) ** 2))
                pcov_reco.append(pcov)

                outAll.append(out)
                #ft_out = fftshift(fft(out))
                #ratios = get_RecoTQSQ(np.real(ft_out)) * 100
                #rmse_temp = RMSE(fsfFIDs, np.real(out))
                #recoRatios.append(ratios)
                rmse.append(RMSE(np.real(mqFID), np.real(out)))

                #usfData.append(['-','-',np.round(np.abs(ratios),4), np.round(rmse,5)])
                if f == 6:
                    fig, ax = plt.subplots(2,1)
                    ax[0].plot(np.real(out), linewidth=0.6, label=f"USF={f}")
                    ax[0].plot(fixedTQTPPI(x_rad, *popt), linewidth=0.6, label="USF Fit")
                    ax[0].plot(fsfFIDs, linewidth=0.6, label="FSF")
                    ax[0].legend(loc="best")
                    ax[1].plot(singleQCsignal(x_rad, popt[0],popt[1],popt[-1]), linewidth=0.6,label="SQ")
                    ax[1].plot(singleQCsignal(x_rad, popt[2],popt[3],popt[-1], TQ=True), linewidth=0.6, label="TQ")
                    ax[1].legend(loc="best")
                    plt.title("Example")
                    plt.show()
                del popt, pcov, tqsq, out

        if csAlg == 'NUSF':
            #rmse.append(RMSE(np.array(fsfFIDs), np.array(usfFIDs)))
            #devTQSQ = np.abs(1- np.array(recoRatios) / np.array(TQSQ_full))* 100 # percent
            csToFile(np.hstack((np.round(TQSQ_full, 5), np.round(recoRatios, 5), np.round(rmse, 5).reshape((len(USF), )), np.round(SQall, 5), np.round(SQall_std, 5), np.round(TQall, 5), np.round(TQall_std, 5))),
                     meas, csAlg, USF, std=np.round(TQSQ_std, 4), snr=numSNRs, numPhaseCycles=numPhaseCycles)
        else:
            # when measurement is used only one FID

            #rmse = RMSE(np.array(fsfFIDs), np.array(usfFIDs))
            #devTQSQ = np.abs(1 - np.array(recoRatios) / np.array(TQSQ_full)) * 100  # percent
            print(f'AT LAST: {rmse}')
            print(f'Shape RMSE: {np.array(rmse).shape}')
            csToFile(np.hstack((np.round(TQSQ_full, 5), np.round(recoRatios, 5), np.round(rmse, 5), np.round(SQall, 5), np.round(SQall_std, 5), np.round(TQall, 5), np.round(TQall_std, 5))),
                     meas, csAlg, USF, std=np.round(TQSQ_std, 5), snr=numSNRs, numPhaseCycles=numPhaseCycles)

        del recoRatios, pcov_reco, rmse, outAll, allMasked, TQSQ_std, fsfFIDs, usfFIDs, TQSQ_full
        #Reco_4 = RecoCS(DataFull=fidInput,CS_Algorithm='IST-S',samplingMethod='PoissonGap',accelerationFactor=8)



"""
rmse = RMSE(np.real(ft_out), np.real(ftInput))
pint = trapzoid(np.real(ft_out), np.real(ftInput))"""

"""if csAlg == "NUSF":
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    fig.suptitle(f"{csAlg}")

    ax1.plot(fidInput, label="Input")
    ax1.legend(loc="best")
    ax1.plot(fixedTQTPPI(x_rad, nusfParAll[0]), linestyle='dashed', alpha=0.7)
    ax2.plot(fidInput, label="Input")
    ax2.legend(loc="best")
    ax2.plot(fixedTQTPPI(x_rad, nusfParAll[1]),linestyle='dashed', alpha=0.7)
    ax3.plot(fidInput)
    ax3.plot(fixedTQTPPI(x_rad, nusfParAll[2]), linestyle='dashed', alpha=0.7)
    ax4.plot(fidInput)
    ax4.plot(fixedTQTPPI(x_rad, nusfParAll[3]), linestyle='dashed', alpha=0.7)
    ax5.plot(fidInput)
    ax5.plot(fixedTQTPPI(x_rad, nusfParAll[4]), linestyle='dashed', alpha=0.7)
    plt.show()
"""

"""plt.plot(rmse)
plt.xticks(np.arange(stop=FIDs.shape[0],step=len(SNRs)),np.round(initialTQSQ*100,2))
plt.show()"""

"""fig, (ax1, ax2, ax3) = plt.subplots(3,1)

fig.suptitle(f"{csAlg}")
ax1.plot(np.round(recoRatios*100,2), linewidth=0.5, label=r"Reconstructed")
ax1.plot(csGen.TQSQgen *100, linewidth=0.5, label="Target")
ax1.plot(targetAtqsq_woNoise, linewidth=0.5, label="Target w/o noise")
ax1.set_ylabel("$A_{TQ}/A_{SQ}$ [\%]")
ax1.legend(loc="best")
ax2.plot(rmse, linewidth=0.5)
ax2.set_ylabel("RMSE")
ax3.plot(pint[0].transpose(), label="Reconstruction", linewidth=0.5)
ax3.plot(pint[1].transpose(), label="Target", linewidth=0.5)
ax3.set_ylabel("Intergrated Spectrum")
ax3.legend(loc="best")
plt.show()"""

"""

print(f"RSME: {RMSE(np.real(ft_out), np.real(fftshift(fft(fidInput))))} US4:  {RMSE(np.real(ft_out4), np.real(fftshift(fft(fidInput))))}\n "
      f"Peak: {trapzoid(np.real(ft_out), np.real(ftInput))} US4:  {trapzoid(np.real(ft_out4), np.real(ftInput))}\n"
      f"Peak sim: {simpson(np.real(ft_out), np.real(ftInput))}")

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(np.real(ftInput), linewidth=0.5, label="Original spectrum")
ax1.plot(np.real(ft_CSdata), linewidth=0.5, label="US 8 spectrum")
ax1.plot(np.real(ft_CSdata4), linewidth=0.5, label="US 4 spectrum")
ax1.legend(loc="best")
ax2.plot(np.real(ft_out), linewidth=0.5, label="Reconstructed")
ax2.legend(loc="best")
plt.show()"""


