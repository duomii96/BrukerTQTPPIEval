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
import os

def fixedTQTPPI(x, params):
    return params[0] * np.sin(x + params[1]) + params[2] * np.sin(3 * x + params[3]) + params[4]

def csToFile(data, type, alg, usf, gt, basePath='/Users/duomii/Desktop/PhD/Scripte/CS/'):
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

    if 'imul' in type:

        dataSplit = int(data.shape[-1] / 2)

        # check if already exists
        if os.path.exists(savePath):

            print("Path exists, data is appended \n")
            # overwrite ? or add/append new data
            with open(savePath + '.txt', 'a') as f:
                f.write('Simulated TQ/SQ Ratios: {}; FID-length: {}\n'.format(*gt))
                f.write(alg + ':\n')
                for idx, el in enumerate(data):
                    f.write("USF: {}".format(usf[idx])+'\n')
                    for dev, rm in zip(el[:dataSplit], el[dataSplit:]):
                        f.write("{:<20} {:<10} {:<10}".format('', dev, rm))
                        f.write('\n')
        else:
            print("Create new folder \n")
            with open(savePath + '.txt', 'w') as f:
                f.write('FID-length: {}\n'.format(gt[-1]))
                f.write(alg + ':\n')
                f.write("{:<20} {:<12} {:<12} {:<12}\n".format('','GT','TQ/SQ Dev. [%]','RMSE (FID)'))
                for idx, el in enumerate(data):

                    f.write("USF: {}".format(usf[idx])+'\n')
                    for dev, rm, tr in zip(el[:dataSplit], el[dataSplit:], np.round(gt[0],4)):
                        f.write("{:<20} {:<12} {:<15} {:<15}".format('',tr , dev, rm))
                        f.write('\n')
    else:
        # Measurement
        if os.path.exists(savePath):

            print("Path exists, data is appended \n")
            # overwrite ? or add/append new data
            with open(savePath + '.txt', 'a') as f:
                f.write("Truth Ratio: {} % \n".format(gt))
                f.write(alg + ':\n')
                for idx, el in enumerate(data):
                    f.write("USF: {}".format(usf[idx]) + '\n')
                    f.write("{:<20} {:<12} {:<12} {:<12}".format('', *el))
                    f.write('\n')
        else:
            print("Create new folder \n")
            with open(savePath + '.txt', 'w') as f:
                f.write("Truth Ratio: {} % \n".format(gt))
                f.write(alg + ':\n')
                f.write("{:<20} {:<12} {:<12} {:<12} {:<12}\n".format('', 'SQ', 'TQ', 'TQ/SQ', 'RMSE'))
                for idx, el in enumerate(data):
                    f.write("USF: {}".format(usf[idx]) + '\n')
                    f.write("{:<20} {:<12} {:<12} {:<12} {:<12}".format('', *el))
                    f.write('\n')

#tests.test_init()
simulation = True


####################### ------ Prepare Data -------------------------------------------------------------
if simulation:
    # Generate FIDs
    csGen = GenerateFIDfixed(PhasecycleAngle=5,NumPhaseCycles=16,StartPhase=45)
    params, TQs = csGen.generateParamsFixed_TQ()
    SNRs = csGen.varySNR(start=40, stop=70)
    FIDs, x , params, SNRs = csGen.generateFIDsFixed(params, SNRs) # shape FIDs: (numFIDS, numphaseSteps)
    numOfTQSQgenerated = params.shape[0] # num SNR = Total num / numOfTQSQgen
    targetAtqsq_woNoise = np.repeat(np.transpose(params[:, 2] / params[:, 0]) * 100, len(SNRs))
    # num FIDs = #parameterCombinations * #SNRvariations
else:
    studyPath = '/Users/duomii/Desktop/PhD/Data/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/' # BSA
    # startFolder =   58; stopFolder = 61; IsFixed=1; #TQTPPI wo180 fix deltaAlpha = 5°, tevo = 5ms
    # startFolder, stopFolder = 135, 137 #129, 152
    folderNum, isFixed = 58, 1
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

print(f"Simulation: {simulation} \n")
print("Which Algorithm ? Either NUSF or IST-D: \n")

csAlg = input()
print(f"Alg. choosen: {csAlg} \n")
if csAlg == 'NUSF':
    nusfParAll = []

USF = [2, 4, 6, 12, 16]
usfData = []
recoRatios, recoRMSE, outAll, allMasked = [], [], [], []
if simulation:
    # for each
    deviationFromGT = []
    fidInput = FIDs
    ftInput = csGen.mqSpectra
    for f in USF:
        ReCo = RecoCS(DataFull=fidInput, CS_Algorithm=csAlg, samplingMethod='PoissonGap', accelerationFactor=f)
        if csAlg == 'NUSF':
            # NUSF Reco returns set of parameters for given FIT. Shape is [1 X Nfit]

            a, _ = ReCo.csReconstruction()

            nusfParAll.append(a)

            sq = a[0] * 100
            tq = a[2] * 100
            recoRatios.append(np.abs(a[2] / a[0] * 100))
            usfData.append(np.abs([sq, tq, a[2] / a[0] * 100]))

        else:
            out = ReCo.csReconstruction()
            outAll.append(out)
            ft_out = fftshift(fft(out))
            # get ground truth value here for estimation of reco error
            recoTQSQ = get_RecoTQSQ(np.abs(ft_out)) * 100 # to make it a percentage
            recoRatios.append(recoTQSQ)

            recoRMSE.append(RMSE(fidInput, np.real(out)))
            deviationFromGT.append(np.abs((recoTQSQ / targetAtqsq_woNoise)) *100)
            del recoTQSQ, ReCo

    csToFile(np.hstack((np.round(deviationFromGT,3), np.round(recoRMSE,3))), 'Simulation', csAlg, USF, gt=[targetAtqsq_woNoise, FIDs.shape[-1]])

else:
    # Measurement is used

    fidInput = np.squeeze(np.real(mqFID))


    for f in USF:

        ReCo = RecoCS(DataFull=fidInput, CS_Algorithm=csAlg, samplingMethod='PoissonGap', accelerationFactor=f)
        ReCo.x = x_rad

        allMasked.append(ReCo.CS_data)

        if csAlg == 'NUSF':
            # NUSF Reco returns set of parameters for given FIT. Shape is [1 X Nfit]

            a, _ = ReCo.csReconstruction()

            nusfParAll.append(a)

            sq = a[0]* 100
            tq  = a[2]* 100
            recoRatios.append(np.abs(a[2] / a[0] * 100))
            usfData.append(np.abs([sq, tq, a[2] / a[0] * 100]))

        else:
            out = ReCo.csReconstruction()
            outAll.append(out)
            ft_out = fftshift(fft(out))
            ratios = get_RecoTQSQ(np.real(ft_out)) * 100
            rmse = RMSE(fidInput, np.real(out))
            recoRatios.append(ratios)
            recoRMSE.append(RMSE(fidInput, np.real(out)))

            usfData.append(['-','-',np.round(np.abs(ratios),4), np.round(rmse,5)])

            del ratios, ft_out
    csToFile(usfData, type='BSA_{}'.format(folderNum), alg=csAlg, usf=USF, gt=truthTQSQRatio)
        #Reco_4 = RecoCS(DataFull=fidInput,CS_Algorithm='IST-S',samplingMethod='PoissonGap',accelerationFactor=8)


print(f"Reconstructed Ratios: {recoRatios}")
#out_4 = Reco_4.csReconstruction()



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


