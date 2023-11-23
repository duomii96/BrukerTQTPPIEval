import matplotlib.pyplot as plt
import numpy as np
import os
from AllPlots import plot_ratioVSNR, data_ratioSNR, data_for_plot, plot_usf
### USF Default : USF = [2, 4, 6, 8, 10, 12, 14, 16]
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams["errorbar.capsize"] =1.5
"""plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": "Palatino",
})"""
plt.rcParams.update({
"font.weight": "bold",
"font.size": 12,
})
phCycles = 1152 # 1152, 576, 288, 192, 128, 96
AllSNR = np.arange(40,200,10)
# Simulated TQ Amplitudes (has to be adjiusted for TQ/SQ):
TQs =  [0.005, 0.01 , 0.02 , 0.03 , 0.04 , 0.05 , 0.06 , 0.07 , 0.08 , 0.09 , 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225, 0.25 ]
## BSA
TQ_idx_selected = 10
#TQ_idx_selected = 4 # roughly 3 -4 %, similar to BSA

snrIndex = 2 # range(40,200)
type='meas'

basePathTofiles = '/Users/duomii/Desktop/PhD/Scripte/CS/'


if 'imul' in type:
    path_final_istd = basePathTofiles + f'Simulation_{phCycles}_IST-D.txt'
    path_final_ists = basePathTofiles + f'Simulation_{phCycles}_IST-S.txt'
    path_final_nusf = basePathTofiles + f'Simulation_{phCycles}_NUSF.txt'

    gt_std = np.load(f'CS_FullDataFitStd_{phCycles}.npy')
    gt_popt = np.load(f'CS_fullFit_popt_{phCycles}.npy')
else:
    path_final_istd = basePathTofiles + f'BSA_IST-D.txt'
    path_final_ists = basePathTofiles + f'BSA_IST-S.txt'
    path_final_nusf = basePathTofiles + f'BSA_NUSF.txt'
    gt_std = np.load(f'CS_FullDataFitStd_meas.npy')
    gt_popt = np.load(f'CS_fullFit_popt_meas.npy')

USF = [2, 4, 6, 8, 10, 12, 14]
usf_idx = 4 # max 7
#fullFit_popt = np.load('CS_fullFit_popt.npy')
#std_full_fit = np.load('CS_FullDataFitStd.npy') # 102 (6 SNR x 17 TQs) lists a 5 stds

"""print("Add specific File name: \n")
suffix = input()
path_final = basePathTofiles + str(suffix)"""

def readIn_CS(savePath, type):
    """
    type: Either measurement or simulation ('meas', 'simul')
    :return:
    """
    # Check if the file exists
    if os.path.exists(savePath):

        if 'imul' in type:
            numSNRs = 11
            print("Loading simulation data...")
            # Open the file in read mode ('r')
            with open(savePath, 'r') as f:
                # Read and process the file line by line
                data_raw = f.readlines()

            # Process the lines as needed
            Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ, SQstd, TQ, TQstd = [], [], [], [], [], [], [], []

            for item in data_raw:
                dataList = item.split()

                if 'SNRs:' in dataList:
                    numSNRs = dataList[-1]
                elif len(dataList) != 8:
                    continue
                else:
                    Full_TQSQ.append(np.double(dataList[0]))
                    TQSQ_us.append(np.double(dataList[1]))
                    TQSQ_std.append(np.double(dataList[2]))
                    rmse.append(np.double(dataList[3]))
                    SQ.append(np.double(dataList[4]))
                    SQstd.append(np.double(dataList[5]))
                    TQ.append(np.double(dataList[6]))
                    TQstd.append(np.double(dataList[7]))

            data = (Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ, SQstd, TQ, TQstd)
        else:
            numSNRs = 0
            # Measurement - either BSA or Agarose
            print("Loading measurement data...")
            with open(savePath, 'r') as f:
                # Read and process the file line by line
                data_raw = f.readlines()
                Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ, SQstd, TQ, TQstd = [], [], [], [], [], [], [], []
            for item in data_raw:
                dataList = item.split()

                if 'Type:' in dataList or len(dataList) !=8:
                    continue
                else:
                    Full_TQSQ.append(np.double(dataList[0]))
                    TQSQ_us.append(np.double(dataList[1]))
                    TQSQ_std.append(np.double(dataList[2]))
                    rmse.append(np.double(dataList[3]))
                    SQ.append(np.double(dataList[4]))
                    SQstd.append(np.double(dataList[5]))
                    TQ.append(np.double(dataList[6]))
                    TQstd.append(np.double(dataList[7]))
            data = (Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ, SQstd, TQ, TQstd)



    else:
        numSNRs =-1
        print(f"File {savePath} does not exist.")
    return data, int(numSNRs)




def data_tq_TqSq(data, usf_idx, type):
    # TQ/SQ vs TQ amplitude while SNR stays the same
    dataIndexRange = np.arange(usf_idx * numSNR * 17+snrIndex,usf_idx * numSNR * 17 + (numSNR * 17)+snrIndex,numSNR) # only for a Signle USF
    Full_TQSQ, TQSQ_us, TQSQ_std, rmse, _,_,_,_ = data
    print(f'Selected US factor: {USF[usf_idx]}')
    tqsq_us_sel = np.array(TQSQ_us)[dataIndexRange]
    std_us_sel = np.array(TQSQ_std)[dataIndexRange]
    Full_TQSQ_sel = np.array(Full_TQSQ)[dataIndexRange]



    return tqsq_us_sel, std_us_sel, Full_TQSQ_sel


def plot_ratioVstq(data1, data2, data3, std1, std2, std3, fully_sampled, numSNR):
    """std1 = np.where(std1 > np.amin(std1) * 2, np.amin(std1) * 2, std1)
    std2 = np.where(std2 > np.amin(std2) * 2, np.amin(std2) * 2, std2)
    std3 = np.where(std3 > np.amin(std3) * 2, np.amin(std3) * 2, std3)"""

    popt_sel, stdFullsel = gt_popt[2::numSNR], gt_std[2::numSNR]

    std_TQ_full_sel = np.array([el[2] for el in stdFullsel])
    std_SQ_full_sel = np.array([el[0] for el in stdFullsel])
    TQ_full_sel, SQ_full_sel = np.array([el[2] for el in popt_sel]), np.array([el[0] for el in popt_sel])
    std_TQSQ_full_sel = fully_sampled * np.sqrt((std_SQ_full_sel / SQ_full_sel) ** 2 + (std_TQ_full_sel / TQ_full_sel) ** 2)


    plt.figure(figsize=(5,5))
    #plt.ylim()
    plt.errorbar(TQs, data1, yerr=std1, label="IST-D")#, marker="v")
    plt.errorbar(TQs, data2, yerr=std2, label="IST-S")#, marker="v")
    plt.errorbar(TQs, data3, yerr=std3, label="US fit")#, marker="v")
    plt.errorbar(TQs, fully_sampled, yerr=std_TQSQ_full_sel, label="FS fit")
    plt.title(f'Phasecycles: {phCycles}, '+ f'USF: {USF[usf_idx]}')
    plt.xlabel("TQ")
    plt.ylabel(r"$A_{TQ}/A_{SQ}$ [\%]")
    plt.legend(loc="best")
    #plt.figaspect(1.)
    plt.tight_layout()
    plt.show()


data_ex_istd, numSNR = readIn_CS(path_final_istd, type=type)
data_ex_ists, numSNR = readIn_CS(path_final_ists, type=type)
data_ex_nusf, numSNR = readIn_CS(path_final_nusf, type=type)



istd_sel, std_istd_sel, full_istd, istd_sq_sel, istd_sqstd_sel, istd_tq_sel, istd_tqstd_sel =  data_for_plot(data_ex_istd, numSNR, tq_idx=TQ_idx_selected, snrIndex=snrIndex,type=type)
ists_sel, std_ists_sel, full_ists, ists_sq_sel, ists_sqstd_sel, ists_tq_sel, ists_tqstd_sel =  data_for_plot(data_ex_ists, numSNR, tq_idx=TQ_idx_selected, snrIndex=snrIndex,type=type)
nusf_sel, std_nusf_sel, full_nusf, nusf_sq_sel, nusf_sqstd_sel, nusf_tq_sel, nusf_tqstd_sel =  data_for_plot(data_ex_nusf, numSNR, tq_idx=TQ_idx_selected, snrIndex=snrIndex,type=type)

# ATQSQ plot #################
plot_usf(istd_sel, ists_sel, nusf_sel, std_istd_sel, std_ists_sel, std_nusf_sel, full_istd,gt_popt, gt_std, TQ_idx_selected, snrIndex,AllSNR,type,'ratio')
# SQ Plot #################
plot_usf(istd_sq_sel, ists_sq_sel, nusf_sq_sel, istd_sqstd_sel, ists_sqstd_sel, nusf_sqstd_sel, full_istd, gt_popt, gt_std, TQ_idx_selected, snrIndex, AllSNR, type,'SQ')
plot_usf(istd_tq_sel, ists_tq_sel, nusf_tq_sel, istd_tqstd_sel, ists_tqstd_sel, nusf_tqstd_sel, full_istd, gt_popt, gt_std, TQ_idx_selected, snrIndex, AllSNR, type,'TQ')

"""tqsq_istd, std_istd, fullIstd = data_tq_TqSq(data_ex_istd, usf_idx, type=type)
tqsq_ists, std_ists, fullIsts = data_tq_TqSq(data_ex_ists, usf_idx, type=type)
tqsq_nusf, std_nusf, fullnusf = data_tq_TqSq(data_ex_nusf, usf_idx, type=type)"""

############# RATIO VS TQ PLOT --------------------- ##########################################
#plot_ratioVstq(tqsq_istd, tqsq_ists, tqsq_nusf, std_istd, std_ists, std_nusf, fullIstd, numSNR)

############ ---------------------------SNR -------------------------------------------------------------
"""
nusf_sel, std_nusf_sel, full_nusf, nusf_sq_sel, nusf_sqstd_sel, nusf_tq_sel, nusf_tqstd_sel = data_ratioSNR(data_ex_nusf, usf_idx, AllSNR, TQ_idx_selected)

ists_sel, std_ists_sel, full_ists, ists_sq_sel, ists_sqstd_sel, ists_tq_sel, ists_tqstd_sel = data_ratioSNR(data_ex_ists, usf_idx, AllSNR, TQ_idx_selected)
istd_sel, std_istd_sel, full_istd, istd_sq_sel, istd_sqstd_sel, istd_tq_sel, istd_tqstd_sel = data_ratioSNR(data_ex_istd, usf_idx, AllSNR, TQ_idx_selected)

plot_ratioVSNR(istd_sel, ists_sel, nusf_sel, std_istd_sel, std_ists_sel, std_nusf_sel, full_istd, AllSNR, TQ_idx_selected, gt_popt, gt_std)
plot_ratioVSNR(istd_sq_sel, ists_sq_sel, nusf_sq_sel, istd_sqstd_sel, ists_sqstd_sel, nusf_sqstd_sel, full_istd, AllSNR, TQ_idx_selected, gt_popt, gt_std, sig='SQ')
plot_ratioVSNR(istd_tq_sel, ists_tq_sel, nusf_tq_sel, istd_tqstd_sel, ists_tqstd_sel, nusf_tqstd_sel, full_istd, AllSNR, TQ_idx_selected, gt_popt, gt_std, sig='TQ')"""