import matplotlib.pyplot as plt
import numpy as np
import os
### USF Default : USF = [2, 4, 6, 8, 10, 12, 14, 16]
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams["errorbar.capsize"] =2
"""plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.serif": "Palatino",
})"""
plt.rcParams.update({
"font.weight": "bold",
"font.size": 16,
})

def data_for_plot(data, numSNRs, tq_idx, snrIndex, type):
    """
    Function to select the given data points from loaded data (i.e. retrieve the data points for diffeent USF's
    SNR 70..
    :param data: tuple .. see function above
    :return:
    """
    if 'imul' in type:
        USF = [2, 4, 6, 8, 10, 12, 14, 16]
        Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ, SQstd, TQ, TQstd = data
        spacing = numSNRs * 17 # 17 different TQ signals, spacing wihtin one USF
        numUSF = int(len(Full_TQSQ)/spacing)

        single_list_idx = int(numSNRs * tq_idx) -1
        final_idx = single_list_idx + snrIndex
        print(f'Index: {single_list_idx}')
        print(f'Final index: {final_idx}')
        Full_sel = np.array(Full_TQSQ)[final_idx::spacing]
        print(f'Fully Sampled TQSQ selected: {Full_sel[0]} \n')
        sq_sel = np.array(SQ)[final_idx::spacing]
        sq_std_sel = np.array(SQstd)[final_idx::spacing]
        tq_sel = np.array(TQ)[final_idx::spacing]
        tq_std_sel = np.array(TQstd)[final_idx::spacing]
        tqsq_us_sel = np.array(TQSQ_us)[final_idx::spacing]
        std_us_sel = np.array(TQSQ_std)[final_idx::spacing]
        rmse_sel = np.array(rmse)[final_idx::spacing]
    else:
        # no need to select
        Full_sel, tqsq_us_sel, std_us_sel, rmse, sq_sel, sq_std_sel, tq_sel, tq_std_sel = data

    return tqsq_us_sel, std_us_sel, Full_sel, sq_sel, sq_std_sel, tq_sel, tq_std_sel
def data_ratioSNR(data, USF_idx, SNRs, TQ_idx ):
    Full_TQSQ, TQSQ_us, TQSQ_std, rmse, SQ_us, SQstd_us, TQ_us, TQstd_us = data
    initIdx = USF_idx * len(SNRs) * 17 + TQ_idx
    endIdx = initIdx + len(SNRs)


    sq_us_sel = np.array(SQ_us)[initIdx:endIdx]
    tq_us_sel = np.array(TQ_us)[initIdx:endIdx]
    sqstd_us_sel = np.array(SQstd_us)[initIdx:endIdx]
    tqstd_us_sel = np.array(TQstd_us)[initIdx:endIdx]

    tqsq_us_sel = np.array(TQSQ_us)[initIdx:endIdx]
    std_us_sel = np.array(TQSQ_std)[initIdx:endIdx]
    Full_TQSQ_sel = np.array(Full_TQSQ)[initIdx:endIdx]

    return tqsq_us_sel, std_us_sel, Full_TQSQ_sel, sq_us_sel, sqstd_us_sel, tq_us_sel, tqstd_us_sel


def plot_ratioVSNR(data1, data2, data3, std1, std2, std3, fully_sampled, SNRs, TQ_idx, gt_popt, gt_std, sig='ratio'):
    """std1 = np.where(std1 > np.amin(std1) * 2, np.amin(std1) * 2, std1)
    std2 = np.where(std2 > np.amin(std2) * 2, np.amin(std2) * 2, std2)
    std3 = np.where(std3 > np.amin(std3) * 2, np.amin(std3) * 2, std3)"""


    # Select popt from gt data fot given TQ. Shape (len(SNR), numPopt))
    popt_sel, stdFullsel = gt_popt[TQ_idx *len(SNRs):TQ_idx *len(SNRs)+len(SNRs)], gt_std[TQ_idx *len(SNRs):TQ_idx *len(SNRs)+len(SNRs)]

    std_TQ_full_sel = np.array([el[2] for el in stdFullsel])
    std_SQ_full_sel = np.array([el[0] for el in stdFullsel])
    TQ_full_sel, SQ_full_sel = np.array([el[2] for el in popt_sel]), np.array([el[0] for el in popt_sel])
    std_TQSQ_full_sel = fully_sampled * np.sqrt((std_SQ_full_sel / SQ_full_sel) ** 2 + (std_TQ_full_sel / TQ_full_sel) ** 2)


    plt.figure(figsize=(5,5))
    #plt.ylim()
    plt.ylim(np.amin((data1, data2, data3)) * 0.5, np.amax((data1, data2, data3)) * 1.5)
    plt.errorbar(SNRs, data1, yerr=std1, label="IST-D")#, marker="v")
    plt.errorbar(SNRs, data2, yerr=std2, label="IST-S")#, marker="v")
    plt.errorbar(SNRs, data3, yerr=std3, label="US fit")#, marker="v")


    if sig == 'ratio':

        plt.errorbar(SNRs, fully_sampled, yerr=std_TQSQ_full_sel, label="FS fit")

        #plt.title(r"$A_{TQ}/A_{SQ}$ = " + f' {np.round(fully_sampled[0], 3)} \%, ' + f'Phasecycles: {phCycles}')
        plt.xlabel('USF')
        plt.ylabel(r"$A_{TQ}/A_{SQ} [\%]$")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    elif sig == 'SQ':
        #sq_full_sel, sq_full_std = popt_sel[TQ_idx_selected][0], stdFullsel[TQ_idx_selected][0]
        plt.errorbar(SNRs, SQ_full_sel, yerr=std_SQ_full_sel, label="FS fit")
        #plt.title(f"SNR = {AllSNR[snrIndex]}, " + f'Phasecycles: {phCycles}')
        plt.xlabel('USF')
        plt.ylabel(r"$A_{SQ} [a.u.]$ ")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        #tq_full_sel, tq_full_std = popt_sel[TQ_idx_selected][2], stdFullsel[TQ_idx_selected][2]
        plt.errorbar(SNRs, TQ_full_sel, yerr=std_TQ_full_sel, label="FS fit")
        #plt.title(f"SNR = {AllSNR[snrIndex]}, " + f'Phasecycles: {phCycles}')
        plt.xlabel("SNR")
        plt.ylabel(r"$A_{TQ} [a.u.]$")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


def plot_usf(data1, data2, data3, std1, std2, std3, fully_sampled, gt_popt, gt_std, tqIdx,snridx,allSNR,type,sig):

    """std1 = np.where(std1 > np.amin(std1)*2, np.amin(std1)*2, std1)
    std2 = np.where(std2 > np.amin(std2)*2, np.amin(std2)*2, std2)
    std3 = np.where(std3 > np.amin(std3)*2, np.amin(std3)*2, std3)"""
    #USF = [2, 4, 6, 8, 10, 12, 14, 16]
    USF = [2, 4, 6, 8, 10, 12, 14]
    if 'imul' in type:
        spacingTQ = len(allSNR)
        startIdx = snridx
        popt_sel, stdFullsel = gt_popt[startIdx::spacingTQ], gt_std[startIdx::spacingTQ]
        sq_full_sel, sq_full_std = popt_sel[tqIdx][0], stdFullsel[tqIdx][0]
        tq_full_sel, tq_full_std = popt_sel[tqIdx][2], stdFullsel[tqIdx][2]
    else:
        # Measurement
        TQ_idx_selected = 0
        popt_sel, stdFullsel = gt_popt, gt_std
        tq_full_sel, tq_full_std = popt_sel[0][2], stdFullsel[0][2]
        sq_full_sel, sq_full_std = popt_sel[0][0], stdFullsel[0][0]

    plt.figure(figsize=(5, 5), dpi=120)
    plt.ylim(np.amin(data1[:-1]) * 0.6, np.amax(data1[:-1]) * 1.4)
    plt.errorbar(USF, data1[:-1], yerr=std1[:-1], label="IST-D")
    plt.errorbar(USF, data2[:-1], yerr=std2[:-1], label="IST-S", )
    plt.errorbar(USF, data3[:-1], yerr=std3[:-1], label="US fit")

    if sig == 'ratio':

        plt.errorbar(USF, fully_sampled[:-1], yerr=std1[:-1], label="FS fit")
        #plt.title(r"$A_{TQ}/A_{SQ}$ = " + f' {np.round(fully_sampled[0], 3)} \%, ' + f'Phasecycles: {phCycles}')
        plt.xlabel(r'$USF$')
        plt.ylabel(r"$A_{TQ}/A_{SQ} [\%]$")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    elif sig == 'SQ':

        plt.errorbar(USF, [np.abs(sq_full_sel)] * len(USF), yerr=[np.abs(sq_full_std)] * len(USF),
                     label="FS fit")
        #plt.title(f"SNR = {AllSNR[snrIndex]}, " + f'Phasecycles: {phCycles}')
        plt.xlabel(r'$USF$')
        plt.ylabel(r"$A_{SQ} [a.u.]$ ")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:

        plt.errorbar(USF, [np.abs(tq_full_sel)] * len(USF), yerr=[np.abs(tq_full_std)] * len(USF), label="FS fit")
        #plt.title(f"SNR = {AllSNR[snrIndex]}, " + f'Phasecycles: {phCycles}')
        plt.xlabel(r"$USF$")
        plt.ylabel(r"$A_{TQ} [a.u.]$")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


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