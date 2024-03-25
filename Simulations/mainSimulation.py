from commonImports import *
import getValues
from PCsimul import PhaseCycle






B0 = 9.4 # T

# TODO: Run and check for errors

# tevoOpt = ln(T2s/T2f)/(1/T2f - 1/T2s)
fixedEval = 0 # When simulating fixed
fleysher = 0 # for fleysher cycling evaluation

T2s_f = 6.75e-3 # s
T2s_l = 30.07e-3 # s
T1s = 38.1e-3 #s
T1f = 37.9e-3 # s

tevoOpt = np.log(T2s_l/T2s_f)/(1/T2s_f-1/T2s_l)


wQbar = 0 # anisotropy (mean value of wQ)
wShift = 0
w0 = getValues.get_w0(B0)
Jen, tauC, wQ, wShift_RMS = getValues.get_JenModel(T1f, T1s, T2s_f, T2s_l, w0)
print(f'Jen: {Jen}, tauC: {tauC}, wQ: {wQ}, wShift_RMS: {wShift_RMS}')

T1s, T1f, T2s, T2f = getValues.get_relaxationTimes_Jen(tauC, wQ, wQbar, Jen, wShift_RMS, w0)
# Ts_ms = np.array([T1s, T1f, T2s, T2f]) * 1000

PC = PhaseCycle(B0, tauC, wQ, wQbar, Jen, wShift, wShift_RMS)
PC.TR = 1000e-3 # s
### ---- Flip angles ----------------------------
PC.flip90 = 90 # can be changed to simulate miscalibrated flip angles
PC.flip180 = 180
#fids_echo, te_s, times = PC.SpinEcho()

"""PC.alphas = np.array([0, 45, 90, 135, 180, 225, 270, 315]) + 90
PC.NumPhaseCycles = 16
PC.tevo0 = 10 * 1e-3 # s
PC.tevoStep = 0.2 * 1e-3 # s"""
PC.NumPhaseCycles = 1

#PC.tevo0 = 8e-3 # s
evoRange = 0.004 # s around evoOpt
evoSteps = 0.001
evoTimes = []
allFIDsp1, allFIDsp2, allTQsP1, allTQsP2 = [], [], [], []
for evoTi in np.arange(tevoOpt - evoRange, tevoOpt + evoRange, evoSteps):
    print(evoTi)
    #FIDs_p1, FIDs_p2, tevos, TQs_p1, TQs_p2, SQs_p1, SQs_p2 = PC.TQTPPI_wo180()
    PC.tevo0 = evoTi
    evoTimes.append(evoTi)
    FIDs_p1, FIDs_p2, tevos, TQs_p1, TQs_p2, SQs_p1, SQs_p2 = PC.TQTPPI_Fleysher()
    #tevos, TQTPPI_FID, FID_corr = PC.get_TQTPPI_FID(tevos, FIDs_p1, FIDs_p2)
    allFIDsp1.append(np.sum(FIDs_p1, axis=0))
    allFIDsp2.append(np.sum(FIDs_p2, axis=0))

if fixedEval:
    ftFIxedFIDs = fft(FID_corr.transpose(), axis=1)
elif fleysher:
    # FFT
    ftP1 = fft(np.array(allFIDsp1))
    ftP2 = fft(np.array(allFIDsp2))
    combined = 0.5 * (np.abs(ftP1) + 1j * np.abs(ftP2))
    plt.figure()
    plt.plot(evoTimes, np.abs(fftshift(combined))[:,1024])
    plt.show()

"""plt.figure()
plt.plot(np.real(TQTPPI_FID))
plt.show()
"""



""" myk_comb = 0.5 * (FIDs_a * np.exp(1j * FIDs_b))
myk_comb = np.concatenate((FIDs_a, FIDs_b), axis=0)
myk_comb_fft = fftshift(fft(myk_comb, 1))
"""
