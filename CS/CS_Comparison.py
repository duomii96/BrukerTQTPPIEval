import matplotlib.pyplot as plt
from CS.CS_Reconstruction import RecoCS
import plotting as plo
from CS.GenerateFixedFID import GenerateFIDfixed
import numpy as np
from CS.ComparisonMetrics import hamming, RMSE, trapzoid, simpson, get_RecoTQSQ
from numpy.fft import fft, ifft, fftshift


csGen = GenerateFIDfixed(PhasecycleAngle=5,NumPhaseCycles=16,StartPhase=45)
params, TQs = csGen.generateParamsFixed_TQ()
SNRs = csGen.varySNR()
FIDs, x , params, SNRs = csGen.generateFIDsFixed(params, SNRs) # shape FIDs: (numFIDS, numphaseSteps)
initialTQSQ = csGen.getTQSQ_initial(params)
# num FIDs = #parameterCombinations * #SNRvariations

fidInput = FIDs
ftInput = fft(fidInput)

ReCo = RecoCS(DataFull=fidInput,CS_Algorithm='IST-D',samplingMethod='PoissonGap',accelerationFactor=8)
#Reco_4 = RecoCS(DataFull=fidInput,CS_Algorithm='IST-S',samplingMethod='PoissonGap',accelerationFactor=4)
out = ReCo.csReconstruction()
#out_4 = Reco_4.csReconstruction()
ft_out = fftshift(fft(out))

recoTQSQ = get_RecoTQSQ(np.real(ft_out))

targetAtqsq_aligned = np.repeat(initialTQSQ*100, len(SNRs))

rmse = RMSE(np.real(ft_out), np.real(ftInput))
pint = trapzoid(np.real(ft_out), np.real(ftInput))


"""plt.plot(rmse)
plt.xticks(np.arange(stop=FIDs.shape[0],step=len(SNRs)),np.round(initialTQSQ*100,2))
plt.show()"""

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(np.round(recoTQSQ*100,2), linewidth=0.5, label=r"Reconstructed")
ax1.plot(targetAtqsq_aligned, linewidth=0.5, label="Target")
ax1.set_ylabel("$A_{TQ}/A_{SQ}$[\%]")
ax1.legend(loc="best")
ax2.plot(rmse, linewidth=0.5)
ax2.set_ylabel("RMSE")
plt.show()

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




