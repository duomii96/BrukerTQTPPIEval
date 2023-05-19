import matplotlib.pyplot as plt

from CS.CS_Reconstruction import RecoCS
import plotting as plo
from CS.GenerateFixedFID import GenerateFIDfixed
import numpy as np


csGen = GenerateFIDfixed(PhasecycleAngle=5,NumPhaseCycles=16,StartPhase=45)
params, TQs = csGen.generateParamsFixed_TQ()
SNRs = csGen.varySNR()
FIDs, x , params, SNRs = csGen.generateFIDsFixed(params, SNRs)

fidInput = FIDs[2,:]

ReCo = RecoCS(DataFull=fidInput,CS_Algorithm='IST-S',samplingMethod='uniform',accelerationFactor=4 )
out = ReCo.csReconstruction()

plo.myplot(FIDs.transpose(), title='IST-S Algorithm')
plt.show()