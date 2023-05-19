import PythonBrukerReadIn as pbr
import numpy as np
import os

spikeComp = 0
filterFID = 0
filterFac = 1
p = 0
filterMz = 0
filtertyp = 0  #(1 lorentz, 0 gauss)
sigma = 0.75
TPMa1 = 4.2
TPMb1 = 3.0
diffab = TPMa1 - TPMb1

data_path = input("Enter the directory path containing the data: ")
data_path = data_path + "/"
ds = pbr.ReadExperiment(path=data_path)

rawComplexData = ds.raw_fid
acqp = ds.acqp
method = ds.method
numPhaseCycles, numPhaseSteps, nR = int(method['NumPhaseCycles']), int(method['NumPhaseSteps']), int(method['Repetitions'])
specDim = int(2 ** np.ceil(np.log2(method['PVM_SpecMatrix'])))

if method['Method'] == '<User:anVsTqtppiSpectro>':
    complexDataAllPhase = np.reshape(rawComplexData, (specDim, numPhaseCycles * numPhaseSteps * 2, nR))
    tmp = complexDataAllPhase[:, 0::2, :] + complexDataAllPhase[:, 1::2, :]
    complexDataAllPhase = tmp
    del tmp
else:
    complexDataAllPhase = np.reshape(rawComplexData, (nR, numPhaseCycles * numPhaseSteps, specDim))
