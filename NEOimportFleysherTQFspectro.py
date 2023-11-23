import PythonBrukerReadIn as pbr
import numpy as np
import scipy
from numpy.fft import fft
from numpy.fft import fftshift as shift



def importFleysherTQF(data_path, spikeComp, phaseCorr, preDCcomp, filter2ndDim, filterFac2ndDim,
                     preFilter, filterFacPre, w0corr, freqDriftVal, postDCcomp, filterFID, filterFacPost, onlyReal):

    ds = pbr.ReadExperiment(path=data_path)

    rawComplexData = ds.raw_fid
    acqp = ds.acqp
    method = ds.method

    numPhaseCycles, numPhaseSteps, nR = int(method['NumPhaseCycles']), int(method['NumPhaseSteps']), int(method['Repetitions'])



    specDim = int(2 ** np.ceil(np.log2(method['PVM_SpecMatrix'])))

    # Reshape data, nr and NPCs NPS either = 12
    complexDataAllPhase = np.reshape(rawComplexData, (nR, numPhaseCycles * numPhaseSteps * 2, specDim))

    if spikeComp:
        for i in range(nR):
            for idx in range(complexDataAllPhase.shape[1]):
                tmp1 = np.concatenate(
                    (complexDataAllPhase[i, idx, :68], np.flipud(complexDataAllPhase[i, idx, :69]) * -1))
                tmp2 = complexDataAllPhase[i, idx, :137]
                tmp3 = tmp2.flatten() - tmp1.flatten()
                complexDataAllPhase[i, idx, :137] = tmp3.squeeze()
    complexDataAllPhase = complexDataAllPhase[:, :, 69:]
    complexDataAllPhase[:, :, 0] = complexDataAllPhase[:, :, 0] * 0.5

    tmp1, tmp2, tmp3 = None, None, None

    temp_0, temp_90 = complexDataAllPhase[:, 0::2, :], complexDataAllPhase[:,1::2, :]
    fft_0, fft_90 = fft(np.real(temp_0), axis=1), fft(np.real(temp_90), axis=1)
    Sig_plus = 0.5 * (fft_0 + 1j * fft_90)
    Sig_minus = 0.5 * (fft_0 - 1j * fft_90)
