"""
Metrics to evaluate the recontructed signal from the compressed sensing algorithms.
"""
import numpy as np
import scipy.integrate as spint


def RMSE(prediction, target):
    res = np.sqrt(np.mean((prediction-target)**2, axis=-1))
    return res

def hamming(prediction, target):
    # Makes no sense to me
    return np.count_nonzero(prediction-target)

def trapzoid(prediction, target):
    # Spectral Integration using Trapezoidal Rule
    predictionInt = np.trapz(prediction)
    targetInt = np.trapz(target)
    return (predictionInt, targetInt)

def simpson(prediction, target):
    # Spectral Integration using Simpson's rule
    predictionInt = spint.simpson(prediction)
    targetInt = spint.simpson(target)
    return (predictionInt, targetInt)

def get_RecoTQSQ(spectra):
    # Return TQSQ as decimal

    what, posSq= np.unravel_index(np.argmax(np.abs(spectra[1, :int(spectra.shape[-1] / 2)]), axis=None),
                                  spectra[:, :int(spectra.shape[-1] / 2)].shape)
    # self.posSq = posSq + (int(self.numPhaseInc / 4)) - 1
    posTq = posSq - 32
    return np.abs(spectra[:, posTq] / spectra[:, posSq])






