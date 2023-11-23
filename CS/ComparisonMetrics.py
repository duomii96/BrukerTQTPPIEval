"""
Metrics to evaluate the recontructed signal from the compressed sensing algorithms.
"""
import numpy as np
import scipy.integrate as spint
from functools import reduce


def RMSE(prediction, target):
    prediction = np.squeeze(prediction)
    if prediction.ndim > 1:
        res = np.abs(np.sqrt(np.mean((prediction-target)**2,axis=-1)))
    else:
        res = np.abs(np.sqrt(np.mean((prediction - target) ** 2)))
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
    try:
        # if multiple dim only
        what, posSq= np.unravel_index(np.argmax(np.abs(spectra[1, :int(spectra.shape[-1] / 2)]), axis=None),
                                      spectra[:, :int(spectra.shape[-1] / 2)].shape)
        posTq = posSq - 32
        ratio = np.abs(spectra[:, posTq]) / np.abs(spectra[:, posSq])
    except:
        (posSq,) = np.unravel_index(np.argmax(np.abs(spectra[:int(spectra.shape[-1] / 2)]), axis=None),
                                       spectra[:int(spectra.shape[-1] / 2)].shape)

        posTq = posSq - 32
        SQ, TQ = spectra[posTq], spectra[posTq]
        ratio = np.abs(spectra[ posTq]) / np.abs(spectra[posSq])
    # self.posSq = posSq + (int(self.numPhaseInc / 4)) - 1

    return ratio





