import numpy as np
import os


def get_NoiseEstimate(spec):
    """
    Takes mqSpectra as input and calculates
    :param spec:
    :return:
    """
    # Estimate noise from spectrum
    # Remove SQ and TQ peaks from spectrum as well as offsett
    numSpectra, specRes = spec.shape
    if numSpectra > specRes:
        spec = spec.transpose()
        temp = numSpectra
        numSpectra = specRes
        specRes = temp
        del temp
    _, posSq1 = np.unravel_index(np.argmax(spec[:, :int(specRes / 2)], axis=None),
                                 spec[:, :int(specRes / 2)].shape)
    _, posSq2 = np.unravel_index(np.argmax(spec[:, int(specRes / 2):], axis=None),
                                 spec[:, int(specRes / 2):].shape)
    posSq2 += int(specRes / 2)
    assert posSq2 > posSq1
    freqSpacing = np.abs(posSq2 - posSq1)
    posTq = posSq1 - freqSpacing


    mask = np.zeros_like(spec, dtype=np.uint)
    mask[:, [posTq, posSq1, posSq1 + int(freqSpacing/2), posSq1 + freqSpacing, posTq + freqSpacing*3]] = 1
    # mask spec
    specForNoise = spec.copy()
    specForNoise = specForNoise[np.invert(mask)]
    # calculate noise
    noiseSpec = np.var(specForNoise, axis=-1)
    return noiseSpec



