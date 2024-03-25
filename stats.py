from commonImports import *
import os

from skimage.restoration import (denoise_wavelet, estimate_sigma)

# estimate sigma is a wavelet based noise estimator



def plot_shifted_spectrum(spectra, x_shift = 3):
    # Plot shifted spectra in one plot to better compare peaks
    # shape of spectra: (Number, k-space -length)
    SQIdx = np.argmax(spectra[0,:])
    fig, ax = plt.subplots()
    x0_values = np.arange(spectra.shape[-1])  # Generating x values for each spectrum
    for index, spectrum in enumerate(np.real(spectra)):
        shifted_x = x0_values + (index * x_shift)  # Shifting x values for each spectrum
        ax.plot(shifted_x, spectrum, label=f'Spectrum {index + 1}', linewidth=0.8)
    ax.axvline(x=SQIdx, color='red', linestyle='--', linewidth=0.7, label='SQ Unshifted')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Intensity')
    ax.legend()
    plt.show()
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


def getTQSQ(spectra):
    """

    :return: TQ, Sq and ratio of both for given Spectrum.
    """
    spectra = np.real(spectra)
    try:
        _,posSq, _ = np.unravel_index(np.argmax(spectra[1, :int(spectra.shape[1]/2), :], axis=None),
            spectra[:, :int(spectra.shape[1] / 2), :].shape)
    except:
        try:
            _, posSq = np.unravel_index(np.argmax(spectra[1, :int(spectra.shape[1]/2)], axis=None),
                                             spectra[:, :int(spectra.shape[1]/2)].shape)
        except:
            (posSq,_) = np.unravel_index(np.argmax(spectra[:int(spectra.shape[0] / 2)], axis=None),
                                     spectra[:int(spectra.shape[0] / 2)].shape)

    # self.posSq = posSq + (int(self.numPhaseInc / 4)) - 1
    posTq = posSq - 32

    try:
        SQpeaks = spectra[:,posSq,:]
        TQpeaks = np.squeeze(spectra[:,posTq,:])
        ratios = TQpeaks / SQpeaks
    except:
        try:
            SQpeaks = spectra[:, posSq]
            TQpeaks = np.squeeze(spectra[:, posTq])
            ratios = TQpeaks / SQpeaks
        except:
            SQpeaks = spectra[posSq]
            TQpeaks = np.squeeze(spectra[posTq])
            ratios = TQpeaks / SQpeaks
    return SQpeaks, TQpeaks, ratios



def waveletDenoise(signal, numWavelets=3, mode='soft'):
    """
    Denoises 1D-Signal- this one is actually for 2D images but can be used here as well.
    BayesShrink is a from of soft Thresholding that estimates each new threshold.
    :param signal:
    :param mode: soft or hard
    :return:
    """

    x_denoise = denoise_wavelet(signal, method='BayesShrink', mode=mode, wavelet_levels=numWavelets, wavelet='sym8', rescale_sigma='True')

    return x_denoise