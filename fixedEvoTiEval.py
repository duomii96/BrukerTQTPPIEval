from commonImports import *


def fixedEvoTimesEval(spectralDataAcq, evoTimes, TQSQspacing, secondDimFit):
    spectralDataAcq = spectralDataAcq
    datashape = np.shape(spectralDataAcq)

    try:
        posSq, _ = np.unravel_index(np.argmax(spectralDataAcq[:int(datashape[0]/2),]), spectralDataAcq.shape)
    except:
        _, posSq = np.unravel_index(np.argmax(spectralDataAcq[1, int(spectralDataAcq.shape[0] / 4):int(spectralDataAcq.shape[0] / 2)], axis=None),
                                         spectralDataAcq[:, int(spectralDataAcq.shape[0] / 4):int(spectralDataAcq.shape[0] / 2)].shape)
    posTq = posSq - TQSQspacing
    print(f'TQ at SpecPos: {posTq} \n')
    if posTq < 1:
        print(f"Not able to find TQ peak for EvoTime: {evoTimes}")
        posTq = 17

    noiseIdcs = np.arange(1, np.ceil(spectralDataAcq.shape[0] / 2) + 1, dtype=np.int)
    mask = (noiseIdcs != posTq) & (noiseIdcs != posSq)
    noiseIdcs = noiseIdcs[mask]

    realData = np.real(spectralDataAcq)
    fixedSQpeaks = np.max(realData, axis=0)

    ratio = realData[posTq, :] / realData[posSq, :] * 100

    normalizedSQpeaks = fixedSQpeaks / np.max(fixedSQpeaks) * 100

    noise_var = np.zeros(spectralDataAcq.shape[1])
    for i in range(spectralDataAcq.shape[1]):
        noise_var[i] = np.var(np.squeeze(realData[noiseIdcs, i]))

    fixedTQpeaks = realData[posTq, :].transpose()


    return fixedTQpeaks, fixedSQpeaks, ratio, noise_var



