from commonImports import *
import PythonBrukerReadIn as pbr
from scipy.optimize import minimize


def NEOreadIn(pathToData):

    spikeComp = 0
    filterFID = 0
    filterFac = 1
    p = 0 # when using phosphor measurements
    filterMz = 0
    filtertyp = 0  #(1 lorentz, 0 gauss)
    sigma = 0.75
    TPMa1 = 4.2
    TPMb1 = 3.0
    usePhaseCorr = 0 # not needed for Single Pulse (oder?)

    # Load method and acqp parameters
    data_path = pathToData
    ds = pbr.ReadExperiment(path=data_path)

    rawComplexData = ds.raw_fid
    acqp = ds.acqp
    method = ds.method


    # Check if the parameters are properly loaded
    #print(method)
    #print(acqp)
    #complexData = rawComplexData[0]  # Assuming rawdata is a list or array-like structure

    if np.ndim(rawComplexData) == 1:
        complexData = np.expand_dims(rawComplexData, axis=0)
    else:
        complexData = rawComplexData

    #sizeCD = np.shape(complexData)

    # Extract relevant information
    specDim = method['PVM_SpecMatrix']
    specDim1 = 2 ** np.ceil(np.log2(method['PVM_SpecMatrix']))

    if specDim != specDim1:
        complexData = complexData[:specDim]
    elif specDim != len(complexData) and specDim == specDim1:
        complexData = np.reshape(complexData, (complexData.shape[-1] // specDim, specDim))

    if method['Method'] == '<User:sr_SP>' and np.min(real(complexData)) < -1e6:
        print("Data multiplied by -1 to switch it around")
        complexData *= -1e0



    # Filtering FID
    filter_fid = True  # Set this to False if not required
    filter_fac = 4  # Adjust filter factor as needed
    sizeCD = np.shape(complexData)
    if filter_fid:
        time_vec_cos = np.linspace(0, 1 / 2 * np.pi, int(sizeCD[0] / filter_fac)) # along phase dimesion
        for idx in range(complexData.shape[0]):
            complexData[idx,:len(time_vec_cos)] *= np.cos(time_vec_cos) ** 2
            complexData[idx, :len(time_vec_cos)] = 0

    # Zero filling
    # test = np.zeros([specDim, sizeCD[1]], dtype=complex)
    # test[:sizeCD[0], :sizeCD[1]] = complexData
    # complexData = test

    complexData0 = np.zeros([sizeCD[0], specDim * 4], dtype=complex)
    complexData0[:sizeCD[0], :specDim] = complexData

    # Phase correction ------------------------------------------------------------------------------------------
    if p != 0:
        # For 19F use ifft and ifftshift
        complex_fft = ifftshift(ifft(complexData, axis=0), axes=0) * complexData.shape[0]
        complex_fft0 = ifftshift(ifft(complexData0, axis=0), axes=0) * complexData.shape[0]
    else:
        # Normal
        complex_fft = fftshift(fft(complexData, axis=0), axes=0)
        complex_fft0 = fftshift(fft(complexData, axis=0), axes=0)

    ### Phase correction

    if usePhaseCorr is None or usePhaseCorr:
        pos = np.argmax(np.abs(complex_fft), axis=0)
        #x = np.zeros(a[1])
        #fval = np.zeros(a[1])

        banana = lambda x: -np.real(complex_fft[pos] * np.exp(1j * x))
        x, fval = minimize(banana, [0], full_output=True)[:2]

        complexDataAllPhaseUnw = complexData * np.exp(1j * x)

    # ---------------------------------------------------------------------------------------------------------
    # Spike compensation ---------------------------------------------------------------

    """   
    if spikeComp:
        for idx in range(complexData.shape[0]):
            tmp1 = np.concatenate((complexData[idx, :68], np.flipud(complexData[idx, :69]) * -1))
            tmp2 = complexData[idx, :137]
            tmp3 = tmp2.flatten() - tmp1.flatten()
            complexData[idx, :137] = tmp3.squeeze()
    complexData = complexData[:,:137]
    complexData[:,0] = complexData[:,0] * 0.5
   """



    if method['Method'] == '<Bruker:SINGLEPULSE>' and len(method.PVM_ArrayPhase) == 4:
        c = np.zeros(4)
        for k in range(4):
            c[k] = np.dot(np.conjugate(complex_fft[:, k]), complex_fft[:, k])
        c /= np.max(c)

    return complexData, complex_fft, method, acqp
