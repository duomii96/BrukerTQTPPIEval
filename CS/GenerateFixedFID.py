import numpy as np





class GenerateFIDfixed:
    def __init__(self, PhasecycleAngle, NumPhaseCycles, StartPhase):
        self.SNR = 10
        self.normalOrFixed = 'fixed'
        self.randomSeedDefault = 1
        self.PhasecycleAngle = PhasecycleAngle
        self.PhasecycleFactor = 360 / PhasecycleAngle
        self.NumPhaseCycles = NumPhaseCycles
        self.StartPhase = StartPhase
        self.EvoTimeStepX = None
        self.EvoTime = 0.4
        self.NumberDataPoints = None
        self.paramsDefaultNormal = np.array(
            [0.3794, 0.03935, 0.6215, 0.01025, 0.1998, 1.093, 9.374, -0.001851, 625.2])  # 2% Agarose default
        self.paramsDefaultFixed = np.array([0.7754, -0.7195, 0.2081, 0.8009, -0.002237])  # 2% Agarose default
        self.generateX()



    @staticmethod
    def fixedTQTPPIfunc(x, params):
        return params[0] * np.sin(x + params[1]) + params[2] * np.sin(3 * x + params[3]) + params[4]
    @staticmethod
    def addGaussianNoise(y, SNR):
        if SNR >= 1000:
            yNoise = y
        else:
            yNoise = np.random.normal(0, np.max(y) / SNR, y.shape) + y
        return yNoise

    @staticmethod
    def addGaussianNoiseSeed(y, SNR, seed):
        np.random.seed(seed)
        if SNR >= 1000:
            yNoise = y
        else:
            yNoise = np.random.normal(0, np.max(y) / SNR, y.shape) + y
        return yNoise

    def generateX(self):
        self.x = np.arange(0,self.NumPhaseCycles * self.PhasecycleFactor * self.PhasecycleAngle,
                          self.PhasecycleAngle ) + self.StartPhase
        self.x *= 2*np.pi / 360


    def fixedTQTPPI(self, x, param):
        return self.fixedTQTPPIfunc(x, param)

    def countNonValFIDs(self, FIDs):
        A = np.zeros((FIDs.shape[0],))
        NumZeroFIDs = 0
        for k in range(FIDs.shape[0]):
            if np.allclose(FIDs[k, :], 0):
                NumZeroFIDs += 1
                A[k] = 1
        return NumZeroFIDs, A

    def generateFIDsFixed(self, params, SNRs):

        n, m = params.shape[0], SNRs.shape[0]
        FIDs = []
        #self.FIDidcs = []
        for i in range(n):
            for j in range(m):
                snr = SNRs[j]
                param = params[i, :]
                FID = self.fixedTQTPPI(self.x, param)
                FID = self.addGaussianNoise(FID, snr)
                #self.FIDidcs.append(m* j+ i)
                FIDs.append(FID)
        return np.array(FIDs), self.x, params, SNRs

    ##--------------------------------------------------------
    ## -------- Parameter Generation -------------------------

    def generateParamsFixed_TQ(self):
        TQs = np.concatenate(([0.005], np.arange(0.01, 0.11, 0.01), np.arange(0.125, 0.251, 0.025)))
        n = len(TQs)
        params = np.tile(self.paramsDefaultFixed, (n, 1))
        params[:, 2] = TQs
        return params, TQs

    def generateParamsFixed_ratioTQSQ(self):
        np.random.seed(0)
        ratioSampleInterval = np.random.uniform(low=0.05, high=0.1, size=(10,))
        n_ratio = len(ratioSampleInterval)
        allTQs = np.concatenate(([0.005], np.arange(0.01, 0.11, 0.01), np.arange(0.125, 0.251, 0.025)))
        if n_ratio != len(allTQs):
            TQs = np.random.choice(allTQs, size=n_ratio, replace=False)
        else:
            TQs = allTQs
        SQs = TQs / ratioSampleInterval
        params = np.tile(self.paramsDefaultFixed, (n_ratio, 1))
        params[:, 0] = SQs
        params[:, 2] = TQs
        return params, TQs, SQs

    def varySNR(self):
        SNRs = np.arange(40, 110, 10)
        return SNRs.reshape(-1, 1)

    def getTQSQ_initial(self, params):
        return params[:, 2] / params[:, 0]



