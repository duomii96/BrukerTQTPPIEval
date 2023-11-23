import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class PhaseCycle:
    def __init__(self):
        self.B0 = 9.4  # T
        self.w0 = None
        self.tauC = None
        self.wQ = None
        self.wQbar = 0
        self.Jen = None
        self.wShift = 0
        self.wShift_RMS = 0
        self.wShift_FID = 0
        self.tevoStep = 0.1000 * 1e-3  # s
        self.tevo0 = 0.1305 * 1e-3  # s
        self.tmix = 0.1500 * 1e-3  # s
        self.startPhase = 90  # degrees
        self.alphas = None
        self.NumPhaseCycles = None
        self.PulseAngles = []
        self.TR = 200e-3  # s  repetition time
        self.dataPoints = 10  # 2048
        self.deadtimeFID = 1e-4  # s
        self.dwelltimeFID = 50e-4  # s
        self.noisePower = 0  # noise Power in Watts, linear, currently only for T3i sequence
        self.flip90 = 90  # can be set to different values to simulate B1+
        self.flip180 = 180  # inhomogeneities
        self.TEstep = 1e-3
        self.TEmax = 400e-3

    @staticmethod
    def general3Pulse(TmnStart, alpha, beta, tevo, tmix, flip1, flip2, flip3):
        # Insert your code here for the general3Pulse method

    @staticmethod
    def general3Pulse3phase(TmnStart, alpha, beta, phase3, tevo, tmix, flip1, flip2, flip3):
        # Insert your code here for the general3Pulse3phase method

    @staticmethod
    def general3Pulse_softPulse(TmnStart, alpha, beta, tevo, tmix, pulseStrength, flip2, flip3):
        # Insert your code here for the general3Pulse_softPulse method

    @staticmethod
    def general3Pulse_w180(TmnStart, alpha, beta, tevo, tmix, flip1, flip2, flip3, flipRefocus):
        # Insert your code here for the general3Pulse_w180 method
        import numpy as np
        from scipy.optimize import fmin

    def phase_cor_FID(fid):
        # Calculate the phase correction factor using optimization
        fidi = fid[0, :]  # Assuming 1st row of FID contains data
        banana = lambda x: -np.mean(np.real(fidi * np.exp(1j * x)))
        x_optimized = fmin(banana, [0])

        # Apply the phase correction to the FID
        if np.max(np.real(fid[0, :] * np.exp(1j * x_optimized[0]))) < 0:
            FID = fid * np.exp(-1j * x_optimized[0])
        else:
            FID = fid * np.exp(1j * x_optimized[0])

        return FID

    def get_TQTPPI_FID(tevos, FIDs_p1, FIDs_p2):
        # Combine FIDs from p1 and p2
        FIDs = FIDs_p1 + FIDs_p2

        # Apply phase correction to the combined FID
        FID_corrected = self.phase_cor_FID(FIDs)

        # Sum FIDs along the first dimension
        TQTPPI_FID = np.sum(FID_corrected, axis=0)

        return tevos, TQTPPI_FID


class NMRFIDSimulator:
    def __init__(self, num_points=2048, decay_constant=0.7, noise_level=0, phaseCycling=True):
        self.num_points = num_points
        self.decay_constant = decay_constant
        self.noise_level = noise_level
        self.phaseCycling = phaseCycling
        if self.phaseCycling:
            self.genPhaseCycles()

    def genPhaseCycles(self, startPhase=90, phaseAngle=45, numCycles=16):

        endPhase = startPhase + numCycles * 360
        list = np.arange(startPhase,endPhase, phaseAngle)
        self.phase_cycle_list = list * 2 * np.pi / 360
    def simulate(self):
        time = np.arange(0, self.num_points)
        signal = np.exp(-self.decay_constant * time)
        noise = np.random.normal(0, self.noise_level, self.num_points)
        fid = np.array(signal + noise, dtype=np.complex)
        if self.phaseCycling:

            fids = np.tile(fid, (len(self.phase_cycle_list),1))

            for idx, phase in enumerate(self.phase_cycle_list):
                fids[idx,:] *=  np.exp(1j * phase)

            return time, fids
        else:
            return time, fid

    def plot_fid(self):
        time, fid = self.simulate()
        plt.figure(figsize=(10, 4))
        plt.plot(time, fid, linewidth=0.6)
        plt.title("Simulated NMR Free Induction Decay (FID)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

# Example usage:
simulator = NMRFIDSimulator(num_points=2048, decay_constant=0.02, noise_level=0.01, phaseCycling=True)
time, fids = simulator.simulate()
