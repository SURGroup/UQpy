"""This module contains functionality for all the stochastic process generation supported by UQpy."""

from UQpy.tools import *
from scipy.linalg import sqrtm


class SRM:
    def __init__(self, n_sim, S, dw, nt, nw, case='uni', g=None):
        # TODO: Error check for all the variables
        # TODO: Division by 2 to deal with zero frequency for all cases
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.n_sim = n_sim
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.m = self.S.shape[0]
            self.n = len(S.shape[1:])
            self.g = g
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.append(self.m, np.ones(self.n, dtype=np.int32) * self.nw))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        B = (2 ** self.n) * np.exp(phi * 1.0j) * np.sqrt(self.S * np.prod(self.dw))
        sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
        samples = np.real(sample)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples

    def _simulate_multi(self, phi):
        # Assembly of S_jk
        S_sqrt = np.sqrt(self.S)
        S_jk = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)
        # Assembly of g_jk
        g_jk = np.zeros_like(S_jk)
        l = 0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                g_jk[i, j] = self.g[l]
                l = l + 1
        g_jk = np.einsum('ij...->ji...', g_jk) + g_jk

        for i in range(self.m):
            g_jk[i, i] = np.ones_like(S_jk[0, 0])
        S = S_jk * g_jk

        S = np.einsum('ij...->...ij', S)
        S1 = S[..., :, :]
        H_jk = np.zeros_like(S1)
        for i in range(len(S1)):
            try:
                H_jk[i] = np.linalg.cholesky(S1[i])
            except:
                H_jk[i] = np.linalg.cholesky(nearestPD(S1[i]))
        H_jk = H_jk.reshape(S.shape)
        H_jk = np.einsum('...ij->ij...', H_jk)
        samples_list = []
        for i in range(self.m):
            samples = 0
            for j in range(i+1):
                B = 2 * H_jk[i, j] * np.sqrt(np.prod(self.dw)) * np.exp(phi[:, j] * 1.0j)
                sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
                samples += np.real(sample)
            samples_list.append(samples)
        samples_list = np.array(samples_list)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return np.einsum('ij...->ji...', samples_list)


class BSRM:
    def __init__(self, n_sim, S, B, dt, dw, nt, nw, case='uni', g=None):
        self.n_sim = n_sim
        self.nw = nw
        self.nt = nt
        self.dw = dw
        self.dt = dt
        self.n = len(S.shape)
        self.S = S
        self.B = B
        self.B_Ampl = np.absolute(B)
        self.B_Real = np.real(B)
        self.B_Imag = np.imag(B)
        self.Biphase = np.arctan2(self.B_Imag, self.B_Real)
        self.Biphase[np.isnan(self.Biphase)] = 0
        self.phi = np.random.uniform(size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
        self._compute_bicoherence()

    def _compute_bicoherence(self):
        self.Bc2 = np.zeros_like(self.B_Real)
        self.PP = np.zeros_like(self.S)
        self.sum_Bc2 = np.zeros(self.nw)
        self.PP[0] = self.S[0]
        self.PP[1] = self.S[1]

        for i in range(self.nw):
            for j in range(int(np.ceil((i + 1) / 2))):
                w1 = i - j
                w2 = j
                if self.B_Ampl[w2, w1] > 0 and self.PP[w2] * self.PP[w1] != 0:
                    self.Bc2[w2, w1] = self.B_Ampl[w2, w1] ** 2 / (self.PP[w2] * self.PP[w1] * self.S[i]) * self.dw
                    self.sum_Bc2[i] = self.sum_Bc2[i] + self.Bc2[w2, w1]
                else:
                    self.Bc2[w2, w1] = 0
            if self.sum_Bc2[i] > 1:
                for j in range(int(np.ceil((i + 1) / 2))):
                    w1 = i - j
                    w2 = j
                    self.Bc2[w2, w1] = self.Bc2[w2, w1] / self.sum_Bc2[i]
                self.sum_Bc2[i] = 1
            self.PP[i] = self.S[i] * (1 - self.sum_Bc2[i])

    def _simulate_bsrm_uni(self):
        Coeff = (2 ** self.n) * np.sqrt(self.S * np.prod(self.dw))
        Phi_e = np.exp(self.phi * 1.0j)
        Biphase_e = np.exp(self.Biphase * 1.0j)
        B = np.sqrt(1 - self.sum_Bc2) * Phi_e
        Bc = np.sqrt(self.Bc2)

        for i in range(self.nw):
            for j in range(1, int(np.ceil((i + 1) / 2))):
                w1 = j
                w2 = i - j
                B[:, i] = B[:, i] + Bc[w1, w2] * Biphase_e[w1, w2] * Phi_e[:, w1] * Phi_e[:, w2]

        B = B * Coeff
        B[np.isnan(B)] = 0
        samples = np.real(np.fft.fftn(B, [self.nt]))
        return samples


class KLE:
    def __init__(self, n_sim, R):
        self.R = R
        self.samples = self._simulate(n_sim)

    def _simulate(self, n_sim):
        lam, phi = np.linalg.eig(self.R)
        nRV = self.R.shape[0]
        xi = np.random.normal(size=(nRV, n_sim))
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi, np.dot(sqrtm(lam), xi))
        samples = np.real(samples)
        samples = samples.T
        return samples


def itam_srm(S, beta, w, t, CDF, mu, sig, parameter1, parameter2, maxii=10):
    S_NGT = S
    S_G0 = S
    S_NG0 = S

    R_NGT = R_to_r(S_to_R(S_NGT, w, t))

    iconverge = 0
    Error0 = 100
    Error1_time = np.zeros([maxii])

    for ii in range(maxii):
        R_G0 = S_to_R(S_G0, w, t)

        # Translation the correlation coefficients from Gaussian to Non-Gaussian case
        R_NG0 = np.zeros_like(R_G0)
        if CDF == 'Lognormal':
            R_NG0 = translate(R_G0, 'Lognormal_Distribution', 'pseudo', mu, sig, parameter1, parameter2)
        elif CDF == 'Beta':
            R_NG0 = translate(R_G0, 'Beta_Distribution', 'pseudo', mu, sig, parameter1, parameter2)
        elif CDF == 'User':
            R_NG0 = translate(R_G0, 'User_Distribution', 'pseudo', mu, sig, parameter1, parameter2)

        R_NG0_Unnormal_Mean = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            if R_NG0[i, 1] != 0:
                R_NG0_Unnormal_Mean[i, :] = (R_NG0[i, :] - mu[i] ** 2)
            else:
                R_NG0_Unnormal_Mean[i, :] = 0

        # Normalize computed non - Gaussian R(Stationary(1D R) & Nonstatioanry(Pseudo R))
        rho = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            if R_G0[i, 1] != 0:
                rho[i, :] = (R_NG0[i, :] - mu[i] ** 2) / sig[i] ** 2
            else:
                rho[i, :] = 0
        R_NG0 = rho

        S_NG0 = R_to_S(R_NG0_Unnormal_Mean, w, t)

        if S_NG0.shape[0] == 1:
            # compute the relative difference between the computed S_NG0 & the target S_NGT
            Err1 = 0
            Err2 = 0
            for j in range(S_NG0.shape[1]):
                Err1 = Err1 + (S_NG0[0, j] - S_NGT[0, j]) ** 2
                Err2 = Err2 + S_NGT[0, j] ** 2
            Error1 = 100 * np.sqrt(Err1 / Err2)
            convrate = (Error0 - Error1) / Error1
            if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
                iconverge = 1
            Error1_time[ii] = Error1
            nError1 = nError1 + 1

        else:  # Pristely_Simpson (S_NG0 -> R_NG0)
            R_NG0_Unnormal = S_to_R(S_NG0, w, t)
            # compute the relative difference between the computed NGACF & the target R(Normalized)
            Err1 = 0
            Err2 = 0
            for i in range(R_NG0.shape[0]):
                for j in range(R_NG0.shape[1]):
                    Err1 = Err1 + (R_NG0[i, j] - R_NGT[i, j]) ** 2
                    Err2 = Err2 + R_NGT[i, j] ** 2
            Error1 = 100 * np.sqrt(Err1 / Err2)
            convrate = abs(Error0 - Error1) / Error1

            if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
                iconverge = 1

            Error1_time[ii] = Error1
            nError1 = nError1 + 1

        # Upgrade the underlying PSDF or ES
        S_G1 = np.zeros_like(S_G0)
        for i in range(S_NG0.shape[0]):
            for j in range(S_NG0.shape[1]):
                if S_NG0[i, j] != 0:
                    S_G1[i, j] = (S_NGT[i, j] / S_NG0[i, j]) ** beta * S_G0[i, j]
                else:
                    S_G1[i, j] = 0

        # Normalize the upgraded underlying PSDF or ES: Method1: Wiener - Khinchine_Simpson(S -> R or ES -> R)
        R_SG1 = S_to_R(S_G1, w, t)

        for i in range(S_G1.shape[0]):
            S_G1[i, :] = S_G1[i, :] / R_SG1[i, i]

        if iconverge == 0 and ii != maxii:
            S_G0 = S_G1
            Error0 = Error1
        else:
            convergeIter = nError1
            print('\n')
            print('Job Finished')
            print('\n')
            break

    S_G_Converged = S_G0
    S_NG_Converged = S_NG0
    return S_G_Converged, S_NG_Converged


def itam_kle(R, t, CDF, mu, sig, parameter1, parameter2):
    # Initial condition
    nError1 = 0
    m = len(t)
    dt = t[1] - t[0]
    T = t[-1]
    # Erasing zero values of variations
    R_NGT = R

    # TF = [False] * m
    # TF = np.array(TF)
    # for i in range(m):
    #     if R[i, i] == 0: TF[i] = True
    # t[TF] = []
    # mu[TF] = []
    # sig[TF] = []
    # if CDF == 'User':
    #     parameter1[TF] = []
    #     parameter2[TF] = []
    # Normalize the non - stationary and stationary non - Gaussian Covariance to Correlation

    R_NGT_Unnormal = R_NGT
    R_NGT = R_to_r(R_NGT)
    # Initial Guess
    R_G0 = R_NGT

    # Iteration Condition
    iconverge = 0
    Error0 = 100
    maxii = 5
    Error1_time = np.zeros(maxii)

    for ii in range(maxii):
        if CDF == 'Lognormal':
            R_NG0 = translate(R_G0, 'Lognormal_Distribution', '', mu, sig, parameter1, parameter2)
        elif CDF == 'Beta':
            R_NG0 = translate(R_G0, 'Beta_Distribution', '', mu, sig, parameter1, parameter2)
        elif CDF == 'User':
            # monotonic increasing CDF
            R_NG0 = translate(R_G0, 'User_Distribution', '', mu, sig, parameter1, parameter2)

        R_NG0_Unnormal = R_NG0
        # Normalize the computed non - Gaussian ACF
        rho = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            for j in range(R_NG0.shape[1]):
                if R_NG0[i, i] != 0 and R_NG0[j, j] != 0:
                    rho[i, j] = (R_NG0[i, j] - mu[i] * mu[j]) / (sig[i] * sig[j])
                else:
                    rho[i, j] = 0
        R_NG0 = rho

        # compute the relative difference between the computed NGACF & the target R(Normalized)
        Err1 = 0
        Err2 = 0
        for i in range(R_NG0.shape[0]):
            for j in range(R_NG0.shape[1]):
                Err1 = Err1 + (R_NGT[i, j] - R_NG0[i, j]) ** 2
                Err2 = Err2 + R_NG0[i, j] ** 2
        Error1 = 100 * np.sqrt(Err1 / Err2)
        convrate = abs(Error0 - Error1) / Error1
        if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
            iconverge = 1
        Error1_time[ii] = Error1
        nError1 = nError1 + 1

        # Upgrade the underlying Gaussian ACF
        R_G1 = np.zeros_like(R_G0)
        for i in range(R_G0.shape[0]):
            for j in range(R_G0.shape[1]):
                if R_NG0[i, j] != 0:
                    R_G1[i, j] = (R_NGT[i, j] / R_NG0[i, j]) * R_G0[i, j]
                else:
                    R_G1[i, j] = 0

        # Eliminate Numerical error of Upgrading Scheme
        R_G1[R_G1 < -1.0] = -0.99999
        R_G1[R_G1 > 1.0] = 0.99999
        R_G1_Unnormal = R_G1

        # Normalize the Gaussian ACF
        R_G1 = R_to_r(R_G1)
        # Iteratively finding the nearest PSD(Qi & Sun, 2006)
        R_G1 = np.array(nearPD(R_G1))
        R_G1 = R_to_r(R_G1)

        # Eliminate Numerical error of finding the nearest PSD Scheme
        R_G1[R_G1 < -1.0] = -0.99999
        R_G1[R_G1 > 1.0] = 0.99999

        if iconverge == 0 and ii != maxii:
            R_G0 = R_G1
            Error0 = Error1
        else:
            convergeIter = nError1
            print('\n')
            print('[Job Finished]')
            print('\n')
            break

    R_G_Converged = R_G0
    R_NG_Converged = R_NG0_Unnormal
    return R_G_Converged, R_NG_Converged


class Translate:
    def __init__(self, nsamples, S_NG, w, t,  CDF, mu, sigma, parameter1, parameter2, beta=1.0):
        self.s_ng = S_NG
        self.w = w
        self.t = t
        self.nw = len(w)
        self.nt = len(t)
        self.dw = w[1] - w[0]
        self.dw = t[1] - t[0]
        self.nsamples = nsamples
        self.CDF = CDF
        self.mu = mu
        self.sigma = sigma
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.beta = beta
        self.s_g = self.get_s_g_from_s_ng()
        self.g_samples = self.generate_g_samples()
        self.ng_samples = self.translate_g_samples()

    def get_s_g_from_s_ng(self):
        s_g = itam_srm(self.s_ng, self.beta, self.w, self.t, self.CDF, self.mu, self.sigma, self.parameter1, self.parameter2)
        return s_g

    def generate_g_samples(self):
        g_samples = SRM(self.s_g, self.nsamples, self.dw, self.dt, self.nw, 'uni')
        return g_samples

    def translate_g_samples(self):
        samples_ng = translate_process(self.g_samples, self.CDF, self.mu, self.sigma, self.parameter1, self.parameter2)
        return samples_ng