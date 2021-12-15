import numpy as np
from idtpy.utils import sampling, exponenial_func, to_db, find_nearest_idx
from copy import deepcopy

class Waveform(object):
    """
    Base model class for IDTs

    Args:
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        phi0 (float): phase offset
        t0 (float): time offset
    """

    def __init__(self, T, phi0=0, t0=0):
        self.T = float(T)
        self.phi0 = float(phi0)
        self.t0 = float(t0)

    def phase(self, t):
        return np.zeros_like(t)

    def amplitude(self, t, amax=1, amin=0):
        t = np.array(t)
        amp = np.ones(t.size) * amin
        amp[self._indexes(t)] = amax
        return amp

    def electrode_positions(self, shp=1):
        return np.array([0])

    def time_response(self, t):
        t = np.array(t)
        ph = self.phase(t)
        amp = self.amplitude(t)
        t_resp = np.exp(-ph * 1j) * amp
        return t_resp

    def freq_response(self, f, shp=1, apodized=False, db=True):
        tn = self.electrode_positions(shp=shp)
        ampn = self.amplitude(tn)

        f_resp = np.zeros_like(f, dtype='complex')
        counts = range(len(tn))
        polarity = [sampling(i, [+1] * shp + [-1] * shp) for i in counts]
        omega = f * 2 * np.pi

        for i, wi in enumerate(omega):
            exp = np.exp(-1j * wi * tn) * ampn
            hi = polarity * exp
            hi = np.sum(hi)
            f_resp[i] = hi
        if not apodized:
            f_resp *= np.power(f, 1 / 2.)
        if db:
            f_resp = to_db(f_resp, masked=False)
        return f_resp

    def time(self, dt=0.01):
        limit = self.T + dt / 2
        time = np.arange(0 + self.t0, limit + self.t0, dt)
        return time

    def freq(self, t):
        pass

    def ideal_compression(self, dt=0.01, centered=True):
        conv = convolve_wfs(self, self, dt)
        time, signal = conv.t, conv.values
        if centered:
            time -= self.T
        return time, signal

    def apply_waveform(self, wf, dt=0.01):
        return convolve_wfs(self, wf, dt)

    def _indexes(self, t):
        t = np.array(t)
        t = t - self.t0
        return np.logical_and(t >= 0, t <= self.T)

    def copy(self):
        return deepcopy(self)

    def __call__(self, t):
        return self.time_response(t)


class Regular(Waveform):
    """
    Model for regular IDT

    Args:
        f (float): resonant frequency
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        phi0 (float): phase offset
        t0 (float): time offset
    """

    def __init__(self, f, T, phi0=0, t0=0):
        super().__init__(T=T, phi0=phi0, t0=t0)
        f = float(f)
        if f <= 0:
            raise ValueError('f must be >= 0')
        self.f = f

    def freq(self, t):
        t = np.array(t)
        ft = np.array([self.f] * len(t))
        idxs = np.logical_not(self._indexes(t))
        ft[idxs] = 0
        return ft

    def phase(self, t):
        t = np.array(t)
        t = t - self.t0
        phi = 2. * np.pi * (self.f * t)
        phi += self.phi0
        return phi

    def electrode_positions(self, shp=1):
        f, T, phi0, t0 = self.f, self.T, self.phi0, self.t0
        se = shp * 2
        ni = -phi0 * se / (2 * np.pi)
        nf = np.ceil(ni + se * T / 2 * (2 * f))
        n = np.arange(ni, nf, 1)
        tpos = n / (se * f) + phi0 / (2 * np.pi * f)
        return tpos + t0


class LinearChirp(Waveform):
    """
    Model for linear chirp IDT
    In a linear chirp IDT, the frequency modulation f(t) is proportional to time t.

    Args:
        fmin (float): initial frequency
        fmax (float): final frequency
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        phi0 (float): phase offset
        t0 (float): time offset
    """

    def __init__(self, fmin, fmax, T, phi0=0, t0=0):
        super().__init__(T=T, phi0=phi0, t0=t0)
        fmin = float(fmin)
        fmax = float(fmax)
        if fmin <= 0:
            raise ValueError('fmin must be >= 0')
        if fmax <= 0:
            raise ValueError('fmax must be >= 0')
        if fmin > fmax:
            raise ValueError('fmin must be < fmax')

        self.fmin = fmin
        self.fmax = fmax
        self.B = fmax - fmin
        self.fmid = self.fmin + self.B / 2.

    def freq(self, t):
        t = np.array(t)
        ft = self.fmin + self.B / self.T * (t - self.t0)
        idxs = np.logical_not(self._indexes(t))
        ft[idxs] = 0
        return ft

    def phase(self, t):
        t = np.array(t)
        t = t - self.t0
        if self.T > 0:
            phi = 2. * np.pi * self.fmin * t + np.pi * self.B / self.T * t ** 2
        else:
            phi = 2. * np.pi * (self.fmin * t)
        phi += self.phi0
        return phi

    def electrode_positions(self, shp=1):
        fmin, B, T, phi0, t0 = self.fmin, self.B, self.T, self.phi0, self.t0
        se = shp * 2
        ni = -phi0 * se / (2 * np.pi)
        nf = np.ceil(ni + se * T / 2 * (2 * fmin + B))
        n = np.arange(ni, nf, 1)
        if B != 0:
            sqr = np.sqrt(1 + 2 * n * B / (T * fmin ** 2 * se) + B * phi0 / (T * fmin ** 2))
            tpos = fmin * T / B * (-1 + sqr)
        else:
            tpos = n / (se * fmin) + phi0 / (2 * np.pi * fmin)
        return tpos + t0


class ExpChirp(Waveform):
    """
    Model for exponential chirp IDT
    In an exponential chirp IDT, the frequency is uniformly increasing (i.e. f_n - f_{n+1} is constant).
    This leads to an exponential dependency of the frequency modulation f(t) with respect to time t.

    Args:
        fmin (float): initial frequency
        fmax (float): final frequency
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        phi0 (float): phase offset
        t0 (float): time offset
    """

    def __init__(self, fmin, fmax, T, phi0=0, t0=0):
        super().__init__(T=T, phi0=phi0, t0=t0)
        fmin = float(fmin)
        fmax = float(fmax)
        if fmin <= 0:
            raise ValueError('fmin must be >= 0')
        if fmax <= 0:
            raise ValueError('fmax must be >= 0')
        if fmin > fmax:
            raise ValueError('fmin must be < fmax')

        self.fmin = fmin
        self.fmax = fmax
        self.B = fmax - fmin
        self.fmid = self.fmin + self.B / 2.
        if self.B == 0:
            reg_idt = Regular(fmin, T, phi0)
            self.phase = reg_idt.phase
            self.freq = reg_idt.freq
            self.time = reg_idt.time
            self.amplitude = reg_idt.amplitude
            self.electrode_positions = reg_idt.electrode_positions
        else:
            self.fit()

    def fit(self):
        fn, tn = self._design_parameters(self.fmin, self.fmax, self.T)
        A, tau, C = self._fit_exp(fn, tn)
        self.A, self.tau, self.C = A, tau, C
        self._tn = tn
        self._fn = fn

    def phase(self, t):
        t = np.array(t)
        t = t - self.t0
        phase = 2. * np.pi * self.A * self.tau * (np.exp(t / self.tau) - 1.)
        phase += 2. * np.pi * self.C * t
        phase += self.phi0
        return phase

    def freq(self, t):
        t = np.array(t)
        t = t - self.t0
        return exponenial_func(t, A=self.A, tau=self.tau, C=self.C)

    def electrode_positions(self, shp=2):
        pos = []
        sp = 2 * shp
        for ti, fi in zip(self._tn, self._fn):
            period = 1 / fi
            dt = period / sp
            for i in range(sp):
                pos.append(ti + i * dt)
        pos = np.array(pos)
        return pos + self.t0

    @staticmethod
    def _design_parameters(fmin, fmax, T, getNp=False):
        " === 1. Find number of steps for frequency === "
        if fmin <= 0:
            print("Minimum frequency is less or equal to 0.")
            return
        Ttemp = -1
        n = 2
        fi = np.linspace(fmin, fmax, n)
        while Ttemp < T:
            fi = np.linspace(fmin, fmax, n)
            Ttemp = np.sum(1 / fi)
            n += 1

        " === 2. Get delay time === "
        td = np.zeros(n - 1)
        for i, f in enumerate(fi):
            td[i] = np.sum(1 / fi[0:i])
        if getNp == False:
            return fi, td
        elif getNp == True:
            n = n - 1
            return fi, td, n

    @staticmethod
    def _fit_exp(fi, td):
        from scipy.optimize import curve_fit
        tau = td[1] / np.log(fi[1] / fi[0])
        popt, pcov = curve_fit(exponenial_func, td, fi, p0=(fi[0], tau, 0))
        return popt


def convolve_wfs(wf1, wf2, dt=0.01):
    ht1 = wf1.time_response(wf1.time(dt)).real
    ht2 = wf2.time_response(wf2.time(dt)).real
    conv = convolve(ht1, ht2)
    time_total = wf1.T + wf2.T
    offset = np.abs(wf1.t0 - wf2.t0)
    time = np.linspace(0, time_total, len(conv)) + offset
    return Convolution(time, conv)


def convolve_wfs_list(waveforms, dt=0.01):
    waveforms = list(waveforms)
    wf1 = waveforms.pop(0)
    wf2 = waveforms.pop(0)
    conv = convolve_wfs(wf1, wf2, dt)
    for wfi in waveforms:
        conv = conv.apply_waveform(wfi, dt)
    return conv


def convolve(a, b, mode='full'):
    a = np.flip(a, 0)
    conv = np.convolve(a, b, mode)
    return conv


def convolve_with_time(a, b, tspan_a, tspan_b, t0_a, t0_b):
    conv = convolve(a, b, 'full')
    time_total = tspan_a + tspan_b
    offset = np.abs(t0_a - t0_b)
    time = np.linspace(0, time_total, len(conv)) + offset
    return time, conv


class Convolution(Waveform):
    def __init__(self, time, values):
        T = np.abs(time[0] - time[-1])
        t0 = time[0]
        super().__init__(T=T, phi0=0, t0=t0)
        self.t = time
        self.values = values
        self.points = len(values)

    def time_response(self, t):
        t = np.array(t)
        ht = np.zeros_like(t)
        idxi = find_nearest_idx(t, self.t0)
        idxf = idxi + self.points
        if idxf > ht.size:
            points = ht.size - idxi
            ht[idxi:] = self.values[:points]
        else:
            ht[idxi:idxf] = self.values
        return ht

    def time(self, dt=0.01):
        return np.arange(self.t[0], self.t[-1], dt)

    @property
    def x(self):
        return self.t

    @property
    def y(self):
        return self.values

    def xy(self, centered=False):
        if centered:
            x = self.t - self.T / 2
        else:
            x = self.t
        return x, self.values

if __name__ == '__main__':
    pass