import unittest
from idtpy import model
import numpy as np
import numpy.testing as npt


class TestWaveform(unittest.TestCase):
    def test_args(self):
        idt = model.Waveform(T=10, phi0=0, t0=0)
        self.assertEqual(idt.T, 10.)
        self.assertEqual(idt.phi0, 0.)
        self.assertEqual(idt.t0, 0.)


class TestRegular(unittest.TestCase):
    def test_args(self):
        idt = model.Regular(f=1, T=10, phi0=0, t0=0)
        self.assertEqual(idt.f, 1)
        t = [-1, 0, 2, 10, 11]
        npt.assert_array_equal(idt._indexes(t), [False, True, True, True, False])
        npt.assert_array_equal(idt.freq(t), [0, 1, 1, 1, 0])
        tn = idt.electrode_positions()
        self.assertEqual(len(tn), 20)
        npt.assert_array_equal(tn, np.arange(0, 10, 10 / 20.))


class TestChirp(unittest.TestCase):
    def test_linear(self):
        idt = model.LinearChirp(fmin=1, fmax=3, T=10, phi0=0, t0=0)
        self.assertEqual(idt.fmin, 1)
        self.assertEqual(idt.fmax, 3)
        self.assertEqual(idt.B, 2)
        f = lambda fmin: model.LinearChirp(fmin=fmin, fmax=1, T=10)
        for fmin in [-1, 2]:
            self.assertRaises(ValueError, f, fmin)
        f = lambda fmax: model.LinearChirp(fmin=3, fmax=fmax, T=10)
        for fmax in [-1, 2]:
            self.assertRaises(ValueError, f, fmax)

        t = [-1, 0, 2, 10, 11]
        npt.assert_array_equal(idt._indexes(t), [False, True, True, True, False])

    def test_exp(self):
        idt = model.ExpChirp(fmin=1, fmax=3, T=10, phi0=0, t0=0)
        self.assertEqual(idt.fmin, 1)
        self.assertEqual(idt.fmax, 3)
        self.assertEqual(idt.B, 2)
        f = lambda fmin: model.ExpChirp(fmin=fmin, fmax=1, T=10)
        for fmin in [-1, 2]:
            self.assertRaises(ValueError, f, fmin)
        f = lambda fmax: model.ExpChirp(fmin=3, fmax=fmax, T=10)
        for fmax in [-1, 2]:
            self.assertRaises(ValueError, f, fmax)

        t = [-1, 0, 2, 10, 11]
        npt.assert_array_equal(idt._indexes(t), [False, True, True, True, False])


if __name__ == '__main__':
    unittest.main()
