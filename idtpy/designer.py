from itertools import islice

import numpy as np

from idtpy import model
from idtpy.shapes import Rectangle, gen_array_rectangle, Group
from idtpy.utils import sampling


class IDT(Group):
    """Base class for other IDTs which is derived from Group.

    Args:
        xn (function): x center position of each electrode
        yn (function): y center position of each electrode
        dxn (function): dx of each electrode
        wn (function): overlap of each electrode
        cun (function): upper contact boolean for each electrode. 'True' creates the contact
        cdn (function): lower contact boolean for each electrode. 'True' creates the contact
        yspan (function): total dy of each electrode
        Ne (int): total number of electrodes
    """

    def __init__(self, xn, yn, dxn, wn, cun, cdn, yspan, Ne):
        for arg in [xn, yn, dxn, wn, cun, cdn, yspan]:
            if not callable(arg):
                raise TypeError(f'{arg.__name__} must be callable')
        Ne = int(Ne)
        if Ne <= 0:
            raise ValueError('Ne must be > 0')
        self.xn = xn
        self.yn = yn
        self.dxn = dxn
        self.wn = wn
        self.cun = cun
        self.cdn = cdn
        self.yspan = yspan
        self.Ne = Ne
        self.clen = lambda n: (self.yspan(n) - self.wn(n)) / 2

        electrodes = self.generate_electrodes()
        super().__init__(electrodes)

    @property
    def electrodes(self):
        """Returs the polygons of the Group object"""
        return self.polygons

    def generate_electrodes(self):
        fdx = self.dxn
        fx0 = self.xn

        def fdy(n):
            dy = self.wn(n)
            if self.cun(n) is True:
                dy += self.clen(n)
            if self.cdn(n) is True:
                dy += self.clen(n)
            return dy

        def fy0(n):
            y0 = self.yn(n)
            if self.cun(n) is True:
                y0 += self.clen(n) / 2
            if self.cdn(n) is True:
                y0 -= self.clen(n) / 2
            return y0

        grects = gen_array_rectangle(fdx, fdy, fx0, fy0, )
        rects = list(islice(grects, self.Ne))
        return rects

    def dummies(self, gap):
        """Return dummy fingers of the IDT

        Args:
            gap (function,int,float): gap between the electrode and the dummy fingers (per each electrode if it is a function)

        Returns:
            Group instance of the dummy fingers
        """
        if isinstance(gap, (int, float)):
            gapn = lambda n: gap
        elif callable(gap):
            gapn = gap
        else:
            raise ValueError('argument gapn must be int, float or a function f(n)')

        fdx = self.dxn
        fx0 = self.xn

        def fdy(n, fbool):
            dy = self.clen(n)
            if fbool(n) is True:
                dy -= gapn(n)
            return dy

        def fy0(n, fbool, polarity=1):
            y0 = self.yn(n)
            if fbool(n) is True:
                y0 += (self.wn(n) + self.clen(n) + gapn(n)) / 2 * polarity
            return y0

        rects = []
        for ud, pol in zip([self.cun, self.cdn], [-1, 1]):
            fdyi = lambda n: fdy(n, ud)
            fy0i = lambda n: fy0(n, ud, pol)
            grects = gen_array_rectangle(fdx, fdyi, fx0, fy0i)
            bools = list(map(ud, range(self.Ne)))
            rects += list(np.array(list(islice(grects, self.Ne)))[bools])

        rects = Group(rects)
        rects.center = self.center
        return rects

    def add_dummies(self, gap):
        """Convenient method to generate and add dummy fingers to the IDT

        Args:
            gap (function,int,float): gap between the electrode and the dummy fingers (per each electrode if it is a function)

        Returns:
            Group instance of the dummy fingers
        """
        dummies = self.dummies(gap)
        self.polygons += dummies.polygons
        return dummies


class Ground(Group):
    def __init__(self, dx, dy, center=(0, 0)):
        rect = Rectangle(dx, dy, center)
        super().__init__([rect])


class Regular(IDT):
    """Regular or uniform IDT

    Args:
        freq (float): resonant frequency
        vsaw (float): SAW speed for calculating the wavelengths
        Np (float): number of periods. Must be multiples of 0.5
        w (float): constant overlap
        l (float): vertical distance from the end of the overlap to the end of the IDT
        Nehp (int): number of electrodes per half period. 1=single-finger, 2=double-finger...
        tfact (float, list of 3): factor for finger width (see the example below)

    Note:
        tfact: If it is list of 3 --> [iniFactor, finalFactor, #fingers on one side].
        Ex: [0.8,0.6,20] means that the first 20 fingers will gradually have 80% to 60% finger width. Symmetrically for the last fingers.
    """

    def __init__(self, freq, vsaw, Np, w=30, l=50, Nehp=2, tfact=1):
        self.freq = freq
        self.vsaw = vsaw
        self.Np = Np
        self.w = w
        self.l = l
        self.Nehp = Nehp
        self.tfact = tfact

        wlen = vsaw / freq
        Nep = 2 * Nehp
        Ne = int(Np * Nep)
        dxn_corr = thickness_correction(tfact, Ne)
        kwargs = dict(
            xn=lambda n: wlen / Nep * n,
            yn=lambda n: 0,
            dxn=lambda n: wlen / (2 * Nep) * dxn_corr[n],
            wn=lambda n: w,
            cun=lambda n: sampling(n, [True] * Nehp + [False] * Nehp),
            cdn=lambda n: sampling(n, [False] * Nehp + [True] * Nehp),
            yspan=lambda n: w + l * 2,
            Ne=Ne,
        )

        super().__init__(**kwargs)

    def contacts(self, gap, dx_factor=3.):
        if self.Nehp == 1:
            rects = self.contacts_single(gap, dx_factor)
        elif self.Nehp > 1:
            rects = self.contacts_n(gap, dx_factor)
        return rects

    def contacts_n(self, gap, dx_factor=1.):
        rects = []
        m = self.Nehp - 1
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            try:
                cu = all([self.cun(n) == self.cun(n + i) for i in range(m + 1)])
                cd = all([self.cdn(n) == self.cdn(n + i) for i in range(m + 1)])
                dx = np.abs(self.xn(n + m) - self.xn(n)) * dx_factor
            except:
                continue
            x0 = (self.xn(n + m) + self.xn(n)) / 2.
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True and cu is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True and cd is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects

    def contacts_single(self, gap, dx_factor=2.):
        rects = []
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            dx = self.dxn(n) * dx_factor
            x0 = self.xn(n)
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects


class LinearChirp(IDT):
    """Linear chirp IDT class

    Args:
        fmin (float): minimum frequency
        fmax (float): maximum frequency
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        vsaw (float): SAW speed for calculating the wavelengths
        w (float): constant overlap
        l (float): vertical distance from the end of the overlap to the end of the IDT
        Nehp (int): number of electrodes per half period. 1=single-finger, 2=double-finger...
        tfact (float, list of 3): factor for finger width (see the example below)

    Note:
        tfact: If it is list of 3 --> [iniFactor, finalFactor, #fingers on one side].
        Ex: [0.8,0.6,20] means that the first 20 fingers will gradually have 80% to 60% finger width. Symmetrically for the last fingers.
    """

    def __init__(self, fmin, fmax, T, vsaw, w=30, l=100, Nehp=2, tfact=1):
        self.fmin = fmin
        self.fmax = fmax
        self.vsaw = vsaw
        self.Nehp = Nehp
        self.w = w
        self.l = l
        self.tfact = tfact

        lin = model.LinearChirp(fmin, fmax, T)
        tn = lin.electrode_positions(shp=Nehp)
        freqn = lin.freq(tn)
        wlen = vsaw / freqn
        Ne = len(tn)
        dxn_corr = thickness_correction(tfact, Ne)
        kwargs = dict(
            xn=lambda n: vsaw * tn[n],
            yn=lambda n: 0,
            dxn=lambda n: wlen[n] / (4 * Nehp) * dxn_corr[n],
            wn=lambda n: w,
            cun=lambda n: sampling(n, [True] * Nehp + [False] * Nehp),
            cdn=lambda n: sampling(n, [False] * Nehp + [True] * Nehp),
            yspan=lambda n: w + l * 2,
            Ne=Ne,
        )
        super().__init__(**kwargs)

    def contacts(self, gap, dx_factor=3.):
        if self.Nehp == 1:
            rects = self.contacts_single(gap, dx_factor)
        elif self.Nehp > 1:
            rects = self.contacts_n(gap, dx_factor)
        return rects

    def contacts_n(self, gap, dx_factor=1.):
        rects = []
        m = self.Nehp - 1
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            try:
                cu = all([self.cun(n) == self.cun(n + i) for i in range(m + 1)])
                cd = all([self.cdn(n) == self.cdn(n + i) for i in range(m + 1)])
                dx = np.abs(self.xn(n + m) - self.xn(n)) * dx_factor
            except:
                continue
            x0 = (self.xn(n + m) + self.xn(n)) / 2.
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True and cu is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True and cd is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects

    def contacts_single(self, gap, dx_factor=2.):
        rects = []
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            dx = self.dxn(n) * dx_factor
            x0 = self.xn(n)
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects


class ExpChirp(IDT):
    """Exponential chirp IDT class

    Args:
        fmin (float): minimum frequency
        fmax (float): maximum frequency
        T (float): length in time. Equivalent to SAW travelling time to cross the IDT from end to end
        vsaw (float): SAW speed for calculating the wavelengths
        w (float): constant overlap
        l (float): vertical distance from the end of the overlap to the end of the IDT
        Nehp (int): number of electrodes per half period. 1=single-finger, 2=double-finger...
        tfact (float, list of 3): factor for finger width (see the example below)

    Note:
        tfact: If it is list of 3 --> [iniFactor, finalFactor, #fingers on one side].
        Ex: [0.8,0.6,20] means that the first 20 fingers will gradually have 80% to 60% finger width. Symmetrically for the last fingers.
    """

    def __init__(self, fmin, fmax, T, vsaw, w=30, l=100, Nehp=2, tfact=1):
        self.fmin = fmin
        self.fmax = fmax
        self.vsaw = vsaw
        self.Nehp = Nehp
        self.w = w
        self.l = l
        self.tfact = tfact

        mod = model.ExpChirp(fmin, fmax, T)
        tn = mod.electrode_positions(shp=Nehp)
        freqn = mod._fn
        wlen = vsaw / freqn
        wlen = [wi for wi in wlen for _ in range(2 * Nehp)]
        Ne = len(tn)
        dxn_corr = thickness_correction(tfact, Ne)
        kwargs = dict(
            xn=lambda n: vsaw * tn[n],
            yn=lambda n: 0,
            dxn=lambda n: wlen[n] / (4 * Nehp) * dxn_corr[n],
            wn=lambda n: w,
            cun=lambda n: sampling(n, [True] * Nehp + [False] * Nehp),
            cdn=lambda n: sampling(n, [False] * Nehp + [True] * Nehp),
            yspan=lambda n: w + l * 2,
            Ne=Ne,
        )
        super().__init__(**kwargs)

    def contacts(self, gap, dx_factor=3.):
        if self.Nehp == 1:
            rects = self.contacts_single(gap, dx_factor)
        elif self.Nehp > 1:
            rects = self.contacts_n(gap, dx_factor)
        return rects

    def contacts_n(self, gap, dx_factor=1.):
        rects = []
        m = self.Nehp - 1
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            try:
                cu = all([self.cun(n) == self.cun(n + i) for i in range(m + 1)])
                cd = all([self.cdn(n) == self.cdn(n + i) for i in range(m + 1)])
                dx = np.abs(self.xn(n + m) - self.xn(n)) * dx_factor
            except:
                continue
            x0 = (self.xn(n + m) + self.xn(n)) / 2.
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True and cu is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True and cd is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects

    def contacts_single(self, gap, dx_factor=2.):
        rects = []
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            dx = self.dxn(n) * dx_factor
            x0 = self.xn(n)
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if self.cun(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
            if self.cdn(n) is True:
                rect = Rectangle(dx, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects


class Split52(IDT):
    """Split52 IDT

    Args:
        freq (float): resonant frequency
        vsaw (float): SAW speed for calculating the wavelengths
        Np (float): number of periods. Must be multiples of 0.5
        w (float): constant overlap
        l (float): vertical distance from the end of the overlap to the end of the IDT
        tfact (float, list of 3): factor for finger width (see the example below)

    Note:
        tfact: If it is list of 3 --> [iniFactor, finalFactor, #fingers on one side].
        Ex: [0.8,0.6,20] means that the first 20 fingers will gradually have 80% to 60% finger width. Symmetrically for the last fingers.
    """

    def __init__(self, freq, vsaw, Np, w=30, l=100, tfact=1):
        self.freq = freq
        self.vsaw = vsaw
        self.Np = Np
        self.w = w
        self.l = l
        self.tfact = tfact

        wlen = vsaw / freq;
        self.wlen = wlen
        Nep = 5;
        self.Nep = Nep
        Ne = int(Np * Nep);
        self.Ne = Ne
        dxn_corr = thickness_correction(tfact, Ne)
        kwargs = dict(
            xn=lambda n: wlen / Nep * n,
            yn=lambda n: 0,
            dxn=lambda n: wlen / (Nep * 2) * dxn_corr[n],
            wn=lambda n: w,
            cun=lambda n: sampling(n, [True, True, False, False, False]),
            cdn=lambda n: sampling(n, [False, False, True, False, True]),
            yspan=lambda n: w + l * 2,
            Ne=Ne,
        )

        super().__init__(**kwargs)

    def contacts(self, gap, dx_factor=1.):
        rects = []
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            try:
                dx_small = np.abs(self.xn(n + 1) - self.xn(n)) * dx_factor
                dx_large = np.abs(self.xn(n + 2) - self.xn(n)) * dx_factor
            except:
                continue
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if n % self.Nep == 0:
                x0 = (self.xn(n + 1) + self.xn(n)) / 2.
                rect = Rectangle(dx_small, dy, (x0, y0 + dy0))
                rects.append(rect)
            elif n % self.Nep == 2:
                x0 = (self.xn(n + 2) + self.xn(n)) / 2.
                rect = Rectangle(dx_large, dy, (x0, y0 - dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects


class Dart(IDT):
    """Unidirection IDT of DART type

    Args:
        freq (float): resonant frequency
        vsaw (float): SAW speed for calculating the wavelengths
        Np (float): number of periods. Must be multiples of 0.5
        w (float): constant overlap
        l (float): vertical distance from the end of the overlap to the end of the IDT
        tfact (float, list of 3): factor for finger width (see the example below)
        direction (str): 'l' (left) or 'r' (right)

    Note:
        tfact: If it is list of 3 --> [iniFactor, finalFactor, #fingers on one side].
        Ex: [0.8,0.6,20] means that the first 20 fingers will gradually have 80% to 60% finger width. Symmetrically for the last fingers.
    """

    def __init__(self, freq, vsaw, Np, w=30, l=100, tfact=1, direction='r'):
        self.freq = freq
        self.vsaw = vsaw
        self.Np = Np
        self.w = w
        self.l = l
        self.tfact = tfact
        self.direction = direction.lower()

        wlen = vsaw / freq;
        self.wlen = wlen
        Nep = 3;
        self.Nep = Nep
        Ne = int(Np * Nep);
        self.Ne = Ne
        dxn_corr = thickness_correction(tfact, Ne)

        kwargs = dict(
            xn=self._xn,
            yn=lambda n: 0,
            dxn=lambda n: self._dxn(n) * dxn_corr[n],
            wn=lambda n: w,
            cun=lambda n: sampling(n, [True, False, False]),
            cdn=lambda n: sampling(n, [False, True, True]),
            yspan=lambda n: w + l * 2,
            Ne=Ne,
        )

        super().__init__(**kwargs)

    def check_direction(self):
        if self.direction not in ['l', 'left', 'r', 'right']:
            raise ValueError(f'direction ({self.direction}) must be left or right')

    def _xn(self, n):
        if self.direction in ['r', 'right']:
            rel_pos = [0, 2, 5]
        elif self.direction in ['l', 'left']:
            rel_pos = [0, 3, 6]
        rel_pos = np.array(rel_pos) * self.wlen / 8
        dx = sampling(n, rel_pos)
        pos = n // self.Nep * self.wlen + dx
        return pos

    def _dxn(self, n):
        if self.direction in ['r', 'right']:
            rel_dx = [1, 1, 3]
        elif self.direction in ['l', 'left']:
            rel_dx = [1, 3, 1]

        rel_dx = np.array(rel_dx) * self.wlen / 8
        dx = sampling(n, rel_dx)
        return dx

    def contacts(self, gap, dx_factor=2.0):
        rects = []
        for n, finger in enumerate(self.electrodes):
            dy = self.clen(n) - gap
            dx = self.dxn(n) * dx_factor
            y0 = self.yn(n)
            dy0 = self.wn(n) / 2. + dy / 2. + gap
            if n % self.Nep == 0:
                x0 = self.xn(n)
                rect = Rectangle(dx, dy, (x0, y0 + dy0))
                rects.append(rect)
        rects = Group(rects)
        return rects


def thickness_correction(factor, points, method=np.linspace):
    correction = np.ones(points)
    try:
        if len(factor) == 3:
            i, f, nf = factor
            correction *= f
            correction[:nf] = method(i, f, nf)
            correction[-nf:] = method(f, i, nf)
        elif len(factor) == 1:
            correction *= factor
    except:
        correction *= factor
    return correction


def idt_contact(idt, dx=250, dy=60, loc='tl', gap=40, clen=25, ext=20):
    """
    Method to create contact rectangles for an IDT
    For more information, see idt_contact.pdf

    Args:
        idt (obj IDT): Instance of IDT
        dx (float): horizontal length from the end of the IDT for the contact
        dy (float): vertical length of the contact bar
        loc (str): location of the contact bar. Allowed: 'tl','tr','bl','br'. Ex: tl = top left
        gap (float): distance between contact and the ground
        clen (float): contact length = overlap of between the contact bar and the IDT
        ext (float): extension = extra horizontal length of the contact bar

    Returns:
        2 Group instances for positive and negative rectangles
    """
    dx = dx + idt.dx + ext
    loc = loc.lower()
    if 't' in loc:
        y0 = idt.ymax - clen + dy / 2
    else:
        y0 = idt.ymin + clen - dy / 2
    if 'l' in loc:
        x0 = idt.xmax + ext - dx / 2
    else:
        x0 = idt.xmin - ext + dx / 2

    cont = Rectangle(dx, dy, (x0, y0))
    ncont = cont.copy()
    ncont.extend(top=gap, bottom=gap, left=gap, right=gap)
    pos = Group([cont])
    neg = Group([ncont])
    return pos, neg


def det_contact(idt, sdx=25, h=100, clen=25, dy=60, left=200, right=200, gap=40, loc='t'):
    """
    Method to create a T-shape contact aimed for a few fingers IDT
    For more information, see det_contact.pdf

    Args:
        idt (obj IDT): Instance of IDT
        sdx (float): horizontal length of the vertical bar
        h (float): vertical length for the ground between the channel and the contact bar
        clen (float): contact length = overlap of between the contact bar and the IDT
        dy (float): vertical length of the contact bar
        left (float): horizontal length for the left side of the contact bar
        right (float): horizontal length for the right side of the contact bar
        gap (float): distance between contact and the ground
        loc (str): location of the contact bar. Allowed: 't'(top),'b'(bottom)

    Returns:
        2 Group instances for positive and negative rectangles
    """
    pos, neg = [], []
    if 't' in loc:
        polarity = 1
    else:
        polarity = -1
    sdy = h + gap
    scont = Rectangle(sdx, sdy, idt.center)
    scont.translate(0, (idt.dy / 2 + sdy / 2 - clen) * polarity)

    nscont = scont.copy()
    nscont.extend(left=gap, right=gap)

    cont = Rectangle(left + right + sdx, dy, scont.center)
    dx0 = (right - left) / 2
    dy0 = (scont.dy / 2 + dy / 2) * polarity
    cont.translate(dx0, dy0)

    ncont = cont.copy()
    ncont.extend(left=gap, right=gap, top=gap, bottom=gap)

    pos = Group([scont, cont])
    neg = Group([nscont, ncont])
    return pos, neg


def channel(idt1, idt2, clen=25, gap=20):
    """
    Method to create a rectangle for the channel between 2 IDTs
    For more information, see channel.pdf

    Args:
        idt1 (obj IDT): Instance of IDT
        idt2 (obj IDT): Instance of IDT
        clen (float): contact length = overlap of between the contact bar and the IDT
        gap (float): horizontal gap between metal and IDTs

    Returns:
        2 Group instances for positive and negative rectangles.
        Note: positive is empty
    """
    idts = idt1 + idt2
    ch = Rectangle(idts.dx, idts.dy, idts.center)
    ch.extend(top=-clen, bottom=-clen, left=gap, right=gap)
    pos, neg = Group([]), Group([ch])
    return pos, neg


def waveguide(dx, dy, gap, center=(0, 0)):
    p1 = Rectangle(dx, dy, center)
    n1 = p1.copy()
    n1.extend(left=gap, right=gap, top=gap, bottom=gap)
    pos = Group([p1])
    neg = Group([n1])
    return pos, neg


if __name__ == '__main__':
    pass