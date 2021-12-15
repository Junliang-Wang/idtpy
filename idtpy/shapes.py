from copy import copy, deepcopy

import matplotlib.pyplot as plt
import numpy as np


class Rectangle(object):
    """Store and manipulate the parameters of a rectangle
    
    Args:
        dx (float): dx length
        dy (float): dy length
        center (tuple of 2): it contains x0 and y0
    """

    def __init__(self, dx, dy, center=(0, 0)):
        self._check_inputs(dx, dy, center)
        self._center = center
        self._dx = dx
        self._dy = dy

    @property
    def vertices(self):
        """Return 4 tuples of (x,y) corresponding to each vertex"""
        x0, y0 = self.center
        dx, dy = self.dx, self.dy
        points = [
            (x0 - dx / 2, y0 - dy / 2),
            (x0 - dx / 2, y0 + dy / 2),
            (x0 + dx / 2, y0 + dy / 2),
            (x0 + dx / 2, y0 - dy / 2),
        ]
        return points

    @property
    def boundary(self):
        """Return vertix coordinates for x and y"""
        x, y = [], []
        for vi in self.vertices:
            x.append(vi[0])
            y.append(vi[1])
        x.append(x[0])
        y.append(y[0])
        return np.array(x), np.array(y)

    @property
    def bounds(self):
        """Return xmin, ymin, xmax, ymax of the object"""
        xmin = self.left[0]
        ymin = self.bottom[1]
        xmax = self.right[0]
        ymax = self.top[1]
        return (xmin, ymin, xmax, ymax)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, coord):
        self._check_center(coord)
        self._center = coord

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, d):
        self._check_dx(d)
        self._dx = d

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, d):
        self._check_dy(d)
        self._dy = d

    @property
    def top(self):
        dx = 0
        dy = self.dy / 2
        return self._coord(dx, dy)

    @property
    def bottom(self):
        dx = 0
        dy = -self.dy / 2
        return self._coord(dx, dy)

    @property
    def left(self):
        dx = -self.dx / 2
        dy = 0
        return self._coord(dx, dy)

    @property
    def right(self):
        dx = +self.dx / 2
        dy = 0
        return self._coord(dx, dy)

    def _coord(self, dx=0, dy=0):
        x0, y0 = self.center
        return (x0 + dx, y0 + dy)

    def reflect(self, x=False, y=False):
        """Reflect along a x, y axis
        
        Args:
            x (bool): flip vertically
            y (bool): flip horizontally
        """
        x0, y0 = copy(self.center)
        if x is True:
            y0 = -y0
        if y is True:
            x0 = -x0
        self.center = (x0, y0)

        return self

    def translate(self, dx=0, dy=0):
        """Translate the object"""
        self.center = self._coord(dx, dy)
        return self

    def extend(self, top=0, bottom=0, left=0, right=0):
        """Add distance to each side"""
        self.dx = self.dx + left + right
        self.dy = self.dy + top + bottom
        x0, y0 = self.center
        self.center = (x0 + (right - left) / 2, y0 + (top - bottom) / 2)
        return self

    def show(self, ax=None, **pkwargs):
        """Convenient method to plot it with matplotlib"""
        if ax is None:
            fig, ax = plt.subplots(1)
        x, y = self.boundary
        ax.plot(x, y, **pkwargs)
        return ax

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        cname = self.__class__.__name__
        out = f'{cname}(dx={self.dx},dy={self.dy},center={self.center})'
        return out

    def copy(self):
        """Returns a copy of itself"""
        return deepcopy(self)

    def _check_inputs(self, dx, dy, center):
        self._check_dx(dx)
        self._check_dy(dy)
        self._check_center(center)

    def _check_dx(self, dx):
        if dx <= 0:
            raise ValueError(f'dx ({dx}) must be > 0')

    def _check_dy(self, dy):
        if dy <= 0:
            raise ValueError(f'dx ({dy}) must be > 0')

    def _check_center(self, center):
        if len(center) != 2:
            raise ValueError(f'center length ({len(center)}) must be 2')


class Group(object):
    """Store and manipulate a list of Rectangles

    Args:
        polygons (list): list of Rectangle instances
    """

    def __init__(self, polygons):
        self.polygons = self._copy_polygons(polygons)

    def _copy_polygons(self, polygons):
        copied = []
        for pol in polygons:
            copied.append(pol.copy())
        return copied

    @property
    def center(self):
        """Return the center of the group"""
        dx = self.xmin + self.dx / 2
        dy = self.ymin + self.dy / 2
        return (dx, dy)

    @center.setter
    def center(self, coord):
        x0, y0 = self.center
        x1, y1 = coord
        dx = x1 - x0
        dy = y1 - y0
        self.translate(dx, dy)

    @property
    def dx(self):
        """Return the dx length of the group"""
        return self.xmax - self.xmin

    @property
    def dy(self):
        """Return the dy length of the group"""
        return self.ymax - self.ymin

    @property
    def bounds(self):
        """Return xmin, ymin, xmax, ymax of the group"""
        limits = []
        for pol in self.polygons:
            limits.append(pol.bounds)
        limits = np.array(limits)
        xmin = np.min(limits[:, 0])
        ymin = np.min(limits[:, 1])
        xmax = np.max(limits[:, 2])
        ymax = np.max(limits[:, 3])
        return (xmin, ymin, xmax, ymax)

    @property
    def xmin(self):
        return self.bounds[0]

    @property
    def ymin(self):
        return self.bounds[1]

    @property
    def xmax(self):
        return self.bounds[2]

    @property
    def ymax(self):
        return self.bounds[3]

    @property
    def top(self):
        dx = 0
        dy = self.dy / 2
        return self._coord(dx, dy)

    @property
    def bottom(self):
        dx = 0
        dy = -self.dy / 2
        return self._coord(dx, dy)

    @property
    def left(self):
        dx = -self.dx / 2
        dy = 0
        return self._coord(dx, dy)

    @property
    def right(self):
        dx = +self.dx / 2
        dy = 0
        return self._coord(dx, dy)

    def translate(self, dx=0, dy=0):
        """Translate the group of Rectangles"""
        for pol in self.polygons:
            pol.translate(dx, dy)
        return self

    def reflect(self, x=False, y=False, local=True):
        """Reflect the group of Rectangles
        
        Args:
            x (bool): flip vertically
            y (bool): flip horizontally
            local (bool): if True, it reflects with respect to its center. Otherwise with respect to the global (0,0)
        """
        if local:
            self.reflect_local(x=x, y=y)
        else:
            self.reflect_global(x=x, y=y)
        return self

    def reflect_global(self, x=False, y=False):
        for pol in self.polygons:
            pol.reflect(x=x, y=y)
        return self

    def reflect_local(self, x=False, y=False):
        """Move to global (0,0), reflect and move back to the original center"""
        _center = copy(self.center)
        self.align('c')
        self.reflect_global(x=x, y=y)
        self.center = _center
        return self

    # def translate_center(self, dx=0, dy=0):
    #     x0, y0 = self.center
    #     self.center = (x0 + dx, y0 + dy)
    #     self.translate(-dx, -dy)
    #     return self

    def align(self, loc):
        """Translate the group such that the global (0,0) is at the relative location

        Args:
            loc (str): 't', 'b', 'l' or 'r'
        """
        loc = str(loc)
        loc = loc.lower()
        if loc in ['top', 't']:
            dx, dy = self.top
        elif loc in ['bottom', 'b']:
            dx, dy = self.bottom
        elif loc in ['left', 'l']:
            dx, dy = self.left
        elif loc in ['right', 'r']:
            dx, dy = self.right
        elif loc in ['center', 'c', 'middle', 'm']:
            dx, dy = self.center
        else:
            raise ValueError(f'location {loc} must be t, b, l, r or m')

        self.center = self._coord(-dx, -dy)
        return self

    def _coord(self, dx=0, dy=0):
        x0, y0 = self.center
        return (x0 + dx, y0 + dy)

    def copy(self):
        """Return a copy of itself"""
        return deepcopy(self)

    def show(self, ax=None, **pkwargs):
        """Convenient method to plot it with matplotlib"""
        if ax is None:
            fig, ax = plt.subplots(1)
        for pol in self.polygons:
            pol.show(ax, **pkwargs)
        return ax

    def __add__(self, element):
        """Return a Group with both polygons"""
        if hasattr(element, 'polygons'):
            return Group(self.polygons + element.polygons)
        else:
            raise ValueError(f'{element} is not valid for addition')

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            result = self.polygons[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return result

    def __getitem__(self, item):
        if not isinstance(item, (int, slice)):
            raise ValueError('Index is not valid')
        return self.polygons[item]

    def __setitem__(self, item, value):
        if not isinstance(item, (int, slice)):
            raise ValueError('Index is not valid')
        if isinstance(value, Rectangle):
            self.polygons[item] = value

    def __delitem__(self, item):
        if not isinstance(item, (int, slice)):
            raise ValueError('Index is not valid')
        del self.polygons[item]

    def __str__(self):
        out = self.__repr__()
        for pol in self.polygons:
            out += pol.__str__() + '\n'
        return out

    def __repr__(self):
        cname = self.__class__.__name__
        out = f'{cname}(...) with {len(self.polygons)} polygons\n'
        return out


def array_rectangle(n, fdx, fdy, fcenter):
    rects = [Rectangle(fdx(i), fdy(i), fcenter(i)) for i in range(n)]
    return rects


def counter(i=0, add=1):
    while True:
        yield i
        i += add


def gen_array_rectangle(fdx, fdy, fx0, fy0, counter=counter):
    counter = counter()
    while True:
        n = next(counter)
        yield Rectangle(fdx(n), fdy(n), (fx0(n), fy0(n)))


if __name__ == '__main__':
    pass
