import unittest
from idtpy.shapes import *
import numpy.testing as npt


class TestRectangle(unittest.TestCase):
    def test_inputs(self):
        rect = Rectangle(dx=10, dy=50)
        self.assertEqual(rect.dx, 10)
        self.assertEqual(rect.dy, 50)
        self.assertEqual(rect.center, (0, 0))
        rect = Rectangle(dx=10, dy=50, center=(-5, 5))
        self.assertEqual(rect.center, (-5, 5))
        self.assertRaises(ValueError, Rectangle, -10, 10)
        self.assertRaises(ValueError, Rectangle, 10, -10)
        self.assertRaises(ValueError, Rectangle, 10, 10, (2,))
        self.assertRaises(ValueError, Rectangle, 10, 10, 'random')

    def test_vertices(self):
        rect = Rectangle(dx=10, dy=50, center=(-5, 5))
        self.assertEqual(rect.vertices[0], (-10, -20))
        self.assertEqual(rect.vertices[1], (-10, 30))
        self.assertEqual(rect.vertices[2], (0, 30))
        self.assertEqual(rect.vertices[3], (0, -20))

    def test_boundaries(self):
        rect = Rectangle(dx=10, dy=50, center=(-5, 5))
        self.assertTrue((rect.boundary[0] == (-10, -10, 0, 0, -10)).all())
        self.assertTrue((rect.boundary[1] == (-20, 30, 30, -20, -20)).all())
        xmin, ymin, xmax, ymax = rect.bounds
        self.assertEqual(xmin, -10)
        self.assertEqual(ymin, -20)
        self.assertEqual(xmax, 0)
        self.assertEqual(ymax, 30)

        self.assertEqual(rect.top, (-5, 30))
        self.assertEqual(rect.bottom, (-5, -20))
        self.assertEqual(rect.left, (-10, 5))
        self.assertEqual(rect.right, (0, 5))

    def test_property_setters(self):
        rect = Rectangle(dx=10, dy=50, center=(-5, 5))
        rect.center = (10, -8)
        self.assertEqual(rect.center, (10, -8))
        rect.dx = 2
        self.assertEqual(rect.dx, 2)
        rect.dy = 8
        self.assertEqual(rect.dy, 8)

    def test_translate(self):
        rect = Rectangle(dx=2, dy=4, center=(0, 0))
        rect.translate(dx=2)
        self.assertEqual(rect.center, (2, 0))
        rect.translate(dy=10)
        self.assertEqual(rect.center, (2, 10))
        rect.translate(dx=1, dy=0.5)
        self.assertEqual(rect.center, (3, 10.5))

    def test_extend(self):
        rect = Rectangle(dx=2, dy=4, center=(0, 0))
        r1 = rect.copy()
        r1.extend(top=2)
        self.assertEqual(r1.dy, 6)
        self.assertEqual(r1.dx, 2)
        self.assertEqual(r1.center, (0, 1))

        r1 = rect.copy()
        r1.extend(bottom=2)
        self.assertEqual(r1.dy, 6)
        self.assertEqual(r1.dx, 2)
        self.assertEqual(r1.center, (0, -1))

        r1 = rect.copy()
        r1.extend(left=2)
        self.assertEqual(r1.dy, 4)
        self.assertEqual(r1.dx, 4)
        self.assertEqual(r1.center, (-1, 0))

        r1 = rect.copy()
        r1.extend(right=2)
        self.assertEqual(r1.dy, 4)
        self.assertEqual(r1.dx, 4)
        self.assertEqual(r1.center, (1, 0))

        r1 = rect.copy()
        r1.extend(top=-2)
        self.assertEqual(r1.dy, 2)
        self.assertEqual(r1.dx, 2)
        self.assertEqual(r1.center, (0, -1))

    def test_reflect(self):
        rect = Rectangle(dx=2, dy=4, center=(-5, 5))
        rect.reflect(x=True)
        self.assertEqual(rect.center, (-5, -5))
        rect.reflect(y=True)
        self.assertEqual(rect.center, (5, -5))
        rect.reflect(x=True, y=True)
        self.assertEqual(rect.center, (-5, 5))


class TestGroup(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        r1 = Rectangle(dx=2, dy=4, center=(0, 0))
        r2 = r1.copy()
        r1.translate(dx=-5, dy=0)
        r2.translate(dx=0, dy=2)
        g = Group([r1, r2])
        self._g = g

    def test_boundaries(self):
        g = self._g.copy()
        self.assertEqual(len(g.polygons), 2)
        self.assertEqual(g.center, (-2.5, 1))
        self.assertEqual(g.dx, 7)
        self.assertEqual(g.dy, 6)

        xmin, ymin, xmax, ymax = g.bounds
        self.assertEqual(xmin, -6)
        self.assertEqual(ymin, -2)
        self.assertEqual(xmax, 1)
        self.assertEqual(ymax, 4)

        self.assertEqual(g.xmin, -6)
        self.assertEqual(g.ymin, -2)
        self.assertEqual(g.xmax, 1)
        self.assertEqual(g.ymax, 4)

        self.assertEqual(g.top, (-2.5, 4.0))
        self.assertEqual(g.bottom, (-2.5, -2.0))
        self.assertEqual(g.left, (-6.0, 1.0))
        self.assertEqual(g.right, (1.0, 1.0))

    def test_translate(self):
        g = self._g.copy()
        g.translate(dx=2)
        self.assertEqual(g.center, (-0.5, 1))
        g.translate(dy=2)
        self.assertEqual(g.center, (-0.5, 3))

    def test_center(self):
        g = self._g.copy()
        self.assertEqual(g.center, (-2.5, 1))
        self.assertEqual(g[0].center, (-5, 0))
        self.assertEqual(g[1].center, (0, 2))
        g.center = (0, 0)
        self.assertEqual(g.center, (0, 0))
        self.assertEqual(g[0].center, (-2.5, -1))
        self.assertEqual(g[1].center, (2.5, 1))
        g.translate(dx=-1, dy=1)
        self.assertEqual(g.center, (-1, 1))

    def test_align(self):
        g = self._g.copy()
        self.assertRaises(ValueError, g.align, 'test')
        self.assertRaises(ValueError, g.align, 1)
        g.center = (0, 0)
        g.align(loc='t')
        self.assertEqual(g.center, (0, -3))
        g.align(loc='b')
        self.assertEqual(g.center, (0, 3))
        g.align(loc='l')
        self.assertEqual(g.center, (3.5, 0))
        g.align(loc='r')
        self.assertEqual(g.center, (-3.5, 0))
        g.align(loc='c')
        self.assertEqual(g.center, (0, 0))

    def test_reflect(self):
        r1 = Rectangle(dx=2, dy=4, center=(0, 0))
        r2 = r1.copy()
        r2.translate(dx=5, dy=2)
        g = Group([r1, r2])
        g1 = g.copy()
        g1.reflect(x=True, y=False, local=True)
        self.assertEqual(g1[0].center, (0, 2))
        self.assertEqual(g1[1].center, (5, 0))
        g1 = g.copy()
        g1.reflect(x=False, y=True, local=True)
        self.assertEqual(g1[0].center, (5, 0))
        self.assertEqual(g1[1].center, (0, 2))
        g1 = g.copy()
        g1.reflect(x=True, y=True, local=True)
        self.assertEqual(g1[0].center, (5, 2))
        self.assertEqual(g1[1].center, (0, 0))


if __name__ == '__main__':
    unittest.main()
