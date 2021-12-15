import gdspy


class GdsAssistant(object):
    def __init__(self, libname='default'):
        gdspy.current_library = gdspy.GdsLibrary(name=libname)
        self.lib = gdspy.current_library

    def save(self, fullpath):
        self.lib.write_gds(fullpath)

    def new_cell(self, name):
        return self.lib.new_cell(name)

    def add_childs_to_cell(self, parent, *childs, **kwargs):
        for child in childs:
            parent.add(gdspy.CellReference(child, **kwargs))
        return parent

    def get_gds_polygons(self, group, layer=0, datatype=0):
        gpolygons = []
        for pol in group:
            points = pol.vertices
            gpolygons.append(gdspy.Polygon(points, layer=layer, datatype=datatype))
        return gpolygons

    def boolean(self, *args, **kwargs):
        return gdspy.boolean(*args, **kwargs)

if __name__ == '__main__':
    pass