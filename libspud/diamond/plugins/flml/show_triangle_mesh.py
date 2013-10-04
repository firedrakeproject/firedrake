from diamond.plugins import register_plugin, cb_decorator
from lxml import etree

# For now, I've put TriangleReader in diamond
from diamond.triangle_reader import TriangleReader

from enthought.mayavi.core.engine import Engine # Error occurs when this is imported
from enthought.mayavi.modules.api import Surface
from enthought.mayavi.core.ui.engine_view import EngineView

def plugin_applies(xpath):
    # Allow plugin to be used at any element which is under any mesh
    return (xpath.startswith('/geometry/mesh'))

@cb_decorator

def handle_click(xml, xpath):
    xml_root = etree.fromstring(xml)

    # This check is needed in case xpath in Diamond is not updated upon changing an attribute
    # (can be removed once the bug in Diamond has been fixed)
    if (len(xml_root.xpath(xpath)) == 0):
        return

    # Track back along xpath to find mesh element
    element = xml_root.xpath(xpath)[0]
    while (element.tag != 'mesh'):
        element = element.getparent()

    # Check whether Triangle format is selected (can only show Triangle meshes)
    triangle_format_element = element.xpath('from_file/format[@name="triangle"]')
    if (triangle_format_element):
        try:
            # Exception occurs if file name is not set
            file_name = element.xpath('from_file')[0].attrib['file_name']
        except:
            return
    else:
        return

    # Find the number of dimensions so that we load the correct file
    ndim = int(xml_root.xpath('/fluidity_options/geometry/dimension')[0].getchildren()[0].text)
    show_mesh(file_name, ndim)

def show_mesh(file_name, ndim):
    mayavi_engine = Engine()
    mayavi_engine.start()
    mayavi_engine.new_scene()

    # View the MayaVi pipeline
    engine_view = EngineView(engine=mayavi_engine)
    ui = engine_view.edit_traits()

    # Setup MayaVi pipeline
    src = TriangleReader()

    if (ndim == 2):
        src.initialize(file_name+'.edge') # Load the 2D .edge file
    else:
        src.initialize(file_name+'.face') # Load the 3D .face file

    mayavi_engine.add_source(src)
    # Add any filters, modules here in the order that they are to appear in the pipeline
    mayavi_engine.add_module(Surface())

register_plugin(plugin_applies, "Show Mesh", handle_click)
