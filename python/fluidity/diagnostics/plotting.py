#!/usr/bin/env python

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301  USA

"""
Graph plotting utilities
"""

import os
import tempfile
import unittest

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.gui as gui
import fluidity.diagnostics.utils as utils

if not gui.GuiDisabledByEnvironment():
    try:
        import matplotlib
    except ImportError:
        debug.deprint("Warning: Failed to import matplotlib module")


def MatplotlibSupport():
    return "matplotlib" in globals()

if MatplotlibSupport():
    matplotlib.use("Cairo")
    try:
        import matplotlib.backends.backend_gtk
    except:
        debug.deprint(
            "Warning: Failed to import matplotlib.backends.backend_gtk module")

if not gui.GuiDisabledByEnvironment():
    try:
        import gtk
    except:
        debug.deprint("Warning: Failed to import gtk module")
    try:
        import pylab
    except ImportError:
        debug.deprint("Warning: Failed to import pylab module")

import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.optimise as optimise


class Plot:

    """
    Class defining a pylab plot
    """

    def __init__(self, plot):
        self._plot = None
        self._SetPlot(plot)

        return

    def __del__(self):
        self._CloseFigure()

        return

    def _CloseFigure(self):
        if not self._plot is None:
            pylab.close(self._figure)

        return

    def _NewFigure(self):
        self._CloseFigure()
        self._figure = pylab.figure()

        return

    def _SetPlot(self, plot):
        self._plot = plot

        return

    def Widget(self, withToolbar=True):
        """
        Return a GTK widget for the current plot
        """

        figureWidget = matplotlib.backends.backend_gtk.FigureCanvasGTK(
            self._figure)
        if withToolbar:
            vBox = gtk.VBox()
            toolbarWidget = matplotlib.backends.backend_gtk.NavigationToolbar2GTK(
                figureWidget, None)
            vBox.pack_start(figureWidget)
            vBox.pack_end(toolbarWidget, expand=False, fill=False)
            return vBox
        else:
            return figureWidget

    def Save(self, filename):
        """
        Save the plot as an image
        """

        self._figure.savefig(filename)

        return


class ScatterPlot(Plot):

    """
    A pylab scatter plot
    """

    def __init__(self, x, y, xLabel=None, yLabel=None, bounds=None):
        self._plot = None
        self.SetData(x, y)

        if not xLabel is None:
            self.SetXLabel(xLabel)

        if not yLabel is None:
            self.SetYLabel(yLabel)

        if not bounds is None:
            self.SetBounds(bounds)

        return

    def SetData(self, x, y):
        assert(len(x) == len(y))

        self._NewFigure()
        self._SetPlot(pylab.scatter(x=x, y=y, marker="+"))

        return

    def SetLegend(self, labels):
        assert(not self._plot is None)

        self._plot.axes.legend(labels, loc="lower left")

        return

    def SetXLabel(self, label):
        assert(not self._plot is None)

        self._plot.axes.set_xlabel(label)

        return

    def SetXLinear(self):
        assert(not self._plot is None)

        self._plot.axes.set_xscale("linear")

        return

    def SetXLog(self):
        assert(not self._plot is None)

        self._plot.axes.set_xscale("log")

        return

    def SetYLabel(self, label):
        assert(not self._plot is None)

        self._plot.axes.set_ylabel(label)

        return

    def SetYLinear(self):
        assert(not self._plot is None)

        self._plot.axes.set_yscale("linear")

        return

    def SetYLog(self):
        assert(not self._plot is None)

        self._plot.axes.set_yscale("log")

        return

    def GetAxisBounds(self):
        axis = self.get_axes()[0]

        return axis.get_xbound(), axis.get_ybound()

    def SetBounds(self, bounds):
        assert(self._plot is not None)

        self._plot.axes.set_xbound(bounds[0])
        self._plot.axes.set_ybound(bounds[1])

        return

# Class written using code from Patrick Farrell


class LinePlot(ScatterPlot):

    def SetData(self, x, y):
        assert(len(x) == len(y))

        if len(y) > 0 and utils.CanLen(y[0]):
            fields = len(y[0])
            if optimise.DebuggingEnabled():
                for comp in y[1:]:
                    assert(len(comp) == fields)

            yData = []
            for i in range(fields):
                yData.append([y[j][i] for j in range(len(y))])
        else:
            fields = 1
            yData = [y]

        self._NewFigure()

        colours = []
        for i in range(fields):
            colourNumber = 256 * 256 * 256 * i / fields

            colourComp = str(hex(colourNumber % 256))[2:]
            while len(colourComp) < 2:
                colourComp = "0" + colourComp
            colour = colourComp
            colourNumber /= 256

            colourComp = str(hex(colourNumber % 256))[2:]
            while len(colourComp) < 2:
                colourComp = "0" + colourComp
            colour = colourComp + colour
            colourNumber /= 256

            colourComp = str(hex(colourNumber % 256))[2:]
            while len(colourComp) < 2:
                colourComp = "0" + colourComp
            colour = "#" + colourComp + colour

            colours.append(colour)

        args = []
        for i in range(fields):
            args += [x, yData[i], colours[i]]

        pylab.plot(*args)
        self._SetPlot(pylab.gca())

        return


class ContourPlot(Plot):

    """
    A pylab contour plot
    """

    def __init__(self, x, y, z, levels=None):
        self._plot = None
        self.SetData(x, y, z, levels=levels)

        return

    def SetData(self, x, y, z, levels=None):
        assert(len(z) == len(y))
        if optimise.DebuggingEnabled():
            for val in z:
                assert(len(val) == len(x))

        self._NewFigure()
        if levels is None:
            self._SetPlot(pylab.contourf(x, y, z))
        else:
            self._SetPlot(pylab.contourf(x, y, z, levels))

        return


class plottingUnittests(unittest.TestCase):

    def testPylabSupport(self):
        import matplotlib
        import matplotlib.backends.backend_gtk  # noqa: testing
        import pylab  # noqa: testing

        self.assertTrue(MatplotlibSupport())

        return

    def testScatterPlot(self):
        plot = ScatterPlot([0.0, 1.0], [1.0, 2.0], xLabel="x", yLabel="y")
        # Set data again, to test figure close
        plot.SetData([0.0, 1.0], [1.0, 2.0])

        tempDir = tempfile.mkdtemp()
        plot.Save(os.path.join(tempDir, "temp.png"))
        plot.Save(os.path.join(tempDir, "temp.svg"))
        filehandling.Rmdir(tempDir, force=True)

        self.assertTrue(isinstance(plot.Widget(), gtk.Widget))

        return

    def testLinePlot(self):
        LinePlot([0.0, 1.0], [1.0, 2.0])
        LinePlot([0.0, 1.0], [[0.0, 1.0], [1.0, 2.0]])
        return

    def testContourPlot(self):
        ContourPlot([0.0, 1.0], [1.0, 2.0], [[2.0, 3.0], [4.0, 5.0]])
        return
