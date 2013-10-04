#!/usr/bin/env python
#
# James Maddison

"""
Plot data in a .stat file
"""

import getopt
import sys
import time

import gtk

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.fluiditytools as fluidity_tools
import fluidity.diagnostics.gui as gui
import fluidity.diagnostics.plotting as plotting


def Help():
    debug.dprint("Usage: statplot [OPTIONS] FILENAME [FILENAME ...]\n" +
                 "\n" +
                 "Options:\n" +
                 "\n" +
                 "-h  Display this help\n" +
                 "-v  Verbose mode", 0)

    return

try:
    opts, args = getopt.getopt(sys.argv[1:], "hv")
except getopt.GetoptError:
    Help()
    sys.exit(-1)

if not ("-v", "") in opts:
    debug.SetDebugLevel(0)

if ("-h", "") in opts:
    Help()
    sys.exit(1)

if len(args) == 0:
    debug.FatalError("Filename must be specified")


class StatplotWindow(gtk.Window):

    def __init__(self, filenames):
        assert(len(filenames) > 0)

        self._filenames = filenames

        gtk.Window.__init__(self)
        self.set_title(self._filenames[-1])
        self.connect("key-press-event", self._KeyPressed)

        self._ReadData()

        # Containers
        self._vBox = gtk.VBox()
        self.add(self._vBox)

        self._hBox = gtk.HBox()
        self._vBox.pack_end(self._hBox, expand=False, fill=False)

        # The plot widget
        self._xField = None
        self._yField = None
        self._xData = None
        self._yData = None
        self._plotWidget = None
        self._plotType = plotting.LinePlot

        # The combos
        paths = self._stat.Paths()
        paths.sort()
        self._xCombo = gui.ComboBoxFromEntries(paths)
        self._xCombo.connect("changed", self._XComboChanged)
        if "ElapsedTime" in paths:
            iter = self._xCombo.get_model().get_iter(
                (paths.index("ElapsedTime"),))
        else:
            iter = self._xCombo.get_model().get_iter_first()
        if not iter is None:
            self._xCombo.set_active_iter(iter)
        self._hBox.pack_start(self._xCombo)

        self._yCombo = gui.ComboBoxFromEntries(paths)
        self._yCombo.connect("changed", self._YComboChanged)
        iter = self._yCombo.get_model().get_iter_first()
        if not iter is None:
            iter2 = self._yCombo.get_model().iter_next(iter)
            if iter2 is None:
                self._yCombo.set_active_iter(iter)
            else:
                self._yCombo.set_active_iter(iter2)
        self._hBox.pack_end(self._yCombo)

        self._vBox.show_all()

        return

    def _ReadData(self):
        stats = []
        for i, filename in enumerate(self._filenames):
            failcount = 0
            while failcount < 5:
                try:
                    stats.append(fluidity_tools.Stat(filename))
                    break
                except (TypeError, ValueError):
                    # We opened the .stat when it was being written to by
                    # fluidity
                    time.sleep(0.2)
                    failcount = failcount + 1
            if failcount == 5:
                raise Exception("Could not open %s" % filename)
        if len(stats) == 1:
            self._stat = stats[0]
        else:
            self._stat = fluidity_tools.JoinStat(*stats)

        return

    def _RefreshData(self, keepBounds=False):
        self._xField = self._xCombo.get_active_text()
        self._xData = self._stat[self._xField]
        self._yField = self._yCombo.get_active_text()
        self._yData = self._stat[self._yField]
        if keepBounds:
            axis = self._plotWidget.get_children()[0].figure.get_axes()[0]
            bounds = (axis.get_xbound(), axis.get_ybound())
        else:
            bounds = None
        self._RefreshPlot(bounds)

        return

    def _RefreshPlot(self, bounds=None, xscale=None, yscale=None):
        if not self._xData is None and not self._yData is None:
            assert(len(self._xData) == len(self._yData))
            if not self._plotWidget is None:
                self._vBox.remove(self._plotWidget)

                axis = self._plotWidget.get_children()[0].figure.get_axes()[0]
                if xscale is None:
                    xscale = axis.get_xscale()
                if yscale is None:
                    yscale = axis.get_yscale()
            else:
                if xscale is None:
                    xscale = "linear"
                if yscale is None:
                    yscale = "linear"

            self._plotWidget = self._plotType(x=self._xData, y=self._yData,
                                              xLabel=self._xField,
                                              yLabel=self._yField).Widget()
            axis = self._plotWidget.get_children()[0].figure.get_axes()[0]
            axis.set_xscale(xscale)
            axis.set_yscale(yscale)
            if not bounds is None:
                axis.set_xbound(bounds[0])
                axis.set_ybound(bounds[1])

            self._vBox.pack_start(self._plotWidget)
            self._plotWidget.show_all()

        return

    def SetXField(self, field):
        self._xField = field
        self._xData = self._stat[self._xField]

        self._RefreshPlot()

        return

    def SetYField(self, field):
        self._yField = field
        self._yData = self._stat[self._yField]

        self._RefreshPlot()

        return

    def _XComboChanged(self, widget):
        self.SetXField(self._xCombo.get_active_text())

        return

    def _YComboChanged(self, widget):
        self.SetYField(self._yCombo.get_active_text())

        return

    def _KeyPressed(self, widget, event):
        char = event.string
        if char == "R":
            self._ReadData()
            self._RefreshData(keepBounds=True)
        elif char == "l":
            self._plotType = plotting.LinePlot
            self._RefreshData(keepBounds=True)
        elif char == "x":
            scale = self._plotWidget.get_children()[
                0].figure.get_axes()[0].get_xscale()
            if scale == "linear":
                self._RefreshPlot(xscale="log")
            else:
                self._RefreshPlot(xscale="linear")
        elif char == "y":
            scale = self._plotWidget.get_children()[
                0].figure.get_axes()[0].get_yscale()
            if scale == "linear":
                self._RefreshPlot(yscale="log")
            else:
                self._RefreshPlot(yscale="linear")
        elif char == "q":
            self.destroy()
        elif char == "r":
            self._ReadData()
            self._RefreshData()
        elif char == "s":
            self._plotType = plotting.ScatterPlot
            self._RefreshData(keepBounds=True)

        return

# The window
window = StatplotWindow(args)
window.set_default_size(640, 480)

# Fire up the GUI
gui.DisplayWindow(window)
