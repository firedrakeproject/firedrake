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
ERA data set handling routines
"""

import datetime
import os
import unittest

import fluidity.diagnostics.debug as debug

try:
    import Scientific.IO.NetCDF as netcdf
except:
    debug.deprint("Warning: Failed to import Scientific.IO.NetCDF module")

import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.structured_fields as structured_fields


class Era15:

    """
    Class for handling ERA15 10m wind data
    """

    # Zero point for time data
    _epoch = datetime.datetime(1900, 1, 1, 0, 0, 0)

    def __init__(self, filename, latitudeName="latitude",
                 longitudeName="longitude", timeName="time"):
        self._file = netcdf.NetCDFFile(filename, "r")
        self._latitudes = self.Values(latitudeName)
        self._longitudes = self.Values(longitudeName)
        self._times = self.Values(timeName)

        # Longitude wrap-around
        self._longitudes.append(self._longitudes[0] + 360.0)

        debug.dprint(self)

        return

    def __del__(self):
        self._file.close()

        return

    def __str__(self):
        return "Latitudes: " + str(len(self._latitudes)) + "\n" + \
               "Longitudes: " + str(len(self._longitudes) - 1) + "\n" + \
               "Times: " + str(len(self._times))

    def File(self):
        """
        Return the raw NetCDF file handle for the ERA15 data file
        """

        return self._file

    def Variable(self, name):
        """
        Return the raw NetCDF variable object for the suppled field
        """

        return self._file.variables[name]

    def VariableLen(self, name):
        """
        Return the length of the supplied field
        """

        return len(self.Variable(name))

    def Units(self, name):
        """
        Return the units of the supplied field
        """

        return self.Variable(name).units

    def AddOffset(self, name):
        """
        Return the offset for the supplied field
        """

        variable = self.Variable(name)
        if hasattr(variable, "add_offset"):
            assert(len(variable.add_offset) == 1)
            return float(variable.add_offset)
        else:
            return 0.0

    def ScaleFactor(self, name):
        """
        Return the scale factor for the supplied field
        """

        variable = self.Variable(name)
        if hasattr(variable, "scale_factor"):
            assert(len(variable.scale_factor) == 1)
            return float(variable.scale_factor)
        else:
            return 1.0

    def FillValue(self, name):
        """
        Return the value denoting "fill value" for the supplied field
        """

        variable = self.Variable(name)
        if hasattr(variable, "fill_value"):
            assert(len(variable.fill_value) == 1)
            return float(variable.fill_value)
        else:
            return calc.Nan()

    def MissingValue(self, name):
        """
        Return the value denoting "missing value" for the supplied field
        """

        variable = self.Variable(name)
        if hasattr(variable, "missing_value"):
            assert(len(variable.missing_value) == 1)
            return float(variable.missing_value)
        else:
            return calc.Nan()

    def Value(self, name, index, *indices):
        """Index into the supplied field, applying offset, scale and fill and
        missing value detection. Permits indexing into the value directly by
        supplying additional indices."""

        variable = self.Variable(name)

        value = variable[index]
        for index in indices:
            value = value[index]
        value = float(value)

        if calc.IsNan(value):
            raise Exception("Nan value")
        elif value == self.FillValue(name):
            raise Exception("Fill value")
        elif value == self.MissingValue(name):
            raise Exception("Missing value")

        return value * self.ScaleFactor(name) + self.AddOffset(name)

    def Values(self, name):
        """
        Return the supplied field as a list
        """

        return [self.Value(name, i) for i in range(self.VariableLen(name))]

    def FieldNames(self):
        """Return a list of all fields in the NetCDF data file (including
        latitude, longitude and time)."""

        return self._file.variables.keys()

    def DatetimeToHours(self, time):

        delta = time - self._epoch

        return delta.days * 24.0 + delta.seconds / 3600.0

    def InterpolatedValue(self, name, latitude, longitude, time):
        """
        Tri-linearly interpolate the supplied field at the supplied latitude,
        longitude and time
        """

        if latitude > self._latitudes[0] or latitude < self._latitudes[-1]:
            debug.deprint("latitude = " + str(latitude))
            debug.deprint("Valid latitude range = " + str(
                (self._latitudes[-1], self._latitudes[0])))
            raise Exception("Invalid latitude")
        while longitude < 0.0:
            longitude += 360.0
        while longitude > 360.0:
            longitude -= 360.0

        if time < self._times[0] or time > self._times[-1]:
            debug.deprint("time = " + str(time))
            debug.deprint("Valid time range = " + str(
                (self._times[0], self._times[-1])))
            raise Exception("Invalid time")

        left = structured_fields.IndexBinarySearch(longitude, self._longitudes)
        right = left + 1
        assert(not right == left)
        longRatio = ((longitude - self._longitudes[left]) /
                     (self._longitudes[right] - self._longitudes[left]))
        left %= len(self._longitudes) - 1
        right %= len(self._longitudes) - 1

        upper = structured_fields.IndexBinaryDecSearch(
            latitude, self._latitudes)
        lower = min(upper + 1, len(self._latitudes) - 1)
        if upper == lower:
            latRatio = 1.0
        else:
            latRatio = ((latitude - self._latitudes[lower]) /
                        (self._latitudes[upper] - self._latitudes[lower]))

        before = structured_fields.IndexBinarySearch(time, self._times)
        after = min(before + 1, len(self._times) - 1)
        if after == before:
            timeRatio = 1.0
        else:
            timeRatio = (time - self._times[before]) / \
                (self._times[after] - self._times[before])

        timeVals = [None, None]
        for i, timeIndex in enumerate([before, after]):
            timeVals[i] = calc.BilinearlyInterpolate(
                self.Value(name, timeIndex, upper, left), self.Value(
                    name, timeIndex, upper, right),
                self.Value(name, timeIndex, lower, left), self.Value(
                    name, timeIndex, lower, right),
                longRatio, latRatio)

        return calc.LinearlyInterpolate(timeVals[0], timeVals[1], timeRatio)

    def U10(self, latitude, longitude, time, name="10u"):
        """
        Tri-linearly interpolate the zonal 10m wind at the supplied latitude,
        longitude and time
        """

        return self.InterpolatedValue(name, latitude, longitude, time)

    def V10(self, latitude, longitude, time, name="10v"):
        """Tri-linearly interpolate the meridional 10m wind at the supplied
        latitude, longitude and time."""

        return self.InterpolatedValue(name, latitude, longitude, time)


class eraUnittests(unittest.TestCase):

    def testNetcdfSupport(self):
        import Scientific.IO.NetCDF  # noqa: testing
        return


class eraDataUnittests(unittest.TestCase):

    def testEra15(self):
        filename = os.path.join(os.path.dirname(__file__), os.path.pardir,
                                "test-data", "ERA15", "output.nc")
        data = Era15(filename)
        data.U10(0.0, 0.0, data.DatetimeToHours(
            datetime.datetime(1992, 1, 1, 0, 0, 0)))
        data.V10(0.0, 0.0, data.DatetimeToHours(
            datetime.datetime(1992, 1, 1, 0, 0, 0)))

        return
