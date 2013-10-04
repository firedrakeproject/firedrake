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
Some mathematical routines
"""

import copy
import math
import unittest

import fluidity.diagnostics.debug as debug

try:
    import numpy
except ImportError:
    debug.deprint("Warning: Failed to import numpy module")
try:
    import scipy.linalg
except ImportError:
    debug.deprint("Warning: Failed to import scipy.linalg module")
try:
    import scipy.stats
except ImportError:
    debug.deprint("Warning: Failed to import scipy.stats module")
try:
    import vtk
except ImportError:
    debug.deprint("Warning: Failed to import vtk module")

import fluidity.diagnostics.optimise as optimise
import fluidity.diagnostics.utils as utils


def NumpySupport():
    return "numpy" in globals()


def LinearlyInterpolate(lower, upper, ratio):
    """
    Linearly interpolate between the given values by the given ratio
    """

    return (upper - lower) * ratio + lower


def BilinearlyInterpolate(uLeft, uRight, bLeft, bRight, xRatio, yRatio):
    """
    Bilinearly interpolate between the given values by the given ratios
    """

    left = LinearlyInterpolate(bLeft, uLeft, yRatio)
    right = LinearlyInterpolate(bRight, uRight, yRatio)

    return LinearlyInterpolate(left, right, xRatio)


def LinearlyInterpolateField(v, xDonor, xTarget):
    """
    Linearly interpolate the given 1D field between the donor and target
    coordinates (which are assumed sorted in increasing order)
    """

    assert(len(v) == len(xDonor))
    assert(len(v) > 1)
    assert(len(xTarget) > 1)
    assert(xTarget[0] >= xDonor[0])
    assert(xTarget[-1] <= xDonor[-1])

    result = numpy.empty(len(xTarget))
    upper = 1
    for i, x in enumerate(xTarget):
        while x > xDonor[upper]:
            upper += 1
        lower = upper - 1
        ratio = (x - xDonor[lower]) / (xDonor[upper] - xDonor[lower])
        result[i] = LinearlyInterpolate(v[lower], v[upper], ratio)

    return result


def Se(vals, returnMean=False):
    """
    Compute the standard error of the supplied array
    """

    mean = MeanVal(vals)
    se = 0.0
    for val in vals:
        se += math.pow(val - mean, 2)
    se /= float(len(vals) - 1)
    se = math.sqrt(se)

    if returnMean:
        return mean, se
    else:
        return se


def DeterminantCE(matrix):
    """Return the determinant of the supplied square matrix using cofactor
    expansion."""

    dim = len(matrix[0])

    if dim == 0:
        return 0.0
    elif dim == 1:
        return matrix[0][0]

    det = 0.0
    sign = 1.0

    for i in range(dim):
        if len(matrix[i]) != dim:
            raise Exception(
                "Unable to calculate determinant of non-square matrix")
        newMatrix = []
        for j in range(dim - 1):
            newMatrix.append([])
            for k in range(dim):
                if k != i:
                    newMatrix[j].append(matrix[j + 1][k])

        det += sign * matrix[0][i] * DeterminantCE(newMatrix)

        sign *= -1.0

    return det


def DeterminantLU(matrix):
    """
    Return the determinant of the supplied square matrix by LU decomposition
    """

    dim = len(matrix[0])

    det = 1.0

    for i in range(dim):
        if len(matrix[i]) != dim:
            raise Exception(
                "Unable to calculate determinant of non-square matrix")
        if matrix[i][i] == 0.0:
            for j in range(i + 1, dim + 1):
                if j == dim:
                    return 0.0
                if matrix[j][i] != 0.0:
                    rowStore = matrix[i]
                    matrix[i] = matrix[j]
                    matrix[j] = rowStore
                    det *= -1.0
                    break
        det *= matrix[i][i]
        for j in range(i + 1, dim):
            rowFactor = matrix[j][i] * 1.0 / matrix[i][i]
            for k in range(dim):
                matrix[j][k] -= matrix[i][k] * rowFactor

    return det


def DeterminantNumpy(matrix):
    """Return the determinant of the supplied square matrix using numpy, via
    LAPACK z/dgetrf (LU factorisation)."""

    a = numpy.array(matrix)
    M = len(matrix)
    a.shape = (M, M)

    return numpy.linalg.det(a)

if NumpySupport():
    # Default to determinant using numpy
    Determinant = DeterminantNumpy
else:
    # Otherwise, drop back to custom LU decomposition
    Determinant = DeterminantLU


def Eigendecomposition(matrix, returnEigenvectors=False):
    """
    Peform an eigendecomposition of the supplied matrix
    """

    a = numpy.array(matrix)
    a.shape = (a.shape[0], a.shape[0])

    return scipy.linalg.eig(a, right=returnEigenvectors)


def EvalPoly(x, poly):
    """
    Evaluation a polynomial at the supplied point
    """

    # Naiive implementation will do for now
    val = 0.0
    for i, coeff in enumerate(poly):
        val += coeff * math.pow(x, len(poly) - i - 1)

    return val


def EvalPolyRange(min, max, poly, N=1000):
    """
    Evaluate a polynomial over the supplied range with supplied coefficients
    """

    X = numpy.array([min + (max - min) * ((float(i) / float(N - 1)))
                    for i in range(N)])
    vals = numpy.empty(N)

    for i, x in enumerate(X):
        vals[i] = EvalPoly(x, poly)

    return X, vals


def NormalisedFft(values, divisions=None, returnPhases=False):
    """
    Perform a normalised fast Fourier transform of the supplied values.
    More efficient if divisions is a power of 2
    """

    if divisions is None:
        divisions = len(values)

    fft = numpy.fft.rfft(values, n=divisions) / divisions

    if returnPhases:
        amp = numpy.array([abs(val) for val in fft])
        phases = numpy.array(
            [numpy.arctan2(val.imag, val.real) for val in fft])

        return amp, phases
    else:
        return abs(fft)


def UnwrappedPhases(phases, min=-math.pi):
    """
    Undo 2-pi wrap-around in a phase time series
    """

    unwrappedPhases = numpy.empty(len(phases))
    unwrappedPhases[0] = phases[0]
    add = 0.0
    for i in range(1, len(unwrappedPhases)):
        if abs(phases[i] - math.pi - min) > 0.5 * math.pi:
            if phases[i] > min + math.pi and phases[i - 1] < min + math.pi:
                add -= 2.0 * math.pi
            elif phases[i] < min + math.pi and phases[i - 1] > min + math.pi:
                add += 2.0 * math.pi
        unwrappedPhases[i] = phases[i] + add

    return unwrappedPhases


class NormalisedLombScargle:

    """
    Class defining a normalised Lomb-Scargle periodogram.
    """

    def __init__(self, x, t):
        assert(len(x) == len(t))

        self._x = copy.deepcopy(x)
        self._t = copy.deepcopy(t)

        self._t.sort()

        self._initialise_math()
        self._initialise_parameters()

        return

    def _initialise_math(self):
        """
        Detect math routines
        """

        if NumpySupport():
            # Use numpy if we can, as the math calls are performance critical
            self._cos = numpy.cos
            self._sin = numpy.sin
            self._atan = numpy.arctan
            self._atan2 = numpy.arctan2
        else:
            # Otherwise, fall back to math
            debug.deprint(
                "Warning: Unable to use numpy module in Lomb-Scargle periodogram")
            self._cos = math.cos
            self._sin = math.sin
            self._atan = math.atan
            self._atan2 = math.atan2

        return

    def _initialise_parameters(self):
        """Initialise the parameters used in evaluating the Lomb-Scargle
        periodogram."""

        self._mean = MeanVal(self._x)

        self._variance = 0.0
        for x in self._x:
            self._variance += (x - self._mean) ** 2
        self._variance /= float(self.DataPointsCount() - 1)

        debug.dprint("Mean = " + str(self._mean), 3)
        debug.dprint("Variance = " + str(self._variance), 3)

        return

    def DataPointsCount(self):
        """
        Return the number of data points
        """

        return len(self._x)

    def Tau(self, omega):
        numerator = 0.0
        denomenator = 0.0
        for t in self._t:
            numerator += self._sin(2.0 * omega * t)
            denomenator += self._cos(2.0 * omega * t)

        return self._atan2(numerator, denomenator) / (2.0 * omega)

    def EvaluatePoint(self, omega):
        """Evalulate the Lomb-Scargle periodogram at the given angular
        frequency. Performs a naiive calculation as in equation (3) of W H
        Press and G B Rybicki, The Astrophysical Journal, 388:277-280, 1989
        (see also (10) of J D Scargle, The Astrophysical Journal, 263:835-853,
        1982)."""

        if AlmostEquals(omega, 0.0):
            return 0.0

        tau = self.Tau(omega)

        A = 0.0
        B = 0.0
        C = 0.0
        D = 0.0

        for i in range(self.DataPointsCount()):
            valMinusMean = self._x[i] - self._mean
            omegaTMinusTau = omega * (self._t[i] - tau)

            c = self._cos(omegaTMinusTau)
            s = self._sin(omegaTMinusTau)

            A += valMinusMean * c
            B += c ** 2

            C += valMinusMean * s
            D += s ** 2

        A **= 2
        C **= 2

        # The power as defined in Press and Rybicki is:
        #   P_N = ((A / B) + (C / D)) / (2.0 * self._variance)
        # I prefer something more directly comparable with an equivalent
        # Fourier amplitude, as follows
        return math.sqrt(((A / B) + (C / D)) / (self.DataPointsCount())) / math.sqrt(2.0)

    def EvaluateDirect(self, omegas):
        """Evaluate the Lomb-Scargle periodogram at the supplied angular
        frequencies, using direct calculation. O len(omega)
        self.DataPointsCount()."""

        return numpy.array([self.EvaluatePoint(omega) for omega in omegas])

    def EvaluateExtirpilating(self, omegas):
        """Evaluate the Lomb-Scargle periodogram at the supplied angular
        frequencies, using the algorithm as in W H Press and G B Rybicki, The
        Astrophysical Journal, 388:277-280, 1989. This method is a place-holder
        for when/if this is implemented."""

        raise Exception("Not yet implemented")

    def Evaluate(self, omegas):
        """Evaluate the Lomb-Scargle periodogram at the supplied angular
        frequencies."""

        return self.EvaluateDirect(omegas)


def DominantModeStructured(amps, dt, N=250):
    """Compute the period and amplitude of the dominant mode in an even data
    series."""

    def Omegas(ts):
        return [2.0 * math.pi / t for t in ts]

    nScans = len(amps)
    times = [i * dt for i in range(nScans)]

    tMin = 2.0 * dt
    tMax = nScans * dt

    return DominantModeUnstructured(amps, times, N=N, tMin=tMin, tMax=tMax)


def DominantModeUnstructured(amps, times, N=250, tMin=None, tMax=None):
    """
    Compute the period and amplitude of the dominant mode in an uneven data
    series.
    """

    def Omegas(ts):
        return [2.0 * math.pi / t for t in ts]

    debug.dprint("Finding dominant mode")

    assert(len(amps) == len(times))
    if tMin is None:
        tMin = Inf()
        for i in range(len(times) - 1):
            tMin = min(tMin, times[i + 1] - times[i])
        tMin *= 2.0
    if tMax is None:
        tMax = 2.0 * (times[-1] - times[0])

    debug.dprint("Period bounds = " + str((tMin, tMax)), 1)

    ls = NormalisedLombScargle(amps, times)

    amp = 0.0
    t = 0.0
    while True:
        debug.dprint("Period bounds = " + str((tMin, tMax)), 2)

        ts = [tMin + (tMax - tMin) * float(i) / float(N - 1) for i in range(N)]
                 # Note the factor of two - we want the least squares harmonic
                 # fit amplitude, not the FFT amplitude
        lsAmps = 2.0 * ls.Evaluate(Omegas(ts))
        lastT = t
        amp = 0.0
        for i, lsAmp in enumerate(lsAmps):
            if lsAmp > amp:
                t = ts[i]
                amp = lsAmp
        if AlmostEquals(t, lastT):
            break
        tMin = t - (tMax - tMin) / float(N - 1)
        tMax = t + (t - tMin)
        N = 4

    debug.dprint("Period = " + str(t))
    debug.dprint("Amplitude = " + str(amp))

    debug.dprint("Done")

    return t, amp


def IndexBinaryLboundSearch(val, values, increasing=True):
    """
    Find the max (min) index into values such that values[index] <= val, where
    if increasing is True (False), assumes the values are in increasing
    (decreasing) order
    """

    if isinstance(values, numpy.ndarray):
        values = values.tolist()

    if not increasing:
        lValues = copy.deepcopy(values)
        values = lValues
        values.reverse()

    if optimise.DebuggingEnabled():
        testValues = copy.deepcopy(values)
        testValues.sort()
        assert(values == testValues)

    minLower = 0
    maxLower = len(values)
    lower = minLower
    while maxLower - minLower > 0:
        # Choose new bounds for lower
        if values[lower] <= val:
            minLower = lower
        else:
            maxLower = lower

        # Calculate a new guess for lower
        oldLower = lower
        lower = int(float(maxLower + minLower) / 2.0)
        if oldLower == lower:
            if maxLower - minLower <= 1:
                break
            else:
                lower += 1

    if increasing:
        return lower
    else:
        return len(values) - lower - 1


def IndexBinaryUboundSearch(val, values, increasing=True):
    """
    Find the min (max) index into values such that values[index] >= val, where
    if increasing is True (False), assumes the values are in increasing
    (decreasing) order
    """

    if isinstance(values, numpy.ndarray):
        values = values.tolist()

    if increasing:
        lValues = copy.deepcopy(values)
        values = lValues
        values.reverse()

    if optimise.DebuggingEnabled():
        testValues = copy.deepcopy(values)
        testValues.sort()
        testValues.reverse()
        assert(values == testValues)

    minUpper = 0
    maxUpper = len(values) - 1
    upper = maxUpper
    while maxUpper - minUpper > 0:
        # Choose new bounds for lower
        if values[upper] < val:
            maxUpper = upper
        else:
            minUpper = upper

        # Calculate a new guess for lower
        oldUpper = upper
        upper = int(float(maxUpper + minUpper) / 2.0)
        if oldUpper == upper:
            if maxUpper - minUpper <= 1:
                break
            else:
                upper -= 1

    if not increasing:
        return upper
    else:
        return len(values) - upper - 1


def SSA(v, n, J=1):
    """
    Perform a singular systems analysis of the supplied array of data. See:
      Inertia-Gravity Wave Generation by Baroclinic Instability, Tom Jacoby,
      First year report, AOPP, September 2007
    """

    N_T = len(v)
    N = N_T - (n - 1) * J

    shape = (n, N)
    debug.dprint("Assembling matrix for SSA, shape = " + str(shape))
    X = numpy.empty(shape)
    for i in range(N):
        for j in range(n):
            X[j, i] = v[i + j * J]
    theta = numpy.dot(X, X.transpose()) / float(N)
    del(X)

    debug.dprint("Performing eigendecomposition")
    return Eigendecomposition(theta, returnEigenvectors=True)


def InterpolatedSSA(v, t, N_T, n, J=1, t0=None, t1=None):
    """
    Perform a singular systems analysis of the supplied non-uniform data using
    linear interpolation
    """

    if t0 is None:
        t0 = t[0]
    if t1 is None:
        t1 = t[-1]

    dt = (t1 - t0) / float(N_T)
    debug.dprint("Interpolation dt: " + str(dt))

    lt = [t0 + i * dt for i in range(N_T)]
    lv = LinearlyInterpolateField(v, t, lt)
    print lv, lt

    return SSA(lv, n, J=J)


def LinearRegression(x, y, returnR=False, returnSe=False):
    """
    Linear regression for x-y data
    """

    m, c, r, two_tailed, se = scipy.stats.linregress(x, y)

    if returnR:
        if returnSe:
            return (m, c), r, se
        else:
            return (m, c), r
    else:
        if returnSe:
            return (m, c), se
        else:
            return (m, c)


def Deg2Rad(degrees):
    """
    Convert the supplied number of degrees to radians
    """

    return degrees * math.pi / 180.0


def Rad2Deg(radians):
    """
    Convert the supplied number of radians to degrees
    """

    return radians * 180.0 / math.pi


def IsEven(val):
    """
    Return whether the supplied value is an even number
    """

    val = int(round(val))

    return 2 * (val / 2) == val


def IsOdd(val):
    """
    Return whether the supplied value is an odd number
    """

    return not IsEven(int(val))


def Epsilon():
    """
    Return the smallest difference epsilon such that 1.0 + epsilon is
    distinguishable from 1.0
    """

    if not hasattr(Epsilon, "_epsilon"):
        # If we haven't yet calculated Epsilon, calculate it
        Epsilon._epsilon = 1.0
        for divide in [2.0, 1.75, 1.5, 1.25]:
            while True:
                newEpsilon = Epsilon._epsilon / divide
                if newEpsilon == Epsilon._epsilon:
                    break
                elif 1.0 + newEpsilon > 1.0:
                    Epsilon._epsilon = newEpsilon
                else:
                    break

    return Epsilon._epsilon


def Tiny():
    """
    Return the smallest positive float that is distinguishable from 0.0
    """

    if not hasattr(Tiny, "_tiny"):
        # If we haven't yet calculated Tiny, calculate it
        Tiny._tiny = 1.0
        for divide in [2.0, 1.75, 1.5, 1.25]:
            while True:
                newTiny = Tiny._tiny / divide
                if newTiny == Tiny._tiny:
                    break
                elif newTiny > 0.0:
                    Tiny._tiny = newTiny
                else:
                    break

    return Tiny._tiny


def Huge():
    """Return the largest positive integer that can be stored (before a long
    integer is needed)."""

    if not hasattr(Huge, "_huge"):
        # If we haven't yet calculated Huge, calculate it
        Huge._huge = 1
        while True:
            newHuge = Huge._huge * 2
            if isinstance(newHuge, int):
                Huge._huge = newHuge
            else:
                break
        add = Huge._huge / 2
        while True:
            newHuge = Huge._huge + add
            if isinstance(newHuge, int):
                Huge._huge = newHuge
            else:
                newAdd = add / 2
                if newAdd in [0, add]:
                    break
                else:
                    add = newAdd

    return Huge._huge


def Inf():
    """
    Return floating point inf
    """

    return float("inf")


def Nan():
    """
    Return floating point nan
    """

    return float("nan")


def IsInf(val):
    """
    Return whether the supplied valus is inf
    """

    if hasattr(math, "isinf"):
        return math.isinf(val)
    else:
        return val + val == val and val + 1 == val


def IsNan(val):
    """
    Return whether the supplied value is floating point nan
    """

    if hasattr(math, "isnan"):
        return math.isnan(val)
    else:
        return not val == val


def MaxVal(inputList):
    """
    Return the maximum value in the supplied list
    """

    maxval = -Inf()
    for val in inputList:
        maxval = max(maxval, val)

    return maxval


def MinVal(inputList):
    """
    Return the minimum value in the supplied list
    """

    minval = Inf()
    for val in inputList:
        minval = min(minval, val)

    return minval


def SumVal(inputList):
    """
    Return the sum of the supplied list
    """

    return sum(inputList)


def MeanVal(inputList):
    """
    Return the mean of the supplied list
    """

    return SumVal(inputList) / float(len(inputList))


def SumSquare(inputList):
    """
    Return the sum of the squares of entries in the supplied list
    """

    sum = 0.0
    for val in inputList:
        sum += val ** 2

    return sum


def L2Norm(inputList):
    """
    Return the norm of the supplied list
    """

    return math.sqrt(SumSquare(inputList))


def AlmostEquals(x, y, tolerance=Epsilon()):
    """
    Return whether the supplied values are almost equal, to within the supplied
    tolerance
    """

    if abs(x) < tolerance:
        return abs(x - y) < tolerance
    else:
        return abs(x - y) / abs(x) < tolerance


def RotatedVector(vector, angle, axis=None):
    """
    Rotate a vector
    """

    if len(vector) in [0, 1]:
        raise Exception(
            "Vector rotation doesn't make sense for 0-1 dimensions")
    elif len(vector) == 2:
        assert(axis is None)
        r = L2Norm(vector)
        angle += math.atan2(vector[1], vector[0])
        return [r * math.cos(angle), r * math.sin(angle)]
    elif len(vector) == 3:
        assert(not axis is None)
        assert(len(axis) == 3)
        if (hasattr(RotatedVector, "_angle") and RotatedVector._angle == angle
                and RotatedVector._axis == list(axis)):
            # If we have cached this rotation matrix, use the cache
            matrix = RotatedVector._matrix
        else:
            # Otherwise, generate a new transform and cache the rotation matrix
            transform = vtk.vtkTransform()
            transform.Identity()
            transform.RotateWXYZ(Rad2Deg(angle), axis[0], axis[1], axis[2])
            matrix = transform.GetMatrix()
            RotatedVector._angle = angle
            RotatedVector._axis = list(axis)
            RotatedVector._matrix = matrix
        return matrix.MultiplyPoint(list(vector) + [0.0])[:-1]
    else:
        raise Exception(
            "Vector rotation not implemented for " + str(len(vector)) + " dimensions")


def RotatedTensor(tensor, angle, axis=None):
    """
    Rotate a tensor
    """

    if len(tensor) == 0:
        return []

    if hasattr(tensor, "shape") and len(tensor.shape) == 1:
        size = int(numpy.sqrt(len(tensor)))
        assert(len(tensor) == size * size)
        tensor = [tensor[i * size:(i + 1) * size] for i in range(size)]

    raise Exception("Not yet implemented")


def CartesianVectorsToPolar(coordinates, data):
    """
    Project the supplied 2D Cartesian (x-y) vectors to polar coordinates
    (r-theta). coordinates must be in Cartesian.
    """

    if optimise.DebuggingEnabled():
        assert(len(coordinates) == len(data))
        for i, coord in enumerate(coordinates):
            assert(len(coord) == 2)
            assert(len(data[i]) == 2)

    newData = numpy.empty((len(data), 2))
    for i, coord in enumerate(coordinates):
        datum = data[i]

        rMag = L2Norm(coord[:2])
        r = [coord[0] / rMag, coord[1] / rMag]
        theta = [-r[1], r[0]]

        newData[i] = [datum[0] * r[0] + datum[1] *
                      r[1], datum[0] * theta[0] + datum[1] * theta[1]]

    return newData


def CartesianVectorsToCylindrical(coordinates, data):
    """Project the supplied 3D Cartesian (x-y-z) vectors to cylindrical
    coordinates (r-phi-z). coordinates must be in Cartesian."""

    if optimise.DebuggingEnabled():
        assert(len(coordinates) == len(data))
        for i, coord in enumerate(coordinates):
            assert(len(coord) == 3)
            assert(len(data[i]) == 3)

    newData = numpy.empty((len(data), 3))
    for i, coord in enumerate(coordinates):
        datum = data[i]

        rMag = L2Norm(coord[:2])
        r = [coord[0] / rMag, coord[1] / rMag]
        phi = [-r[1], r[0]]

        newData[i, :] = [datum[0] * r[0] + datum[1] * r[1],
                         datum[0] * phi[0] + datum[1] * phi[1], datum[2]]

    return newData


def CylindricalVectorsToCartesian(coordinates, data):
    """Project the supplied cylindrical coordinates (r-phi-z) vectors to 3D
    Cartesian (x-y-z). coordinates must be in Cartesian."""

    if optimise.DebuggingEnabled():
        assert(len(coordinates) == len(data))
        for i, coord in enumerate(coordinates):
            assert(len(coord) == 3)
            assert(len(data[i]) == 3)

    newData = numpy.empty((len(data), 3))
    for i, coord in enumerate(coordinates):
        datum = data[i]

        rMag = L2Norm(coord[:2])
        x = [coord[0] / rMag, -coord[1] / rMag]
        y = [-x[1], x[0]]

        newData[i, :] = [datum[0] * x[0] + datum[1] *
                         x[1], datum[0] * y[0] + datum[1] * y[1], datum[2]]

    return newData


def Maxima(data):
    """
    Return the indices of maxima in the supplied data
    """

    if len(data) < 2:
        return []

    maxima = []

    increasing = data[1] > data[0]
    if not increasing:
        maxima.append(0)
    for i in range(2, len(data)):
        if data[i] > data[i - 1]:
            increasing = True
        elif increasing:
            increasing = False
            maxima.append(i - 1)

    return numpy.array(maxima)


def Minima(data):
    """
    Return the indices of minima in the supplied data
    """

    if len(data) < 2:
        return []

    minima = []

    decreasing = data[1] < data[0]
    if not decreasing:
        minima.append(0)
    for i in range(2, len(data)):
        if data[i] < data[i - 1]:
            decreasing = True
        elif decreasing:
            decreasing = False
            minima.append(i - 1)

    return numpy.array(minima)


def Factorise(integer):
    """
    Quick and dirty integer factorisation
    """

    assert(integer > 0)

    factors = [1]
    for possible in range(2, integer / 2 + 1):
        if integer % possible == 0:
            factors.append(possible)
    if integer > 1:
        factors.append(integer)

    return factors


def Factorial(integer):
    """
    Factorial function
    """

    assert(integer >= 0)

    if integer <= 4:
        return [1, 1, 2, 6, 24][integer]

    factorial = 24
    mult = 5
    while mult <= integer:
        factorial *= mult
        mult += 1

    return factorial


def CrossProduct(vector1, vector2):
    """
    Compute a cross product
    """

    assert(len(vector1) == 3)
    assert(len(vector2) == 3)

    return numpy.array([vector1[1] * vector2[2] - vector1[2] * vector2[1],
                       -vector1[0] * vector2[2] + vector1[2] * vector2[0],
                        vector1[0] * vector2[1] - vector1[1] * vector2[0]])


class calcUnittests(unittest.TestCase):

    def testNumpySupport(self):
        import numpy  # noqa: testing
        self.assertTrue(NumpySupport())

        return

    def testScipySupport(self):
        import scipy  # noqa: testing
        import scipy.linalg  # noqa: testing
        import scipy.stats  # noqa: testing

        return

    def testLinearlyInterpolate(self):
        self.assertAlmostEquals(LinearlyInterpolate(0.0, 2.0, 0.6), 1.2)

        return

    def testBilinearlyInterpolate(self):
        self.assertAlmostEquals(
            BilinearlyInterpolate(0.0, 2.0, 0.0, 2.0, 0.6, 0.6), 1.2)

        return

    def testSe(self):
        vals = [0.0, 1.0]
        self.assertAlmostEquals(Se(vals), 1.0 / math.sqrt(2.0))

        return

    def testDeterminant(self):
        self.assertAlmostEquals(DeterminantCE([[1.0, 0.0], [0.0, 2.0]]), 2.0)
        self.assertAlmostEquals(DeterminantCE([[1.0, 2.0], [2.0, 4.0]]), 0.0)
        self.assertAlmostEquals(DeterminantLU([[1.0, 0.0], [0.0, 2.0]]), 2.0)
        self.assertAlmostEquals(DeterminantLU([[1.0, 2.0], [2.0, 4.0]]), 0.0)
        self.assertAlmostEquals(
            DeterminantNumpy([[1.0, 0.0], [0.0, 2.0]]), 2.0)
        self.assertAlmostEquals(
            DeterminantNumpy([[1.0, 2.0], [2.0, 4.0]]), 0.0)

        return

    def testEigendecomposition(self):
        w = Eigendecomposition(
            numpy.array([[1.0, 0.0], [0.0, 1.0]]), returnEigenvectors=False)
        self.assertEquals(len(w), 2)
        self.assertAlmostEquals(w[0], 1.0)
        self.assertAlmostEquals(w[1], 1.0)

        w = Eigendecomposition(
            numpy.array([[1.0, 0.0], [0.0, 0.0]]), returnEigenvectors=False)
        self.assertEquals(len(w), 2)
        self.assertTrue(AlmostEquals(w[0], 1.0) or AlmostEquals(w[1], 1.0))
        self.assertTrue(AlmostEquals(w[0], 0.0) or AlmostEquals(w[1], 0.0))

        return

    def testNormalisedFft(self):
        fft = NormalisedFft([i for i in range(16)])
        self.assertAlmostEquals(fft[0], 7.5)

        fft = NormalisedFft([math.sin(2.0 * math.pi * (i / 4.0))
                            for i in range(16)])
        self.assertAlmostEquals(fft[0], 0.0)
        self.assertEquals(utils.IndexOfMax(fft), 4)
        self.assertAlmostEquals(fft[4], 0.5)

        amp, phases = NormalisedFft(
            [i for i in range(16)], returnPhases=True)
        self.assertAlmostEquals(amp[0], 7.5)

        amp, phases = NormalisedFft([math.cos(2.0 * math.pi * (i / 4.0))
                                    for i in range(16)], returnPhases=True)
        self.assertAlmostEquals(amp[0], 0.0)
        self.assertEquals(utils.IndexOfMax(amp), 4)
        self.assertAlmostEquals(phases[4], 0.0)
        self.assertAlmostEquals(amp[4], 0.5)

        for phase in [math.pi / float(i) for i in range(2, 5)]:
            amp, phases = NormalisedFft([math.cos(2.0 * math.pi * (i / 4.0) - phase)
                                        for i in range(16)], returnPhases=True)
            self.assertAlmostEquals(amp[0], 0.0)
            self.assertEquals(utils.IndexOfMax(amp), 4)
            self.assertAlmostEquals(phases[4], -phase)
            self.assertAlmostEquals(amp[4], 0.5)

        return

    def testUnwrappedPhases(self):
        phases = UnwrappedPhases(
            [0.0, (3.0 / 4.0) * math.pi, (-3.0 / 4.0) * math.pi])
        self.assertEquals(len(phases), 3)
        self.assertAlmostEquals(phases[0], 0.0)
        self.assertAlmostEquals(phases[1], (3.0 / 4.0) * math.pi)
        self.assertAlmostEquals(phases[2], (5.0 / 4.0) * math.pi)

        phases = UnwrappedPhases(
            [0.0, -(3.0 / 4.0) * math.pi, (3.0 / 4.0) * math.pi])
        self.assertEquals(len(phases), 3)
        self.assertAlmostEquals(phases[0], 0.0)
        self.assertAlmostEquals(phases[1], -(3.0 / 4.0) * math.pi)
        self.assertAlmostEquals(phases[2], -(5.0 / 4.0) * math.pi)

        phases = UnwrappedPhases(
            [math.pi, (7.0 / 4.0) * math.pi, (1.0 / 4.0) * math.pi], min=0.0)
        self.assertEquals(len(phases), 3)
        self.assertAlmostEquals(phases[0], math.pi)
        self.assertAlmostEquals(phases[1], (7.0 / 4.0) * math.pi)
        self.assertAlmostEquals(phases[2], (9.0 / 4.0) * math.pi)

        phases = UnwrappedPhases(
            [math.pi, (1.0 / 4.0) * math.pi, (7.0 / 4.0) * math.pi], min=0.0)
        self.assertEquals(len(phases), 3)
        self.assertAlmostEquals(phases[0], math.pi)
        self.assertAlmostEquals(phases[1], (1.0 / 4.0) * math.pi)
        self.assertAlmostEquals(phases[2], (-1.0 / 4.0) * math.pi)

        return

    def testNormalisedLombScargle(self):
        times = [i / float(20) * 2.0 * math.pi for i in range(20)]
        vals = [math.sin(time) + 0.3 * math.sin(3.0 * time) + 0.7 * math.cos(5.0 * time)
                for time in times]
        ls = NormalisedLombScargle(vals, times)
        omegas = [float(i) for i in range(11)]
        nfft = ls.Evaluate(omegas)
        self.assertEquals(utils.IndexOfMax(nfft), 1)

        times = [i / float(20) * 2.0 * math.pi for i in range(30)]
        vals = [math.sin(time) + 0.3 * math.sin(3.0 * time) + 0.7 * math.cos(5.0 * time)
                for time in times]
        ls = NormalisedLombScargle(vals, times)
        omegas = [float(i) for i in range(11)]
        nfft = ls.Evaluate(omegas)
        self.assertEquals(utils.IndexOfMax(nfft), 1)

        times = [i / float(20) * 2.0 * math.pi for i in range(20)]
        times += [0.1, 1.0, 2.0]
        times.sort()
        vals = [math.sin(time) + 0.3 * math.sin(3.0 * time) + 0.7 * math.cos(5.0 * time)
                for time in times]
        ls = NormalisedLombScargle(vals, times)
        omegas = [float(i) for i in range(11)]
        nfft = ls.Evaluate(omegas)
        self.assertEquals(utils.IndexOfMax(nfft), 1)

        vals = [math.cos(2.0 * math.pi * (i / 4.0)) for i in range(16)]
        times = [2.0 * math.pi * i / float(len(vals))
                 for i in range(len(vals))]
        ls = NormalisedLombScargle(vals, times)
        omegas = [float(i) for i in range(11)]
        nfft = ls.Evaluate(omegas)
        self.assertAlmostEquals(nfft[0], 0.0)
        self.assertAlmostEquals(nfft[4], 0.5)

        return

    def testDominantModeStructured(self):
        dt = 1.2
        amps = [1.5 * math.sin(2.0 * math.pi * dt * float(i) / float(9))
                for i in range(25)]

        t, amp = DominantModeStructured(amps, dt)

        self.assertAlmostEquals(t, 9.0, 0)
        self.assertAlmostEquals(amp, 1.5, 1)

        return

    def testIndexBinaryLboundSearch(self):
        values = [0.0, 1.0, 2.0]
        self.assertEquals(
            IndexBinaryLboundSearch(-0.1, values, increasing=True), 0)
        self.assertEquals(
            IndexBinaryLboundSearch(0.1, values, increasing=True), 0)
        self.assertEquals(
            IndexBinaryLboundSearch(1.1, values, increasing=True), 1)
        self.assertEquals(
            IndexBinaryLboundSearch(2.1, values, increasing=True), 2)

        values = [0, 1, 2]
        self.assertEquals(
            IndexBinaryLboundSearch(1, values, increasing=True), 1)

        values = [0.0, 1.0, 2.0]
        self.assertEquals(IndexBinaryLboundSearch(-0.1, values), 0)
        self.assertEquals(IndexBinaryLboundSearch(0.1, values), 0)
        self.assertEquals(IndexBinaryLboundSearch(1.1, values), 1)
        self.assertEquals(IndexBinaryLboundSearch(2.1, values), 2)

        values = [0, 1, 2]
        self.assertEquals(IndexBinaryLboundSearch(1, values), 1)

        values = [2.0, 1.0, 0.0]
        self.assertEquals(
            IndexBinaryLboundSearch(-0.1, values, increasing=False), 2)
        self.assertEquals(
            IndexBinaryLboundSearch(0.1, values, increasing=False), 2)
        self.assertEquals(
            IndexBinaryLboundSearch(1.1, values, increasing=False), 1)
        self.assertEquals(
            IndexBinaryLboundSearch(2.1, values, increasing=False), 0)

        values = [2, 1, 0]
        self.assertEquals(
            IndexBinaryLboundSearch(1, values, increasing=False), 1)

        return

    def testIndexBinaryUboundSearch(self):
        values = [0.0, 1.0, 2.0]
        self.assertEquals(
            IndexBinaryUboundSearch(-0.1, values, increasing=True), 0)
        self.assertEquals(
            IndexBinaryUboundSearch(0.1, values, increasing=True), 1)
        self.assertEquals(
            IndexBinaryUboundSearch(1.1, values, increasing=True), 2)
        self.assertEquals(
            IndexBinaryUboundSearch(2.1, values, increasing=True), 2)

        values = [0, 1, 2]
        self.assertEquals(
            IndexBinaryUboundSearch(1, values, increasing=True), 1)

        values = [0.0, 1.0, 2.0]
        self.assertEquals(IndexBinaryUboundSearch(-0.1, values), 0)
        self.assertEquals(IndexBinaryUboundSearch(0.1, values), 1)
        self.assertEquals(IndexBinaryUboundSearch(1.1, values), 2)
        self.assertEquals(IndexBinaryUboundSearch(2.1, values), 2)

        values = [0, 1, 2]
        self.assertEquals(IndexBinaryUboundSearch(1, values), 1)

        values = [2.0, 1.0, 0.0]
        self.assertEquals(
            IndexBinaryUboundSearch(-0.1, values, increasing=False), 2)
        self.assertEquals(
            IndexBinaryUboundSearch(0.1, values, increasing=False), 1)
        self.assertEquals(
            IndexBinaryUboundSearch(1.1, values, increasing=False), 0)
        self.assertEquals(
            IndexBinaryUboundSearch(2.1, values, increasing=False), 0)

        values = [2, 1, 0]
        self.assertEquals(
            IndexBinaryUboundSearch(1, values, increasing=False), 1)

        return

    def testLinearRegression(self):
        eq, r = LinearRegression(
            [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], returnR=True)
        self.assertAlmostEquals(eq[0], 1.0)
        self.assertAlmostEquals(eq[1], 2.0)
        self.assertAlmostEquals(r, 1.0)

        return

    def testDeg2Rad(self):
        self.assertAlmostEquals(Deg2Rad(90.0), math.pi / 2.0)

        return

    def testRad2Deg(self):
        self.assertAlmostEquals(Rad2Deg(math.pi / 2.0), 90.0)

        return

    def testIsEven(self):
        self.assertTrue(IsEven(0))
        self.assertTrue(IsEven(-2))
        self.assertTrue(IsEven(2))
        self.assertFalse(IsEven(1))
        self.assertFalse(IsEven(-1))
        self.assertTrue(IsEven(2.0))

        return

    def testIsOdd(self):
        self.assertFalse(IsOdd(-2))
        self.assertFalse(IsOdd(2))
        self.assertTrue(IsOdd(1))
        self.assertTrue(IsOdd(-1))
        self.assertTrue(IsOdd(1.0))

        return

    def testEpsilon(self):
        self.assertTrue(1.0 + Epsilon() > 1.0)
        for divide in [2.0, 1.75, 1.5, 1.25]:
            self.assertTrue(
                1.0 + (Epsilon() / divide) in [1.0, 1.0 + Epsilon()])

        return

    def testTiny(self):
        self.assertTrue(Tiny() > 0.0)
        for divide in [2.0, 1.75, 1.5, 1.25]:
            self.assertTrue(Tiny() / divide in [0.0, Tiny()])

        return

    def testHuge(self):
        self.assertTrue(isinstance(Huge(), int))
        self.assertFalse(isinstance(Huge() + 1, int))
        self.assertTrue(isinstance(Huge() + 1, long))

        return

    def testInf(self):
        self.assertTrue(isinstance(Inf(), float))
        self.assertTrue(Inf() > 0.0)
        self.assertEquals(1.0 / Inf(), 0.0)

        return

    def testNan(self):
        self.assertTrue(isinstance(Nan(), float))
        self.assertFalse(Nan() == Nan())
        self.assertFalse(1.0 * Nan() == Nan())
        self.assertFalse(0.0 + Nan() == Nan())

        return

    def testIsInf(self):
        self.assertTrue(IsInf(Inf()))
        self.assertTrue(IsInf(-Inf()))
        self.assertFalse(IsInf(0.0))

        return

    def testIsNan(self):
        self.assertTrue(IsNan(Nan()))
        self.assertFalse(IsNan(0.0))

        return

    def testMaxVal(self):
        self.assertEquals(MaxVal([1, 2, 3, 4, 10, 6, 7, 8, 9]), 10)

        return

    def testMinVal(self):
        self.assertEquals(MinVal([1, 2, 3, 4, -10, 6, 7, 8, 9]), -10)

        return

    def testSumVal(self):
        self.assertAlmostEquals(SumVal([]), 0.0)
        self.assertAlmostEquals(SumVal([1.0, -1.0]), 0.0)
        self.assertAlmostEquals(SumVal([1.0, 2.0]), 3.0)

        return

    def testMeanVal(self):
        self.assertAlmostEquals(MeanVal([0.0, 1.0]), 0.5)

        return

    def testAlmostEquals(self):
        self.assertTrue(AlmostEquals(0.0, 0.1, tolerance=0.11))
        self.assertFalse(AlmostEquals(0.0, 0.1, tolerance=0.09))
        self.assertTrue(AlmostEquals(2.0, 2.2, tolerance=0.11))
        self.assertFalse(AlmostEquals(2.0, 2.2, tolerance=0.09))

        return

    def testRotatedVector(self):
        vec = RotatedVector([1.0, 0.0], math.pi / 4.0)
        self.assertEquals(len(vec), 2)
        self.assertAlmostEquals(vec[0], 1.0 / math.sqrt(2))
        self.assertAlmostEquals(vec[1], 1.0 / math.sqrt(2))

        vec = RotatedVector(
            [1.0, 0.0, 0.0], math.pi / 4.0, axis=(0.0, 0.0, 1.0))
        self.assertEquals(len(vec), 3)
        self.assertAlmostEquals(vec[0], 1.0 / math.sqrt(2))
        self.assertAlmostEquals(vec[1], 1.0 / math.sqrt(2))
        self.assertAlmostEquals(vec[2], 0.0)

        vec = RotatedVector(
            [1.0, 0.0, 0.0], math.pi / 4.0, axis=(0.0, 1.0, 0.0))
        self.assertEquals(len(vec), 3)
        self.assertAlmostEquals(vec[0], 1.0 / math.sqrt(2))
        self.assertAlmostEquals(vec[1], 0.0)
        self.assertAlmostEquals(vec[2], -1.0 / math.sqrt(2))

        vec = RotatedVector(
            [1.0, 0.0, 0.0], math.pi / 4.0, axis=(1.0, 0.0, 0.0))
        self.assertEquals(len(vec), 3)
        self.assertAlmostEquals(vec[0], 1.0)
        self.assertAlmostEquals(vec[1], 0.0)
        self.assertAlmostEquals(vec[2], 0.0)

        vec = RotatedVector(
            [0.0, 0.0, 1.0], math.pi / 4.0, axis=(1.0, 0.0, 0.0))
        self.assertEquals(len(vec), 3)
        self.assertAlmostEquals(vec[0], 0.0)
        self.assertAlmostEquals(vec[1], -1.0 / math.sqrt(2))
        self.assertAlmostEquals(vec[2], 1.0 / math.sqrt(2))

        return

    def testRotatedTensor(self):
        return

    def testCartesianVectorsToCylindrical(self):
        data = CartesianVectorsToCylindrical(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.assertEquals(len(data), 2)
        self.assertEquals(len(data[0]), 3)
        self.assertEquals(len(data[1]), 3)
        self.assertAlmostEquals(data[0][0], 1.0)
        self.assertAlmostEquals(data[0][1], 1.0)
        self.assertAlmostEquals(data[0][2], 0.0)
        self.assertAlmostEquals(data[1][0], 1.0)
        self.assertAlmostEquals(data[1][1], -1.0)
        self.assertAlmostEquals(data[1][2], 1.0)

        return

    def testCylindricalVectorsToCartesian(self):
        data = CylindricalVectorsToCartesian(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.assertEquals(len(data), 2)
        self.assertEquals(len(data[0]), 3)
        self.assertEquals(len(data[1]), 3)
        self.assertAlmostEquals(data[0][0], 1.0)
        self.assertAlmostEquals(data[0][1], 1.0)
        self.assertAlmostEquals(data[0][2], 0.0)
        self.assertAlmostEquals(data[1][0], -1.0)
        self.assertAlmostEquals(data[1][1], 1.0)
        self.assertAlmostEquals(data[1][2], 1.0)

        return

    def testMaxima(self):
        maxima = Maxima([0.0, 1.0, 0.0])
        self.assertEquals(len(maxima), 1)
        self.assertEquals(maxima[0], 1)

        maxima = Maxima([0.0, 1.0, 2.0, 0.0])
        self.assertEquals(len(maxima), 1)
        self.assertEquals(maxima[0], 2)

        maxima = Maxima([0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        self.assertEquals(len(maxima), 2)
        self.assertEquals(maxima[0], 2)
        self.assertEquals(maxima[1], 4)

        return

    def testMinima(self):
        minima = Minima([0.0, -1.0, 0.0])
        self.assertEquals(len(minima), 1)
        self.assertEquals(minima[0], 1)

        minima = Minima([0.0, -1.0, -2.0, 0.0])
        self.assertEquals(len(minima), 1)
        self.assertEquals(minima[0], 2)

        minima = Minima([0.0, -1.0, -2.0, 0.0, -1.0, 0.0])
        self.assertEquals(len(minima), 2)
        self.assertEquals(minima[0], 2)
        self.assertEquals(minima[1], 4)

        return

    def testFactorise(self):
        self.assertEquals(Factorise(1), [1])
        self.assertEquals(Factorise(5), [1, 5])
        self.assertEquals(Factorise(10), [1, 2, 5, 10])

        return

    def testFactorial(self):
        self.assertEquals(Factorial(0), 1)
        self.assertEquals(Factorial(1), 1)
        self.assertEquals(Factorial(2), 2)
        self.assertEquals(Factorial(3), 6)
        self.assertEquals(Factorial(6), 720)

        return

    def testCrossProduct(self):
        cross = CrossProduct([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.assertEquals(len(cross), 3)
        self.assertAlmostEquals(cross[0], 0.0)
        self.assertAlmostEquals(cross[1], 0.0)
        self.assertAlmostEquals(cross[2], 1.0)

        return
