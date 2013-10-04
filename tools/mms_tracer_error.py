#!/usr/bin/env python

from numpy import sqrt, size, zeros, abs, max

import vtktools


def tetvol(X, Y, Z):
    X12 = X[1] - X[0]
    X13 = X[2] - X[0]
    X14 = X[3] - X[0]
    Y12 = Y[1] - Y[0]
    Y13 = Y[2] - Y[0]
    Y14 = Y[3] - Y[0]
    Z12 = Z[1] - Z[0]
    Z13 = Z[2] - Z[0]
    Z14 = Z[3] - Z[0]
    VOL = X12 * (Y13 * Z14 - Y14 * Z13) + X13 * \
        (Y14 * Z12 - Y12 * Z14) + X14 * (Y12 * Z13 - Y13 * Z12)
    return abs(VOL / 6)


def triarea(X, Y):
    X12 = X[1] - X[0]
    X13 = X[2] - X[0]
    Y12 = Y[1] - Y[0]
    Y13 = Y[2] - Y[0]
    AREA = (X12 * Y13 - Y12 * X13)
    return abs(AREA / 2)


def l2(file, numericalfield, analyticalfield):
    ug = vtktools.vtu(file)
    ug.GetFieldNames()
    uv = ug.GetScalarField(numericalfield)
    ex = ug.GetScalarField(analyticalfield)
    pos = ug.GetLocations()
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    NE = ug.ugrid.GetNumberOfCells()
    ML = zeros(size(x), float)
    for ele in range(NE):
        ndglno = ug.GetCellPoints(ele)
        if(size(ndglno) == 4):
            t = tetvol(x[ndglno], y[ndglno], z[ndglno])
        elif(size(ndglno) == 3):
            t = triarea(x[ndglno], y[ndglno])
        for nod in ndglno:
            ML[nod] = ML[nod] + t / size(ndglno)

    err_x = ex - uv

    norm_x = 0.0
    for nod in range(size(x)):
        norm_x = norm_x + ML[nod] * (err_x[nod]) ** 2

    norm_x = sqrt(abs(norm_x))
    return (norm_x)


def inf(file, numericalfield, analyticalfield):
    ug = vtktools.vtu(file)
    ug.GetFieldNames()
    uv = ug.GetScalarField(numericalfield)
    ex = ug.GetScalarField(analyticalfield)

    err_x = ex - uv

    norm_x = max(abs(err_x))
    return (norm_x)
