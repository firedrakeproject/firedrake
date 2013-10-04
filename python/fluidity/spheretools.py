# A module for converting between spherical and polar coordinates

# Usage


def polar2cart(r):
    # convert polar coordinates to Cartesian coordinates
    from math import sin, cos, pi
    from numpy import array

    def val(X, t):
        theta = 2 * pi * X[0] / 360
        phi = 2 * pi * X[1] / 360
        return r * array([cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)])
    return val


def cart2polar(X):
    # convert Cartesian coordinates to polar coordinates
    from math import asin, atan2, sqrt
    from numpy import array
    x = X[0]
    y = X[1]
    z = X[2]
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = asin(z / r)
    theta = atan2(y, x)
    return array([theta, phi])


def spherical_basis_vecs():
    # return unit vectors in theta, phi
    from math import sin, cos, pi
    from numpy import array

    def val(X, t):
        theta = 2 * pi * X[0] / 360
        phi = 2 * pi * X[1] / 360
        return array([[-sin(theta), cos(theta), 0.0],
                      [-cos(theta) * sin(phi), -sin(theta) * sin(phi), cos(phi)]]).T
    return val


def spherical_down():
    # return down
    from math import sin, cos
    from numpy import array

    def val(X, t):
        ([theta, phi]) = cart2polar(X)
        return -array([cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)])
    return val


def coriolis(omega):
    # return f=2 * Omega sin(phi)
    from math import sin, pi

    def val(X, t):
        phi = 2 * pi * X[1] / 360
        return 2 * omega * sin(phi)
    return val
