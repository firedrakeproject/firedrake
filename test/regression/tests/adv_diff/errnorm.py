from math import sqrt


def errnorm(filename):
    with open(filename, "r") as f:
        return sqrt(float(f.read()))
