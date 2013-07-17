from math import log, sqrt


def convergence(filename1, filename2):
    with open(filename1) as f1:
        with open(filename2) as f2:
            return log(sqrt(float(f1.read())) / sqrt(float(f2.read())), 2)
