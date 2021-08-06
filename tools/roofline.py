"""
The generation of a roofline plot given the flop count
and arithmetic intensity for a given script.
"""

import numpy
import datetime
import matplotlib.pyplot as plt 
from firedrake.petsc import PETSc
#from firedrake.PyOP2.pyop2.base import ParLoop

print(PETSc.Log.getPerfInfoAllStages())

def roofline(kernel):
    flops = kernel.flop_count
    intensity = 0.25
    double_precision = 0.4
    msb = 11.23750
    mmb = 52.8
    time = datetime.datetime.now()
    fig_name = "Roofline plot {}".format(time.ctime())

    x_range = [-6, 10] #Probably need to auto generate this, or have large range then generate xlim & ylim?
    x = numpy.logspace(x_range[0], x_range[1], base=2, num = 1000)
    y = []
    for i in x: 
        # The minimum of the memory streaming bandwidth and maximum memory bandwidth
        # How to obtain msb & mmb?
        y.append(min(i * msb, mmb))

    plt.loglog(x, y, c='black', label='Roofline')
    plt.loglog(intensity, double_precision, 'o', c='crimson', linewidth=0)
    #plt.savefig(figname)
    plt.show()

# Annotate graph with regions, multiple roofs