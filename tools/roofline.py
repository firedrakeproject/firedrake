"""
The generation of a roofline plot given the flop count
and arithmetic intensity for a given script.
"""

import numpy
import datetime
import matplotlib.pyplot as plt 
from firedrake.petsc import PETSc

def roofline():
    flops = 0
    bytes = 0
    log = PETSc.Log.getPerfInfoAllStages()['Main Stage']
    data = log.items()

    for d in data:
        flops += d[1]['flops']
        bytes += d[1]['bytes']
    
    intensity = flops/bytes
    streaming = 11.23750
    maximum = 52.8
    time = datetime.datetime.now()
    fig_name = "Roofline_plot_{}".format(time.ctime())

    x_range = [-6, 10] 
    x = numpy.logspace(x_range[0], x_range[1], base=2, num=1000)
    y = []
    for i in x: 
        # The minimum of the memory streaming bandwidth and maximum memory bandwidth
        y.append(min(i * streaming, maximum))

    plt.loglog(x, y, c='black', label='Roofline')
    plt.loglog(intensity, flops, 'o', c='crimson', linewidth=0)
    plt.xlabel("Arithmetic Intensity [FLOPs/byte]")
    plt.ylabel("Performance [FLOPs]")
    plt.show()
    #plt.savefig(figname)

# Annotate graph with regions, multiple roofs
