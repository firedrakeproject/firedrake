"""
The generation of a roofline plot given the flop count
and arithmetic intensity for a given script.
"""

import numpy
import matplotlib.pyplot as plt 
from firedrake.petsc import PETSc
from collections import defaultdict
from functools import partial
from contextlib import contextmanager

class Roofline:
    def __init__(self, streaming_limit, flop_limit):
        """ """
        self.data = defaultdict(partial(defaultdict, float))
        self.streaming_limit = streaming_limit
        self.flop_limit = flop_limit
    
    def start_collection(self, region_name=None):
        """ """
        start = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[region_name]
        for event, info in start.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time'):
                event_data[n] -= info[n]

    def stop_collection(self, region_name=None):
        """ """
        stop = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[region_name]
        for event, info in stop.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time'):
                event_data[n] += info[n]

    @contextmanager
    def collecting(self, region_name=None):
        """ """
        self.start_collection(region_name)
        try: 
            yield
        finally:
            self.stop_collection(region_name)

    def roofline(self, region_name=None, event_name=None, axes=None):
        """ """
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(111)
        
        if event_name is not None:
            data = self.data[region_name][event_name]
        else: 
            data = defaultdict(float)
            for event in self.data[region_name].values():
                data['flops'] += event['flops']
                data['bytes'] += event['bytes'] 
                data['time'] += event['time']
        
        intensity = data['flops']/data['bytes']
        flop_rate = data['flops']/data['time'] * 1e-9
        
        x_range = [-6, 10] 
        x = numpy.logspace(x_range[0], x_range[1], base=2, num=100)
        y = []
        for points in x: 
            # The minimum of the memory streaming bandwidth and compute limit
            y.append(min(points * self.streaming_limit, self.flop_limit))

        axes.loglog(x, y, c='black', label='Roofline')
        axes.loglog(intensity, flop_rate, 'o', linewidth=0, label=event_name or 'Total')
        axes.legend(loc='best')
        axes.set_xlabel("Arithmetic Intensity [FLOPs/byte]")
        axes.set_ylabel("Performance [GFLOPs/s]")
        return axes 
