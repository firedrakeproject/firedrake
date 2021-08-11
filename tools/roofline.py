"""
The generation of a roofline plot given the flop count
and arithmetic intensity for a given script.
"""

import firedrake
import numpy
import matplotlib.pyplot as plt 
from firedrake.petsc import PETSc
from collections import defaultdict
from functools import partial
from contextlib import contextmanager

class Roofline:
    def __init__(self, streaming_limit, flop_limit):
        """ """
        self.data = defaultdict(partial(defaultdict, partial(defaultdict, float)))
        self.streaming_limit = streaming_limit
        self.flop_limit = flop_limit
    
    def start_collection(self, event_type=None):
        """ """
        start = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[event_type]
        for event, info in start.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] -= info[n]

    def stop_collection(self, event_type=None):
        """ """
        stop = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[event_type]
        for event, info in stop.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] += info[n]
                if event_data['count'] == 0:
                    print(data[event][0])
        
        if len(data) == 0:
            firedrake.logging.warn("Requested event has 0 occurrences.")

    @contextmanager
    def collecting(self, event_type=None):
        """ """
        self.start_collection(event_type)
        try: 
            yield
        finally:
            self.stop_collection(event_type)

    def roofline(self, event_type=None, event_name=None, axes=None):
        """ """
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(111)
        
        if event_name is not None:
            data = self.data[event_type][event_name]
        else: 
            data = defaultdict(float)
            for event in self.data[event_type].values():
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
