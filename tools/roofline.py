"""
The generation of a roofline plot for a given script, given the 
Maximum Memory Bandwidth and Memory Streaming Bandwidth of the CPU.
"""

import firedrake
import numpy
import pickle
import matplotlib.pyplot as plt 
from firedrake.petsc import PETSc
from collections import defaultdict
from functools import partial
from contextlib import contextmanager

class Roofline:
    def __init__(self, streaming_limit, flop_limit):
        """The generation of a roofline performance model, for given code.

        :arg streaming_limit: Peak Memory Streaming Bandwidth (GB/s)
        :arg flop_limit: Peak Floating Point Performance (GFLOPS/s)
        """
        self.data = defaultdict(partial(defaultdict, partial(defaultdict, float)))
        self.streaming_limit = streaming_limit
        self.flop_limit = flop_limit
    
    def start_collection(self, region_name=None):
        """The start point of data collection for the Roofline model."""
        start = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[region_name]
        for event, info in start.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] -= info[n]

    def stop_collection(self, region_name=None):
        """The end point of data collection for the Roofline model."""
        stop = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[region_name]
        for event, info in stop.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] += info[n]

    @contextmanager
    def collecting(self, region_name=None):
        """Automated inclusion of stop_collection at the end of a script if not called."""
        self.start_collection(region_name)
        try: 
            yield
        finally:
            self.stop_collection(region_name)

    def roofline(self, region_name=None, event_name=None, axes=None):
        """The generation of a roofline plot.

        :arg data_type: Choice between 'flops', 'bytes', and 'time'
        :arg axes: Existing axes to add roofline plot to
        :arg data: Load previously saved data stored as a pickle file
        :returns: Roofline plot axes
        """

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
        flop_rate = (data['flops']/data['time']) * 1e-9
        
        x_range = [-6, 10] 
        x = numpy.logspace(x_range[0], x_range[1], base=2, num=100)
        y = []
        for points in x: 
            # The minimum of the memory streaming bandwidth and compute limit
            y.append(min(points * self.streaming_limit, self.flop_limit))
        
        # Shading of compute-limited and memory-limited regions
        mem_lim = self.flop_limit/self.streaming_limit
        x_mem, y1_mem, y2_mem = [], [], []
        x_comp, y1_comp, y2_comp = [], [], []
        for i in range(1, len(x)):
            if x[i-1] <= (mem_lim):
                x_mem.append(x[i])
                y1_mem.append(y[i])
                y2_mem.append(y[0])
            if x[i] >= mem_lim:
                x_comp.append(x[i])
                y1_comp.append(y[i])
                y2_comp.append(y[0])

        axes.loglog(x, y, c='black', label='Roofline')
        axes.loglog(intensity, flop_rate, 'o', linewidth=0, label=region_name or 'Total')
        axes.fill_between(x=x_mem, y1=y1_mem, y2=y2_mem, color='mediumspringgreen', alpha=0.1, label='Memory-bound region')
        axes.fill_between(x=x_comp, y1=y1_comp, y2=y2_comp, color='darkorange', alpha=0.1, label='Compute-bound region')
        axes.legend(loc='best')
        axes.set_xlabel("Operational Intensity [FLOPs/byte]")
        axes.set_ylabel("Performance [GFLOPs/s]")
        return axes 

    def save(self, name):
        """Save Roofline object as a pickle file.
        
        :arg name: Name assigned to pickle file containing the data
        """
        f_name = '{}.p'.format(name)
        with open(f_name, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, name):
        """Load roofline data from a pickle file
        
        :arg name: Name assigned to pickle file containing the data
        """
        f_name = '{}.p'.format(name)
        with open(f_name, "rb") as f:
            return pickle.load(f)
