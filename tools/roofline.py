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
    def __init__(self, streaming_limit, flop_limit, event_type=None):
        """The generation of a roofline performance model, for given code.

        :arg self: self
        :arg streaming_limit: Memory Streaming Bandwidth (GB/s)
        :arg flop_limit: CPU's Maximum Memory Bandwidth (GB/s)
        :arg event_type: Only examine data for the specified PETSc event
        """
        self.data = defaultdict(partial(defaultdict, partial(defaultdict, float)))
        self.streaming_limit = streaming_limit
        self.flop_limit = flop_limit
        self.event_type = event_type
    
    def start_collection(self):
        """The start point of data collection for the Roofline model."""
        event_type = self.event_type
        start = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[event_type]
        for event, info in start.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] -= info[n]

    def stop_collection(self):
        """The end point of data collection for the Roofline model."""
        event_type = self.event_type
        stop = PETSc.Log.getPerfInfoAllStages()['Main Stage']
        data = self.data[event_type]
        for event, info in stop.items():
            event_data = data[event]
            for n in ('flops', 'bytes', 'time', 'count'):
                event_data[n] += info[n]
        
        if len(data) == 0:
            firedrake.logging.warn("Requested event(s) occurs 0 times.")

    @contextmanager
    def collecting(self):
        """Automated inclusion of stop_collection at the end of a script if not called."""
        event_type = self.event_type
        self.start_collection(event_type)
        try: 
            yield
        finally:
            self.stop_collection(event_type)

    def roofline(self, data_type=None, axes=None, saved_data=None):
        """The generation of a roofline plot.

        :arg self: Self
        :arg data_type: Choice between 'flops', 'bytes', and 'time'
        :arg axes: Existing axes to add roofline plot to
        :arg data: Load previously saved data stored as a pickle file
        :returns: Roofline plot axes
        """
        
        event_type = self.event_type
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(111)
        
        if saved_data is not None:
            self.data = saved_data

        if data_type is not None:
            data = self.data[event_type][data_type]
        else: 
            data = defaultdict(float)
            for event in self.data[event_type].values():
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
        axes.loglog(intensity, flop_rate, 'o', linewidth=0, label=data_type or 'Total')
        axes.fill_between(x=x_mem, y1=y1_mem, y2=y2_mem, color='mediumspringgreen', alpha=0.1, label='Memory-bound region')
        axes.fill_between(x=x_comp, y1=y1_comp, y2=y2_comp, color='darkorange', alpha=0.1, label='Compute-bound region')
        axes.legend(loc='best')
        axes.set_xlabel("Arithmetic Intensity [FLOPs/byte]")
        axes.set_ylabel("Performance [GFLOPs/s]")
        plt.show()
        self.axes = axes
        return axes 

    def save_data(self, name):
        """Save PETSc performance data as a .txt file.
        
        :arg name: Name assigned to .txt file containing the data
        """
        data = self.data
        f_name = '{}.p'.format(name)
        pickle.dump(data, open(f_name, "wb"))

    def save_axes(self, name):
        """Save roofline plot axes as a pickle file
        
        :arg name: Name assigned to pickle file containing the axes
        """
        axes = self.axes
        f_name = '{}.p'.format(name)
        pickle.dump(axes, open(f_name, "wb"))
