"""In this file two counters are provided, which are used to ensure unique naming.
   loopy_kernel_counter is used for naming kernels,
   loopy_indexset_counter is used for naming the indices within one kernel.
   Whenever the counter are accessed through their counter functions, 
   they are increased within the same step.
"""

loopy_kernel_counter = 0
loopy_indexset_counter = 0

def knl_counter():
    global loopy_kernel_counter
    c = loopy_kernel_counter
    loopy_kernel_counter += 1
    return c

def indexset_counter():
    global loopy_indexset_counter
    c = loopy_indexset_counter
    loopy_indexset_counter += 1
    return c