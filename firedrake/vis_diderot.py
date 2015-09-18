import ctypes
from os import path

from cfunction import *

__all__ = ['simple_d2s']


#################### Visualizing with Diderot ##############################################
#
#             Overview
# file  -  The path to the Diderot program
# calls -  Initializing functions available for that program
# field#k(d)[d1 .. dn]- Describes type of field the diderot program expects.
#     is a $C^{k}$ continuous function in $\Re^{d} \rightarrow\Re^{d_1} \times \cdots \times \Re^{d_n}$.

#               Naming convention of the arguments
# resU, resV:int         - resolution of output image
# physical*:int          - physical size of field.
#                               ex. a unit square mesh physicalx0=0, and physicalxn=1.0
# stepSize:float         - scales spread over resolution.
#                               ex. a unit square mesh res*=100 steps=.01
# f:void *               - pointer to field
# type:int               - type of output file. 0 for nrrd else png file
# name:string            - name of output file for correct output type
#
#                More details
# The resolution (resU,resV) of the image can be described with
# -  the physical size (physical*) of the field
#           i.e. lerp(physicalx0, )
# -  or the stepSize which scales the sampling over the resolution
#           i.e. [resU,resV]*stepSize
######################################## simple #########################################################

# file         - fem/examples/d2s/simple_lerp
# calls        - callDiderot2_lerp()
# field#k(2)[] - 2d scalar field
def simple2_lerp(name,f, resU, resV, physicalx0,physicalxn,physicaly0,physicalyn,type):
    
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), '../diderot/simple_lerp_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/simple_lerp.o')
    call = make_c_evaluate(f, "callDiderot2_lerp", ldargs=[init_file, diderot_file, "-lteem"])
    
    return call(ctypes.c_char_p(name),type,p_cf, resU, resV, physicalx0,physicalxn,physicaly0,physicalyn)

# file          - fem/examples/d2s/simple_sample
# calls         - callDiderot2_step()
# field#k(2)[]  - 2d scalar field
def simple2_sample(name,f, resU, resV, stepSize,type):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), '../diderot/simple_sample_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/simple_sample.o')
    call = make_c_evaluate(f, "callDiderot2_step", ldargs=[init_file, diderot_file, "-lteem"])
    
    return call(ctypes.c_char_p(name),type,p_cf, resU, resV, ctypes.c_float(stepSize))

# file         - fem/examples/d3s/simple
# calls        - callDiderot3()
# field#k(3)[] - 3d scalar field
def simple3(name,f, resU, resV,resW, physicalx0,physicalxn,physicaly0,physicalyn,physicalz0,physicalzn ):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), '../diderot/simple_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/simple.o')
    call = make_c_evaluate(f, "callDiderot3", ldargs=[init_file, diderot_file, "-lteem"])
    
    return call(ctypes.c_char_p(name),p_cf, resU, resV, resW,physicalx0,physicalxn,physicaly0,physicalyn,physicalz0,physicalzn)



########################################## iso ##########################################################
# Note- Look at .diderot file and rewrite
# file          - fem/examples/d2s/iso_sample
# call          - callDiderot2_iso
# field#k(2)[]  - 2d scalar field
# stepsMax:int  - max number of steps
# isovalue:float- isovalue
def iso_sample(name,f, resU, resV, stepSize,stepsMax,isovalue,epsilon):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), '../diderot/iso_sample_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/iso_sample.o')
    call = make_c_evaluate(f, "callDiderot2_iso", ldargs=[init_file, diderot_file, "-lteem"])
    
    return call(ctypes.c_char_p(name),p_cf, resU, resV, ctypes.c_float(stepSize),stepsMax,ctypes.c_float(isovalue),ctypes.c_float(epsilon))



########################################## Mips ##########################################################
# Maximum Intensity Projection
# file          - fem/examples/d3s/mip
# call          - callDiderot3_noCam()
# field#k(3)[]  - 3d scalar field
# steps:float   - rayStep
def mip(name,f, resU, resV, steps):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), '../diderot/mip_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/mip.o')
    call = make_c_evaluate(f, "callDiderot3_noCam", ldargs=[init_file, diderot_file, "-lteem"])
    
    return call(ctypes.c_char_p(name),p_cf, resU, resV, ctypes.c_float(steps))


