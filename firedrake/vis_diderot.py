import ctypes
from os import path

from firedrake.function import make_c_evaluate
#__all__ = ['simple_d2s']


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

# file         - fem/basic/basic_d2s
# calls        - callDiderot2_lerp()
# field#k(2)[] - 2d scalar field
def tmp(f):
    p_cf = f._ctypes
    init_file = path.join(path.dirname(__file__), '/Users/chariseechiw/fire/firedrake/diderot/store/tmp_init.o')
    diderot_file = path.join(path.dirname(__file__), '/Users/chariseechiw/fire/firedrake/diderot/store/tmp.o')
    call = make_c_evaluate(f, "cat", ldargs=[init_file, diderot_file, "-lteem"])
    return 1


def basic_d2s_lerp(name, f, resU, resV, physicalx0, physicalxn, physicaly0, physicalyn, type):

    p_cf = f._ctypes
    init_file = path.join(path.dirname(__file__), '../diderot/store/basic_d2s_lerp_init.o')
    diderot_file = path.join(path.dirname(__file__), '../diderot/store/basic_d2s_lerp.o')
    call = make_c_evaluate(f, "callDiderot", ldargs=[init_file, diderot_file, "-lteem"])

    return call(ctypes.c_char_p(name), type, p_cf, resU, resV, physicalx0, physicalxn, physicaly0, physicalyn)


# file          - fem/examples/d2s/simple_sample
# calls         - callDiderot2_step()
# field#k(2)[]  - 2d scalar field
# def basic_d2s_sample(name,f, resU, resV, stepSize, type):
#     p_cf = cFunction(f)
#     init_file = path.join(path.dirname(__file__), '../diderot/store/basic_d2s_sample_init.o')
#     diderot_file = path.join(path.dirname(__file__),'../diderot/store/basic_d2s_sample.o')
#     call = make_c_evaluate(f, "callDiderot", ldargs=[init_file, diderot_file, "-lteem"])
#
#     return call(ctypes.c_char_p(name),type,p_cf, resU, resV, ctypes.c_float(stepSize))
