from pyop2 import compilation
from ffc import compile_element

__all__ = ['c_evaluate']


def c_evaluate(function):
    function_space = function.function_space()
    ufl_element = function_space.ufl_element()

    (src,) = compile_element(ufl_element)
    with open("locate.c") as f:
        src += f.read()

    return compilation.load(src, "c", "evaluate", cppargs=["-I."])
