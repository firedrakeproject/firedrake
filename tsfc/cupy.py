from collections import namedtuple

CuPyKernel = namedtuple("CuPyKernel", "code default_entrypoint")
CuPyEntrypoint = namedtuple("CuPyEntrypoint", "name args arg_order")

def generate(code, args, scalar_type, name, index_names, log=None):
    code_str = code[0].replace("cupy_kernel", name)
    print(args)
    return CuPyKernel(code = code_str, default_entrypoint=CuPyEntrypoint(name=name, args=tuple(args), arg_order=code[1])), "CupyKernel" 
