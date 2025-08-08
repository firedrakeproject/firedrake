from collections import namedtuple

CuPyKernel = namedtuple("CuPyKernel", "code default_entrypoint temps")
CuPyEntrypoint = namedtuple("CuPyEntrypoint", "name args arg_order")

def generate(code, args, scalar_type, name, index_names, log=None):
    from firedrake.device import compute_device
    if compute_device.kernel_type == "triton":
        temps =tuple([arr[0] for arr in compute_device.kernel_data["arrays"] if arr[1] is not None])
    code_str = code[0].replace(f"{compute_device.kernel_type}_kernel", name)
    print(args)
    return CuPyKernel(code = code_str, default_entrypoint=CuPyEntrypoint(name=name, args=tuple(args), arg_order=code[1]), temps = temps), "GPUKernel" 
