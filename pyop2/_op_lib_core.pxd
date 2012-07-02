"""
Cython header file for OP2 C library
"""
cdef extern from "op_lib_core.h":
    ctypedef struct op_set_core:
        int index, size
        char * name
        int core_size, exec_size, nonexec_size
        pass
    ctypedef op_set_core * op_set

    ctypedef struct op_map_core:
        pass
    ctypedef op_map_core * op_map

    ctypedef struct op_dat_core:
        pass
    ctypedef op_dat_core * op_dat

    ctypedef struct op_arg:
        pass

    ctypedef struct op_kernel:
        pass

    ctypedef enum op_access:
        pass

    op_set op_decl_set_core(int, char *)

    op_map op_decl_map_core(op_set, op_set, int, int *, char *)

    op_dat op_decl_dat_core(op_set, int, char *, int, char *, char *)

    op_arg op_arg_dat_core(op_dat, int, op_map, int, char *, op_access)

    op_arg op_arg_gbl_core(char *, int, char *, int, op_access)
