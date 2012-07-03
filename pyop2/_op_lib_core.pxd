"""
Cython header file for OP2 C library
"""
cdef extern from "op_lib_core.h":
    ctypedef struct op_set_core:
        int size
        int exec_size

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

cdef extern from "op_rt_support.h":
    ctypedef struct op_plan:
        char * name
        op_set set
        int nargs
        int ninds
        int part_size
        op_map * maps
        op_dat * dats
        int * idxs
        op_access * accs
        int * nthrcol
        int * thrcol
        int * offset
        int * ind_map
        int ** ind_maps
        int * ind_offs
        int * ind_sizes
        int * nindirect
        short * loc_map
        short ** loc_maps
        int nblocks
        int * nelems
        int ncolors_core
        int ncolors_owned
        int ncolors
        int * ncolblk
        int * blkmap
        int * nsharedCol
        int nshared
        float transfer
        float transfer2
        int count

    op_plan * op_plan_core(char *, op_set, int, int, op_arg *,
                           int, int *)
