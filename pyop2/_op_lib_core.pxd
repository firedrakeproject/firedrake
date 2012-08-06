# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Cython header file for OP2 C library
"""
cdef extern from "op_lib_core.h":
    ctypedef struct op_set_core:
        int size
        int core_size
        int exec_size
        int nonexec_size

    ctypedef op_set_core * op_set

    ctypedef struct op_map_core:
        pass
    ctypedef op_map_core * op_map

    ctypedef struct op_sparsity_core:
        op_map * rowmaps
        op_map * colmaps
        int nmaps
        int dim[2]
        size_t nrows
        size_t ncols
        int * nnz
        int total_nz
        int * rowptr
        int * colidx
        size_t max_nonzeros
        char * name
    ctypedef op_sparsity_core * op_sparsity

    ctypedef struct op_mat_core:
        int index
        int dim[2]
        int size
        void * mat
        void * mat_array
        char * type
        op_sparsity sparsity
        char * data
        char * lma_data
    ctypedef op_mat_core * op_mat

    ctypedef struct op_dat_core:
        pass
    ctypedef op_dat_core * op_dat

    ctypedef struct op_arg:
        pass

    ctypedef struct op_kernel:
        pass

    ctypedef enum op_access:
        OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX

    op_set op_decl_set_core(int, char *)

    op_map op_decl_map_core(op_set, op_set, int, int *, char *)

    op_sparsity op_decl_sparsity_core(op_map *, op_map *, int, int *, int,
                                      char *)

    op_dat op_decl_dat_core(op_set, int, char *, int, char *, char *)

    op_arg op_arg_dat_core(op_dat, int, op_map, int, char *, op_access)

    op_arg op_arg_gbl_core(char *, int, char *, int, op_access)

cdef extern from "op_lib_mat.h":
    void op_solve(op_mat mat, op_dat b, op_dat x)

    void op_mat_zero ( op_mat mat )

    op_mat op_decl_mat(op_sparsity, int *, int, char *, int, char *)

    void op_mat_get_values ( op_mat mat, double **v, int *m, int *n)

    void op_mat_zero_rows ( op_mat mat, int n, int *rows, double val)

    void op_mat_assemble ( op_mat mat )

    void op_mat_get_array ( op_mat mat )

    void op_mat_put_array ( op_mat mat )

cdef extern from "op_lib_c.h":
    void op_init(int, char **, int)

    void op_exit()

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

    void op_rt_exit()


cdef extern from "dlfcn.h":
    void * dlopen(char *, int)
    int RTLD_NOW
    int RTLD_GLOBAL
    int RTLD_NOLOAD


cdef extern from "mpi.h":
    cdef void emit_ifdef '#if defined(OPEN_MPI) //' ()
    cdef void emit_endif '#endif //' ()
