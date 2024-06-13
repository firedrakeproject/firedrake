# Code for projections and other fun stuff involving supermeshes.
import firedrake
import ctypes
import sys
from firedrake.cython.supermeshimpl import assemble_mixed_mass_matrix as ammm, intersection_finder
from firedrake.mg.utils import get_level
from firedrake.petsc import PETSc
from firedrake.mg.kernels import to_reference_coordinates, compile_element
from firedrake.utility_meshes import UnitTriangleMesh, UnitTetrahedronMesh
from firedrake.functionspace import FunctionSpace
from firedrake.assemble import assemble
from firedrake.ufl_expr import TestFunction, TrialFunction
import firedrake.mg.utils as utils
from firedrake.utils import complex_mode, ScalarType
import ufl
from ufl import inner, dx
import numpy
from pyop2.sparsity import get_preallocation
from pyop2.compilation import load
from pyop2.mpi import COMM_SELF
from pyop2.utils import get_petsc_dir


__all__ = ["assemble_mixed_mass_matrix", "intersection_finder"]


class BlockMatrix(object):
    def __init__(self, mat, dimension):
        self.mat = mat
        self.dimension = dimension

    def mult(self, mat, x, y):
        sizes = self.mat.getSizes()
        for i in range(self.dimension):
            start = i
            stride = self.dimension

            xa = x.array_r[start::stride]
            ya = y.array_r[start::stride]
            xi = PETSc.Vec().createWithArray(xa, size=sizes[1], comm=x.comm)
            yi = PETSc.Vec().createWithArray(ya, size=sizes[0], comm=y.comm)
            self.mat.mult(xi, yi)
            y.array[start::stride] = yi.array_r

    def multTranspose(self, mat, x, y):
        sizes = self.mat.getSizes()
        for i in range(self.dimension):
            start = i
            stride = self.dimension

            xa = x.array_r[start::stride]
            ya = y.array_r[start::stride]
            xi = PETSc.Vec().createWithArray(xa, size=sizes[0], comm=x.comm)
            yi = PETSc.Vec().createWithArray(ya, size=sizes[1], comm=y.comm)
            self.mat.multTranspose(xi, yi)
            y.array[start::stride] = yi.array_r


@PETSc.Log.EventDecorator()
def assemble_mixed_mass_matrix(V_A, V_B):
    """
    Construct the mixed mass matrix of two function spaces,
    using the TrialFunction from V_A and the TestFunction
    from V_B.
    """

    if len(V_A) > 1 or len(V_B) > 1:
        raise NotImplementedError("Sorry, only implemented for non-mixed spaces")

    if V_A.ufl_element().mapping() != "identity" or V_B.ufl_element().mapping() != "identity":
        msg = """
Sorry, only implemented for affine maps for now. To do non-affine, we'd need to
import much more of the assembly engine of UFL/TSFC/etc to do the assembly on
each supermesh cell.
"""
        raise NotImplementedError(msg)

    mesh_A = V_A.mesh()
    mesh_B = V_B.mesh()

    dim = mesh_A.geometric_dimension()
    assert dim == mesh_B.geometric_dimension()
    assert dim == mesh_A.topological_dimension()
    assert dim == mesh_B.topological_dimension()

    (mh_A, level_A) = get_level(mesh_A)
    (mh_B, level_B) = get_level(mesh_B)

    if mesh_A is mesh_B:
        def likely(cell_A):
            return [cell_A]
    else:
        if (mh_A is None or mh_B is None) or (mh_A is not mh_B):

            # No mesh hierarchy structure, call libsupermesh for
            # intersection finding
            intersections = intersection_finder(mesh_A, mesh_B)
            likely = intersections.__getitem__
        else:
            # We do have a mesh hierarchy, use it

            if abs(level_A - level_B) > 1:
                raise NotImplementedError("Only works for transferring between adjacent levels for now.")

            # What are the cells of B that (probably) intersect with a given cell in A?
            if level_A > level_B:
                cell_map = mh_A.fine_to_coarse_cells[level_A]

                def likely(cell_A):
                    return cell_map[cell_A]

            elif level_A < level_B:
                cell_map = mh_A.coarse_to_fine_cells[level_A]

                def likely(cell_A):
                    return cell_map[cell_A]

    assert V_A.value_size == V_B.value_size
    orig_value_size = V_A.value_size
    if V_A.value_size > 1:
        V_A = firedrake.FunctionSpace(mesh_A, V_A.ufl_element().sub_elements[0])
    if V_B.value_size > 1:
        V_B = firedrake.FunctionSpace(mesh_B, V_B.ufl_element().sub_elements[0])

    assert V_A.value_size == 1
    assert V_B.value_size == 1

    preallocator = PETSc.Mat().create(comm=mesh_A._comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)

    rset = V_B.dof_dset
    cset = V_A.dof_dset

    nrows = rset.layout_vec.getSizes()
    ncols = cset.layout_vec.getSizes()

    preallocator.setLGMap(rmap=rset.scalar_lgmap, cmap=cset.scalar_lgmap)
    preallocator.setSizes(size=(nrows, ncols), bsize=1)
    preallocator.setUp()

    zeros = numpy.zeros((V_B.cell_node_map().arity, V_A.cell_node_map().arity), dtype=ScalarType)
    for cell_A, dofs_A in enumerate(V_A.cell_node_map().values):
        for cell_B in likely(cell_A):
            dofs_B = V_B.cell_node_map().values_with_halo[cell_B, :]
            preallocator.setValuesLocal(dofs_B, dofs_A, zeros)
    preallocator.assemble()

    dnnz, onnz = get_preallocation(preallocator, nrows[0])

    # Unroll from block to AIJ
    dnnz = dnnz * cset.cdim
    dnnz = numpy.repeat(dnnz, rset.cdim)
    onnz = onnz * cset.cdim
    onnz = numpy.repeat(onnz, cset.cdim)
    preallocator.destroy()

    assert V_A.value_size == V_B.value_size
    rdim = V_B.dof_dset.cdim
    cdim = V_A.dof_dset.cdim

    #
    # Preallocate M_AB.
    #
    mat = PETSc.Mat().create(comm=mesh_A._comm)
    mat.setType(PETSc.Mat.Type.AIJ)
    rsizes = tuple(n * rdim for n in nrows)
    csizes = tuple(c * cdim for c in ncols)
    mat.setSizes(size=(rsizes, csizes),
                 bsize=(rdim, cdim))
    mat.setPreallocationNNZ((dnnz, onnz))
    mat.setLGMap(rmap=rset.lgmap, cmap=cset.lgmap)
    # TODO: Boundary conditions not handled.
    mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, False)
    mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
    mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, False)
    mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
    mat.setUp()

    evaluate_kernel_A = compile_element(ufl.Coefficient(V_A), name="evaluate_kernel_A")
    evaluate_kernel_B = compile_element(ufl.Coefficient(V_B), name="evaluate_kernel_B")

    # We only need one of these since we assume that the two meshes both have CG1 coordinates
    to_reference_kernel = to_reference_coordinates(mesh_A.coordinates.ufl_element())

    if dim == 2:
        reference_mesh = UnitTriangleMesh(comm=COMM_SELF)
    else:
        reference_mesh = UnitTetrahedronMesh(comm=COMM_SELF)
    evaluate_kernel_S = compile_element(ufl.Coefficient(reference_mesh.coordinates.function_space()), name="evaluate_kernel_S")

    V_S_A = FunctionSpace(reference_mesh, V_A.ufl_element())
    V_S_B = FunctionSpace(reference_mesh, V_B.ufl_element())
    M_SS = assemble(inner(TrialFunction(V_S_A), TestFunction(V_S_B)) * dx)
    M_SS = M_SS.M.handle[:, :]
    node_locations_A = utils.physical_node_locations(V_S_A).dat.data_ro_with_halos
    node_locations_B = utils.physical_node_locations(V_S_B).dat.data_ro_with_halos
    num_nodes_A = node_locations_A.shape[0]
    num_nodes_B = node_locations_B.shape[0]

    to_reference_kernel = to_reference_coordinates(mesh_A.coordinates.ufl_element())

    supermesh_kernel_str = """
    #include "libsupermesh-c.h"
    #include <petsc.h>
    %(to_reference)s
    %(evaluate_S)s
    %(evaluate_A)s
    %(evaluate_B)s
#define complex_mode %(complex_mode)s

    #define PrintInfo(...) do { if (PetscLogPrintInfo) printf(__VA_ARGS__); } while (0)
    #define FPrintInfo(...) do { if (PetscLogPrintInfo) fprintf(stderr, __VA_ARGS__); } while (0)
    static void print_array(PetscScalar *arr, int d)
    {
        for(int j=0; j<d; j++)
            FPrintInfo("%%+.2f ", arr[j]);
    }
    static void print_coordinates(PetscScalar *simplex, int d)
    {
        for(int i=0; i<d+1; i++)
        {
            PrintInfo("\t");
            print_array(&simplex[d*i], d);
            PrintInfo("\\n");
        }
    }
#if complex_mode
    static void seperate_real_and_imag(PetscScalar *simplex, double *real_simplex, double *imag_simplex, int d)
    {
        for(int i=0; i<d+1; i++)
        {
            for(int j=0; j<d; j++)
            {
                real_simplex[d*i+j] = creal(simplex[d*i+j]);
                imag_simplex[d*i+j] = cimag(simplex[d*i+j]);
            }
        }
    }
    static void merge_back_to_simplex(PetscScalar* simplex, double* real_simplex, double* imag_simplex, int d)
    {
        print_coordinates(simplex,d);
        for(int i=0; i<d+1; i++)
        {
            for(int j=0; j<d; j++)
            {
                simplex[d*i+j] = real_simplex[d*i+j]+imag_simplex[d*i+j]*_Complex_I;
            }
        }
    }
#endif
    int supermesh_kernel(PetscScalar* simplex_A, PetscScalar* simplex_B, PetscScalar* simplices_C,  PetscScalar* nodes_A,  PetscScalar* nodes_B,  PetscScalar* M_SS, PetscScalar* outptr, int num_ele)
    {
#define d %(dim)s
#define num_nodes_A %(num_nodes_A)s
#define num_nodes_B %(num_nodes_B)s

        double simplex_ref_measure;
        PrintInfo("simplex_A coordinates\\n");
        print_coordinates(simplex_A, d);
        PrintInfo("simplex_B coordinates\\n");
        print_coordinates(simplex_B, d);
        int num_elements = num_ele;

        if (d == 2) simplex_ref_measure = 0.5;
        else if (d == 3) simplex_ref_measure = 1.0/6;

        PetscScalar R_AS[num_nodes_A][num_nodes_A];
        PetscScalar R_BS[num_nodes_B][num_nodes_B];
        PetscScalar coeffs_A[%(num_nodes_A)s] = {0.};
        PetscScalar coeffs_B[%(num_nodes_B)s] = {0.};

        PetscScalar reference_nodes_A[num_nodes_A][d];
        PetscScalar reference_nodes_B[num_nodes_B][d];

#if complex_mode
        double real_simplex_A[d*(d+1)];
        double imag_simplex_A[d*(d+1)];
        seperate_real_and_imag(simplex_A, real_simplex_A, imag_simplex_A, d);
        double real_simplex_B[d*(d+1)];
        double imag_simplex_B[d*(d+1)];
        seperate_real_and_imag(simplex_B, real_simplex_B, imag_simplex_B, d);

        double real_simplices_C[num_elements*d*(d+1)];
        double imag_simplices_C[num_elements*d*(d+1)];
        for (int ii=0; ii<num_elements*d*(d+1); ++ii) imag_simplices_C[ii] = 0.;

        %(libsupermesh_intersect_simplices)s(real_simplex_A, real_simplex_B, real_simplices_C, &num_elements);

        merge_back_to_simplex(simplex_A, real_simplex_A, imag_simplex_A, d);
        merge_back_to_simplex(simplex_B, real_simplex_B, imag_simplex_B, d);
        for(int s=0; s<num_elements; s++)
        {
            PetscScalar* simplex_C = &simplices_C[s * d * (d+1)];
            double* real_simplex_C = &real_simplices_C[s * d * (d+1)];
            double* imag_simplex_C = &imag_simplices_C[s * d * (d+1)];
            merge_back_to_simplex(simplex_C, real_simplex_C, imag_simplex_C, d);
        }
#else
        %(libsupermesh_intersect_simplices)s(simplex_A, simplex_B, simplices_C, &num_elements);
#endif
        PrintInfo("Supermesh consists of %%i elements\\n", num_elements);

        // would like to do this
        //PetscScalar MAB[%(num_nodes_A)s][%(num_nodes_B)s] = (PetscScalar (*)[%(num_nodes_B)s])outptr;
        // but have to do this instead because we don't grok C
        PetscScalar (*MAB)[num_nodes_A] = (PetscScalar (*)[num_nodes_A])outptr;
        PetscScalar (*MSS)[num_nodes_A] = (PetscScalar (*)[num_nodes_A])M_SS; // note the underscore

        for ( int i = 0; i < num_nodes_B; i++ ) {
            for (int j = 0; j < num_nodes_A; j++) {
                MAB[i][j] = 0.0;
            }
        }

        for(int s=0; s<num_elements; s++)
        {
            PetscScalar* simplex_S = &simplices_C[s * d * (d+1)];
            double simplex_S_measure;
#if complex_mode
            double real_simplex_S[d*(d+1)];
            double imag_simplex_S[d*(d+1)];
            seperate_real_and_imag(simplex_S, real_simplex_S, imag_simplex_S, d);

            %(libsupermesh_simplex_measure)s(real_simplex_S, &simplex_S_measure);

            merge_back_to_simplex(simplex_S, real_simplex_S, imag_simplex_S, d);
#else
            %(libsupermesh_simplex_measure)s(simplex_S, &simplex_S_measure);
#endif
            PrintInfo("simplex_S coordinates with measure %%f\\n", simplex_S_measure);
            print_coordinates(simplex_S, d);

            PrintInfo("Start mapping nodes for V_A\\n");
            PetscScalar physical_nodes_A[num_nodes_A][d];
            for(int n=0; n < num_nodes_A; n++) {
                PetscScalar* reference_node_location = &nodes_A[n*d];
                PetscScalar* physical_node_location = physical_nodes_A[n];
                for (int j=0; j < d; j++) physical_node_location[j] = 0.0;
                pyop2_kernel_evaluate_kernel_S(physical_node_location, simplex_S, reference_node_location);
                PrintInfo("\\tNode ");
                print_array(reference_node_location, d);
                PrintInfo(" mapped to ");
                print_array(physical_node_location, d);
                PrintInfo("\\n");
            }
            PrintInfo("Start mapping nodes for V_B\\n");
            PetscScalar physical_nodes_B[num_nodes_B][d];
            for(int n=0; n < num_nodes_B; n++) {
                PetscScalar* reference_node_location = &nodes_B[n*d];
                PetscScalar* physical_node_location = physical_nodes_B[n];
                for (int j=0; j < d; j++) physical_node_location[j] = 0.0;
                pyop2_kernel_evaluate_kernel_S(physical_node_location, simplex_S, reference_node_location);
                PrintInfo("\\tNode ");
                print_array(reference_node_location, d);
                PrintInfo(" mapped to ");
                print_array(physical_node_location, d);
                PrintInfo("\\n");
            }
            PrintInfo("==========================================================\\n");
            PrintInfo("Start pulling back dof from S into reference space for A.\\n");
            for(int n=0; n < num_nodes_A; n++) {
                for(int i=0; i<d; i++) reference_nodes_A[n][i] = 0.;
                to_reference_coords_kernel(reference_nodes_A[n], physical_nodes_A[n], simplex_A);
                PrintInfo("Pulling back ");
                print_array(physical_nodes_A[n], d);
                PrintInfo(" to ");
                print_array(reference_nodes_A[n], d);
                PrintInfo("\\n");
            }
            PrintInfo("Start pulling back dof from S into reference space for B.\\n");
            for(int n=0; n < num_nodes_B; n++) {
                for(int i=0; i<d; i++) reference_nodes_B[n][i] = 0.;
                to_reference_coords_kernel(reference_nodes_B[n], physical_nodes_B[n], simplex_B);
                PrintInfo("Pulling back ");
                print_array(physical_nodes_B[n], d);
                PrintInfo(" to ");
                print_array(reference_nodes_B[n], d);
                PrintInfo("\\n");
            }

            PrintInfo("Start evaluating basis functions of V_A at dofs for V_A on S\\n");
            for(int i=0; i<num_nodes_A; i++) {
                coeffs_A[i] = 1.;
                for(int j=0; j<num_nodes_A; j++) {
                    R_AS[i][j] = 0.;
                    pyop2_kernel_evaluate_kernel_A(&R_AS[i][j], coeffs_A, reference_nodes_A[j]);
                }
                print_array(R_AS[i], num_nodes_A);
                PrintInfo("\\n");
                coeffs_A[i] = 0.;
            }
            PrintInfo("Start evaluating basis functions of V_B at dofs for V_B on S\\n");
            for(int i=0; i<num_nodes_B; i++) {
                coeffs_B[i] = 1.;
                for(int j=0; j<num_nodes_B; j++) {
                    R_BS[i][j] = 0.;
                    pyop2_kernel_evaluate_kernel_B(&R_BS[i][j], coeffs_B, reference_nodes_B[j]);
                }
                print_array(R_BS[i], num_nodes_B);
                PrintInfo("\\n");
                coeffs_B[i] = 0.;
            }
            PrintInfo("Start doing the matmatmat mult\\n");

            for ( int i = 0; i < num_nodes_B; i++ ) {
                for (int j = 0; j < num_nodes_A; j++) {
                    for ( int k = 0; k < num_nodes_B; k++) {
                        for ( int l = 0; l < num_nodes_A; l++) {
                            MAB[i][j] += (simplex_S_measure/simplex_ref_measure) * R_BS[i][k] * MSS[k][l] * R_AS[j][l];
                        }
                    }
                }
            }
        }
        return num_elements;
    }
    """ % {
        "evaluate_S": str(evaluate_kernel_S),
        "evaluate_A": str(evaluate_kernel_A),
        "evaluate_B": str(evaluate_kernel_B),
        "to_reference": str(to_reference_kernel),
        "num_nodes_A": num_nodes_A,
        "num_nodes_B": num_nodes_B,
        "libsupermesh_simplex_measure": "libsupermesh_triangle_area" if dim == 2 else "libsupermesh_tetrahedron_volume",
        "libsupermesh_intersect_simplices": "libsupermesh_intersect_tris_real" if dim == 2 else "libsupermesh_intersect_tets_real",
        "dim": dim,
        "complex_mode": 1 if complex_mode else 0
    }

    dirs = get_petsc_dir() + (sys.prefix, )
    includes = ["-I%s/include" % d for d in dirs]
    libs = ["-L%s/lib" % d for d in dirs]
    libs = libs + ["-Wl,-rpath,%s/lib" % d for d in dirs] + ["-lpetsc", "-lsupermesh"]
    lib = load(
        supermesh_kernel_str, "c", "supermesh_kernel",
        cppargs=includes,
        ldargs=libs,
        argtypes=[ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp],
        restype=ctypes.c_int,
        comm=mesh_A._comm
    )

    ammm(V_A, V_B, likely, node_locations_A, node_locations_B, M_SS, ctypes.addressof(lib), mat)
    if orig_value_size == 1:
        return mat
    else:
        (lrows, grows), (lcols, gcols) = mat.getSizes()
        lrows *= orig_value_size
        grows *= orig_value_size
        lcols *= orig_value_size
        gcols *= orig_value_size
        size = ((lrows, grows), (lcols, gcols))
        context = BlockMatrix(mat, orig_value_size)
        blockmat = PETSc.Mat().createPython(size, context=context, comm=mat.comm)
        blockmat.setUp()
        return blockmat
