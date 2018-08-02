import numpy
import firedrake
import FIAT
from pyop2 import op2
from tsfc.fiatinterface import create_element
from tsfc import compile_expression_at_points
from firedrake.mesh import _from_cell_list, Mesh
from firedrake.mg.kernels import compile_element, to_reference_coordinates
from firedrake.parameters import parameters as default_parameters
from firedrake.libsupermesh import create_supermesh


def supermesh_wrapping(kernel_AB, get_map_AB, itspace_A, domain_A, kernel_coeffs_AB, coords_arg, Vtest=None):
    domains = set(coeff.ufl_domain() for coeff in kernel_coeffs_AB)
    domains.add(domain_A)
    domains.discard(None)
    if len(domains) == 1:
        # no supermeshing: everything is passed back unchanged
        return kernel_AB, get_map_AB, itspace_A, domain_A, kernel_coeffs_AB, coords_arg, []
    elif len(domains) > 2:
        raise NotImplemented("Cannot integrate form with more than 2 domains")

    domains.remove(domain_A)
    domain_B = domains.pop()
    super_mesh = SuperMesh(domain_A, domain_B)

    kernel_C = supermesh_kernel(kernel_AB, kernel_coeffs_AB, domain_A, domain_B, Vtest=Vtest)

    def get_map_C(x, bcs=None, decoration=None):
        map_AB = get_map_AB(x, bcs=bcs, decoration=decoration)
        return super_mesh.supermesh_pull_back_map(map_AB)

    coords_arg = make_coords_arg(super_mesh.mesh_C, get_map_AB)
    extra_coords_args = [make_coords_arg(domain_A, get_map_C), make_coords_arg(domain_B, get_map_C)]

    return kernel_C, get_map_C, super_mesh.mesh_C.cell_set, super_mesh.mesh_C, kernel_coeffs_AB, coords_arg, extra_coords_args


def supermesh_kernel(kernel, coeffs, mesh_A, mesh_B, Vtest=None):
    tdim = mesh_A.topological_dimension()
    assert tdim == mesh_B.topological_dimension()
    Anodes = Vtest.cell_node_map().arity if Vtest else 1
    Asize = Vtest.ufl_element().value_size() if Vtest else 1

    # first, interpolate reference coordinates for the nodes of every coeff and arg
    coeffs_and_args = coeffs.copy()
    if Vtest:
        coeffs_and_args.append(Vtest)
    interpolate_reference_coordinates = ''
    interpolate_kernels = ''
    for k, coeff in enumerate(coeffs_and_args):
        # kernel to interpolate ref coords of simplex in mesh_C to ref coords for nodes of coeff
        interpolate_name = 'interpolate_kernel_%d' % k
        interpolate_kernel = _interpolate_x_kernel(coeff, mesh_A.coordinates)
        interpolate_kernel.name = interpolate_name

        nnodes = coeff.cell_node_map().arity
        coeff_size = coeff.ufl_element().value_size()
        if coeff.ufl_domain() is mesh_A:
            Xref = 'XrefA'
        elif coeff.ufl_domain() is mesh_B:
            Xref = 'XrefB'

        interpolate_kernels += str(interpolate_kernel)
        interpolate_reference_coordinates += '''
    double wC_%(k)d[%(nnodes_times_coeff_size)d];
    double XrefC_%(k)d[%(nnodes)d][%(tdim)d];
    for ( int i = 0; i < %(nnodes)d; i++ ) {
        for ( int j = 0; j < %(tdim)d; j++ ) {
            XrefC_%(k)d[i][j] = 0.0;
        }
    }
    %(interpolate_name)s(XrefC_%(k)d, %(Xref)s);
    ''' % {'k': k,
           'nnodes': nnodes,
           'tdim': tdim,
           'nnodes_times_coeff_size': nnodes*coeff_size,
           'interpolate_name': interpolate_name,
           'Xref': Xref}

    # then, for coefficients only, we interpolate them in these nodes using the reference coordinates
    evaluate_coeffs_on_C = ''
    evaluate_kernels = ''
    for k, coeff in enumerate(coeffs):
        # kernel to evaluate coefficient for each ref. coord
        evaluate_name = 'evaluate_kernel_%d' % k
        evaluate_kernel = compile_element(coeff, name=evaluate_name)
        nnodes = coeff.cell_node_map().arity
        coeff_size = coeff.ufl_element().value_size()
        if coeff.ufl_domain() is mesh_A:
            Xref = 'XrefA'
        elif coeff.ufl_domain() is mesh_B:
            Xref = 'XrefB'

        evaluate_kernels += str(evaluate_kernel)
        evaluate_coeffs_on_C += '''
    for ( int i = 0; i < %(nnodes)d; i++ ) {
        for ( int j = 0; j < %(coeff_size)d; j++ ) {
            wC_%(k)d[i*%(coeff_size)d + j] = 0.0;
        }
        %(evaluate_name)s(wC_%(k)d + i*%(coeff_size)d, w_%(k)d, XrefC_%(k)d[i]);
    }''' % {
            'k': k,
            'nnodes': nnodes,
            'coeff_size': coeff_size,
            'evaluate_name': evaluate_name}

    if Vtest:
        # we want to use the last of the reference coordinates XrefC from before
        k = len(coeffs)
        # kernel to evaluate coefficient for each ref. coord
        evaluate_name = 'evaluate_kernel_%d' % k
        evaluate_kernel = compile_element(firedrake.TestFunction(Vtest), Vtest, name=evaluate_name)
        evaluate_kernels += str(evaluate_kernel)

        # we first assemble into a local vec A_C
        A_Cdecl = '''
    double A_C[%(Anodes_times_Asize)d];
    for ( int i = 0; i < %(Anodes_times_Asize)d; i++ ) {
        A_C[i] = 0.0;
    }''' % {'Anodes_times_Asize': Anodes * Asize}
        transform_argument = '''
    for ( int i = 0; i < %(Anodes)d; i++ ) {
        %(evaluate_name)s(A, A_C + i*%(Asize)d, XrefC_%(k)d[i]);
    }''' % {'k': k,
            'evaluate_name': evaluate_name,
            'Anodes': Anodes,
            'Asize': Asize}
    else:
        # don't need a temporary, assemble straight into A
        A_Cdecl = ''
        # no transformation needed
        transform_argument = ''

    # kernel to compute the refence coordinates of the supermesh C triangle within triangle A, and triangle B
    to_reference_kernel = to_reference_coordinates(mesh_A.coordinates.ufl_element())

    xnodes = mesh_A.coordinates.cell_node_map().arity
    kernel_source = '''
%(original_kernel)s
%(to_reference)s
%(interpolate_kernels)s
%(evaluate_kernels)s

static inline void supermesh_kernel(double A[%(Anodes_times_Asize)d], const double *restrict coords_C,%(coeffs)s, const double *restrict coords_A, const double *restrict coords_B) {
    double XrefA[%(xnodes_times_tdim)d], XrefB[%(xnodes_times_tdim)d];
    for ( int i = 0; i < %(xnodes)d; i++ ) {
        to_reference_coords_kernel(XrefA + i*%(tdim)d, coords_C + i*%(tdim)d, coords_A);
        to_reference_coords_kernel(XrefB + i*%(tdim)d, coords_C + i*%(tdim)d, coords_B);
    }
    %(interpolate_reference_coordinates)s
    %(evaluate_coeffs_on_C)s
    %(A_Cdecl)s
    %(original_kernel_name)s(%(A_or_A_C)s, coords_C,%(coeffs_C)s);
    %(transform_argument)s
}''' % {
        'original_kernel': str(kernel.code()),
        'original_kernel_name': kernel.name,
        'to_reference': str(to_reference_kernel),
        'evaluate_kernels': evaluate_kernels,
        'interpolate_kernels': interpolate_kernels,
        'interpolate_reference_coordinates': interpolate_reference_coordinates,
        'evaluate_coeffs_on_C': evaluate_coeffs_on_C,
        'transform_argument': transform_argument,
        'Anodes_times_Asize': Anodes * Asize,
        'A_Cdecl': A_Cdecl,
        'A_or_A_C': 'A_C' if Vtest else 'A',
        'tdim': tdim,
        'xnodes': xnodes,
        'xnodes_times_tdim': xnodes*tdim,
        'coeffs': ','.join(' const double *restrict w_%d' % k for k in range(len(coeffs))),
        'coeffs_C': ','.join(' wC_%d' % k for k in range(len(coeffs)))}
    opts = default_parameters['coffee']
    kernel_C = op2.Kernel(kernel_source, 'supermesh_kernel', opts)
    return kernel_C


class SuperMesh:
    def __init__(self, mesh_A, mesh_B):
        self.mesh_A = mesh_A
        self.mesh_B = mesh_B

        nodes_C, self.cell_map_CA, self.cell_map_CB = create_supermesh(mesh_A, mesh_B)
        shp_C = nodes_C.shape
        nodes_C = nodes_C.reshape((shp_C[0]*shp_C[1], shp_C[2]))
        cells_C = numpy.arange(shp_C[0]*shp_C[1]).reshape((shp_C[0], shp_C[1]))

        # this assert doesn't work - why?
        # assert mesh_A.comm == mesh_B.comm
        plex_C = _from_cell_list(mesh_A.topological_dimension(), cells_C, nodes_C, mesh_A.comm)
        self.mesh_C = Mesh(plex_C, reorder=False)
        self.mesh_C.init()

    def supermesh_pull_back_map(self, map):
        if map.iterset == self.mesh_A.cell_set:
            return _compose_map(self.cell_map_CA, self.mesh_C.cell_set, map)
        elif map.iterset == self.mesh_B.cell_set:
            return _compose_map(self.cell_map_CB, self.mesh_C.cell_set, map)
        else:
            # Need to think what other cases we can handle
            raise NotImplementedError


# FIXME: this is copy-pasta from interpolation._interpolator()
# I'm sure there's a better way to do this
def _interpolate_x_kernel(V, coords):
    """Kernel to interpolate coordinates to nodes"""
    to_element = create_element(V.ufl_element(), vector_is_mixed=False)
    to_pts = []

    if V.ufl_element().mapping() != "identity":
        raise NotImplementedError("Can only interpolate onto elements "
                                  "with affine mapping. Try projecting instead")

    for dual in to_element.dual_basis():
        if not isinstance(dual, FIAT.functional.PointEvaluation):
            raise NotImplementedError("Can only interpolate onto point "
                                      "evaluation operators. Try projecting instead")
        pts, = dual.pt_dict.keys()
        to_pts.append(pts)

    ast, oriented, coefficients = compile_expression_at_points(coords, to_pts, coords)
    return ast


def _compose_map(int_list_XY, set_X, map_YZ):
    """Compose map given by int_list_XY and op2 map_YZ: map_XZ(x) = map_YZ(int_list_XY(x)) for all x in set_X"""
    int_map_XZ = map_YZ.values[int_list_XY]
    map_XZ = op2.Map(set_X, map_YZ.toset, map_YZ.arity, values=int_map_XZ)
    return map_XZ


def make_coords_arg(mesh, get_map):
    coords = mesh.coordinates
    arg = coords.dat(op2.READ, get_map(coords)[op2.i[0]])
    return arg
