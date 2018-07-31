import FIAT
import ufl
from pyop2 import op2
from tsfc.fiatinterface import create_element
from tsfc import compile_expression_at_points
from firedrake.mesh import _from_cell_list, Mesh
from firedrake.mg.kernels import compile_element, to_reference_coordinates
from firedrake.parameters import parameters as default_parameters


def supermesh_wrapping(form, kernel_AB, get_map_AB, itspace_A, domain_A, kernel_coeffs_AB, coords_arg):
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

    kernel_C = supermesh_kernel(kernel_AB, kernel_coeffs_AB, domain_A, domain_B)

    def get_map_C(x, bcs=None, decoration=None):
        map_AB = get_map_AB(x, bcs=bcs, decoration=decoration)
        return super_mesh.supermesh_pull_back_map(map_AB)

    coords_arg = make_coords_arg(super_mesh.mesh_C, get_map_AB)
    extra_coords_args = [make_coords_arg(domain_A, get_map_C), make_coords_arg(domain_B, get_map_C)]

    return kernel_C, get_map_C, super_mesh.mesh_C.cell_set, super_mesh.mesh_C, kernel_coeffs_AB, coords_arg, extra_coords_args


def supermesh_kernel(kernel, coeffs, mesh_A, mesh_B):
    tdim = mesh_A.topological_dimension()
    assert tdim == mesh_B.topological_dimension()

    evaluate_coeffs_on_C = ''
    evaluate_kernels = ''
    interpolate_kernels = ''
    for i, coeff in enumerate(coeffs):
        # kernel to interpolate ref coords of simplex in mesh_C to ref coords for nodes of coeff
        interpolate_name = 'interpolate_kernel_%(i)d' % {'i': i}
        interpolate_kernel = _interpolate_x_kernel(coeff, mesh_A.coordinates)
        interpolate_kernel.name = interpolate_name
        # kernel to evaluate coefficient for each ref. coord
        evaluate_name = 'evaluate_kernel_%(i)d' % {'i': i}
        evaluate_kernel = compile_element(coeff, name=evaluate_name)

        ctype = evaluate_kernel.args[0].typ
        nnodes = coeff.cell_node_map().arity
        coeff_size = ufl.product(coeff.ufl_shape)
        if coeff.ufl_domain() is mesh_A:
            Xref = 'XrefA'
        elif coeff.ufl_domain() is mesh_B:
            Xref = 'XrefB'

        evaluate_kernels += str(evaluate_kernel)
        interpolate_kernels += str(interpolate_kernel)
        evaluate_coeffs_on_C += '''
    %(ctype)s wC_%(i)d[%(nnodes_times_coeff_size)d];
    double XrefC_%(i)d[%(nnodes)d][%(tdim)d];
    for ( int i = 0; i < %(nnodes)d; i++ ) {
        for ( int j = 0; j < %(tdim)d; j++ ) {
            XrefC_%(i)d[i][j] = 0.0;
        }
    }
    %(interpolate_name)s(XrefC_%(i)d, %(Xref)s);
    for ( int i = 0; i < %(nnodes)d; i++ ) {
        for ( int j = 0; j < %(coeff_size)d; j++ ) {
            wC_%(i)d[i*%(coeff_size)d + j] = 0.0;
        }
        %(evaluate_name)s(wC_%(i)d + i*%(coeff_size)d, w_%(i)d, XrefC_%(i)d[i]);
    }''' % {
            'ctype': ctype,
            'i': i,
            'nnodes': nnodes,
            'coeff_size': coeff_size,
            'tdim': tdim,
            'nnodes_times_coeff_size': nnodes*coeff_size,
            'evaluate_name': evaluate_name,
            'interpolate_name': interpolate_name,
            'Xref': Xref}

    to_reference_kernel = to_reference_coordinates(mesh_A.coordinates.ufl_element())

    xnodes = mesh_A.coordinates.cell_node_map().arity
    kernel_source = '''
%(original_kernel)s
%(to_reference)s
%(interpolate_kernels)s
%(evaluate_kernels)s

static inline void supermesh_kernel(double A[%(Asize)d], const double *restrict coords_C,%(coeffs)s, const double *restrict coords_A, const double *restrict coords_B) {
    double XrefA[%(xnodes_times_tdim)d], XrefB[%(xnodes_times_tdim)d];
    for ( int i = 0; i < %(xnodes)d; i++ ) {
        to_reference_coords_kernel(XrefA + i*%(tdim)d, coords_C + i*%(tdim)d, coords_A);
        to_reference_coords_kernel(XrefB + i*%(tdim)d, coords_C + i*%(tdim)d, coords_B);
    }
    %(evaluate_coeffs_on_C)s;
    %(original_kernel_name)s(A, coords_C,%(coeffs_C)s);
}''' % {
        'original_kernel': str(kernel.code()),
        'original_kernel_name': kernel.name,
        'to_reference': str(to_reference_kernel),
        'evaluate_kernels': evaluate_kernels,
        'interpolate_kernels': interpolate_kernels,
        'evaluate_coeffs_on_C': evaluate_coeffs_on_C,
        'Asize': 1,
        'tdim': tdim,
        'xnodes': xnodes,
        'xnodes_times_tdim': xnodes*tdim,
        'coeffs': ','.join(' const double *restrict w_%(i)d' % {'i': i} for i in range(len(coeffs))),
        'coeffs_C': ','.join(' wC_%(i)d' % {'i': i} for i in range(len(coeffs)))}
    print(kernel_source)
    opts = default_parameters['coffee']
    kernel_C = op2.Kernel(kernel_source, 'supermesh_kernel', opts)
    return kernel_C


class SuperMesh:
    def __init__(self, mesh_A, mesh_B):
        self.mesh_A = mesh_A
        self.mesh_B = mesh_B

        # magic
        nodes_C = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]]
        cells_C = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]

        # more magic
        self.cell_map_CA = [0, 0, 1, 1]
        self.cell_map_CB = [0, 1, 1, 0]

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
