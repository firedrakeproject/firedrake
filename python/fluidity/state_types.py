#
#
# Note for future develoeprs: Wrap any
#  scipy calls in try blocks. Scipy
#  is not avaiable on many supercomputers
#  but we still want state_types
#
#

import numpy
import copy


class State:

    def __init__(self, n=""):
        self.scalar_fields = {}
        self.vector_fields = {}
        self.tensor_fields = {}
        self.csr_matrices = {}
        self.meshes = {}
        self.halos = {}
        self.name = n

    def __repr__(self):
        return '(State) %s' % self.name

    def print_fields(self):
        print "scalar: ", self.scalar_fields, "\nvector:", \
            self.vector_fields, "\ntensor:", self.tensor_fields, \
            "\ncsr_matrices:", self.csr_matrices


class Field:

    def __init__(self, n, ft, op, description, uid, mesh=None):
        self.name = n
        self.field_type = ft
        self.option_path = op
        self.description = description
        self.uid = uid
        if mesh:
            self.set_mesh(mesh)

    def __repr__(self):
        return '(%s) %s' % (self.description, self.name)

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.element_count = self.mesh.element_count
        self.ele_count = self.mesh.element_count

    def shape(self):
        return self.mesh.shape

    def ele_loc(self, ele_number):
        # Return the number of nodes of element ele_number
        return self.mesh.ele_loc(ele_number)

    def ele_nodes(self, ele_number):
        # Return a pointer to a vector containing the global node numbers of
        # element ele_number, i.e. all node numbers associated with the element
        # ele_number
        return self.mesh.ele_nodes(ele_number)

    def ele_shape(self, ele_number):
        return self.mesh.shape

    def addto(self, node, val):
        '''Add val to node of this field. If node is a scalar then val must
        have the shape of one data item in this field. If node is a sequence
        then the leading dimension of val must match the length of node and
        the remaining dimensions must match the shape of a data item in this
        field.'''
        try:
            for ii, vv in zip(node, val):
                self.val[ii] = self.val[ii] + vv
        except TypeError:
            # In this case it's presumably scalars.
            self.val[node] = self.val[node] + val

    def set(self, node, val):
        '''Set node of this field to val.  If node is a scalar then val must
        have the shape of one data item in this field. If node is a sequence
        then the leading dimension of val must match the length of node and
        the remaining dimensions must match the shape of a data item in this
        field.'''

        try:
            for ii, vv in zip(node, val):
                self.val[ii] = vv
        except TypeError:
            self.val[node] = val

    def __setitem__(self, node, val):
        self.set(node, val)

    def node_val(self, node):
        if node != int(node):
            raise TypeError
        if node > self.node_count:
            raise IndexError
        return self.val[node]

    def __getitem__(self, node):
        return self.node_val(node)

    def ele_val(self, ele_number):
        # Return the values of field at the nodes of ele_number
        return numpy.array(map(self.node_val, self.ele_nodes(ele_number)))

    def ele_val_at_quad(self, ele_number):
        # Return the values of field at the quadrature points of ele_number
        shape_n = self.ele_shape(ele_number).n
        ele_val = self.ele_val(ele_number)
        return numpy.array(numpy.dot(ele_val, shape_n))

    def ele_region_id(self, ele_number):
        return self.mesh.ele_region_id(ele_number)

    def remap_ele(self, ele_number, mesh):
        assert self.mesh.continuity >= mesh.continuity
        if mesh.continuity >= 0:  # if we are CG
            assert self.mesh.shape.degree <= mesh.shape.degree
        assert not (self.mesh.shape.type == "bubble" and
                    mesh.shape.type == "lagrangian")

        # we should check for periodic/nonperiodic here, but
        # our mesh type doesn't know whether it's periodic or not ...

        # we really ought to cache locweight, as it's constant
        # for each element, and only depends on the target mesh --
        # but we currently don't, sorry
        locweight = numpy.zeros((mesh.shape.loc, self.mesh.shape.loc))
        for i in range(mesh.shape.loc):
            for j in range(self.mesh.shape.loc):
                locweight[i, j] = self.mesh.shape.eval_shape(
                    j, mesh.shape.local_coords(i))

        return numpy.dot(locweight, self.ele_val(ele_number))


class ScalarField(Field):

    "A scalar field"
    description = "ScalarField"

    def __init__(self, n, v, ft, op, uid, mesh=None):
        Field.__init__(self, n, ft, op, self.description, uid, mesh)
        self.val = v
        self.node_count = self.val.shape[0]


class VectorField(Field):

    "A vector field"
    description = "VectorField"

    def __init__(self, n, v, ft, op, dim, uid, mesh=None):
        Field.__init__(self, n, ft, op, self.description, uid, mesh)
        self.val = v
        self.dimension = dim
        self.node_count = self.val.shape[0]


class TensorField(Field):

    "A tensor field"
    description = "VectorField"

    def __init__(self, n, v, ft, op, dim0, dim1, uid, mesh=None):
        Field.__init__(self, n, ft, op, self.description, uid, mesh)
        self.val = v
        self.dimension = numpy.array([dim0, dim1])
        self.node_count = self.val.shape[0]

# This is an example of wrapping up a class in a try block
# to prevent scipy being imported
try:
    import scipy
    import scipy.sparse

    class CsrMatrix(scipy.sparse.csr_matrix):

        "A csr matrix"

        def __init__(self, *args, **kwargs):
            try:
                scipy.sparse.csr_matrix.__init__(self, *args, **kwargs)
                self.format = 'csr'
            except TypeError:  # old version of scipy
                pass
except ImportError:
    class CsrMatrix(object):

        def __init__(self, *args, **kwargs):
            raise ImportError("No such module scipy.sparse")


class Halo:

    "A dummy halo"

    def __init__(self, *args, **kwargs):
        pass


class Mesh:

    "A mesh"

    def __init__(self, ndglno, elements, element_classes, nodes, node_classes,
                 continuity, name, parent, option_path, region_ids, uid,
                 node_halo=None, element_halo=None):
        self.ndglno = ndglno
        self.element_count = elements
        self._element_classes = element_classes
        self.node_count = nodes
        self._node_classes = node_classes
        self.continuity = continuity
        self.name = name
        self.parent = parent
        self.option_path = option_path
        self.region_ids = region_ids
        self.shape = Element(
            0, 0, 0, 0, [], [], [], 0, 0, 0, 0, "unknown", "unknown")
        self.node_halo = node_halo
        self.element_halo = element_halo
        self.uid = uid
        self.coords = None
        self.faces = None

    def __repr__(self):
        return '(Mesh) %s' % self.name

    def ele_shape(self, ele_number):
        # Returns the shape of this mesh
        return self.shape

    def ele_loc(self, ele_number):
        # Returns the loc of the shape of this mesh
        return self.shape.loc

    def ele_nodes(self, ele_number):
        # Return all nodes associated with the element ele_number
        base = self.shape.loc * ele_number
        nodes = []
        for i in range(self.shape.loc):
            # Subtract 1, since the nodes are numbered from 1 in ndglno
            nodes.append(self.ndglno[base + i] - 1)
        return nodes

    def ele_region_id(self, ele_number):
        # Return the region_id of element ele_number
        return self.region_ids[ele_number]


class Faces:

    "Mesh faces"

    def __init__(self, surface_node_list, face_element_list, boundary_ids):
        self.surface_node_list = surface_node_list
        self.face_element_list = face_element_list
        self.boundary_ids = boundary_ids
        self.boundaries = {}
        self.face_list = None
        self.surface_mesh = None


class Element:

    "An element"

    def __init__(self, dim, loc, ngi, degree, n, dn, coords, size_spoly_x,
                 size_spoly_y, size_dspoly_x, size_dspoly_y, family, type):
        self.dimension = dim  # 2d or 3d?
        self.loc = loc  # Number of nodes
        self.ngi = ngi  # Number of gauss points
        self.degree = degree  # Polynomial degree of element
        # Shape functions: n is for the primitive function, dn is for partial
        # derivatives, dn_s is for partial derivatives on surfaces
        # n is loc x ngi, dn is loc x ngi x dim
        self.n = n
        self.dn = dn
        self.coords = coords
        self.family = family
        self.type = type

        # Initialize spoly and dspoly, make sure to transpose due to Fortran
        # silliness
        self.spoly = [[Polynomial([], 0) for inner in range(size_spoly_y)]
                      for outer in range(size_spoly_x)]
        self.dspoly = [[Polynomial([], 0) for inner in range(size_dspoly_y)]
                       for outer in range(size_dspoly_x)]

    # Methods for setting up attributes
    def set_quadrature(self, quadrature):
        self.quadrature = quadrature

    def set_surface_quadrature(self, quadrature):
        self.surface_quadrature = quadrature

    def set_polynomial_s(self, poly, x, y):
        self.spoly[x - 1][y - 1] = poly

    def set_polynomial_ds(self, poly, x, y):
        self.dspoly[x - 1][y - 1] = poly

    def eval_shape(self, node, coords):
        result = 1.0
        for i in range(len(self.spoly)):
            result = result * self.spoly[i][node].eval(coords[i])
        return result

    def local_coords(self, node):
        return self.coords[node, :]


class Quadrature:

    "Quadrature"

    def __init__(self, w, locations, dim, degree, loc, ngi):
        # Dimension of the elements for which quadrature is required.
        self.dimension = dim
        self.degree = degree  # Degree of accuracy of quadrature
        self.loc = loc        # Number of vertices of the element.
        self.ngi = ngi        # Number of quadrature points
        self.weights = w       # Quadrature weights
        self.locations = locations  # Locations of quadrature points


class Polynomial:

    "Polynomial"

    def __init__(self, coefs, degree):
        self.coefficients = coefs
        self.degree = degree

    def __repr__(self):
        return '(Polynomial)' + str(self.coefficients)

    def eval(self, x):
        val = 0.0
        for i in range(self.degree + 1):
            val = val + self.coefficients[i] * x ** i
        return val


class Transform:

    "Transform with information about the detwei and Jacobian"
    # Note that so far only the dim == ldim == (2||3) have been tested

    def __init__(self, ele_num, field):
        self.ele_num = ele_num
        self.element = field.mesh.shape
        self.field = field
        # Jacobian matrix and its inverse at each quadrature point
        # (dim x dim x field.mesh.shape.ngi)
        # Facilitates access to this information externally
        self.J = [numpy.zeros((field.dimension, self.element.dimension))
                  for gi in range(self.element.ngi)]
        self.invJ = [numpy.zeros((self.element.dimension, field.dimension))
                     for gi in range(self.element.ngi)]
        self.detwei = numpy.zeros(self.element.ngi)
        self.det = numpy.zeros(self.element.ngi)
        # Calculate detwei, i.e. the gauss weights transformed by the
        # coordinate transform from real to computational space
        self.transform_to_physical_detwei(field)

    def set_J(self, J, gi):
        # Set J for the specified quadrature point and calculate its inverse
        self.J[gi] = J
        S = numpy.linalg.svd(J, compute_uv=False)
        self.det[gi] = abs(
            reduce(lambda x, y: x * y, [s for s in S if s > 0], 1))
        self.detwei[gi] = self.det[gi] * self.element.quadrature.weights[gi]
        self.invJ[gi] = numpy.linalg.pinv(J)

    def transform_to_physical_detwei(self, field):
        X = numpy.transpose(numpy.matrix(field.ele_val(self.ele_num)))
        for gi in range(self.element.ngi):
            J = numpy.dot(X, self.element.dn[:, gi, :])
            self.set_J(J, gi)

    def grad(self, shape):
        newshape = copy.copy(shape)
        newshape.dn = numpy.zeros((shape.loc, shape.ngi, self.field.dimension))

        for gi in range(self.field.mesh.shape.ngi):
            for i in range(shape.loc):
                newshape.dn[i, gi, :] = numpy.dot(shape.dn[i, gi, :], self.invJ[gi])
        return newshape

    def shape_shape(self, shape1, shape2, coeff=None):
        # For each node in each element shape1, shape2 calculate the
        # coefficient of the integral int(shape1shape2)dV.
        #
        # In effect, this calculates a mass matrix.
        m = numpy.zeros((shape1.loc, shape2.loc))

        if coeff is None:
            for i in range(shape1.loc):
                for j in range(shape2.loc):
                    m[i, j] = numpy.dot(shape1.n[i] * shape2.n[j], self.detwei)
        else:
            assert len(coeff) == len(self.detwei)
            for i in range(shape1.loc):
                for j in range(shape2.loc):
                    m[i, j] = numpy.dot(
                        shape1.n[i] * shape2.n[j], numpy.array(self.detwei) * coeff)
        return m

    def shape_dshape(self, shape, dshape):
        # For each node in element shape and transformed gradient dshape
        # calculate the coefficient of the integral int(shape dshape)dV

        # The dimensions of dshape are: (nodes, gauss points, dimensions)

        # dshape is usually the element returned by calling element.grad()

        # dshape_loc = size(self.gradient)
        dshape_loc = len(dshape.dn)
        dshape_dim = dshape.dn.shape[2]

        shape_dshape = numpy.zeros((shape.loc, dshape_loc, dshape_dim))
        for i in range(shape.loc):
            for j in range(dshape_loc):
                shape_dshape[i, j, :] = (numpy.matrix(self.detwei) *
                                         numpy.matrix(self.spread(shape.n[i], dshape_dim) * dshape.dn[j])).reshape((dshape_dim,))
        return shape_dshape

    def spread(self, arr, dim):
        # Simple version of the Fortran spread function which returns
        # an array of a higher dimension with the same values as the original
        # one in the new dimension
        if dim == 1:
            return arr
        elif dim == 2:
            return numpy.column_stack((arr, arr))
        elif dim == 3:
            return numpy.column_stack((arr, arr, arr))


def test_shape_dshape(state):
    # Tests shape_dshape (surprised?) - pass in the state with the coordiate
    # field
    cf = state.vector_fields['Coordinate']
    lumpmass = ScalarField("LumpMass", numpy.zeros(cf.node_count), "", "")
    lumpmass.set_mesh(cf.mesh)

    # Set our linear function f to the x values of the Coordinate Mesh
    f = ScalarField("Linear", cf.val1, "", "")
    f.set_mesh(cf.mesh)

    psi = VectorField("psi", numpy.zeros(cf.node_count), numpy.zeros(
        cf.node_count), numpy.zeros(cf.node_count), "", "", 2)
    psi.set_mesh(cf.mesh)

    for el in range(lumpmass.element_count()):
        el_f = lumpmass.ele_nodes(el)
        t = Transform(el, cf)
        masslump_local = t.shape_shape(cf.mesh.shape, cf.mesh.shape).sum(1)
        lumpmass.addto(el_f, masslump_local)
        rhs_local = t.shape_dshape(f.mesh.shape, t.grad(f.mesh.shape))
        rhs_f = 0
        for i in range(rhs_local.shape[1]):
            rhs_f = rhs_f + numpy.matrix(rhs_local[:, i, :]) * f.node_val(el_f[i])
        psi.addto(el_f, rhs_f)

    print psi.val[0] / lumpmass.val

    return 0
