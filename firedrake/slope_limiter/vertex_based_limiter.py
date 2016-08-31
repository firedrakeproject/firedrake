from __future__ import absolute_import
from firedrake import dx, assemble, LinearSolver, MIN, MAX
from firedrake.interpolation import Interpolator
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.parloops import par_loop, READ, RW
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.slope_limiter.limiter import Limiter
from firedrake.constant import Constant

__all__ = ("KuzminLimiter",)


def KuzminLimiter(space):
    # Returns a P1DG or P2DG limiter as determined by the given space
    if space.ufl_element().degree() == 1:
        return KuzminLimiter1D(space)
    elif space.ufl_element().degree() == 2:
        return KuzminLimiter2D(space)
    else:
        raise NotImplementedError("Given FunctionSpace instance "
                                  "must be of degree 1 or 2")


class KuzminLimiter1D(Limiter):
    """
    A vertex based limiter for P1DG fields.

    This limiter implements the vertex-based limiting scheme described in
    Dmitri Kuzmin, "A vertex-based hierarchical slope limiter for p-adaptive
    discontinuous Galerkin methods". J. Comp. Appl. Maths (2010)
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """

    def __init__(self, space):
        """
        Initialise limiter

        :param space : FunctionSpace instance
        """

        self.P1DG = space
        self.P1CG = FunctionSpace(self.P1DG.mesh(), 'CG', 1)  # for min/max limits
        self.P0 = FunctionSpace(self.P1DG.mesh(), 'DG', 0)  # for centroids

        # Storage containers for cell means, max and mins
        self.centroids = Function(self.P0)
        self.max_field = Function(self.P1CG)
        self.min_field = Function(self.P1CG)
        self.centroid_solver = self._construct_centroid_solver()
        self.max_interpolate = Interpolator(self.centroids, self.max_field, access=MAX)
        self.min_interpolate = Interpolator(self.centroids, self.min_field, access=MIN)

        # Perform limiting loop
        self._limit_kernel = """
double alpha = 1.0;
double qavg = qbar[0][0];
for (int i=0; i < q.dofs; i++) {
    if (q[i][0] > qavg)
        alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(q[i][0] - qavg)));
    else if (q[i][0] < qavg)
        alpha = fmin(alpha, fmin(1, (qavg - qmin[i][0])/(qavg - q[i][0])));
}

for (int i=0; i<q.dofs; i++) {
    q[i][0] = qavg + alpha*(q[i][0] - qavg);
}
                             """

    def _construct_centroid_solver(self):
        """
        Constructs a linear problem for computing the centroids

        :return: LinearSolver instance
        """
        u = TrialFunction(self.P0)
        v = TestFunction(self.P0)
        a = assemble(u * v * dx)
        return LinearSolver(a, solver_parameters={'ksp_type': 'preonly',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'})

    def _update_centroids(self, field):
        """
        Update centroid values
        """
        b = assemble(TestFunction(self.P0) * field * dx)
        self.centroid_solver.solve(self.centroids, b)

    def compute_bounds(self, field):
        """
        Only computes min and max bounds of neighbouring cells
        """
        self._update_centroids(field)
        self.max_field.assign(-1.0e30)  # small number
        self.min_field.assign(1.0e30)  # big number

        # FIXME: Relies on implementation detail in PyOP2
        # since op2.RW does not work the way it was expected to
        # we are using op2.WRITE in all cases and it happens to
        # read and write from the buffer
        self.min_interpolate.interpolate()
        self.max_interpolate.interpolate()

    def apply_limiter(self, field):
        """
        Only applies limiting loop on the given field
        """

        par_loop(self._limit_kernel, dx,
                 {"qbar": (self.centroids, READ),
                  "q": (field, RW),
                  "qmax": (self.max_field, READ),
                  "qmin": (self.min_field, READ)})

    def apply(self, field):
        """
        Re-computes centroids and applies limiter to given field
        """
        assert field.function_space() == self.P1DG, \
            'Given field does not belong to this objects function space'

        self.compute_bounds(field)
        self.apply_limiter(field)


class KuzminLimiter2D(Limiter):
    """
    A vertex based limiter for P2DG fields.

    This limiter implements the vertex-based limiting scheme described in
    Dmitri Kuzmin, "A vertex-based hierarchical slope limiter for p-adaptive
    discontinuous Galerkin methods". J. Comp. Appl. Maths (2010)
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """

    def __init__(self, space):
        """
        Initialise limiter

        :param space : FunctionSpace instance
        """
        self.P2DG = space
        self.P2DT = FunctionSpace(self.P2DG.mesh(), 'Discontinuous Taylor', 2)
        self.P1CG = FunctionSpace(self.P2DG.mesh(), 'CG', 1)
        self.P0 = FunctionSpace(self.P2DG.mesh(), 'DG', 0)  # for centroids

        # Find dimension of mesh
        self.dim = self.P2DG.mesh().topological_dimension()
        if self.dim > 1:
            raise NotImplementedError("Only Interval Meshes are currently supported")

        # Create fields
        self.taylor_field = Function(self.P2DT)

        # field gradients bound containers
        self.max_field = Function(self.P1CG)
        self.min_field = Function(self.P1CG)
        self.dx_max_field = Function(self.P1CG)
        self.dx_min_field = Function(self.P1CG)

        # Algorithm to find optimal alpha value
        self.calculate_alpha = """
void calculate_alpha(double* alpha, double** max_field, double** min_field,
                     double vertex_value, double center_value, int i) {
    if (vertex_value > max_field[i][0]) {
        *alpha = fmin(*alpha, fmin(1, (max_field[i][0] - center_value)/(vertex_value - center_value)));
    } else if (vertex_value < min_field[i][0]) {
        *alpha = fmin(*alpha, fmin(1, (min_field[i][0] - center_value)/(vertex_value - center_value)));
    }
}
"""

        # Cache projectors and fiat element tables
        self._init_projections()
        self._create_tables()

    def apply_limiter(self, field):
        """
        Apply the limiter
        Must compute bounds before calling this method.

        Parameters
        ----------
        field: Function which belongs to P2DG space that will be limited
        """

        par_loop("""
        %(calculate_alpha)s
        %(table)s
        %(table_x)s
        double alpha = 1.0;
        double alpha_dx = 1.0;
        double t_c = t[0][0];
        double t_dx = t[1][0];
        for(int i = 0; i < max.dofs; i++) {
            double u_c = 0;
            double u_dx = 0;

            // interpolate the cell average and derivative to the vertex linearly
            for ( int j = 0; j < 2; j++ ) {
                u_c += table[i][j] * t[j][0];
            }
            for ( int j = 1; j < t.dofs; j++ ) {
                u_dx += table_x[i][j] * t[j][0];
            }

            calculate_alpha(&alpha, max, min, u_c, t_c, i);
            calculate_alpha(&alpha_dx, max_dx, min_dx, u_dx, t_dx, i);
        }

        t[1][0] *= fmax(alpha, alpha_dx);
        t[2][0] *= alpha_dx;
        """ % {'calculate_alpha': self.calculate_alpha, 'table': self.str_table, 'table_x': self.str_table_x},
            dx,
            {"t": (self.taylor_field, RW),
             "max": (self.max_field, READ),
             "min": (self.min_field, READ),
             "max_dx": (self.dx_max_field, READ),
             "min_dx": (self.dx_min_field, READ)})

        self._project(f_in=self.taylor_field, f_out=field, space=self.P2DG, projector=self.P2DG_projector)

    def compute_bounds(self, field):

        # Project to taylor basis
        self._project(f_in=field, f_out=self.taylor_field,
                      space=self.P2DT, projector=self.P2DT_projector)

        # Throw away higher derivatives and calculate max/min at vertices
        # Setup fields
        self.min_field.assign(1.0e20)
        self.max_field.assign(-1.0e20)

        self.dx_min_field.assign(1.0e20)
        self.dx_max_field.assign(-1.0e20)
        # Calculate max/min of cell gradients at vertices
        par_loop("""
                 for(int i = 0; i < max.dofs; i++) {
                    min[i][0] = fmin(t[0][0], min[i][0]);
                    max[i][0] = fmax(t[0][0], max[i][0]);

                    dx_max[i][0] = fmax(t[1][0], dx_max[i][0]);
                    dx_min[i][0] = fmin(t[1][0], dx_min[i][0]);
                 }
                 """,
                 dx,
                 {"dx_max": (self.dx_max_field, RW),
                  "dx_min": (self.dx_min_field, RW),
                  "min": (self.min_field, RW),
                  "max": (self.max_field, RW),
                  "t": (self.taylor_field, RW)})

    def _project(self, f_in, f_out, space, projector):
        v = TestFunction(space)
        rhs = assemble(f_in*v*dx)
        with rhs.dat.vec_ro as v:
            with f_out.dat.vec_ro as v_out:
                projector.M.handle.mult(v, v_out)

    def _init_projections(self):
        # Stores projectors
        self.P2DG_projector = self._create_projection(self.P2DG)
        self.P2DT_projector = self._create_projection(self.P2DT)

    def _create_projection(self, space):
        # Creates a Projection
        u = TrialFunction(space)
        v = TestFunction(space)

        return assemble(u*v*dx, inverse=True)

    def _create_tables(self):
        # Field fiat element, dimension and cell
        fiat_element = self.P2DT.fiat_element
        cell = fiat_element.get_reference_element()
        vertices = cell.get_vertices()

        key = {1: (0, ),
               2: (0, 0)}[self.dim]

        key_x = {1: (1, ),
                 2: (1, 0)}[self.dim]

        table = fiat_element.tabulate(0, vertices)[key]
        table_x = fiat_element.tabulate(1, vertices)[key_x]

        self.str_table = create_c_table(table, "table")
        self.str_table_x = create_c_table(table_x, "table_x")

    def apply(self, field):
        assert field.function_space() == self.P2DG, \
            'Given field does not belong to this objects function space'

        self.compute_bounds(field)
        self.apply_limiter(field)


def create_c_table(table, name):
    # Returns a C compatible string to be pasted in the kernel
    return "static const double %s[%d][%d] = %s;" % \
           ((name,) + table.T.shape + ("{{" + "}, \n{".join([", ".join(map(str, x)) for x in table.T]) + "}}", ))
