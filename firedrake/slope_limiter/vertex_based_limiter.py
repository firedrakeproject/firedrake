from __future__ import absolute_import
from firedrake import dx, assemble, LinearSolver, MIN, MAX, project, Projector
from firedrake.interpolation import Interpolator
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.parloops import par_loop, READ, RW
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.slope_limiter.limiter import Limiter
from firedrake.constant import Constant
import numpy as np

__all__ = ("VertexBasedLimiter",)


def VertexBasedLimiter(space):
    # Returns a P1DG or P2DG limiter as determined by the given space
    if space.ufl_element().degree() == 1:
        return VertexBasedLimiter1(space)
    elif space.ufl_element().degree() == 2:
        return VertexBasedLimiter2(space)
    else:
        raise NotImplementedError("Given FunctionSpace instance "
                                  "must be of degree one or 2")


class VertexBasedLimiter1(Limiter):
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


class VertexBasedLimiter2(Limiter):
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
        self.P1DG = FunctionSpace(self.P2DG.mesh(), 'DG', 1)
        self.P1CG = FunctionSpace(self.P2DG.mesh(), 'CG', 1)
        self.P2TG = FunctionSpace(self.P2DG.mesh(), 'Discontinuous Taylor', 2)

        # Create fields
        self.taylor_field = Function(self.P2TG)
        self.linear_field = Function(self.P1DG)

        # field bound containers
        self.dx_max_field = Function(self.P1CG)
        self.dx_min_field = Function(self.P1CG)
        self.dy_max_field = Function(self.P1CG)
        self.dy_min_field = Function(self.P1CG)

        # P1DG limiter
        self.p1dg_limiter = VertexBasedLimiter1(self.P1DG)

        self._init_projections()
        self._create_tables()

    def revert(self, field):

        # project back and apply to original field
        self._project(f_in=self.taylor_field, f_out=field,
                      space=self.P2DG, projector=self.P2DG_projector)

    def apply_limiter(self, field):
        """
        Apply the limiter
        Must compute bounds before calling this method.
        """
        par_loop("""
        %(table_x)s
        %(table_y)s
        double alpha_x = 1.0;
        double alpha_y = 1.0;
        double dx_avg = q[1][0];
        double dy_avg = q[2][0];

        // Loop over degrees of freedom
        for ( int i = 0; i < dx_maxf.dofs; i++ ) {

            // interpolate the cell to the vertices
            double dxi = 0;
            double dyi = 0;
            for ( int j = 0; j < q.dofs; j++ ) {
                dxi += dx_table[i][j] * q[j][0];
                dyi += dy_table[i][j] * q[j][0];
            }

            // Calculate alpha_x
            if (dxi > dx_avg) {
                alpha_x = fmin(alpha_x, fmin(1, (dx_maxf[i][0] - dx_avg)/(dxi - dx_avg)));
            } else if (dxi < dx_avg) {
                alpha_x = fmin(alpha_x, fmin(1, (dx_avg - dx_minf[i][0])/(dx_avg - dxi)));
            }

            // Calculate alpha_y
            if (dyi > dy_avg) {
                alpha_y = fmin(alpha_y, fmin(1, (dy_maxf[i][0] - dy_avg)/(dyi - dy_avg)));
            } else if (dyi < dy_avg) {
                alpha_y = fmin(alpha_y, fmin(1, (dy_avg - dy_minf[i][0])/(dy_avg - dyi)));
            }

        }

        // Calculate alpha_e1 (Same as P1DG limiter)
        double alpha_e1 = 1.0;
        double qavg = qbar[0][0];
        for (int i=0; i < q_lin.dofs; i++) {
            if (q_lin[i][0] > qavg)
                alpha_e1 = fmin(alpha_e1, fmin(1, (qmax[i][0] - qavg)/(q_lin[i][0] - qavg)));
            else if (q_lin[i][0] < qavg)
                alpha_e1 = fmin(alpha_e1, fmin(1, (qavg - qmin[i][0])/(qavg - q_lin[i][0])));
        }

        double alpha_e2 = fmin(alpha_x, alpha_y);
        alpha_e1 = fmax(alpha_e1, alpha_e2);

        // Apply the values of alpha to limit function
        q[1][0] = alpha_e1 * q[1][0];
        q[2][0] = alpha_e1 * q[2][0];
        for(int i = 3; i < q.dofs; i++) {
            q[i][0] = alpha_e2 * q[i][0];
        }

                 """ % {'table_x': self.str_table_x, 'table_y': self.str_table_y},
                 dx,
                 {"q": (self.taylor_field, RW),
                  "q_lin": (self.linear_field, RW),
                  "qbar": (self.p1dg_limiter.centroids, READ),
                  "qmax": (self.p1dg_limiter.max_field, READ),
                  "qmin": (self.p1dg_limiter.min_field, READ),
                  "dx_maxf": (self.dx_max_field, READ),
                  "dx_minf": (self.dx_min_field, READ),
                  "dy_maxf": (self.dy_max_field, READ),
                  "dy_minf": (self.dy_min_field, READ)})

    def compute_bounds(self, field, taylor_field=None):

        # Project to taylor basis
        # self.taylor_field = project(field, self.P2TG)
        self._project(f_in=field, f_out=self.taylor_field,
                      space=self.P2TG, projector=self.P2TG_projector)

        # Make copy so can zero out higher derivatives
        copy_field = Function(self.taylor_field)

        # higher derivatives
        self.dx_max_field.assign(-1.0e50)  # small number
        self.dx_min_field.assign(1.0e50)  # big number

        self.dy_max_field.assign(-1.0e50)  # small number
        self.dy_min_field.assign(1.0e50)  # big number

        # Throw away higher derivatives and calculate max/min at vertices
        par_loop("""
                 for(int i = 3; i < q.dofs; i++) {
                     q[i][0] = 0;
                 }

                 for(int i = 0; i < dx_maxf.dofs; i++) {
                    dx_maxf[i][0] = fmax(dx_maxf[i][0], q[1][0]);
                    dx_minf[i][0] = fmin(dx_minf[i][0], q[1][0]);

                    dy_maxf[i][0] = fmax(dy_maxf[i][0], q[2][0]);
                    dy_minf[i][0] = fmin(dy_minf[i][0], q[2][0]);
                 }
                 """,
                 dx,
                 {"q": (copy_field, RW),
                  "dx_maxf": (self.dx_max_field, RW),
                  "dx_minf": (self.dx_min_field, RW),
                  "dy_maxf": (self.dy_max_field, RW),
                  "dy_minf": (self.dy_min_field, RW)})

        # Project back to linear field to apply limiter
        self._project(f_in=copy_field, f_out=self.linear_field,
                      space=self.P1DG, projector=self.P1DG_projector)

        # Compute max and min fields using P1DG limiter
        self.p1dg_limiter.compute_bounds(self.linear_field)

    def _project(self, f_in, f_out, space, projector):
        v = TestFunction(space)
        rhs = assemble(f_in*v*dx)
        with rhs.dat.vec_ro as v:
            with f_out.dat.vec_ro as v_out:
                projector.M.handle.mult(v, v_out)

    def _create_projection(self, space):
        u = TrialFunction(space)
        v = TestFunction(space)

        return assemble(u*v*dx, inverse=True)

    def _init_projections(self):
        # Create Projections
        self.P1DG_projector = self._create_projection(self.P1DG)
        self.P2DG_projector = self._create_projection(self.P2DG)
        self.P2TG_projector = self._create_projection(self.P2TG)

    def _create_tables(self):
        taylor_element = self.P2TG.fiat_element
        dim = self.P1DG.mesh().topological_dimension()
        if dim == 2:
            tables = taylor_element.tabulate(1, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            dx_table = tables[(0, 1)]
            dy_table = tables[(1, 0)]
        elif dim == 1:
            tables = taylor_element.tabulate(2, [[0.0], [1.0]])
            dx_table = tables[(1,)]
            dy_table = tables[(2,)]
        else:
            raise NotImplementedError("Only 1 and 2 dimensional meshes supported")

        self.str_table_x = create_c_table(dx_table, "dx_table")
        self.str_table_y = create_c_table(dy_table, "dy_table")

    def apply(self, field):
        assert field.function_space() == self.P2DG, \
            'Given field does not belong to this objects function space'

        self.compute_bounds(field)
        self.apply_limiter(field)
        self.revert(field)


def create_c_table(table, name):
    # Returns a C compatible string to be pasted in the kernel
    return "static const double %s[%d][%d] = %s;" % \
            ((name,) + table.T.shape + ("{{" + "}, \n{".join([", ".join(map(str, x)) for x in table.T]) + "}}", ))
