from __future__ import absolute_import
from firedrake import dx, assemble, LinearSolver, MIN, MAX, project, Projector
from firedrake.interpolation import Interpolator
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.parloops import par_loop, READ, RW, WRITE
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.slope_limiter.limiter import Limiter
from firedrake.constant import Constant
import numpy as np

__all__ = ("KuzminLimiter",)


def KuzminLimiter(space):
    # Returns a P1DG or P2DG limiter as determined by the given space
    if space.ufl_element().degree() == 1:
        return KuzminLimiter1(space)
    elif space.ufl_element().degree() == 2:
        return KuzminLimiter2(space)
    else:
        raise NotImplementedError("Given FunctionSpace instance "
                                  "must be of degree one or 2")


class KuzminLimiter1(Limiter):
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
        # alpha_0 = Function(self.P0)

        par_loop(self._limit_kernel, dx,
                 {"qbar": (self.centroids, READ),
                  "q": (field, RW),
                  "qmax": (self.max_field, READ),
                  "qmin": (self.min_field, READ)})

        # self.ts = FunctionSpace(self.P1DG.mesh(), "Discontinuous Taylor", 1)
        # self.tf = Function(self.ts)
        # self.tf.project(field)
        # self._create_tables()
        # self.min_field.assign(1e8)
        # self.max_field.assign(-1e8)
        # par_loop("""
        # for(int i = 0; i < min.dofs; i++ ) {
        #     min[i][0] = fmin(min[i][0], t[0][0]/scale[0]);
        #     max[i][0] = fmax(max[i][0], t[0][0]/scale[0]);
        # }
        # """, dx,
        #          {"min": (self.min_field, RW),
        #           "max": (self.max_field, RW),
        #           "scale": (self.volume, READ),
        #           "t": (self.tf, READ)})
        #
        # par_loop("""
        # %s
        # double alpha = 1.0;
        # double qavg = t[0][0]/scale[0];
        # for(int i = 0; i < qmax.dofs; i++) {
        #     // interpolate the cell to the vertex
        #     double ci = 0;
        #     for ( int j = 0; j < t.dofs; j++ ) {
        #         ci += table[i][j] * t[j][0];
        #     }
        #     //ci = q[i][0];
        #     if (ci > qavg)
        #         alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(ci - qavg)));
        #     else if (ci < qavg)
        #         alpha = fmin(alpha, fmin(1, (qavg - qmin[i][0])/(qavg - ci)));
        # }
        # for(int i = 1; i < t.dofs; i++) {
        #     t[i][0] *= alpha;
        # }
        #
        # /* if(abs(alpha_0[0][0]- alpha) > 0.001) {
        #     printf("qbar[0][0], t[0][0]: %%g %%g diff %%g\\n", qbar[0][0], qavg, qbar[0][0] - qavg);
        #     printf("%%f %%f \\n", alpha_0[0][0], alpha);
        # } */
        # """ % self.str_table, dx,
        #          {"qbar": (self.centroids, READ),
        #           "q": (field, RW),
        #           "t": (self.tf, RW),
        #           "qmax": (self.max_field, READ),
        #           "qmin": (self.min_field, READ),
        #           "scale": (self.volume, READ),
        #           "alpha_0": (alpha_0, RW)})
        #
        # field.project(self.tf)

    def _create_tables(self):
        taylor_element = self.ts.fiat_element
        cell = taylor_element.get_reference_element()
        self.volume = Constant(cell.volume())
        vertices = cell.get_vertices()
        dim = cell.get_spatial_dimension()
        key = {1: (0, ),
               2: (0, 0),
               3: (0, 0, 0)}[dim]
        table = taylor_element.tabulate(0, vertices)[key]
        self.str_table = create_c_table(table, "table")

    def apply(self, field):
        """
        Re-computes centroids and applies limiter to given field
        """
        assert field.function_space() == self.P1DG, \
            'Given field does not belong to this objects function space'

        self.compute_bounds(field)
        self.apply_limiter(field)


def create_c_table(table, name):
    # Returns a C compatible string to be pasted in the kernel
    return "static const double %s[%d][%d] = %s;" % \
            ((name,) + table.T.shape + ("{{" + "}, \n{".join([", ".join(map(str, x)) for x in table.T]) + "}}", ))


class KuzminLimiter2(Limiter):
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
        self.P1TG = FunctionSpace(self.P2DG.mesh(), 'Discontinuous Taylor', 1)
        self.P0 = FunctionSpace(self.P1DG.mesh(), 'DG', 0)  # for centroids

        # Create fields
        self.taylor_field = Function(self.P2TG)
        self.lin_taylor_field = Function(self.P1TG)
        self.lin_taylor_fieldx = Function(self.P1TG)
        self.lin_taylor_fieldy = Function(self.P1TG)

        # field gradients bound containers
        self.max_field = Function(self.P1CG)
        self.min_field = Function(self.P1CG)
        self.dx_max_field = Function(self.P1CG)
        self.dx_min_field = Function(self.P1CG)
        self.dy_max_field = Function(self.P1CG)
        self.dy_min_field = Function(self.P1CG)

        # TESTING
        self.taylor2 = Function(self.P2TG)
        # P1DG limiter
        self.p1dg_limiter = KuzminLimiter1(self.P1DG)

        self._init_projections()
        self._create_tables()
        self.alpha1 = Function(self.P0)
        self.alpha2 = Function(self.P0)
        self.centroids = Function(self.P0)
        self.p1f = Function(self.P1DG)
        self.p1fx = Function(self.P1DG)
        self.p1fy = Function(self.P1DG)
        self._limit_kernel = """
%s
%s
%s
double alpha = 1.0;
double alpha_x = 1.0;
double alpha_y = 1.0;
double t_c = t[0][0] / scale[0];
double t_x = t[1][0];
double t_y = t[2][0];

void calculate_alpha(double* alpha, double** max_field, double** min_field,
                     double vertex_value, double center_value, int i) {
    if(fabs(vertex_value - center_value) > 0.00001) {
        if (vertex_value > center_value && vertex_value > max_field[i][0]) {
            *alpha = fmin(*alpha, fmin(1, (max_field[i][0] - center_value)/(vertex_value - center_value)));
        } else if (vertex_value < center_value && vertex_value < min_field[i][0]) {
            *alpha = fmin(*alpha, fmin(1, (min_field[i][0] - center_value)/(vertex_value - center_value)));
        }
    }
}

void unconstrained_vertex(double* vertex, double center, double dx, double dy, int i) {
    *vertex = center;
    *vertex += table[i][1] * dx;
    *vertex += table[i][2] * dy;
}

//remove higher
for (int i=0; i < max.dofs; i++) {
    double u_c = 0;
    double u_x = 0;
    double u_y = 0;

    for(int j= 0; j < 3; j++) {
        u_c += table[i][j] * t[j][0];
    }

    /*
    u_x = t_x + table_x[i][1] * t[3][0] + table_y[i][2] * t[4][0];

    u_y = t_y + table_x[i][1] * t[4][0] + table_y[i][2] * t[5][0];
    */

    for(int j= 0; j < t.dofs; j++) {
        u_x += table_x[i][j] * t[j][0];
        u_y += table_y[i][j] * t[j][0];
    }

    //printf("%%g ", u_c);
    //unconstrained_vertex(&u_c, t_c, t[1][0], t[2][0], i);
    //printf("%%g\\n", u_c);
    //unconstrained_vertex(&u_x, t_x, t[3][0], t[4][0], i);
    //unconstrained_vertex(&u_y, t_y, t[4][0], t[5][0], i);

    //Find alpha
    calculate_alpha(&alpha, max, min, u_c, t_c, i);
    calculate_alpha(&alpha_x, max_x, min_x, u_x, t_x, i);
    calculate_alpha(&alpha_y, max_y, min_y, u_y, t_y, i);

    //printf("alpha_x: %%g, avg %%g, max %%g min %%g vert %%g \\n", alpha, t_x, max_x[i][0], min_x[i][0], u_x);

/*
    unconstrained_vertex(&u_c, t_c, t[1][0]*alpha, t[2][0]*alpha, i);
    unconstrained_vertex(&u_x, t_x, t[3][0]*alpha_x, t[4][0]*alpha_x, i);
    unconstrained_vertex(&u_y, t_y, t[4][0]*alpha_y, t[5][0]*alpha_y, i);

    if((abs(u_c - max[i][0]) > 1e-10 && u_c > max[i][0]) || (u_c < min[i][0] && abs(u_c - min[i][0]) > 1e-10)) {
        puts("error c");
        printf("max: %%g min: %%g val%%g\\n", max[i][0], min[i][0],u_c);
    }
    if((abs(u_x - max_x[i][0]) > 1e-10 && u_x > max_x[i][0]) || (u_x < min_x[i][0] && abs(u_x - min_x[i][0]) > 1e-10)) {
        puts("error x");
        printf("max: %%g min: %%g val%%g\\n", max_x[i][0], min_x[i][0],u_x);
    }
    if(u_y > max_y[i][0] + 1e-10 || u_y < min_y[i][0] - 1e-10) {
        puts("error y");
        printf("max: %%g min: %%g val%%g\\n", max_y[i][0], min_y[i][0],u_y);
    }
    */
}

alpha_x = fmin(alpha_x, alpha_y);
alpha = fmax(alpha, alpha_x);


t[1][0] *= alpha;
t[2][0] *= alpha;
t[3][0] *= alpha_x;
t[4][0] *= alpha_x;
t[5][0] *= alpha_x;
""" % (self.str_table, self.str_table_x, self.str_table_y)

    def apply_limiter(self, field):
        """
        Apply the limiter
        Must compute bounds before calling this method.

        Parameters
        ----------
        field: Function which belongs to P2DG space that will be limited
        """
        # Find dimension of mesh
        dim = self.P1DG.mesh().topological_dimension()

        if dim == 1:
            p1f = Function(self.P1DG).project(field)
            par_loop("""
            %s
            %s
            double alpha = 1.0;
            double alpha_x = 1.0;
            double qavg = t[0][0] / scale[0];
            double t_x = t[1][0] / scale[0];
            for(int i = 0; i < qmax.dofs; i++) {
                // interpolate the cell to the vertex
                double ci = 0;
                double dxi = 0;
                for ( int j = 0; j < 2; j++ ) {
                    ci += table[i][j] * t[j][0];
                }
                for ( int j = 1; j < t.dofs; j++ ) {
                    dxi += table_x[i][j] * t[j][0];
                }
                /* ci = q[i][0]; */
                //Find alpha
                if (ci > qavg)
                    alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(ci - qavg)));
                else if (ci < qavg)
                    alpha = fmin(alpha, fmin(1, (qmin[i][0] - qavg)/(ci - qavg)));

                //Find alpha_x
                if (dxi > t_x)
                    alpha_x = fmin(alpha_x, fmin(1, (max_x[i][0] - t_x)/(dxi - t_x)));
                else if (dxi < t_x)
                    alpha_x = fmin(alpha_x, fmin(1, (min_x[i][0] - t_x)/(dxi - t_x)));
            }

            t[1][0] *= fmax(alpha, alpha_x);
            t[2][0] *= alpha_x;
            """ % (self.str_table, self.str_table_x), dx,
                     {"q": (p1f, RW),
                      "t": (self.taylor_field, RW),
                      "scale": (self.volume, READ),
                      "qmax": (self.max_field, READ),
                      "max_x": (self.dx_max_field, READ),
                      "min_x": (self.dx_min_field, READ),
                      "qmin": (self.min_field, READ)})

            self._project(f_in=self.taylor_field, f_out=field, space=self.P2DG, projector=self.P2DG_projector)

        else:  # dim==2
            # self._project(self.taylor_field, self.p1f, self.P1DG, self.P1DG_projector)
            par_loop(self._limit_kernel,
                     dx, {"t": (self.taylor_field, RW),
                          "scale": (self.volume, READ),
                          "max": (self.max_field, READ),
                          "min": (self.min_field, READ),
                          "max_x": (self.dx_max_field, READ),
                          "min_x": (self.dx_min_field, READ),
                          "max_y": (self.dy_max_field, READ),
                          "min_y": (self.dy_min_field, READ)})

            self._project(self.taylor_field, field, self.P2DG, self.P2DG_projector)

    def compute_bounds(self, field):

        # Project to taylor basis
        # self.taylor_field.interpolate(field)
        # self._project(f_in=field, f_out=self.taylor_field,
        #               space=self.P2TG, projector=self.P2TG_projector)
        self.taylor_field.project(field, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

        dim = self.P1DG.mesh().topological_dimension()

        # Throw away higher derivatives and calculate max/min at vertices
        if dim == 1:
            # Setup fields
            self.min_field.assign(1.0e20)
            self.max_field.assign(-1.0e20)

            self.dx_min_field.assign(1.0e20)
            self.dx_max_field.assign(-1.0e20)
            # Calculate max/min of cell gradients at vertices
            par_loop("""
                     for(int i = 0; i < max.dofs; i++) {
                        min[i][0] = fmin(q[0][0], min[i][0]);
                        max[i][0] = fmax(q[0][0], max[i][0]);

                        dx_maxf[i][0] = fmax(dx_maxf[i][0], q[1][0]);
                        dx_minf[i][0] = fmin(dx_minf[i][0], q[1][0]);
                     }
                     """,
                     dx,
                     {"dx_maxf": (self.dx_max_field, RW),
                      "dx_minf": (self.dx_min_field, RW),
                      "min": (self.min_field, RW),
                      "max": (self.max_field, RW),
                      "q": (self.taylor_field, RW)})

        elif dim == 2:
            # Calculate max/min of cell gradients at vertices
            self.min_field.assign(1.0e20)
            self.dx_min_field.assign(1.0e20)
            self.dy_min_field.assign(1.0e20)

            self.max_field.assign(-1.0e20)
            self.dx_max_field.assign(-1.0e20)
            self.dy_max_field.assign(-1.0e20)

            # Center of the cell is scaled by volume of the cell
            par_loop("""
                    //printf("center val: %g\\n", q[0][0] / scale[0]);
                     for(int i = 0; i < dx_max.dofs; i++) {
                        //printf("maxsofar: %g, center val: %g\\n",max[i][0], q[0][0] / scale[0]);
                        min[i][0] = fmin(q[0][0] / scale[0], min[i][0]);
                        max[i][0] = fmax(q[0][0] / scale[0], max[i][0]);

                        dx_max[i][0] = fmax(dx_max[i][0], q[1][0]);
                        dx_min[i][0] = fmin(dx_min[i][0], q[1][0]);

                        dy_max[i][0] = fmax(dy_max[i][0], q[2][0]);
                        dy_min[i][0] = fmin(dy_min[i][0], q[2][0]);
                     }
                     """,
                     dx,
                     {"dx_max": (self.dx_max_field, RW),
                      "dx_min": (self.dx_min_field, RW),
                      "dy_max": (self.dy_max_field, RW),
                      "dy_min": (self.dy_min_field, RW),
                      "min": (self.min_field, RW),
                      "max": (self.max_field, RW),
                      "scale": (self.volume, READ),
                      "q": (self.taylor_field, RW)})

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
        self.P1TG_projector = self._create_projection(self.P1TG)
        self.P2TG_projector = self._create_projection(self.P2TG)

    def _create_tables(self):
        taylor_element = self.P2TG.fiat_element
        cell = taylor_element.get_reference_element()
        self.volume = Constant(cell.volume())
        vertices = cell.get_vertices()
        dim = cell.get_spatial_dimension()
        key = {1: (0, ),
               2: (0, 0),
               3: (0, 0, 0)}[dim]

        key_x = {1: (1, ),
                 2: (1, 0),
                 3: (1, 0, 0)}[dim]

        if dim >= 2:
            key_y = {2: (0, 1), 3: (0, 1, 0)}[dim]
            key_xy = {2: (1, 1), 3: (1, 1, 0)}[dim]
            table_y = taylor_element.tabulate(1, vertices).get(key_y, None)
            table_xy = taylor_element.tabulate(2, vertices)[key_xy]
            self.str_table_y = create_c_table(table_y, "table_y")
            self.str_table_xy = create_c_table(table_xy, "table_xy")
            if dim >= 3:
                key_z = {3: (0, 0, 1)}[dim]

        table = taylor_element.tabulate(2, vertices)[key]
        table_x = taylor_element.tabulate(1, vertices)[key_x]
        self.str_table = create_c_table(table, "table")
        self.str_table_x = create_c_table(table_x, "table_x")

    def apply(self, field):
        assert field.function_space() == self.P2DG, \
            'Given field does not belong to this objects function space'

        self.compute_bounds(field)
        self.apply_limiter(field)

