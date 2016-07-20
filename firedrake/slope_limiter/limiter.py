from firedrake import *
from abc import *

class _Limiter(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, field):
        return NotImplemented

    @abstractmethod
    def compute_bounds(self, field):
        return NotImplemented

class VertexBased(_Limiter):
    """
    Vertex Based limiter for P1DG fields
    """
    def __init__(self, space):
        """
        Initialise limiter

        :param space : FunctionSpace instance
        """

        self.P1DG = space
        self.P1CG = FunctionSpace(self.P1DG.mesh(), 'CG', 1)  # for min/max limits
        self.P0 = FunctionSpace(self.P1DG.mesh(), 'DG', 0)    # for centroids

        # Storage containers for cell means, max and mins
        self.centroids = Function(self.P0)
        self.max_field = Function(self.P1CG)
        self.min_field = Function(self.P1CG)

        self.centroid_solvers = {}

        # Update min and max loop
        self._min_max_loop = """
for(int i = 0; i < maxq.dofs; i++) {
    maxq[i][0] = fmax(maxq[i][0],q[0][0]);
    minq[i][0] = fmin(minq[i][0],q[0][0]);
}
                             """
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

    def _construct_centroid_solver(self, field):
        """
        Constructs a linear problem for computing the centroids
        and adds it to the cache

        :param field: Discretised field associated with the P1DG field
        :return: LinearVariationalSolver instance
        """

        if field not in self.centroid_solvers:
            tri = TrialFunction(self.P0)
            test = TestFunction(self.P0)

            a = tri * test * dx
            l = field * test * dx

            params = {'ksp_type': 'preonly'}
            problem = LinearVariationalProblem(a, l, self.centroids)
            solver = LinearVariationalSolver(problem, solver_parameters=params)
            self.centroid_solvers[field] = solver
        return self.centroid_solvers[field]

    def _update_centroids(self, field):
        """
        Update centroid values
        """
        solver = self._construct_centroid_solver(field)
        solver.solve()

    def compute_bounds(self, field):
        """
        Compute min and max bounds of neighbouring cells
        """
        self._update_centroids(field)
        self.max_field.assign(-1.0e10)  # small number
        self.min_field.assign(1.0e10)   # big number

        par_loop(self._min_max_loop,
                 dx,
                 {"maxq": (self.max_field, RW),
                  "minq": (self.min_field, RW),
                  "q": (self.centroids, READ)})

    def _apply_limiter(self, field):
        """
        Perform limiting loop on the given field
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
        self._apply_limiter(field)
