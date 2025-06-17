"""Utilities supporting spacetime finite element approaches."""
from .mesh import MeshGeometry, ExtrudedMesh, Mesh
from numbers import Real
from ufl.core.expr import Expr
from ufl import (as_vector, Measure, dx, ds_h, dS_h, interpolate,
    SpatialCoordinate)
from ufl.sobolevspace import SobolevSpace
from finat.ufl import FiniteElementBase, TensorProductElement
from .functionspaceimpl import WithGeometry 
from .functionspace import FunctionSpace
from .function import Function
from .assemble import assemble
from collections.abc import Iterable
from bcs import DirichletBC


class SpaceTime:
    """An object encapsulating a mesh extruded into the temporal direction.
    
    Parameters
    ----------
    spatial_mesh
        The mesh representing the spatial domain.
    dt
        The size(s) of one or more timesteps to be solved at once.

    Attributes
    ----------
    time_extent
        The total time to be solved at once (equal to the sum of dt).
    """

    def __init__(self, spatial_mesh: MeshGeometry,
                 dt: Real | tuple[Real, ...]):
        self.spatial_mesh = spatial_mesh
        self.dt = dt
        if isinstance(dt, Iterable):
            layers = len(dt)
            self.time_extent = sum(dt)
        else:
            layers = 1
            self.time_extent = dt
        self.mesh = ExtrudedMesh(spatial_mesh, layers=layers,
                                 layer_height=dt,
                                 name=spatial_mesh.name + "_spacetime")
        self._initial_mesh = self._timepoint_mesh(time=0.0)
        self._final_mesh = self._timepoint_mesh()

        # Build the time displaced mesh for initial condition updates.
        displaced_coords = Function(self.mesh.coordinates.function_space)
        x = list(SpatialCoordinate(self.mesh))
        x[-1] += self.time_extent
        displaced_coords.interpolate(x)
        self._displaced_mesh = Mesh(displaced_coords)

    def geometric_dimension(self) -> int:
        """Return the geometric dimension of the spatial mesh."""
        return self.spatial_mesh.geometric_dimension()
    
    def topological_dimension(self) -> int:
        """Return the topological dimension of the spatial mesh."""
        return self.spatial_mesh.topological_dimension()

    def grad(self, expr: Expr) -> Expr:
        '''Return the spatial gradient vector of the expression.'''
        return as_vector([expr.dx(i)
                          for i in range(self.geometric_dimension())])

    def div(self, expr: Expr) -> Expr:
        """Return the spatial divergence of the expression."""
        return sum(self.grad(expr))
    
    def curl(self, expr: Expr) -> Expr:
        """Return the spatial curl of the expression."""
        dim = self.geometric_dimension()
    
        if 1 < dim <= 3:
            return as_vector([expr[j].dx(i) - expr[i].dx(j)
                              for i in range(self.dim -1)
                              for j in range(self.dim - 1)])
        else:
            raise ValueError('Curl not defined in this dimension')

    def dt(self, expr: Expr) -> Expr:
        """Return the temporal derivative of an expression."""
        return expr.dx(self.geometric_dimension())
    

    def dx(self, *args, **kwargs) -> Measure:
        """The spacetime integration measure.
        
        All arguments are passed through to the underlying mesh measure."""
        return dx(*args, domain=self.mesh, **kwargs)
    
    def ds(self, *args, **kwargs) -> Measure:
        """The spatial exterior boundary measure on the spacetime mesh.
        
        Returns
        -------
            The exterior horizontal surface measure on the spacetime mesh. All
            arguments are passed through to that measure.
        """
        return ds_h(*args, domain=self.mesh, **kwargs)
    
    def dS(self, *args, **kwargs) -> Measure:
        """The spatial interior boundary measure on the spacetime mesh.
        
        Returns
        -------
            The interior horizontal surface measure on the spacetime mesh. All
            arguments are passed through to that measure.
        """
        return dS_h(*args, domain=self.mesh, **kwargs)

    def function_space(self,
                       space_element: FiniteElementBase,
                       time_element: FiniteElementBase,
                       sobolev_space: SobolevSpace | None = None
                       ) -> WithGeometry:
        """Create a spacetime function space.
        
        Parameters
        ----------
        space_element
            The spatial element defined on the spatial cell.
        time_element
            The time element defined on the interval.
        sobolev_space
            HDiv or HCurl if the resulting function space should be of
            one of those Sobolev spaces.

        Result
        ------
        FunctionSpace
            A function space on the spacetime mesh.
        """
        element = TensorProductElement(space_element, time_element)
        if sobolev_space:
            # Enable the creation of HDiv and HCurl elements.
            element = SobolevSpace(element)
        return FunctionSpace(self.mesh,
                             TensorProductElement(space_element, time_element))
    
    @property
    def spatial_coordinate(self):
        """The symbolic spatial coordinate on the spacetime mesh."""
        return as_vector(list(SpatialCoordinate(self.mesh))[:-1])
    
    @property
    def temporal_coordinate(self):
        """The symbolic time coordinate on the spacetime mesh."""

    def extract_spatial_function(self, function: Function,
                                 function_space: WithGeometry,
                                 time: Real | None = None) -> Function:
        """Extract the value of a function at a particular time.
        
        Parameters
        ----------
        function
            The spacetime function whose value is to be extracted at a given
            time.
        function_space
            The spatial function space into which to extract the value.
        time
            The time point at which to extract the value. If `None` then the
            end point of the interval is used.

        Returns
        -------
        Function
            The spatial function whose 
        """
        if time == 0.0:
            mesh_i = self._initial_mesh
        elif time is None:
            mesh_i = self._final_mesh
        else:
            mesh_i = self._timepoint_mesh(time)

        fs_i = function_space.reconstruct(mesh_i)
        fn_i = assemble(interpolate(function, fs_i))
        fs_out = function.function_space().reconstruct(self.spatial_mesh)
        fn_out = Function(fs_out)
        fn_out.dat.data_wo[:] = fn_i.dat.data_ro         
        return fn_out

    def _timepoint_mesh(self, time):
        fs = VectorFunctionSpace(self.spatial_mesh,
                                 self.spatial_mesh.ufl_element(),
                                 dim=self.geometric_dimension())
        X = assemble(interpolate(
            as_vector(list(SpatialCoordinate(self.spatial_mesh)) + [time])
        ))

class InitialCondition(DirichletBC):
    """An initial value for a spacetime problem.
    
    Parameters
    ----------
    spacetime: SpaceTime
        The spacetime on which this initial condition is defined. 
    V: FunctionSpace
        The spacetime function space for which this is an initial condition.
    value: Expr
        A function of space. 

    """
    def __init__(self, spacetime: SpaceTime, V: WithGeometry, value: Expr):
        g = Function(V).interpolate(value, allow_missing_dofs=True)
        super().__init__(V, g, "bottom")

    def update(self, solution: Function):
        """Update the initial condition with the final state of a solution."""
        if solution.function_space() != self.function_space():
            raise ValueError("Can only update initial condition "
                             "from function on the same space.")

        displaced_fn = Function(V.reconstruct(spacetime._displaced_mesh))
        # Note that this will be worth caching at some stage.
        displaced_fn.interpolate(solution, allow_missing_dofs=True)
        self.function_arg.dat.data_wo[:] = displaced_fn.dat.data_ro
