Multicomponent flow -- microfluidic mixing of hydrocarbons
===================================================

.. rst-class:: emphasis

    We show how Firedrake can be used to simulate multicomponent flow;
    specifically the microfluidic mixing of benzene and cyclohexane.

    The demo was contributed by `Aaron Baier-Reinio
    <mailto:baierreinio@maths.ox.ac.uk>`__ and `Kars Knook
    <mailto:knook@maths.ox.ac.uk>`__.

We consider a steady, isothermal, nonreacting mixture of benzene and cyclohexane in
a two-dimensional microfluidic container :math:`\Omega \subset \mathbb{R}^2`.
We model the mixture using the Stokes--Onsager--Stefan--Maxwell (SOSM) equations.
Momentum transport is modelled using the steady compressible 
Stokes momentum equation for a Newtonian fluid,

.. math::

    -\nabla \cdot \big\{
        2 \eta \epsilon(v) + (\zeta - 2\eta / d) (\nabla \cdot v) \mathbb{I}
    \big\} + \nabla p = 0 \quad \textrm{in}\ \Omega.

Here, the unknowns are the :math:`\mathbb{R}^2`-valued velocity :math:`v`,
scalar pressure :math:`p` and density :math:`\rho`.
Moreover :math:`\epsilon (v)` denotes the symmetric gradient of :math:`v`,
:math:`\eta, \zeta > 0` are the shear and bulk viscosities respectively,
:math:`d=2` the spatial dimension and
:math:`\mathbb{I}` the :math:`d \times d` identity matrix.

Let :math:`n` denote the number of chemical species in the mixture
(in this example :math:`n=2`, i.e. benzene and cyclohexane).
The continuity equation for the molar concentration :math:`c_i`
of species :math:`i \in \{1:n\}` in the absence of chemical reactions is

.. math::

    \partial_t c_i + \frac{1}{M_i} \nabla \cdot J_i = 0
    \quad \textrm{in}\ \Omega \quad \forall i \in \{1 : n \},

where :math:`M_i > 0` is the molar mass of species :math:`i` and
:math:`J_i` its mass flux.
As we are considering steady flow, the continuity equations simplify to

.. math::

    \frac{1}{M_i} \nabla \cdot J_i = 0
    \quad \textrm{in}\ \Omega \quad \forall i \in \{1 : n \}.

The mass fluxes must be modelled with a constitutive law.
A basic Fickian model may use :math:`J_i = M_i (c_i v - D_i \nabla c_i)`
where :math:`c_i v` represents advection and :math:`-D_i \nabla c_i` Fickian diffusion.
The Fickian approach is appropriate for dilute mixtures 
(i.e. mixtures where all of the species but one are present in trace amounts),
but typically is not thermodynamically consistent in the non-dilute regime, 
and fails to account for cross-diffusional effects.
We remedy these drawbacks by employing the Onsager--Stefan--Maxwell (OSM) equations
(also known as Maxwell--Stefan equations :cite:`Krishna:1997`),
which, in the present isothermal setting, implicitly determine the mass fluxes through the relations

.. math::

    -\frac{1}{M_i} \nabla \mu_i + \frac{1}{\rho} \nabla p &= 
    \sum_{\substack{j=1 \\ j \neq i}}^n \frac{RT c_j}{\mathscr{D}_{ij} M_i c_T}
    \Bigg( \frac{J_i}{M_i c_i} - \frac{J_j}{M_j c_j} \Bigg)
    \quad \textrm{in}\ \Omega
    \quad \forall i \in \{1 : n \}.

Here :math:`\mu_i` is the chemical potential of species `i`,
:math:`R` the ideal gas constant,
:math:`T` the temperature,
:math:`\mathscr{D}_{ij}` Stefan--Maxwell diffusion coefficients
and :math:`c_T = \sum_{j=1}^n c_j` the total concentration.
Onsager reciprocal relations imply that 
:math:`\mathscr{D}_{ij} = \mathscr{D}_{ji} \ \forall i \neq j` while
:math:`\mathscr{D}_{jj}` is undefined.
Since :math:`n=2` there is only one Stefan--Maxwell diffusion coefficient
:math:`\mathscr{D}_{12} = \mathscr{D}_{21}`
and in the code below we denote it by :code:`D_sm`.

Only :math:`n-1` of the OSM equations are linearly independent; the :math:`J_i`'s
are fully determined by a mass-average constraint
:math:`v = \frac{1}{\rho} \sum_{j=1}^n J_j`.
We will incorporate this numerically by introducing an augmentation parameter
:math:`\gamma > 0` and reformulating the OSM equations as

.. math::

    -\frac{1}{M_i} \nabla \mu_i + \frac{1}{\rho} \nabla p
    + \frac{\gamma}{\rho} v &= 
    \sum_{\substack{j=1 \\ j \neq i}}^n
    \frac{\gamma}{\rho^2} J_j + 
    \frac{RT c_j}{\mathscr{D}_{ij} M_i c_T}
    \Bigg( \frac{J_i}{M_i c_i} - \frac{J_j}{M_j c_j} \Bigg)
    \quad \textrm{in}\ \Omega
    \quad \forall i \in \{1 : n \}.

In our numerics we will also (weakly) enforce that
:math:`\nabla \cdot v = \nabla \cdot (\frac{1}{\rho} \sum_{j=1}^n J_j )`
as this turns out to yield a well-posed discrete scheme.

This completes the description of transport phenomena in the mixture.
However, to obtain a closed set of equations, we must also model how the free energy and
volume of the mixture depend on temperature, pressure and composition.
To describe this we must introduce mole fractions :math:`x_i := c_i / c_T`;
note that by definition :math:`\sum_{j=1}^n x_j = 1`.
Thermodynamics requires that the :math:`\mu_i`'s' and :math:`c_T` satisfy

.. math::

    \mu_i &= g_i(T, p, x_1, \ldots, x_n)
    \quad \textrm{in}\ \Omega
    \quad \forall i \in \{1 : n \}, \\
    \frac{1}{c_T} &= \sum_{j=1}^n x_j V_j(T, p, x_1, \ldots, x_n) \quad \textrm{in}\ \Omega,

where :math:`g_i: \mathbb{R}^{n+2} \rightarrow \mathbb{R}` are partial molar Gibbs functions 
and :math:`V_i: \mathbb{R}^{n+2} \rightarrow \mathbb{R}` partial molar volume functions.
These functions are derived from partial derivatives of the Gibbs free energy of the mixture.
In this demo we employ a Margules model :cite:`Perry:2007` for the :math:`g_i`'s.
We also assume that the partial molar volumes are constant
(for liquids this is usually a reasonable assumption), from which it follows that

.. math::

    \frac{1}{c_T} = \sum_{j=1}^n \frac{x_j}{c_j^{\textrm{pure}}} \quad \textrm{in}\ \Omega,

where :math:`c_j^{\textrm{pure}}` is the (known) concentration of pure species `j`.
Note that the above equation is an example of a volumetric equation of state.



Defining parameters for the numerics and nondimensionalised physical quantities::

    from firedrake import *

    k = 3                                           # The polynomial degree (for the velocity spaces)
    deg_max = 15                                    # Maximum quadrature degree
    gamma = Constant(1e-1)                          # Augmentation parameter, dimensionless
    v_ref_1 = Constant(0.4e-6)                      # Initialisng reference inlet velocity of benzene (m/s)

    RT = Constant(8.314 * 298.15)                   # Ideal gas constant times temperature, J / mol
    eta = Constant(6e-4)                            # Shear viscosity, Pa s
    zeta = Constant(1e-7)                           # Bulk viscosity, Pa s
    D_sm = Constant(2.1e-9)                         # Stefan--Maxwell diffusivity, m^2 / s
    L_ref = Constant(2e-3)                          # Reference length, m

    # Constants for the pure species
    M_1 = Constant(0.078)                           # Molar mass of Benzene, kg / mol
    M_2 = Constant(0.084)                           # Molar mass of Cyclohexane, kg / mol
    rho_pure_1 = Constant(876)                      # Density of pure Benzene, kg / m^3
    rho_pure_2 = Constant(773)                      # Density of pure Cyclohexane, kg / m^3
    c_pure_1 = rho_pure_1 / M_1                     # Concentration of pure Benzene, mol / m^3
    c_pure_2 = rho_pure_2 / M_2                     # Concentration of pure Cyclohexane, mol / m^3

    # Constants for the equimolar mixture
    c_equi_tot = (c_pure_1 * c_pure_2) / ((0.5 * c_pure_2) + (0.5 * c_pure_1))      # Total equimolar concentration, mol / m^3
    c_equi_1 = 0.5 * c_equi_tot                     # Equimolar concentration of Benzene, mol / m^3
    c_equi_2 = 0.5 * c_equi_tot                     # Equimolar concentration of Cyclohexane, mol / m^3
    rho_equi = (M_1 * c_equi_1) + (M_2 * c_equi_2)  # Equimolar density, kg / m^3

    # Reference density and related constants
    rho_ref = rho_equi                              # Reference density, kg / m^3
    c_ref = c_equi_tot                              # Reference concentration, mol / m^3
    M_ref = rho_ref / c_ref                         # Reference molar mass, kg / mol

    # Reference convective velocities
    v_ref_2 = (c_pure_1 / c_pure_2) * v_ref_1       # Reference inflow velocity of Cyclohexane, m / s
    v_ref = 0.5 * (v_ref_1 + v_ref_2)               # Reference velocity, m / s

    # Derived quantities
    v_sm_ref = D_sm / L_ref                         # Reference diffusion velocity, m / s
    p_ref = eta * v_ref / L_ref                     # Reference pressure, Pa
    lame_ND = (zeta / eta) - 1.0                    # Non-dimensionalised Lame parameter, dimensionless
    Pe = v_ref / v_sm_ref                           # Peclet number, dimensionless
    Me = p_ref / (RT * c_ref)                       # Pressure diffusion number, dimensionless
    Me_1 = p_ref / (RT * c_pure_1)                  # Non-dimensionalised partial molar volume of Benzene, dimensionless
    Me_2 = p_ref / (RT * c_pure_2)                  # Non-dimensionalised partial molar volume of Cyclohexane, dimensionless
    M_1_ND = M_1 / M_ref                            # Non-dimensionalised molar mass of Benzene, dimensionless
    M_2_ND = M_2 / M_ref                            # Non-dimensionalised molar mass of Cyclohexane, dimensionless

    # Margules model parameters
    A_12 = Constant(0.4498)                         # Dimensionless
    A_21 = Constant(0.4952)                         # Dimensionless

Pure benzene and cyclohexane are piped in through opposing inlets on the left-hand side of the microfluidic container. 
These chemicals form a nonideal mixture before flowing out through the outlet on the right-hand side.
The mesh of the two-dimensional microfluidic container is created using :doc:`netgen <netgen_mesh.py>`. 
We use fourth order curved elements to adequately capture the geometry of the container. ::

    import netgen.occ as ngocc

    wp = ngocc.WorkPlane()
    wp.MoveTo(0, 1)
    wp.Spline([ngocc.Pnt(1, 0), ngocc.Pnt(0, -1)])
    wp.LineTo(1, -1)
    wp.Spline([ngocc.Pnt(3, -0.5), ngocc.Pnt(4, -0.5)], tangents={ 1 : ngocc.gp_Vec2d(1, 0) })
    wp.LineTo(4, 0.5)
    wp.Spline([ngocc.Pnt(3, 0.5), ngocc.Pnt(1, 1)], tangents={ 0 : ngocc.gp_Vec2d(-1, 0) })
    wp.LineTo(0, 1)

    shape = wp.Face()
    shape.edges.Max(ngocc.Y).name = "inlet_1"
    shape.edges.Min(ngocc.Y).name = "inlet_2"
    shape.edges.Max(ngocc.X).name = "outlet"

    ngmesh = ngocc.OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.055)
    mesh = Mesh(Mesh(ngmesh).curve_field(4))        # fourth order curved elements

    inlet_1_ids = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "inlet_1"]
    inlet_2_ids = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "inlet_2"]
    outlet_ids = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "outlet"]
    walls_ids = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == ""]

    ds = ds(mesh, degree=deg_max)
    dx = dx(mesh, degree=deg_max)
    x_sc, y_sc = SpatialCoordinate(mesh)

The mesh is displayed below.

.. image:: microfluidic_container.png
    :align: center
    :width: 60%

Thermodynamic constitutive relations::

    # Margules model for chemical potentials, assuming constant partial molar volumes
    def mu_relation(x_1, x_2, p):
        mu_1 = (Me_1 * p) + ln(x_1) \
            + (x_2 ** 2) * (A_12 + (2.0 * (A_21 - A_12) * x_1))
        mu_2 = (Me_2 * p) + ln(x_2) \
            + (x_1 ** 2) * (A_21 + (2.0 * (A_12 - A_21) * x_2))

        return (mu_1, mu_2)

    # Volumetric equation of state, assuming constant partial molar volumes
    def conc_relation(x_1, x_2):
        x_1_nm = x_1 / (x_1 + x_2)
        x_2_nm = x_2 / (x_1 + x_2)

        c_tot = 1.0 / ((x_1_nm * (c_ref / c_pure_1)) + (x_2_nm * (c_ref / c_pure_2)))
        c_1 = x_1_nm * c_tot
        c_2 = x_2_nm * c_tot

        return (c_tot, c_1, c_2)

Defining the test and trial functions::

    W_h = FunctionSpace(mesh, "BDM", k)             # Momentum space
    V_h   = VectorFunctionSpace(mesh, "CG", k)      # Bulk velocity space
    U_h = FunctionSpace(mesh, "DG", k - 1)          # Chemical potential space
    P_h   = FunctionSpace(mesh, "CG", k - 1)        # Pressure space
    X_h = FunctionSpace(mesh, "DG", k - 1)          # Molar fraction space
    R_h = FunctionSpace(mesh, "CG", k - 1)          # Density reciprocal space
    L_h = FunctionSpace(mesh, "R", 0)               # Real space

    Z_h = W_h * W_h * V_h * U_h * U_h * P_h * X_h * X_h * R_h * L_h * L_h
    PETSc.Sys.Print("Mesh has %d cells, with %d finite element DOFs" % (mesh.num_cells(), Z_h.dim()))

    sln = Function(Z_h)
    mm_1, mm_2, v, mu_aux_1, mu_aux_2, p, x_1, x_2, rho_inv, l_1, l_2 = split(sln)
    mu_1 = mu_aux_1 + l_1
    mu_2 = mu_aux_2 + l_2
    u_1, u_2, u, w_1, w_2, q, y_1, y_2, r, s_1, s_2 = TestFunctions(Z_h)

Defining the variational formulation::

    c_tot, c_1, c_2 = conc_relation(x_1, x_2)

    # The Stokes viscous terms
    A_visc = 2.0 * inner(sym(grad(v)), sym(grad(u))) * dx
    A_visc += lame_ND * inner(div(v), div(u)) * dx

    # The augmented Onsager transport matrix terms
    A_osm = Pe * (1.0 / c_tot) * ((c_2 / (M_1_ND * M_1_ND * c_1)) * inner(mm_1, u_1) \
            + (c_1 / (M_2_ND * M_2_ND * c_2)) * inner(mm_2, u_2) \
            - (1.0 / (M_1_ND * M_2_ND)) * (inner(mm_1, u_2) + inner(mm_2, u_1))) * dx
    A_osm += Pe * gamma * inner(v - (rho_inv * (mm_1 + mm_2)), u - (rho_inv * (u_1 + u_2))) * dx

    # The diffusion driving force terms and Stokes pressure term
    B_blf = ((Me * inner(p, div(rho_inv * (u_1 + u_2)))) - inner(p, div(u))) * dx
    B_blf -= ((1.0 / M_1_ND) * inner(mu_1, div(u_1)) + (1.0 / M_2_ND) * inner(mu_2, div(u_2))) * dx

    # The div(mass-average constraint) and continuity equation terms
    BT_blf = (inner(q, div(rho_inv * (mm_1 + mm_2))) - inner(q, div(v))) * dx
    BT_blf -= ((1.0 / M_1_ND) * inner(w_1, div(mm_1)) + (1.0 / M_2_ND) * inner(w_2, div(mm_2))) * dx

    # The total residual
    tot_res = A_visc + A_osm + B_blf + BT_blf

    # The thermodynamic constitutive relation and density reciprocal terms
    mu_1_cr, mu_2_cr = mu_relation(x_1, x_2, p)
    tot_res += (inner(mu_1 - mu_1_cr, y_1) + inner(mu_2 - mu_2_cr, y_2)) * dx

    tot_res += inner(1.0 / rho_inv, r) * dx
    tot_res -= inner((M_1_ND * c_1) + (M_2_ND * c_2), r) * dx

    # The density consistency terms
    nml = FacetNormal(mesh)
    tot_res -= q * inner((rho_inv * (mm_1 + mm_2)) - v, nml) * ds

    # integral constraints
    tot_res += inner(x_1 + x_2 - 1, s_1) * dx
    tot_res += inner((M_1_ND * c_1) - (M_2_ND * c_2), s_2) * ds(outlet_id)

We enforce parabolic profiles on :math:`J_i \cdot n` at inflow :math:`i` and on the outflow.
The magnitude of the parabolic profiles are :math:`M_ic_i^\text{ref}v_i^\text{ref}` respectively.
Elsewhere on the boundary :math:`J_i \cdot n = 0` for each :math:`i`. Instead of specifying
the value of the bulk velocity on the inflows and outflow, we enforce :math:`\rho v \cdot n = (J_1 + J_2 )\cdot n`
and :math:`\rho v \times n = 0` in these regions. This means that :math:`v` equals an unknown quantity
so we need to use :code:`EquationBC` instead of :code:`DirichletBC`. ::

    J_1_inflow_bc_func = -M_1_ND * 2.0 * x_sc * (x_sc - 1.0) * (v_ref_1 / v_ref) * (c_pure_1 / c_ref) * as_vector([2.0, -1.0])
    J_2_inflow_bc_func = -M_2_ND * 2.0 * x_sc * (x_sc - 1.0) * (v_ref_2 / v_ref) * (c_pure_2 / c_ref) * as_vector([2.0, 1.0])
    rho_v_inflow_1_bc_func = J_1_inflow_bc_func
    rho_v_inflow_2_bc_func = J_2_inflow_bc_func

    J_1_outflow_bc_func = -M_1_ND * 2.0 * (y_sc + 0.5) * (y_sc - 0.5) * (v_ref_1 / v_ref) * (c_pure_1 / c_ref) * as_vector([1.0, 0.0])
    J_2_outflow_bc_func = -M_2_ND * 2.0 * (y_sc + 0.5) * (y_sc - 0.5) * (v_ref_2 / v_ref) * (c_pure_2 / c_ref) * as_vector([1.0, 0.0])
    rho_v_outflow_bc_func = J_1_outflow_bc_func + J_2_outflow_bc_func

    # Boundary conditions on the bulk velocity are enforced via EquationBC
    v_inflow_1_bc = EquationBC(inner(v - rho_inv * rho_v_inflow_1_bc_func, u) * ds(inlet_1_id, degree=deg_max) == 0, sln, inlet_1_id, V=Z_h.sub(2))
    v_inflow_2_bc = EquationBC(inner(v - rho_inv * rho_v_inflow_2_bc_func, u) * ds(inlet_2_id, degree=deg_max) == 0, sln, inlet_2_id, V=Z_h.sub(2))
    v_outflow_bc = EquationBC(inner(v - rho_inv * rho_v_outflow_bc_func, u) * ds(outlet_id, degree=deg_max) == 0, sln, outlet_id, V=Z_h.sub(2))

    # The boundary conditions on the fluxes
    flux_bcs = [DirichletBC(Z_h.sub(0), J_1_inflow_bc_func, inlet_1_id),
                DirichletBC(Z_h.sub(0), J_1_outflow_bc_func, outlet_id),
                DirichletBC(Z_h.sub(0), as_vector([0.0, 0.0]), inlet_2_id),
                DirichletBC(Z_h.sub(0), as_vector([0.0, 0.0]), walls_ids),
                DirichletBC(Z_h.sub(1), J_2_inflow_bc_func, inlet_2_id),
                DirichletBC(Z_h.sub(1), J_2_outflow_bc_func, outlet_id),
                DirichletBC(Z_h.sub(1), as_vector([0.0, 0.0]), inlet_1_id),
                DirichletBC(Z_h.sub(1), as_vector([0.0, 0.0]), walls_ids),
                v_inflow_1_bc,
                v_inflow_2_bc,
                v_outflow_bc,
                DirichletBC(Z_h.sub(2), as_vector([0.0, 0.0]), walls_ids)]

Furthermore, we use :code:`FixAtPointBC` from :doc:`Steady Boussinesq problem with integral constraints.<demos/boussinesq.py>` 
to remove the pressure null space and fix :math:`\mu_i = 0` for each :math:`i`. ::

    import firedrake.utils as firedrake_utils

    class FixAtPointBC(firedrake.DirichletBC):
        r'''A special BC object for pinning a function at a point.

        :arg V: the :class:`.FunctionSpace` on which the boundary condition should be applied.
        :arg g: the boundary condition value.
        :arg bc_point: the point at which to pin the function.
            The location of the finite element DOF nearest to bc_point is actually used.
        '''
        def __init__(self, V, g, bc_point):
            super(FixAtPointBC, self).__init__(V, g, bc_point)
            if isinstance(bc_point, tuple):
                bc_point = as_vector(bc_point)
            self.bc_point = bc_point

        @firedrake_utils.cached_property
        def nodes(self):
            V = self.function_space()
            x = firedrake.SpatialCoordinate(V.mesh())
            xdist = x - self.bc_point

            test = firedrake.TestFunction(V)
            trial = firedrake.TrialFunction(V)
            xphi = firedrake.assemble(ufl.inner(xdist * test, xdist * trial) * ufl.dx, diagonal=True)
            phi = firedrake.assemble(ufl.inner(test, trial) * ufl.dx, diagonal=True)
            with xphi.dat.vec as xu, phi.dat.vec as u:
                xu.pointwiseDivide(xu, u)
                min_index, min_value = xu.min()     # Find the index of the DOF closest to bc_point

            nodes = V.dof_dset.lgmap.applyInverse([min_index])
            nodes = nodes[nodes >= 0]
            return nodes

    # The auxiliary constraints on the chemical potentials and pressure
    aux_point = as_vector([4, 0])       # point on the middle of the outlet
    aux_point_bcs = [FixAtPointBC(Z_h.sub(3), Constant(0.0), aux_point),
                    FixAtPointBC(Z_h.sub(4), Constant(0.0), aux_point),
                    FixAtPointBC(Z_h.sub(5), Constant(0.0), aux_point)]

We provide a naive initial guess based on an equimolar constant distribution of benzene and cyclohexane::

    mm_1, mm_2, v, mu_aux_1, mu_aux_2, p, x_1, x_2, rho_inv, l_1, l_2 = sln.subfunctions
    x_1.interpolate(Constant(0.5))
    x_2.interpolate(Constant(0.5))
    rho_inv.interpolate(1.0 / ((M_1_ND * c_1) + (M_2_ND * c_2)))

and define the nonlinear variational solver object::

    NLVP = NonlinearVariationalProblem(tot_res, sln, bcs=flux_bcs+aux_point_bcs)
    NLVS = NonlinearVariationalSolver(NLVP)

Newton's method applied directly to the problem with :math:`v_1^\text{ref}=0.4\times 10^{-5}`
with the naive initial guess does not converge. Hence, we apply parameter continuation to :math:`v_1^\text{ref}`
to find a better initial guess. We start by solving the problem for :math:`v_1^\text{ref}=0.4\times 10^{-6}` 
with the naive initial guess and use its solution as initial guess for the problem with 
:math:`v_1^\text{ref}=0.1\times 10^{-5}`. We repeat this trick with :math:`v_1^\text{ref}=0.2\times 10^{-5}`
and :math:`v_1^\text{ref}=0.3\times 10^{-5}` before solving for :math:`v_1^\text{ref}=0.4\times 10^{-5}`. 
We can reuse the nonlinear variational solver object each iteration, but have to reassign :code:`v_ref_1` 
to the new value before calling the :code:`solve()` method. Finally, we write each solution to the same 
VTK file using the :code:`time` keyword argument.::

    from firedrake.output import VTKFile
    outfile = VTKFile("out/sln.pvd")
    cont_vals = [1.0, 2.5, 5, 7.5, 10.0]
    n_cont = len(cont_vals)

    for i in range(n_cont):
        print(f"Solving for v_ref_1 = {0.4e-6*cont_vals[i]}")
        v_ref_1.assign(Constant(0.4e-6*cont_vals[i]))
        NLVS.solve()

        p += assemble(-p * dx) / assemble(1 * dx(mesh))     # normalise p to have 0 mean

        mu_1_out = Function(U_h)
        mu_2_out = Function(U_h)
        rho_out = Function(R_h)
        c_tot_out = Function(R_h)
        c_1_out = Function(R_h)
        c_2_out = Function(R_h)
        
        mu_1_out.interpolate(mu_1)
        mu_2_out.interpolate(mu_2)
        rho_out.interpolate(1.0 / rho_inv)
        c_tot_out.interpolate(c_tot)
        c_1_out.interpolate(c_1)
        c_2_out.interpolate(c_2)

        mm_1.rename("mm_1")
        mm_2.rename("mm_2")
        v.rename("v")
        mu_1_out.rename("mu_1")
        mu_2_out.rename("mu_2")
        p.rename("p")
        x_1.rename("x_1")
        x_2.rename("x_2")
        rho_inv.rename("rho_inv")
        rho_out.rename("rho")
        c_tot_out.rename("c_tot")
        c_1_out.rename("c_1")
        c_2_out.rename("c_2")

        outfile.write(mm_1, mm_2, v, mu_1_out, mu_2_out, p, x_1, x_2, rho_inv, \
                        rho_out, c_tot_out, c_1_out, c_2_out, time=i)


The mole fraction and stream lines of benzene for :math:`v_1^\text{ref}=0.4\times 10^{-6}` 
and :math:`v_1^\text{ref}=0.4\times 10^{-5}` are displayed below on the left and right respectively.
Thanks to parameter continuation and higher-order discretisation methods, we can effectively solve 
for low species concentrations and sharp solution gradients.

+---------------------------+---------------------------+
| .. image:: benzene_0.png  | .. image:: benzene_4.png  |
|    :width: 100%           |    :width: 100%           |
+---------------------------+---------------------------+
