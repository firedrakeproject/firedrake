We import the needed library from FireDrake and NetGen.

.. code:: ipython3

    from firedrake import *
    import firedrake.cython.dmcommon as dmcommon
    from firedrake import mesh
    from ngsolve import Mesh,Draw
    from netgen.geom2d import CSG2d, Circle, Rectangle


.. parsed-literal::

    firedrake:WARNING OMP_NUM_THREADS is not set or is set to a value greater than 1, we suggest setting OMP_NUM_THREADS=1 to improve performance
    <frozen importlib._bootstrap>:219: RuntimeWarning: mpi4py.MPI.File size changed, may indicate binary incompatibility. Expected 32 from C header, got 40 from PyObject


we define a geometry using NetGen constructive geometry tools.

.. code:: ipython3

    geo = CSG2d()
    
    # define some primitives
    circle = Circle( center=(0,0), radius=1.0, mat="mat1", bc="bc_circle" )
    rect = Rectangle( pmin=(0,0), pmax=(1.5,1.5), mat="mat2", bc="bc_rect" )
    
    # use operators +, - and * for union, difference and intersection operations
    domain1 = circle - rect
    domain2 = circle * rect
    domain2.Mat("mat3").Maxh(0.1) # change domain name and maxh
    domain3 = rect-circle
    
    # add top level objects to geometry
    geo.Add(domain1)
    geo.Add(domain2)
    geo.Add(domain3)

Generating a mesh using NetGen and passing it to FireDrake.

.. code:: ipython3

    ngmesh = geo.GenerateMesh(maxh=0.3)
    mesh = mesh.Mesh(ngmesh)


.. parsed-literal::

    NetGen Mesh !


Solving the usual Poisson problem on the NetGen mesh.

.. code:: ipython3

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(1+0*x)
    a = inner(grad(u), grad(v)) * dx
    l = inner(f, v) * dx
    
    print("Problem Seted Up !")
    
    u = Function(V)
    
    bc = DirichletBC (V , 0.0 , [1, 2, 3, 10, 11, 12]) # Boundary condition
    print("Assembling ...")
    A = assemble (a , bcs = bc )
    b = assemble (l)
    bc.apply(b)
    print("Assembled !")
    print("Solving ...");
    solve (A, u, b, solver_parameters ={"ksp_type": "preonly", "pc_type": "lu"})
    print("Solved !");
    File("Poisson.pvd").write(u)


.. parsed-literal::

    Problem Seted Up !
    Assembling ...
    Assembled !
    Solving ...
    Solved !

