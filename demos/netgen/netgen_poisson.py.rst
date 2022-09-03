Poisson Problem on a Netgen Unstructured Mesh
==============================================

The purpose of this demo is to summarise how to construct and use a Netgen mesh in Firedrake. In particular we will consider as model problem the Poisson problem. First we need to import all the needed libraries: ::

  from firedrake import *
  from netgen.geom2d import SplineGeometry
  import numpy as np
  
Once the necessary libraries have been imported we can specify the geometry we are interested in studying. In this particular case I'll be considering a square ::math:: `[0,\pi]^2`. ::

  geo = SplineGeometry()
  geo.AddRectangle((0, 0), (np.pi, np.pi), bc="rect")

Once the geometry has been defined we can let Netgen take care of generating a mesh using the method ::code:: `GenerateMesh` , the option ::code:: `maxh` specifies the largest diameter allowed in the mesh. This method will return a Netgen mesh object that can be passed to the Firedrake mesh method to obtain a mesh we can use in Firedrake. ::

  ngmesh = geo.GenerateMesh(maxh=0.1)
  msh = mesh.Mesh(ngmesh)

We then define the space we need in order to solve the Poisson problem and define the variational form corresponding to the weak solution of the Poisson problem with homogeneous Dirichlet boundary condition, i.e. find ::math:: `u\in \mathcal{W}_{0}^{1,2}(\Omega)` such that: 
.. math::

  \int_{\Omega} \nabla u \cdot \nabla v \;dx= \int fv \;dx

where :math: `f\in \mathcal{L}^2(\Omega)` is the source for our Poisson problem. For this specific example we are considering :math: `f(x,y)=2sin(x)sin(y)` ::

  V = FunctionSpace(msh, "CG", 1)
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Function(V)
  x, y = SpatialCoordinate(msh)
  f.interpolate(2*sin(x)*sin(y))
  a = inner(grad(u), grad(v))*dx
  l = inner(f, v) * dx

We define homogeneous boundary conditions for the problem, using the ::code:: `GetBCIDs` function which given a label returns the corresponding ID in Netgen. ::

  bc = DirichletBC(V, 0.0, ngmesh.GetBCIDs("rect"))

Last we assemble the stiffness matrix, the load vector and solve the problem. ::

  A = assemble(a, bcs=bc)
  b = assemble(l)
  bc.apply(b)
  sol = Function(V)
  sol.rename("Solution")
  solve(A, sol, b)

We can also export the computed solution in a Paraview compatible file. ::

  File("netgen_poisson.pvd").write(sol)

A python script version of this demo can be found `here <netgen_poisson.py>`__.
