Netgen Unstructured Mesh
============================================== 

This tutorial was contributed by `Umberto Zerbinati <mailto:umberto.zerbinati@oriel.ox.ac.uk>`__, and is based on the documentation for NetGen in the `NGSolve Docs <https://docu.ngsolve.org/latest/>`__.

The purpose of this demo is to summarise how to construct and use a Netgen mesh in Firedrake.
First thing first, what is NetGen? NetGen is an automated 2D/3D open-source  masher developed by Joachim Schoeberl, which is capable of dealing both with simplicial and cubes based mesh.
We will first have a look at how different geometries can be implemented using NetGen, then we will consider as an example the Poisson problem and discuss boundary conditions.
Last an example on how to use mesh refinement features included in NetGen can be used in order to construct and adaptive scheme. 


Constructive (2D) Solid Geometry
---------------------------------
Using the Constructive Solid Geometry (CSG) implemented in NetGen we can construct a geometry starting from basics two dimensional sets such as rectangles and disks. Let's have a look at an example.
First we need to import the NetGen library required in order to construct a `SplineGeometry` and initialise one, ::

   from firedrake import *
   from netgen.geom2d import SplineGeometry
   geo = SplineGeometry()

Now using the two of the predefined CSG geometries included in NetGen, i.e. square and circle, we construct a square with an inscribed disk that we will later flag for a finer refinement, using the `SetMaterial` and `SetDomainMaxH` methods. ::

   geo = SplineGeometry()
   geo.AddRectangle(p1=(-1,-1),
                    p2=(1,1),
                    bc="rectangle",
                    leftdomain=1,
                    rightdomain=0)
   geo.AddCircle(c=(0,0),
                 r=0.5,
                 bc="circle",
                 leftdomain=2,
                 rightdomain=1)
   #Flaging for the inside of the disk with a different material IDs
   geo.SetMaterial (1, "outer")
   geo.SetMaterial (2, "inner")
   geo.SetDomainMaxH(2, 0.02)

We can now construct a mesh for the geometry we have defined and dump it into a PVD file in order for us to visualise it later, we will do so using the `GenerateMesh` method inside of the `SplineGeometry` class, ::

   ngmsh = geo.GenerateMesh(maxh=0.1)
   #Generating a Firedrake mesh
   msh = Mesh(ngmsh)
   File("MeshExample1.pvd").write(msh)


We can also "hardwire" a geometry in NetGen specifying the points and the splines making the boundary of the domain.
In particular we add points to an instance of the `SplineGeometry` class using the method `AppendPoint`, while spline can be added using the `Append` method.
When using the `Append` method we can specify the type of Spline we would like the piece of boundary we are adding to be and the boundary label we would like to use in order to identify it (in this particular case we use the "bnd" tag for all the curves making up the boundary).
Here is showed how to construct a "Pacman" like domain using this specific methods. ::

   geo = SplineGeometry()
   pnts =[(0,0),(1,0),(1,1),
          (0,1),(-1,1),(-1,0),
          (-1,-1),(0,-1)]
   p1,p2,p3,p4,p5,p6,p7,p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
   curves = [[["line",p1,p2],"line"],
             [["spline3",p2,p3,p4],"curve"],
             [["spline3",p4,p5,p6],"curve"],
             [["spline3",p6,p7,p8],"curve"],
             [["line",p8,p1],"line"]]
   [geo.Append(c,bc=bc) for c,bc in curves]
   ngmsh = geo.GenerateMesh(maxh=0.4)
   #Generating a Firedrake mesh
   msh = Mesh(ngmsh)
   File("MeshExample2.pvd").write(msh)

Example: Poisson Problem
------------------------- 
Lets now have a look at some features that can be useful when solving a variational problem using a NetGen mesh.
We will consider as example the Poisson problem with constant source data and homogeneous boundary conditions on the boundary the PacMan.
The only method that should be new to the reader should be the `GetBCIDs` which allows to find the IDs of the boundary we have labeled in NetGen. As usual we begin defining the `FunctionSpace` that will be used in our discretisation, defining trial and test functions and the source data ::

   V = FunctionSpace(msh, "CG", 1)
   u = TrialFunction(V)
   v = TestFunction(V)
   
   f = Function(V)
   x, y = SpatialCoordinate(msh)
   f.interpolate(1+0*x)

Now we can define the bilinear form and linear function that characterize the weak formulation of the Poisson problem in abstract form, i.e.

.. math::
   
   Find \; u\in H^1_0(\Omega)\; s.t.\;a(u,v) := \int_{\Omega} \nabla u\cdot \nabla v \; d\vec{x} = l(v) := \int_{\Omega} fv\; d\vec{x}\qquad v\in H^1_0(\Omega). 

Notice that if we would like to impose Neumann boundary condition given the fact that they are natural boundary conditions we would just add an additional boundary integral in the bilinear form. ::
   
   a = inner(grad(u), grad(v))*dx
   l = inner(f, v) * dx
   
Now we are ready to assemble the stiffness matrix for the problem. Since we want to enforce Dirichlet boundary conditions we construct a `DirichletBC` object and we use the `GetBCIDs` method from the NetGen mesh in order to map the label we have given when describing the geometry to the PETSc `DMPLEX` IDs. ::

   sol = Function(V)
   labels = sum(map(ngmsh.GetBCIDs,["line","curve"]),[])
   bc = DirichletBC (V , 0.0 , labels)
   A = assemble (a , bcs = bc )
   b = assemble (l)
   bc.apply(b)
   solve (A, sol, b, solver_parameters ={"ksp_type": "preonly", "pc_type": "lu"})
   File("Poisson.pvd").write(sol)

Mesh Refinement
----------------
In this section we will discuss how to use the mesh refinement methods wrapped from NetGen C++ interface.
In particular we will be considering an Laplace eigenvalue problem on the same PacMan domain presented above, i.e.

.. math::

   Find \; u\in H^1_0(\Omega)\; and \; \lambda \in \mathbb{R}\; s.t.\; \int_{\Omega} \nabla(u)\cdot\nabla(v)\;d\vec{x} = \lambda \int_{\Omega}uv\;d\vec{x}\qquad \forall v\in H^1_0(\Omega).

This script is based on a code developed by Professor Daniele Boffi and based on a code from Professor Douglas Arnold for the source problem.
We begin by defining some quantities of interest such as the desired tolerance, the maximum number of iterations and the exact eigenvalue. Also the PETSc monitor options is switched on.

::
   
   
   from firedrake.petsc import PETSc
   from slepc4py import SLEPc
   import numpy as np

   tolerance = 1e-16
   max_iterations = 10    
   exact = 3.375610652693620492628**2;
   Err = {}; 
   opts = PETSc.Options()
   opts.setValue("monitor", None)

We create a function to solve the eigenvalue problem using SLEPc. We begin initialising the `FunctionSpace`, the bilinear forms and linear functionals needed in the variational problem.
Then a SLEPc Eigenvalue Problem Solver (`EPS`) is initialised and set up to use a shift and invert (`SINVERT`) spectral transformation where the preconditioner factorisation is computed using MUMPS. ::

   def Solve(msh,ngmsh):
        V = FunctionSpace(msh, "CG", 2)
        u = TrialFunction(V)
        v = TestFunction(V)
        #Stifness matrix
        a = inner(grad(u), grad(v))*dx#+1e8*inner(u,v)*ds
        #Mass matrix
        m = (u*v)*dx
        uh = Function(V)
        labels = sum(map(ngmsh.GetBCIDs,["line","curve"]),[])
        bc = bc = DirichletBC(V, 0.0, labels)
        A = assemble(a,bcs=bc); 
        M = assemble(m);
        Asc, Msc = A.M.handle, M.M.handle
        #Setting up SLEPc   
        E = SLEPc.EPS().create()
        E.setType(SLEPc.EPS.Type.ARNOLDI)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        E.setDimensions(1,SLEPc.DECIDE)
        E.setOperators(Asc,Msc)
        ST = E.getST();
        ST.setType(SLEPc.ST.Type.SINVERT)
        PC = ST.getKSP().getPC();
        PC.setType("lu");
        PC.setFactorSolverType("mumps");
        E.setST(ST)
        E.solve()
        vr, vi = Asc.getVecs()
        with uh.dat.vec_wo as vr:
                lam = E.getEigenpair(0, vr, vi)
        return lam, uh, V
   
We will also need a function that mark the elements that need to be marked according to an error indicator, i.e.

.. math::
   \eta = \sum_{K\in \mathcal{T}_h(\Omega)} h^2\int_K|\lambda u_h + \Delta u_h|^2\;d\vec{x}+\frac{h}{2}\int_{E\subset \partial K} | [\![ \nabla u\cdot n_E]\!] | ^2\; ds

In order to do so we begin computing the value of the indicator using a piecewise constant function space. ::

   def Mark(msh,uh,lam):
        W = FunctionSpace(msh, "DG", 0)
        w = TestFunction(W)
        R_T = lam.real*uh + div(grad(uh))
        n = FacetNormal(V.mesh())
        h = CellDiameter(msh)
        R_dT = dot(grad(uh),n)
        eta = assemble(h**2*R_T**2*w*dx + (h("+")+h("-"))*(R_dT("+")-R_dT("-"))**2*(w("+")+w("-"))*dS)
        eta = eta.dat.data
        eta_max = max(eta)
        sum_eta = sum(eta)
        # stop error estimate is less than tolerance
        if sum_eta < tolerance:
                exit()
        frac = .95
        delfrac = .05
        part = .5
        marked = np.zeros(eta.size, dtype='bool')
        sum_marked_eta = 0.
        while sum_marked_eta < part*sum_eta:
                new_marked = (~marked) & (eta > frac*eta_max)
                sum_marked_eta += sum(eta[new_marked])
                marked += new_marked
                frac -= delfrac
        return marked

Last we define the method that will take care of refining the mesh on the marked elements. ::

   def Refine(msh,ngmsh,marked):
        i = 0;
        for el in ngmsh.Elements2D():
            plex_idx = msh._cell_numbering.getOffset(i);
            if marked[plex_idx]:
                el.SetRefinementFlag(1)
            else:
                el.SetRefinementFlag(0)
            i = i+1
        ngmsh.RefineFlaged(0,True)
        return ngmsh
        

It is now time to define the solve, mark and refine loop that is at the heart of the adaptive method described here. ::
  
   for i in range(max_iterations):
        print("Refinement cycle {}".format(i))
        lam, uh,V = Solve(msh,ngmsh);
        marked = Mark(msh,uh,lam);
        ngmsh = Refine(msh,ngmsh,marked); 
        msh = Mesh(ngmsh);
        outfile = File("Eig.pvd")
        outfile.write(uh)

Constructive (3D) Solid Geometry
---------------------------------
In this section we will focus our attention on three dimensional constructive solid geometry, in particular we will have a look at the operator `+,-,*~` which have been override in order to have a special meaning when applied to two instances of the class `CSGeometry`.
It is important to notice that the same operators can be used also when working with a `SplineGeometry` and their action will have the same meaning that is presented here.
The `+,-,*` operator have respectively the meaning of union, intersection and set difference operator. We will build a cube using the planes intersection and remove from it a portion of sphere. ::

   from netgen.csg import *
   left  = Plane (Pnt(0,0,0), Vec(-1,0,0) )
   right = Plane (Pnt(1,1,1), Vec( 1,0,0) )
   front = Plane (Pnt(0,0,0), Vec(0,-1,0) )
   back  = Plane (Pnt(1,1,1), Vec(0, 1,0) )
   bot   = Plane (Pnt(0,0,0), Vec(0,0,-1) )
   top   = Plane (Pnt(1,1,1), Vec(0,0, 1) )
   cube = left*right*front*back*bot*top
   cube.bc("cube")
   sphere = Sphere(Pnt(0.6,0.6,0.6),0.5)
   geo = CSGeometry()
   geo.Add(cube-sphere)
   ngmsh = geo.GenerateMesh(maxh=0.1)
   msh = Mesh(ngmsh)
   File("MeshExample3.pvd").write(msh)
   
Open Cascade Technology
-------------------------
Last we will have a look at the NetGen Open Cascade Technology interface, which has been recently included. We will follow the tutorial presented in the `NetGen docs <https://docu.ngsolve.org/nightly/i-tutorials/unit-4.4-occ/bottle.html>`__, which it self comes from the OCCT tutorial `here <https://dev.opencascade.org/doc/overview/html/occt__tutorial.html>`__ presented.
The idea is to draw a "flask" using the OCCT interface and solve the linear elasticity equations to compute the stress tensor on the flask when its subject to gravity.
We begin importing the NetGen Open Cascade interface and constructing the bottom of the flask using many different method such as `Axes, Face, Pnt, Segment, ...` (all the details this methods can be found in `NetGen docs <https://docu.ngsolve.org/nightly/i-tutorials/unit-4.4-occ/bottle.html>`__.

::

   from netgen.occ import *
   myHeight = 70
   myWidth = 50
   myThickness = 30
   pnt1 = Pnt(-myWidth / 2., 0, 0)
   pnt2 = Pnt(-myWidth / 2., -myThickness / 4., 0)
   pnt3 = Pnt(0, -myThickness / 2., 0)
   pnt4 = Pnt(myWidth / 2., -myThickness / 4., 0)
   pnt5 = Pnt(myWidth / 2., 0, 0)
   seg1 = Segment(pnt1, pnt2)
   arc = ArcOfCircle(pnt2, pnt3, pnt4)
   seg2 = Segment(pnt4, pnt5)
   wire = Wire ([seg1, arc, seg2])
   mirrored_wire = wire.Mirror(Axis((0,0,0), X))
   w = Wire([wire, mirrored_wire])
   f = Face (w)
   f.bc("bottom")

Once the bottom part of the flask has been constructed we then extrude it to create the main body. We now construct the neck of the flask and fuse it with the main body. ::

   body = f.Extrude( myHeight*Z )
   body = body.MakeFillet (body.edges, myThickness / 12.)
   neckax = Axes(body.faces.Max(Z).center, Z)
   myNeckRadius = myThickness / 4.
   myNeckHeight = myHeight / 10
   neck = Cylinder(neckax, myNeckRadius, myNeckHeight);
   body = body+neck
   fmax = body.faces.Max(Z)
   thickbody = body.MakeThickSolid([fmax], -myThickness / 50, 1.e-3)
   
Last we are left with construct the threading of the flask neck and fuse it to the the rest of the flask body. In order to do this we are going to need the value of pi, which we grab from the Python math package.::

   import math
   cyl1 = Cylinder(neckax, myNeckRadius * 0.99, 1).faces[0]
   cyl2 = Cylinder(neckax, myNeckRadius * 1.05, 1).faces[0]
   aPnt = Pnt(2.*math.pi, myNeckHeight / 2.)
   aDir = Dir( 2.*math.pi, myNeckHeight / 4. )
   anAx2d = gp_Ax2d(aPnt, aDir)
   aMajor = 2. * math.pi
   aMinor = myNeckHeight / 10
   arc1 = Ellipse(anAx2d, aMajor, aMinor).Trim(0, math.pi)
   arc2 = Ellipse(anAx2d, aMajor, aMinor/4).Trim(0, math.pi)
   seg = Segment(arc1.start, arc1.end)
   wire1 = Wire( [Edge(arc1, cyl1), Edge(seg, cyl1)] )
   wire2 = Wire( [Edge(arc2, cyl2), Edge(seg, cyl2)] )
   threading = ThruSections ([wire1, wire2])
   bottle = thickbody+threading
   geo = OCCGeometry(bottle)

As usual we generate a mesh for the described geometry and use the Firedrake-NetGen interface to import as a PETSc DMPLEX. ::

   ngmsh = geo.GenerateMesh(maxh=5)
   msh = Mesh(ngmsh)
   File("MeshExample4.pvd").write(msh)

We are now ready to solve the linear elasticity equations. We will assume that our flask is made of a  linear hyper elastic material and therefore we can write the stress tensor as,

.. math::

   \sigma(u) := 2\mu\epsilon(u)+\lambda\nabla\cdot u
   
where epsilon denotes the symmetric part of the gradient of the argument, lambda and mu are the usual Lamé parameters. As usual we construct the discrete finale element space we will be using inn order to solve this problem and define the variational formulation for this particular problem, i.e.

.. math::
   
   Find\; u\in H^1_0(\Omega)\; s.t.\; \int_{\Omega} \sigma(u):\epsilon(u)\;d\vec{x} = \int_{\Omega} fv\;d\vec{x}, \qquad v\in H^1_0(\Omega)

::
   
   
   V = VectorFunctionSpace(msh, "CG", 1)
   W = FunctionSpace(msh, "CG", 1)
   u = TrialFunction(V)
   v = TestFunction(V)
   sol = Function(V)
   labels = sum(map(ngmsh.GetBCIDs,["bottom"]),[])
   bc = DirichletBC (V , 0.0 , labels)
   f = as_vector([0,0,-9.8])
   mu = Constant(1)
   lambda_ = Constant(0.25)
   Id = Identity(3)
   def epsilon(u):
        return 0.5*(grad(u) + grad(u).T)
   def sigma(u):
        return lambda_*div(u)*Id + 2*mu*epsilon(u)
   a = inner(sigma(u), epsilon(v))*dx
   l = inner(f, v)*dx;

Last we solve the problem using an LU preconditioner computed using MUMPs and plot the L2 norm of the stress tensor of our solution. ::

   uh = Function(V)
   solve(a == l, uh, bcs=bc, solver_parameters={"ksp_monitor": None,"ksp_rtol":1e-20,"ksp_atol":1e-8,"pc_type":"cholesky","pc_factor_mat_solver_type":"mumps"})
   abs_uh = interpolate(inner(uh,uh)**(1/2),W)
   abs_stress = interpolate(inner(sigma(uh),sigma(uh))**(1/2),W)
   abs_uh.rename("uh")
   abs_stress.rename("stress")
   File("Elasticity.pvd").write(abs_uh,abs_stress)
