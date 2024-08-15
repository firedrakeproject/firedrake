.. only:: html

  .. contents::


================================
 Preconditioning infrastructure
================================

Firedrake has tight coupling with the `PETSc <https://petsc.org>`__
library which provides support for a wide range of preconditioning
strategies, see the relevant `PETSc documentation
<https://petsc.org/release/manual/ksp/#preconditioners>`__ for an
overview.

In addition to these algebraic approaches, Firedrake offers a flexible
framework for defining preconditioners that need to construct or apply
auxiliary operators. The basic approach is described in
:cite:`Kirby2017`. Here we provide a brief overview of the
preconditioners available in Firedrake that use this approach. To use
these preconditioners, one sets ``"pc_type": "python"`` and
``"pc_python_type": "fully_qualified.NameOfPC"`` in the
``solver_parameters``.

Additive Schwarz methods
========================

Small-block overlapping additive Schwarz preconditioners built on top
of `PCASM
<https://petsc.org/release/manualpages/PC/PCASM.html>`__ that can
be used as components of robust multigrid schemes when using geometric
multigrid.

:class:`.ASMPatchPC`
   Abstract base class for which one must implement
   :meth:`.ASMPatchPC.get_patches` to extract sets of
   degrees of freedom. Needs to be used with assembled sparse matrices
   (``"mat_type": "aij"``).
:class:`.ASMStarPC`
   Constructs patches by gathering degrees of freedom in the star of
   specified mesh entities.
:class:`.ASMVankaPC`
   Constructs patches using the Vanka scheme by gathering degrees of
   freedom in the closure of the star of specified mesh entities.
:class:`.ASMLinesmoothPC`
   Constructs patches gathering degrees of freedom in vertical columns
   on :func:`extruded meshes <.ExtrudedMesh>`.
:class:`.ASMExtrudedStarPC`
   Like :class:`.ASMStarPC` but on extruded meshes.

In addition to these algebraic approaches to constructing patches,
Firedrake also interfaces with `PCPATCH
<https://petsc.org/release/manualpages/PC/PCPATCH.html>`__ for
both linear and nonlinear overlapping Schwarz methods. The approach is
described in detail in :cite:`Farrell2019d`. These preconditioners can
be used with both sparse matrices and Firedrake's :doc:`matrix-free
operators <matrix-free>`, and can be applied either additively or
multiplicatively within an MPI rank and additively between ranks.

:class:`.PatchPC`
   Small-block overlapping Schwarz smoother with topological
   definition of patches. Does not support extruded meshes.
:class:`.PatchSNES`
   Nonlinear overlapping Schwarz smoother with topological definition
   of patches. Does not support extruded meshes.
:class:`.PlaneSmoother`
   A Python construction class for :class:`.PatchPC` and
   :class:`.PatchSNES` that approximately groups mesh
   entities into lines or planes (useful for advection-dominated
   problems).

Multigrid methods
=================

Firedrake has support for rediscretised geometric multigrid on both
normal and extruded meshes, with regular refinement. This is obtained
by constructing a :func:`mesh hierarchy <.MeshHierarchy>`
and then using ``"pc_type": "mg"``. In addition to this basic support,
it also has out of the box support for a number of problem-specific
preconditioners.

:class:`.HypreADS`
   An interface to Hypre's `auxiliary space divergence solver
   <https://hypre.readthedocs.io/en/latest/solvers-ads.html>`__.
   Currently only implemented for lowest-order Raviart-Thomas
   elements.
:class:`.HypreAMS`
   An interface to Hypre's `auxiliary space Maxwell solver
   <https://hypre.readthedocs.io/en/latest/solvers-ams.html>`__.
   Currently only implemented for lowest order Nedelec elements of the
   first kind.
:class:`.PMGPC`
   Generic p-coarsening rediscretised linear multigrid. If the problem
   is built on a mesh hierarchy then the coarse grid can do further
   h-multigrid with geometric coarsening.
:class:`.P1PC`
   Coarsening directly to linear elements.
:class:`.PMGSNES`
   Generic p-coarsening nonlinear multigrid. If the problem is built
   on a mesh hierarchy then the coarse grid can do further h-multigrid
   with geometric coarsening.
:class:`.P1SNES`
   Coarsening directly to linear elements.
:class:`.GTMGPC`
   A two-level non-nested multigrid scheme for the hybridised mixed
   method that operates on the trace variable, using the approach
   of :cite:`Gopalakrishnan2009`. The Firedrake implementation is
   described in :cite:`Betteridge2021a`.

Assembled and auxiliary operators
=================================

Many preconditioning schemes call for auxiliary operators, these are
facilitated by variations on Firedrake's
:class:`~.AssembledPC` which can be used to deliver an
assembled operator inside a nested solver where the outer matrix is a
matrix-free operator. Matrix-free operators can be used "natively"
with PETSc's ``"jacobi"`` preconditioner, since they can provide their
diagonal cheaply. For more complicated things, one must assemble an
operator instead.

:class:`.AssembledPC`
   Assemble an operator as a sparse matrix and
   then apply an inner preconditioner. For example, this might be used
   to assemble a coarse grid in an (otherwise matrix-free) multigrid
   solver.
:class:`.AuxiliaryOperatorPC`
   Abstract base class for preconditioners built from assembled
   auxiliary operators. One should subclass this preconditioner and
   override the :meth:`.PCSNESBase.form` method. This can be
   used to provide bilinear forms to the solver that were not there
   in the original problem, for example, the pressure mass matrix for
   block preconditioners of the Stokes equations.
:class:`.FDMPC`
   An auxiliary operator that uses piecewise-constant coefficients
   that is assembled in the basis of shape functions that diagonalize
   separable problems in the interior of each cell. Currently
   implemented for quadrilateral and hexahedral cells. The assembled
   matrix becomes as sparse as a low-order refined preconditioner, to
   which one may apply other preconditioners such as :class:`.ASMStarPC` or
   :class:`.ASMExtrudedStarPC`. See details in :cite:`Brubeck2022`.
:class:`.MassInvPC`
   Preconditioner for applying an inverse mass matrix.
:class:`~.PCDPC`
   A preconditioner providing the Pressure-Convection-Diffusion
   approximation to the Schur complement for the Navier-Stokes
   equations. Note that this implementation only treats problems with
   characteristic velocity boundary conditions correctly.


Hybridisation and element-wise static condensation
==================================================

Firedrake has a number of preconditioners that use the :mod:`Slate
<.slate.slate>` facility for element-wise linear algebra on
assembled tensors. These are described in detail in :cite:`Gibson2018`.

:class:`.HybridizationPC`
   A preconditioner for hybridisable H(div) mixed methods that breaks
   the vector-valued space, and enforces continuity through
   introduction of a trace variable. The (now-broken) problem is
   eliminated element-wise onto the trace space to leave a
   single-variable global problem, whose solver can be configured.
:class:`.SCPC`
   A preconditioner that performs element-wise static condensation
   onto a single field.

.. bibliography:: _static/references.bib _static/firedrake-apps.bib
   :filter: docname in docnames
