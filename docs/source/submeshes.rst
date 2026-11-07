.. only:: html

   .. contents::

Submeshes
=========

A :func:`~.Submesh` is a mesh made from selected cells or facets of a parent
mesh.  Functions on the parent mesh and on the submesh can be transferred
between compatible function spaces, and forms can couple the two meshes.

Creating a submesh
------------------

The ``subdomain_id`` argument selects entities using a label on the parent
mesh.  With the default ``subdim``, :func:`~.Submesh` selects cells.  For
example, the following marks the cells in the left half of a square and
constructs a submesh from them:

.. code-block:: python

   from firedrake import *

   mesh = UnitSquareMesh(4, 4)
   x, y = SpatialCoordinate(mesh)
   DG0 = FunctionSpace(mesh, "DG", 0)
   cell_marker = Function(DG0).interpolate(conditional(x < 0.5, 1, 0))
   mesh.mark_entities(cell_marker, 1)
   submesh = Submesh(mesh, subdomain_id=1)

To select facets, set ``subdim`` to one less than the topological dimension.
For example, this constructs a submesh of the parent boundary:

.. code-block:: python

   boundary = Submesh(
       mesh,
       subdim=mesh.topological_dimension - 1,
       subdomain_id="on_boundary",
   )

The ``subdomain_id`` can also be a sequence, in which case the submesh is the
union of the selected subdomains.  The submesh retains a relation to its
parent, which allows Firedrake to transfer data and assemble forms that use
both meshes.

Assigning between meshes
------------------------

:meth:`~.Function.assign` copies values at matching degrees of freedom.  It is
appropriate for a value or a simple weighted sum of functions.  Assigning
from the parent mesh to the submesh is direct because every submesh degree of
freedom has a corresponding parent degree of freedom:

.. code-block:: python

   V = FunctionSpace(mesh, "CG", 1)
   V_sub = FunctionSpace(submesh, "CG", 1)

   u = Function(V).interpolate(x + y)
   u_sub = Function(V_sub).assign(u)

The reverse direction has parent degrees of freedom outside the submesh.  Set
``allow_missing_dofs=True`` to leave those degrees of freedom unchanged:

.. code-block:: python

   u_from_submesh = Function(V).assign(u_sub, allow_missing_dofs=True)

Without this option, an assignment fails when the target has degrees of
freedom that do not occur on the source mesh.  The option is only needed for
transfers across submeshes when such unmatched degrees of freedom exist.

Interpolating between meshes
----------------------------

:meth:`~.Function.interpolate` evaluates a UFL expression in the target
space.  Unlike :meth:`~.Function.assign`, it supports general expressions,
such as products and nonlinear functions:

.. code-block:: python

   v_sub = Function(V_sub).interpolate(sin(u) + u**2)

The functional form of the operator can also be assembled explicitly with
:func:`~.interpolate`:

.. code-block:: python

   v_sub = assemble(interpolate(sin(u) + u**2, V_sub))

As with assignment, interpolation from a submesh to its parent may encounter
target degrees of freedom that are not defined by the source.  In that case,
use ``allow_missing_dofs=True``; newly created target functions receive zero
at those degrees of freedom by default:

.. code-block:: python

   v_from_submesh = Function(V).interpolate(
       v_sub, allow_missing_dofs=True,
   )

Integrating across a parent mesh and a submesh
----------------------------------------------

To assemble a form that contains arguments from both meshes, define an
intersection measure for each integration domain.  Here ``dx_parent`` uses
the parent mesh as its primary domain and makes the submesh measure available
for cross-mesh integration:

.. code-block:: python

   dx_parent = dx(mesh, intersect_measures=(dx(submesh),))
   dx_submesh = dx(submesh, intersect_measures=(dx(mesh),))

The following integral couples a parent-mesh function to a submesh function.
The label selects the cells from which ``submesh`` was constructed:

.. code-block:: python

   parent_submesh_integral = assemble(u * u_sub * dx_parent(1))

The same integral can be written with the submesh as the primary domain:

.. code-block:: python

   submesh_integral = assemble(u * u_sub * dx_submesh)

For an exterior-facet submesh, use ``ds`` on the parent mesh and ``dx`` on the
facet mesh.  For example, using the ``boundary`` submesh constructed above:

.. code-block:: python

   V_boundary = FunctionSpace(boundary, "CG", 1)
   u_boundary = Function(V_boundary).interpolate(u)
   ds_parent = ds(mesh, intersect_measures=(dx(boundary),))
   boundary_integral = assemble(u * u_boundary * ds_parent)

For an interior-facet submesh, use ``dS`` on the parent mesh instead of
``ds``.
