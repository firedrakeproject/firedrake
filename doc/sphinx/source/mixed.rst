.. _mixed:

Mixed Types
===========

When solving linear systems of equations as they arise for instance in the
finite-element method (FEM), one is often interested in *coupled* solutions of
more than one quantity. In fluid dynamics, a common example is solving a
coupled system of velocity and pressure as it occurs in some formulations of
the Navier-Stokes equations.

Mixed Set, DataSet, Map and Dat
-------------------------------

PyOP2 provides the mixed types :class:`~pyop2.MixedSet`
:class:`~pyop2.MixedDataSet`, :class:`~pyop2.MixedMap` and
:class:`~pyop2.MixedDat` for a :class:`~pyop2.Set`, :class:`~pyop2.DataSet`,
:class:`~pyop2.Map` and :class:`~pyop2.Dat` respectively. A mixed type is
constructed from a list or other iterable of its base type and provides the
same attributes and methods. Under most circumstances types and mixed types
behave the same way and can be treated uniformly. Mixed types allow iteration
over their constituent parts and for convenience the base types are also
iterable, yielding themselves.

A :class:`~pyop2.MixedSet` is defined from a list of sets: ::

  s1, s2 = op2.Set(N), op2.Set(M)
  ms = op2.MixedSet([s1, s2])

There are a number of equivalent ways of defining a
:class:`~pyop2.MixedDataSet`: ::

  mds = op2.MixedDataSet([s1, s2], (1, 2))
  mds = op2.MixedDataSet([s1**1, s2**2])
  mds = op2.MixedDataSet(ms, (1, 2))
  mds = ms**(1, 2)

A :class:`~pyop2.MixedDat` with no associated data is defined in one of the
following ways: ::

  md = op2.MixedDat(mds)
  md = op2.MixedDat([s1**1, s2**2])
  md = op2.MixedDat([op2.Dat(s1**1), op2.Dat(s2**2)])

Finally, a :class:`~pyop2.MixedMap` is defined from a list of maps, all of
which must share the same source :class:`~pyop2.Set`: ::

  it = op2.Set(S)
  mm = op2.MixedMap([op2.Map(it, s1, 2), op2.Map(it, s2, 3)])

Block Sparsity and Mat
----------------------

When declaring a :class:`~pyop2.Sparsity` on pairs of mixed maps, the
resulting sparsity pattern has a square block structure with as many block
rows and columns as there are components in the :class:`~pyop2.MixedDataSet`
forming its row and column space. In the most general case a
:class:`~pyop2.Sparsity` is constructed as follows: ::

  it = op2.Set(...)  # Iteration set
  sr0, sr1 = op2.Set(...), op2.Set(...)  # Sets for row spaces
  sc0, sc1 = op2.Set(...), op2.Set(...)  # Sets for column spaces
  # MixedMaps for the row and column spaces
  mr = op2.MixedMap([op2.Map(it, sr0, ...), op2.Map(it, sr1, ...)])
  mc = op2.MixedMap([op2.Map(it, sc0, ...), op2.Map(it, sc1, ...)])
  # MixedDataSets for the row and column spaces
  dsr = op2.MixedDataSet([sr0**1, sr1**1])
  dsc = op2.MixedDataSet([sc0**1, sc1**1])
  # Blocked sparsity
  sparsity = op2.Sparsity((dsr, dsc), [(mr, mc), ...])

The relationships of each component of the mixed maps and datasets to the
blocks of the :class:`~pyop2.Sparsity` is shown in the following diagram:

.. figure:: images/mixed_sparsity.svg
  :align: center

  The contribution of sets, maps and datasets to the blocked sparsity. 

Block sparsity patterns are computed separately for each block as described in
:ref:`sparsity_pattern` and the same validity rules apply. A
:class:`~pyop2.Mat` defined on a block :class:`~pyop2.Sparsity` has the same
block structure, which is implemented using a PETSc_ MATNEST_.
