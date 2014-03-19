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
