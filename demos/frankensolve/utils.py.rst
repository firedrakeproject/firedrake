Here's some magic Lawrence wrote.::

  from firedrake import *
  from firedrake.petsc import PETSc
  from ufl import Form, as_vector
  from ufl.corealg.map_dag import MultiFunction
  from ufl.algorithms.map_integrands import map_integrand_dags
  from ufl.constantvalue import Zero
  from firedrake.ufl_expr import Argument
  import numpy
  import collections

  def find_sub_block(iset, ises):
      found = []
      sfound = set()
      while True:
          match = False
          for i, iset_ in enumerate(ises):
              if i in sfound:
                  continue
              lsize = iset_.getSize()
              if lsize > iset.getSize():
                  continue
              indices = iset.indices
              tmp = PETSc.IS().createGeneral(indices[:lsize])
              if tmp.equal(iset_):
                  found.append(i)
                  sfound.add(i)
                  iset = PETSc.IS().createGeneral(indices[lsize:])
                  match = True
                  continue
          if not match:
              break
      if iset.getSize() > 0:
          return None
      return found

  class ExtractSubBlock(MultiFunction):

     """Extract a sub-block from a form.

     :arg test_indices: The indices of the test function to extract.
     :arg trial_indices: THe indices of the trial function to extract.
     """

     def __init__(self, test_indices=(), trial_indices=()):
         self.blocks = {0: test_indices,
                        1: trial_indices}
         super(ExtractSubBlock, self).__init__()

     def split(self, form):
         """Split the form.

         :arg form: the form to split.
         """
         args = form.arguments()
         if len(args) == 0:
             raise ValueError
         if all(len(a.function_space()) == 1 for a in args):
             assert (len(idx) == 1 for idx in self.blocks.values())
             assert (idx[0] == 0 for idx in self.blocks.values())
             return (form, )
         f = map_integrand_dags(self, form)
         if len(f.integrals()) == 0:
             return ()
         return (f, )

     expr = MultiFunction.reuse_if_untouched

     def multi_index(self, o):
         return o

     def argument(self, o):
         V = o.function_space()
         if len(V) == 1:
             # Not on a mixed space, just return ourselves.
             return o

         V_is = V.split()
         indices = self.blocks[o.number()]
         if len(indices) == 1:
             W = V_is[indices[0]]
         else:
             W = MixedFunctionSpace([V_is[i] for i in indices])

         a = Argument(W, o.number(), part=o.part())
         args = []
         a_ = split(a)
         for i in range(len(V_is)):
             if i in indices:
                 c = indices.index(i)
                 a__ = a_[c]
                 if len(a__.ufl_shape) == 0:
                     args += [a__]
                 else:
                     args += [a__[j] for j in numpy.ndindex(a__.ufl_shape)]
             else:
                 args += [Zero() for j in numpy.ndindex(V_is[i].ufl_element().value_shape())]
         return as_vector(args)
   







