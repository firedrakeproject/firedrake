from __future__ import absolute_import

import weakref

from functools import partial
from pyop2 import op2

import firedrake
from firedrake.petsc import PETSc
import firedrake.solving_utils
from . import utils
from . import ufl_utils

__all__ = ["NLVSHierarchy"]


def _fs_from_dm(x):
    hierarchy, level = utils.get_level(x)
    return hierarchy[level]


def coarsen_problem(problem):
    u = problem.u
    h, lvl = utils.get_level(u)
    if lvl == -1:
        raise RuntimeError("No hierarchy to coarsen")
    if lvl == 0:
        return None
    new_u = ufl_utils.coarsen_thing(problem.u)
    new_bcs = [ufl_utils.coarsen_thing(bc) for bc in problem.bcs]
    new_J = ufl_utils.coarsen_form(problem.J)
    new_Jp = ufl_utils.coarsen_form(problem.Jp)
    new_F = ufl_utils.coarsen_form(problem.F)

    new_problem = firedrake.NonlinearVariationalProblem(new_F,
                                                        new_u,
                                                        bcs=new_bcs,
                                                        J=new_J,
                                                        Jp=new_Jp,
                                                        form_compiler_parameters=problem.form_compiler_parameters,
                                                        nest=problem._nest)
    return new_problem


def make_transfer(dmc, dmf, typ=None):
    assert typ in ("interpolation", "injection")
    hierarchy, level = utils.get_level(dmc)

    V_c = _fs_from_dm(dmc)
    V_f = _fs_from_dm(dmf)

    cctx = dmc.getAppCtx()
    fctx = dmf.getAppCtx()

    cbcs = cctx._problems[level].bcs
    fbcs = fctx._problems[level+1].bcs

    fine_map = hierarchy.cell_node_map(level)
    coarse_map = V_c.cell_node_map()
    # TODO: Would like to be able to use nest=False,
    # Also, don't need off-diagonal blocks allocated
    sparsity = op2.Sparsity((V_f.dof_dset,
                             V_c.dof_dset),
                            (fine_map,
                             coarse_map),
                            "%s" % typ,
                            nest=True)
    mat = op2.Mat(sparsity, PETSc.ScalarType)

    split = hierarchy.split()
    for i in range(len(V_f)):
        if mat.sparsity.shape > (1, 1):
            fbcs_ = []
            for bc in fbcs:
                if bc.function_space().index == i:
                    fbcs_.append(bc)
            cbcs_ = []
            for bc in cbcs:
                if bc.function_space().index == i:
                    cbcs_.append(bc)
        else:
            fbcs_ = fbcs
            cbcs_ = cbcs
        fine_map = split[i].cell_node_map(level, fbcs_)
        coarse_map = V_c[i].cell_node_map(cbcs_)
        if typ == "interpolation":
            kernel = split[i]._prolong_matrix
        else:
            kernel = split[i]._inject_matrix
        op2.par_loop(kernel,
                     fine_map.iterset,
                     mat[i, i](op2.WRITE, (fine_map[op2.i[0]],
                                           coarse_map[op2.i[1]])))
    mat.assemble()
    mat._force_evaluation()
    mat = mat.handle
    if typ == "interpolation":
        return mat, None
    else:
        return mat


class NLVSHierarchy(object):

    _id = 0

    def __init__(self, problem, **kwargs):
        """
        Solve a :class:`.NonlinearVariationalProblem` on a hierarchy of meshes.

        :arg problem: A :class:`.NonlinearVariationalProblem` or
             iterable thereof (if specifying the problem on each level
             by hand).
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
             :class:`.MixedVectorSpaceBasis`) spanning the null space of the
             operator.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
            PETSc flag options should be specified with `bool`
            values (``True`` for on, ``False`` for off).
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.

        .. note::

           This solver is set up for use with geometric multigrid,
           that is you can use ``"snes_type": "fas"`` or
           ``"pc_type": "mg"`` transparently.
        """
        # Do this first so __del__ doesn't barf horribly if we get an
        # error in __init__
        parameters, nullspace, options_prefix \
            = firedrake.solving_utils._extract_kwargs(**kwargs)

        if options_prefix is not None:
            self._opt_prefix = options_prefix
            self._auto_prefix = False
        else:
            self._opt_prefix = "firedrake_nlvsh_%d_" % NLVSHierarchy._id
            self._auto_prefix = True
            NLVSHierarchy._id += 1

        if isinstance(problem, firedrake.NonlinearVariationalProblem):
            # We just got a single problem so coarsen up the hierarchy
            problems = []
            while True:
                if problem:
                    problems.append(problem)
                else:
                    break
                problem = coarsen_problem(problem)
            problems.reverse()
        else:
            # User has provided list of problems
            problems = problem
        ctx = firedrake.solving_utils._SNESContext(problems)

        if nullspace is not None:
            raise NotImplementedError("Coarsening nullspaces no yet implemented")
        snes = PETSc.SNES().create()

        snes.setDM(problems[-1].dm)
        self.problems = problems
        self.snes = snes
        self.ctx = ctx
        self.ctx.set_function(self.snes)
        self.ctx.set_jacobian(self.snes)

        self.snes.setOptionsPrefix(self._opt_prefix)

        # Allow command-line arguments to override dict parameters
        opts = PETSc.Options()
        for k, v in opts.getAll().iteritems():
            if k.startswith(self._opt_prefix):
                parameters[k[len(self._opt_prefix):]] = v

        self.parameters = parameters

    def __del__(self):
        if self._auto_prefix and hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict)
        self._parameters = val
        firedrake.solving_utils.update_parameters(self, self.snes)

    def solve(self):
        dm = self.snes.getDM()

        dm.setAppCtx(weakref.proxy(self.ctx))
        dm.setCreateMatrix(self.ctx.create_matrix)

        hierarchy, _ = utils.get_level(dm)
        # FIXME: Need to set this up on the subDMs
        for V in hierarchy[:-1]:
            dm = V._dm
            dm.setAppCtx(weakref.proxy(self.ctx))
            dm.setCreateInterpolation(partial(make_transfer, typ="interpolation"))
            dm.setCreateInjection(partial(make_transfer, typ="injection"))
            dm.setCreateMatrix(self.ctx.create_matrix)

        self.snes.setFromOptions()

        for bc in self.problems[-1].bcs:
            bc.apply(self.problems[-1].u)

        with self.problems[-1].u.dat.vec as v:
            self.snes.solve(None, v)

        firedrake.solving_utils.check_snes_convergence(self.snes)
