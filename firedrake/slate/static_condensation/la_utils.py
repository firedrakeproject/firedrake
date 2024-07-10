from collections import namedtuple
from pyop2.utils import as_tuple

from firedrake.formmanipulation import split_form
from firedrake.parameters import parameters
from firedrake.slate.slate import DiagonalTensor, Tensor, AssembledVector
from firedrake.petsc import PETSc


LAContext = namedtuple("LAContext",
                       ["lhs", "rhs", "field_idx"])
LAContext.__doc__ = """\
Context information for systems of equations after
applying algebraic transformation via Slate-supported
operations. This object provides the symbolic expressions
for the transformed linear system of equations.

:param lhs: The resulting expression for the transformed
            left-hand side matrix.
:param rhs: The resulting expression for the transformed
            right-hand side vector.
:param field_idx: An integer or iterable of integers
                  (if the system is mixed) denoting
                  which field(s) the resulting solution
                  is defined on.
"""


def condense_and_forward_eliminate(A, b, elim_fields, prefix, pc):
    """Returns Slate expressions for the operator and
    right-hand side vector after eliminating specified
    unknowns.

    :arg A: a `slate.Tensor` corresponding to the
            mixed UFL operator.
    :arg b: a `firedrake.Function` corresponding
            to the right-hand side.
    :arg elim_fields: a `tuple` of indices denoting
                      which fields to eliminate.
    :arg prefix: an option prefix for the condensed field.
    :arg pc: a Preconditioner instance.
    :returns: a tuple of `LAContext` and `SchurComplementBuilder`
    """

    if not isinstance(A, Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    # Ensures field indices are in increasing order
    id_0, id_1 = elim_fields[0], elim_fields[-1]
    elim_fields = list(as_tuple(elim_fields))
    elim_fields.sort()

    all_fields = list(range(len(A.arg_function_spaces[0])))

    condensed_fields = list(set(all_fields) - set(elim_fields))
    condensed_fields.sort()

    _A = A.blocks
    _b = AssembledVector(b).blocks

    # NOTE: Does not support non-contiguous field elimination
    e_idx0 = elim_fields[0]
    e_idx1 = elim_fields[-1]
    f_idx0 = condensed_fields[0]
    f_idx1 = condensed_fields[-1]

    # Finite element systems where static condensation
    # is possible have the general form:
    #
    #  | A_ee A_ef || x_e |   | b_e |
    #  |           ||     | = |     |
    #  | A_fe A_ff || x_f |   | b_f |
    #
    # where subscript `e` denotes the coupling with fields
    # that will be eliminated, and `f` denotes the condensed
    # fields.
    outer_id_0 = slice(e_idx0, e_idx1 + 1)
    outer_id_1 = slice(f_idx0, f_idx1 + 1)
    Aff = _A[outer_id_1, outer_id_1]
    Aef = _A[outer_id_0, outer_id_1]
    Afe = _A[outer_id_1, outer_id_0]
    Aee = _A[outer_id_0, outer_id_0]

    bf = _b[outer_id_1]
    be = _b[outer_id_0]

    # The reduced operator and right-hand side are:
    #  S = A_ff - A_fe * A_ee.inv * A_ef
    #  r = b_f - A_fe * A_ee.inv * b_e
    # TODO check the vidx and pidx indices are correct
    schur_builder = SchurComplementBuilder(prefix, Aee, Afe, Aef, pc, id_0, id_1, non_zero_saddle_mat=Aff)
    r, S = schur_builder.build_schur(be, non_zero_saddle_rhs=bf)

    field_idx = [idx for idx in range(f_idx0, f_idx1)]

    return LAContext(lhs=S, rhs=r, field_idx=field_idx), schur_builder


def backward_solve(A, b, x, schur_builder, reconstruct_fields):
    """Returns a sequence of linear algebra contexts containing
    Slate expressions for backwards substitution.

    :arg A: a `slate.Tensor` corresponding to the
            mixed UFL operator.
    :arg b: a `firedrake.Function` corresponding
            to the right-hand side.
    :arg x: a `firedrake.Function` corresponding
            to the solution.
    :arg schur_builder: a `SchurComplementBuilder`
    :arg reconstruct_fields: a `tuple` of indices denoting
                             which fields to reconstruct.
    :returns: a list of `LAContext` for the reconstruction
    """

    if not isinstance(A, Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    all_fields = list(range(len(A.arg_function_spaces[0])))
    nfields = len(all_fields)
    reconstruct_fields = as_tuple(reconstruct_fields)

    _b = b.subfunctions
    _x = x.subfunctions

    # Ordering matters
    systems = []

    # Reconstruct one unknown from one determined field:
    #
    # | A_ee  A_ef || x_e |   | b_e |
    # |            ||     | = |     |
    # | A_fe  A_ff || x_f |   | b_f |
    #
    # where x_f is known from a previous computation.
    # Returns the system:
    #
    # A_ee x_e = b_e - A_ef * x_f.
    if nfields == 2:
        id_e, = reconstruct_fields
        id_f, = [idx for idx in all_fields if idx != id_e]

        A_eeinv = schur_builder.A00_inv_hat
        A_ef = schur_builder.KT
        b_e = AssembledVector(_b[id_e])
        x_f = AssembledVector(_x[id_f])

        r_e = b_e - A_ef * x_f
        local_system = LAContext(lhs=A_eeinv, rhs=r_e, field_idx=(id_e,))

        systems.append(local_system)

    # Reconstruct two unknowns from one determined field:
    #
    # | A_e0e0  A_e0e1  A_e0f || x_e0 |   | b_e0 |
    # | A_e1e0  A_e1e1  A_e1f || x_e1 | = | b_e1 |
    # | A_fe0   A_fe1   A_ff  || x_f  |   | b_f  |
    #
    # where x_f is the known field. Returns two systems to be
    # solved in order (determined from the reverse order of indices
    # e0 and e1):
    #
    # Solve for e1 first (obtained from eliminating x_e0):
    #
    # S_e1 x_e1 = r_e1
    #
    # where
    #
    # S_e1 = A_e1e1 - A_e1e0 * A_e0e0.inv * A_e0e1, and
    # r_e1 = b_e1 - A_e1e0 * A_e0e0.inv * b_e0
    #      - (A_e1f - A_e1e0 * A_e0e0.inv * A_e0f) * x_f,
    #
    # And then solve for x_e0 given x_f and x_e1:
    #
    # A_e0e0 x_e0 = b_e0 - A_e0e1 * x_e1 - A_e0f * x_f.
    elif nfields == 3:
        if len(reconstruct_fields) != nfields - 1:
            raise NotImplementedError("Implemented for 1 determined field")

        # Order of reconstruction doesn't need to be in order
        # of increasing indices
        id_e0, id_e1 = reconstruct_fields
        id_f, = [idx for idx in all_fields if idx not in reconstruct_fields]

        _, A_e0e1, A_e1e0, _ = schur_builder.list_split_mixed_ops
        A_e0f, A_e1f = schur_builder.list_split_trace_ops_transpose
        A_e0e0inv = schur_builder.A00_inv_hat

        x_e1 = AssembledVector(_x[id_e1])
        x_f = AssembledVector(_x[id_f])

        b_e0 = AssembledVector(_b[id_e0])
        b_e1 = AssembledVector(_b[id_e1])

        # Solve for e1
        Sf = A_e1f - A_e1e0 * A_e0e0inv * A_e0f
        r_e1 = b_e1 - A_e1e0 * A_e0e0inv * b_e0 - Sf * x_f

        Sinv = schur_builder.inner_S_inv_hat
        systems.append(LAContext(lhs=Sinv, rhs=r_e1, field_idx=(id_e1,)))

        # Solve for e0
        r_e0 = b_e0 - A_e0e1 * x_e1 - A_e0f * x_f
        systems.append(LAContext(lhs=A_e0e0inv, rhs=r_e0, field_idx=(id_e0,)))

    else:
        msg = "Not implemented for systems with %s fields" % nfields
        raise NotImplementedError(msg)

    return systems


class SchurComplementBuilder(object):

    """A Slate-based Schur complement expression builder. The expression is
    used in the trace system solve and parts of it in the reconstruction
    calls of the other two variables of the hybridised system.
    How the Schur complement if constructed, and in particular how the local inverse of the
    mixed matrix is built, is controlled with PETSc options. All corresponding PETSc options
    start with ``hybridization_localsolve``.
    The following option sets are valid together with the usual set of hybridisation options:

    .. code-block:: text

        {'localsolve': {'ksp_type': 'preonly',
                        'pc_type': 'fieldsplit',
                        'pc_fieldsplit_type': 'schur'}}

    A Schur complement is requested for the mixed matrix inverse which appears inside the
    Schur complement of the trace system solve. The Schur complements are then nested.
    For details see defition of :meth:`build_schur`. No fieldsplit options are set so all
    local inverses are calculated explicitly.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'schur',
                       'fieldsplit_1': {'ksp_type': 'default',
                                        'pc_type': 'python',
                                        'pc_python_type': __name__ + '.DGLaplacian'}}

    The inverse of the Schur complement inside the Schur decomposition of the mixed matrix inverse
    is approximated by a default solver (LU in the matrix-explicit case) which is preconditioned
    by a user-defined operator, e.g. a DG Laplacian, see :meth:`build_inner_S_inv`.
    So :math:`P_S * S * x = P_S * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                        'pc_type': 'fieldsplit',
                        'pc_fieldsplit_type': 'schur',
                        'fieldsplit_1': {'ksp_type': 'default',
                                        'pc_type': 'python',
                                        'pc_python_type': __name__ + '.DGLaplacian',
                                        'aux_ksp_type': 'preonly'}
                                        'aux_pc_type': 'jacobi'}}}}

    The inverse of the Schur complement inside the Schur decomposition of the mixed matrix inverse
    is approximated by a default solver (LU in the matrix-explicit case) which is preconditioned
    by a user-defined operator, e.g. a DG Laplacian. The inverse of the preconditioning matrix is
    approximated through the inverse of only the diagonal of the provided operator, see
    :meth:`build_Sapprox_inv`. So :math:`diag(P_S).inv * S * x = diag(P_S).inv * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'schur',
                       'fieldsplit_0': {'ksp_type': 'default',
                                        'pc_type': 'jacobi'}

    The inverse of the :math:`A_{00}` block of the mixed matrix is approximated by a default solver
    (LU in the matrix-explicit case) which is preconditioned by the diagonal matrix of :math:`A_{00},
    see :meth:`build_A00_inv`. So :math:`diag(A_{00}).inv * A_{00} * x = diag(A_{00}).inv * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'None',
                       'fieldsplit_0':  ...
                       'fieldsplit_1':  ...

    All the options for ``fieldsplit_`` are still valid if ``'pc_fieldsplit_type': 'None'.`` In this case
    the mixed matrix inverse which appears inside the Schur complement of the trace system solve
    is calculated explicitly, but the local inverses of :math:`A_{00}` and the Schur complement
    in the reconstructions calls are still treated according to the options in ``fieldsplit_``.

    """

    def __init__(self, prefix, Atilde, K, KT, pc, vidx, pidx, non_zero_saddle_mat=None):
        # set options, operators and order of sub-operators
        self.Atilde = Atilde
        self.K = K
        self.KT = KT
        self.vidx = vidx
        self.pidx = pidx

        # prefixes
        self.prefix = prefix + "localsolve_"
        self._retrieve_options(pc)

        self.non_zero_saddle_mat = non_zero_saddle_mat

        # Check if Atilde is mixed
        all_fields = list(range(len(Atilde.arg_function_spaces[0])))
        self.nfields = len(all_fields)
        self._split_mixed_operator()

        # build all inverses
        self.A00_inv_hat = self.build_A00_inv()
        if self.nfields > 1:
            self.schur_approx = self.retrieve_user_S_approx(pc, self.schur_approx) if self.schur_approx else None
            self.inner_S = self.build_inner_S()
            self.inner_S_approx_inv_hat = self.build_Sapprox_inv()
            self.inner_S_inv_hat = self.build_inner_S_inv()

    def _split_mixed_operator(self):
        split_mixed_op = dict(split_form(self.Atilde.form))
        id0, id1 = (self.vidx, self.pidx)
        A00 = Tensor(split_mixed_op[(id0, id0)])
        self.list_split_mixed_ops = [A00, None, None, None]

        if self.nfields > 1:
            A01 = Tensor(split_mixed_op[(id0, id1)])
            A10 = Tensor(split_mixed_op[(id1, id0)])
            A11 = Tensor(split_mixed_op[(id1, id1)])
            self.list_split_mixed_ops = [A00, A01, A10, A11]

            split_trace_op = dict(split_form(self.K.form))
            K0 = Tensor(split_trace_op[(0, id0)])
            K1 = Tensor(split_trace_op[(0, id1)])
            self.list_split_trace_ops = [K0, K1]

            split_trace_op_transpose = dict(split_form(self.KT.form))
            K0 = Tensor(split_trace_op_transpose[(id0, 0)])
            K1 = Tensor(split_trace_op_transpose[(id1, 0)])
            self.list_split_trace_ops_transpose = [K0, K1]

    def _check_options(self, valid_options):
        opts = PETSc.Options(self.prefix)
        for key, supported in valid_options:
            try:
                value = opts.getString(key)
            except KeyError:
                continue

            if value not in supported:
                raise ValueError(f"Unsupported value ({value}) for '{self.prefix + key}'. "
                                 f"Should be one of {supported}")

    def _retrieve_options(self, pc):
        get_option = lambda key: PETSc.Options(self.prefix).getString(key, default="")

        # Get options for Schur complement decomposition
        self._check_options([("ksp_type", {"preonly"}), ("pc_type", {"fieldsplit"}), ("pc_fieldsplit_type", {"schur"})])
        self.nested = (get_option("ksp_type") == "preonly"
                       and get_option("pc_type") == "fieldsplit"
                       and get_option("pc_fieldsplit_type") == "schur")

        # Get preconditioning options for A00
        fs0, fs1 = ("fieldsplit_"+str(idx) for idx in (self.vidx, self.pidx))
        self._check_options([(fs0+"ksp_type", {"preonly", "default"}), (fs0+"pc_type", {"jacobi"})])
        self.preonly_A00 = get_option(fs0+"_ksp_type") == "preonly"
        self.jacobi_A00 = get_option(fs0+"_pc_type") == "jacobi"

        # Get preconditioning options for the Schur complement
        self._check_options([(fs1+"ksp_type", {"preonly", "default"}), (fs1+"pc_type", {"jacobi", "python"})])
        self.preonly_S = get_option(fs1+"_ksp_type") == "preonly"
        self.jacobi_S = get_option(fs1+"_pc_type") == "jacobi"

        # Get user supplied operator and its options
        self.schur_approx = (get_option(fs1+"_pc_python_type") if get_option(fs1+"_pc_type") == "python" else False)
        self._check_options([(fs1+"aux_ksp_type", {"preonly", "default"}), (fs1+"aux_pc_type", {"jacobi"})])
        self.preonly_Shat = get_option(fs1+"_aux_ksp_type") == "preonly"
        self.jacobi_Shat = get_option(fs1+"_aux_pc_type") == "jacobi"

        if self.jacobi_Shat or self.jacobi_A00:
            assert parameters["slate_compiler"]["optimise"], ("Local systems should only get preconditioned with "
                                                              "a preconditioning matrix if the Slate optimiser replaces "
                                                              "inverses by solves.")

    def build_inner_S(self):
        """Build the inner Schur complement."""
        _, A01, A10, A11 = self.list_split_mixed_ops
        return A11 - A10 * self.A00_inv_hat * A01

    def inv(self, A, P, prec, preonly=False):
        """ Calculates the inverse of an operator A.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with the preconditioner P
            if prec is True.
            The inverse of A may be just approximated with the inverse of P
            if prec and replace.
        """
        return (P if prec and preonly else
                (P*A).inv * P if prec else
                A.inv)

    def build_inner_S_inv(self):
        """ Calculates the inverse of the schur complement.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with the preconditioner P.
        """
        A = self.inner_S
        P = self.inner_S_approx_inv_hat
        prec = bool(self.schur_approx) or self.jacobi_S
        return self.inv(A, P, prec, self.preonly_S)

    def build_Sapprox_inv(self):
        """ Calculates the inverse of preconditioner to the Schur complement,
            which can be either the schur complement approximation provided by the user
            or jacobi.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with jacobi.
        """
        prec = (bool(self.schur_approx) and self.jacobi_Shat) or self.jacobi_S
        A = self.schur_approx if self.schur_approx else self.inner_S
        P = DiagonalTensor(A).inv
        preonly = self.preonly_Shat if self.schur_approx else True
        return self.inv(A, P, prec, preonly)

    def build_A00_inv(self):
        """ Calculates the inverse of :math:`A_{00}`, the (0,0)-block of the mixed matrix Atilde.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with jacobi.
        """
        A, _, _, _ = self.list_split_mixed_ops
        P = DiagonalTensor(A).inv
        return self.inv(A, P, self.jacobi_A00, self.preonly_A00)

    def retrieve_user_S_approx(self, pc, usercode):
        """Retrieve a user-defined :class:firedrake.preconditioners.AuxiliaryOperator from the PETSc Options,
        which is an approximation to the Schur complement and its inverse is used
        to precondition the local solve in the reconstruction calls (e.g.).
        """
        _, _, _, A11 = self.list_split_mixed_ops
        test, trial = A11.arguments()
        if usercode != "":
            (modname, funname) = usercode.rsplit('.', 1)
            mod = __import__(modname)
            fun = getattr(mod, funname)
            if isinstance(fun, type):
                fun = fun()
            return Tensor(fun.form(pc, test, trial)[0])
        else:
            return None

    def build_schur(self, rhs, non_zero_saddle_rhs=None):
        """The Schur complement in the operators of the trace solve contains
        the inverse on a mixed system.  Users may want this inverse to be treated
        with another Schur complement.

        Let the mixed matrix Atilde be called A here.
        Then, if a nested schur complement is requested, the inverse of Atilde
        is rewritten with help of a a Schur decomposition as follows.

        .. code-block:: text

                A.inv=[[I, -A00.inv * A01] * [[A00.inv, 0    ] * [[I,            0]
                      [0,  I             ]]  [0,        S.inv]]  [-A10* A00.inv, I]]
                      --------------------   -----------------   ------------------
                             block1                block2              block3
                with the (inner) schur complement S = A11 - A10 * A00.inv * A01
        """

        if self.nested:
            _, A01, A10, _ = self.list_split_mixed_ops
            K0, K1 = self.list_split_trace_ops
            KT0, KT1 = self.list_split_trace_ops_transpose
            R = [rhs.blocks[self.vidx],
                 rhs.blocks[self.pidx]]
            # K * block1
            K_Ainv_block1 = [K0, -K0 * self.A00_inv_hat * A01 + K1]
            # K * block1 * block2
            K_Ainv_block2 = [K_Ainv_block1[0] * self.A00_inv_hat,
                             K_Ainv_block1[1] * self.inner_S_inv_hat]
            # K * block1 * block2 * block3
            K_Ainv_block3 = [K_Ainv_block2[0] - K_Ainv_block2[1] * A10 * self.A00_inv_hat,
                             K_Ainv_block2[1]]
            # K * block1 * block2 * block3 * broken residual
            schur_rhs = (K_Ainv_block3[0] * R[0] + K_Ainv_block3[1] * R[1])
            # K * block1 * block2 * block3 * K.T
            schur_comp = K_Ainv_block3[0] * KT0 + K_Ainv_block3[1] * KT1
        else:
            P = DiagonalTensor(self.Atilde).inv
            Atildeinv = self.inv(self.Atilde, P, self.jacobi_A00, self.preonly_A00)
            schur_rhs = self.K * Atildeinv * rhs
            schur_comp = self.K * Atildeinv * self.KT
        if self.non_zero_saddle_mat or non_zero_saddle_rhs:
            assert self.non_zero_saddle_mat and non_zero_saddle_rhs, "The problem is not a saddle point system and you missed to pass either A11 or the corresponding part in the rhs."
            # problem is not a saddle point problem
            schur_rhs = non_zero_saddle_rhs - schur_rhs
            schur_comp = self.non_zero_saddle_mat - schur_comp
        return schur_rhs, schur_comp
