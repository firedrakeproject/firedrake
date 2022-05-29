from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict
from loopy.kernel.data import ValueArg

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Delta, Index, Division,
                 Action, MatfreeSolveContext)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from firedrake.parameters import target
import itertools
from tsfc.loopy import profile_insns
from petsc4py import PETSc

from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
import pymbolic.primitives as pym
from loopy.symbolic import SubArrayRef
from firedrake.slate.slac.optimise import optimise
from enum import IntEnum

# FIXME Move all slac loopy in separate file


class _AssemblyStrategy(IntEnum):
    TERMINALS_FIRST = 0
    WHEN_NEEDED = 1


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the
    integrals of forms.
    """
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class SymbolWithFuncallIndexing(ast.Symbol):
    """A functionally equivalent representation of a `coffee.Symbol`,
    with modified output for rank calls. This is syntactically necessary
    when referring to symbols of Eigen::MatrixBase objects.
    """

    def _genpoints(self):
        """Parenthesize indices during loop assignment"""
        pt = lambda p: "%s" % p
        pt_ofs = lambda p, o: "%s*%s+%s" % (p, o[0], o[1])
        pt_ofs_stride = lambda p, o: "%s+%s" % (p, o)
        result = []

        if not self.offset:
            for p in self.rank:
                result.append(pt(p))
        else:
            for p, ofs in zip(self.rank, self.offset):
                if ofs == (1, 0):
                    result.append(pt(p))
                elif ofs[0] == 1:
                    result.append(pt_ofs_stride(p, ofs[1]))
                else:
                    result.append(pt_ofs(p, ofs))
        result = ', '.join(i for i in result)

        return "(%s)" % result


class Transformer(Visitor):
    """Replaces all out-put tensor references with a specified
    name of :type: `Eigen::Matrix` with appropriate shape. This
    class is primarily for COFFEE acrobatics, jumping through
    nodes and redefining where appropriate.

    The default name of :data:`"A"` is assigned, otherwise a
    specified name may be passed as the :data:`name` keyword
    argument when calling the visitor.
    """

    def visit_object(self, o, *args, **kwargs):
        """Visits an object and returns it.

        e.g. string ---> string
        """
        return o

    def visit_list(self, o, *args, **kwargs):
        """Visits an input of COFFEE objects and returns
        the complete list of said objects.
        """
        newlist = [self.visit(e, *args, **kwargs) for e in o]
        if all(newo is e for newo, e in zip(newlist, o)):
            return o

        return newlist

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        """Visits a COFFEE FunDecl object and reconstructs
        the FunDecl body and header to generate
        ``Eigen::MatrixBase`` C++ template functions.

        Creates a template function for each subkernel form.

        .. code-block:: c++

            template <typename Derived>
            static inline void foo(Eigen::MatrixBase<Derived> const & A, ...)
            {
              [Body...]
            }
        """
        name = kwargs.get("name", "A")
        new = self.visit_Node(o, *args, **kwargs)
        ops, okwargs = new.operands()
        if all(new is old for new, old in zip(ops, o.operands()[0])):
            return o

        ret, kernel_name, kernel_args, body, pred, headers, template = ops

        body_statements, _ = body.operands()
        decl_init = "const_cast<Eigen::MatrixBase<Derived> &>(%s_);\n" % name
        new_dec = ast.Decl(typ="Eigen::MatrixBase<Derived> &", sym=name,
                           init=decl_init)
        new_body = [new_dec] + body_statements
        eigen_template = "template <typename Derived>"

        new_ops = (ret, kernel_name, kernel_args,
                   new_body, pred, headers, eigen_template)

        return o.reconstruct(*new_ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        """Visits a declared tensor and changes its type to
        :template: result `Eigen::MatrixBase<Derived>`.

        i.e. double A[n][m] ---> const Eigen::MatrixBase<Derived> &A_
        """
        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        newtype = "const Eigen::MatrixBase<Derived> &"

        return o.reconstruct(newtype, ast.Symbol("%s_" % name))

    def visit_Symbol(self, o, *args, **kwargs):
        """Visits a COFFEE symbol and redefines it as a Symbol with
        FunCall indexing.

        i.e. A[j][k] ---> A(j, k)
        """
        name = kwargs.get("name", "A")
        if o.symbol != name:
            return o

        return SymbolWithFuncallIndexing(o.symbol, o.rank, o.offset)


def slate_to_gem(expression, options):
    """Convert a slate expression to gem.

    :arg expression: A slate expression.
    :returns: A singleton list of gem expressions and a mapping from
        gem variables to UFL "terminal" forms.
    """

    mapper, gem2slate = slate2gem(expression, options)
    return mapper, gem2slate


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
@_slate2gem.register(sl.BlockAssembledVector)
@_slate2gem.register(sl.TensorShell)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    assert expr not in self.gem2slate.values()
    var = Variable(None, shape)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Block)
def _slate2gem_block(expr, self):
    child, = map(self, expr.children)
    child_shapes = expr.children[0].shapes
    offsets = tuple(sum(shape[:idx]) for shape, (idx, *_)
                    in zip(child_shapes.values(), expr._indices))
    return view(child, *(slice(idx, idx+extent) for idx, extent in zip(offsets, expr.shape)))


@_slate2gem.register(sl.DiagonalTensor)
def _slate2gem_diagonal(expr, self):
    if not self.matfree:
        A, = map(self, expr.children)
        assert A.shape[0] == A.shape[1]
        i, j = (Index(extent=s) for s in A.shape)
        idx = (i, i) if expr.vec else (i, j)
        jdx = (i,) if expr.vec else (i, j)
        return ComponentTensor(Product(Indexed(A, (i, i)), Delta(*idx)), jdx)
    else:
        child, = expr.children
        if child.terminal:
            P = self(sl.Tensor(child.form, diagonal=True))
            return P
        else:
            raise NotImplementedError("Diagonals on non-terminal Slate expressions are \
                                        not implemented in a matrix-free manner yet.")


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    tensor, = expr.children
    if expr.diagonal:
        # optimise inverse on diagonal tensor by translating to
        # matrix which contains the reciprocal values of the diagonal tensor
        if self.matfree:
            tensor, = tensor.children
            var = self(sl.Reciprocal(sl.Tensor(tensor.form, diagonal=True)))
            i, j = (Index(extent=s) for s in expr.shape)
            return ComponentTensor(Product(Indexed(var, (i,)), Delta(i, j)), (i, j))
        else:
            A, = map(self, expr.children)
            i, j = (Index(extent=s) for s in A.shape)
            return ComponentTensor(Product(Division(Literal(1.), Indexed(A, (i, i))),
                                           Delta(i, j)), (i, j))
    else:
        return Inverse(self(tensor))


@_slate2gem.register(sl.Reciprocal)
def _slate2gem_reciprocal(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Division(Literal(1.), Indexed(child, indices)), indices)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Action)
def _slate2gem_action(expr, self):
    assert expr not in self.gem2slate.values()
    children = list(map(self, expr.children))
    var = Action(*children, expr.pick_op)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    if expr.matfree:
        assert expr not in self.gem2slate.values()
        children = list(map(self, expr.children))
        if expr.preconditioner:
            # assert isinstance(expr.preconditioner, sl.Inverse), "Preconditioner has to be an inverse"
            # assert expr.preconditioner not in self.gem2slate.values()
            prec = self(expr.preconditioner)
            Ponr = self(expr.Ponr)
        else:
            prec = None
            Ponr = None
        ctx = {'matfree': expr.matfree, 'Aonx': self(expr.Aonx),'Aonp': self(expr.Aonp),
                'preconditioner': prec, 'Ponr': Ponr, "diag_prec": expr.diag_prec,
               'rtol': expr.rtol, 'atol': expr.atol, 'max_it': expr.max_it}
        var = Solve(*children, ctx=MatfreeSolveContext(**ctx))
        self.gem2slate[var.name] = expr
        return var
    else:
        return Solve(*map(self, expr.children))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Product(Literal(-1),
                          Indexed(child, indices)),
                          indices)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.children)
    indices = tuple(make_indices(len(A.shape)))
    var = ComponentTensor(Sum(Indexed(A, indices),
                          Indexed(B, indices)),
                          indices)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.children)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                    Indexed(B, tuple([k] + j)))
    var = ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Hadamard)
def _slate2gem_hadamard(expr, self):
    A, B = map(self, expr.children)
    idx = tuple(make_indices(len(A.shape)))
    AB = Product(Indexed(A, idx),
                 Indexed(B, idx))
    var = ComponentTensor(AB, idx)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A


def slate2gem(expression, options):
    mapper = Memoizer(_slate2gem)
    mapper.gem2slate = OrderedDict()
    mapper.matfree = options["replace_mul"]
    m = mapper(expression)
    return m, mapper.gem2slate


def depth_first_search(graph, node, visited, schedule):
    """A recursive depth-first search (DFS) algorithm for
    traversing a DAG consisting of Slate expressions.

    :arg graph: A DAG whose nodes (vertices) are Slate expressions
                with edges connected to dependent expressions.
    :arg node: A starting vertex.
    :arg visited: A set keeping track of visited nodes.
    :arg schedule: A list of reverse-postordered nodes. This list is
                   used to produce a topologically sorted list of
                   Slate nodes.
    """
    if node not in visited:
        visited.add(node)

        for n in graph[node]:
            depth_first_search(graph, n, visited, schedule)

        schedule.append(node)


def topological_sort(exprs):
    """Topologically sorts a list of Slate expressions. The
    expression graph is constructed by relating each Slate
    node with a list of dependent Slate nodes.

    :arg exprs: A list of Slate expressions.
    """
    graph = OrderedDict((expr, set(traverse_dags([expr])) - {expr})
                        for expr in exprs)

    schedule = []
    visited = set()
    for n in graph:
        depth_first_search(graph, n, visited, schedule)

    return schedule


def merge_loopy(slate_loopy, output_arg, builder, gem2slate, wrapper_name, ctx_g2l, strategy="terminals_first", slate_expr=None, tsfc_parameters=None, slate_parameters=None):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""

    if strategy == _AssemblyStrategy.TERMINALS_FIRST:
        slate_loopy_prg = slate_loopy
        slate_loopy = slate_loopy[builder.slate_loopy_name]
        tensor2temp, tsfc_kernels, insns, builder, events, preamble = assemble_terminals_first(builder, gem2slate, slate_loopy, wrapper_name)
        # Construct args
        args, tmp_args = builder.generate_wrapper_kernel_args(tensor2temp.values())
        kernel_args = [output_arg] + args
        args = [output_arg.loopy_arg] + [a.loopy_arg for a in args] + tmp_args
        for a in slate_loopy.args:
            if a.name not in [arg.name for arg in args] and a.name.startswith("S"):
                ac = a.copy(address_space=lp.AddressSpace.LOCAL)
                args.append(ac)

        # Inames come from initialisations + loopyfying kernel args and lhs
        domains = slate_loopy.domains + builder.bag.index_creator.domains

        # Help scheduling by setting within_inames_is_final on everything
        insns_new = []
        for i, insn in enumerate(insns):
            if insn:
                insns_new.append(insn.copy(depends_on=frozenset({}),
                                 priority=len(insns)-i,
                                 within_inames_is_final=True))

        # Generates the loopy wrapper kernel
        slate_wrapper = lp.make_function(domains, insns_new, args, name=wrapper_name,
                                         seq_dependencies=True, target=target,
                                         silenced_warnings=["single_writer_after_creation", "unused_inames"],
                                         preambles=preamble)

        # Prevent loopy interchange by loopy
        slate_wrapper = lp.prioritize_loops(slate_wrapper, ",".join(builder.bag.index_creator.inames.keys()))

        # Register kernels
        loop = []
        for k in tsfc_kernels:
            if k:
                loop += [k.items()]
        loop += [{slate_loopy.name: slate_loopy_prg}.items()]

        for l in loop:
            (name, knl), = tuple(l)
            if knl:
                slate_wrapper = lp.merge([slate_wrapper, knl])
                slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)
        return slate_wrapper, tuple(kernel_args), events

    elif strategy == _AssemblyStrategy.WHEN_NEEDED:
        coeffs, _ = builder.collect_coefficients(artificial=False)
        builder.bag.copy_coefficients(coeffs)
        tensor2temp, builder, slate_loopy, kernel_args, events = assemble_when_needed(builder, gem2slate,
                                                                 slate_loopy, slate_expr,
                                                                 ctx_g2l, tsfc_parameters,
                                                                 slate_parameters, True, {}, output_arg)
        return slate_loopy, tuple(kernel_args), events


def assemble_terminals_first(builder, gem2slate, slate_loopy, wrapper_name):
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs, _ = builder.collect_coefficients(artificial=False)
    builder.bag = SlateWrapperBag(coeffs, name=wrapper_name)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(gem2slate, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), gem2slate.values()))
    calls_and_kernels_and_events = tuple((c, k, e) for terminal in terminal_tensors
                                         for c, k, e in builder.generate_tsfc_calls(terminal, tensor2temp[terminal]))
    if calls_and_kernels_and_events:  # tsfc may not give a kernel back
        tsfc_calls, tsfc_kernels, tsfc_events = zip(*calls_and_kernels_and_events)
    else:
        tsfc_calls = ()
        tsfc_kernels = ()

    # Add profiling for inits
    inits, slate_init_event, preamble_init = profile_insns("inits_"+wrapper_name, inits, PETSc.Log.isActive())

    # Add profiling for inits
    inits, slate_init_event, preamble_init = profile_insns("inits_"+name, inits, PETSc.Log.isActive())

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))

    inits, slate_wrapper_event, preamble = profile_insns(wrapper_name, inits, PETSc.Log.isActive())
    
    events = tsfc_events + (slate_wrapper_event, slate_init_event) if PETSc.Log.isActive() else ()
    preamble = preamble+preamble_init if PETSc.Log.isActive() else None
    return tensor2temp, tsfc_kernels, insns, builder, events, preamble


def assemble_when_needed(builder, gem2slate, slate_loopy, slate_expr, ctx_g2l, tsfc_parameters, slate_parameters, init_temporaries=True, tensor2temp={}, output_arg=None, matshell=False):
    # FIXME This function needs some refactoring
    # Essentially there are 4 codepath: 1) insn is no matrix-free special insn
    #                                   2) insn is an Action
    #                                   3) insn is a Solve
    #                                       a) the matrix is terminal
    #                                       b) the matrix is a TensorShell
    # My idea would be to subclass CallInstruction from loopy to ActionInstruction, SolveInstruction, ...
    # and then we can use a dispatcher

    # need imports here bc m partially initialized module error
    from firedrake.slate.slac.kernel_builder import LocalLoopyKernelBuilder
    from firedrake.slate.slac.compiler import gem_to_loopy

    insns = []
    tensor2temps = tensor2temp
    knl_list = {}
    gem2pym = ctx_g2l.gem_to_pymbolic
    all_events = ()

    # invert dict
    pyms = [pyms.name if isinstance(pyms, pym.Variable) else pyms.assignee_name for pyms in gem2pym.values()]
    pym2gem = OrderedDict(zip(pyms, gem2pym.keys()))
    slate_loopy_name = builder.slate_loopy_name
    for insn in slate_loopy[slate_loopy_name].instructions:
        # TODO specialise the call instruction node and dispatch based on its type
        if (not isinstance(insn, lp.kernel.instruction.CallInstruction)
            or not (insn.expression.function.name.startswith("action")
                    or insn.expression.function.name.startswith("mtf"))):
            # normal instructions can stay as they are
            insns.append(insn)
        else:
            # gem node correponding to current instruction
            gem_action_node = pym2gem[insn.assignee_name]

            # slate node corresponding to current instructions
            if isinstance(gem_action_node, Solve):
                # FIXME something is happening to the solve action node hash
                # so that gem node cannot be found in gem2slate even though it is there
                # so we save solve node by name for now
                # [Update: the reason for that is
                # that the gem preprocessing in the loopy kernel generation is rewriting some of
                # the gem expression and the gem action node here is the preprocessed]
                slate_node = gem2slate[insn.assignee_name]
            else:
                slate_node = gem2slate[gem_action_node]

            # get information about the coefficient we act on
            coeff_name = insn.expression.parameters[1].subscript.aggregate.name
            tensor_shell_node, slate_coeff_node = slate_node.children
            if isinstance(slate_node, sl.Action) and not matshell:
                # ----- this is the code path for "pure" Actions ----

                # get a terminal tensor for the action
                # and generate a ufl coefficient->name dict
                # for the coefficient c in action(ufl.form, c)
                terminal = slate_node.action()
                coeff = slate_node.ufl_coefficient
                names = {coeff: (coeff_name, slate_node.coeff.shape)}

                # separate action and non-action coefficients, needed because
                # figuring out which coefficients needs to be in the kernel data
                # is different for original coefficients and action coefficients
                action_builder = LocalLoopyKernelBuilder(slate_node, builder.tsfc_parameters, builder.slate_loopy_name)
                action_builder.bag.index_creator = builder.bag.index_creator
                old_coeffs = builder.bag.coefficients
                original_coeffs, action_coeffs = builder.collect_coefficients(expr=terminal, names=names, artificial=True)
                new_coeffs = {}
                for c in original_coeffs:
                    if c in old_coeffs:
                        new_coeffs[c] = old_coeffs[c]
                    else:
                        new_coeffs[c] = original_coeffs[c]       
                action_builder.bag.copy_coefficients(new_coeffs, action_coeffs)

                # temporaries that have calls assigned, which get inlined later,[
                # need to be initialised, so e.g. the lhs of an action
                # that is because TSFC generates kernels with an output like A = A + ...]
                if terminal not in tensor2temps.keys():
                    # gem terminal node corresponding to lhs of the instructions
                    gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                    inits, tensor2temp = action_builder.initialise_terminals({gem_inlined_node: terminal}, action_builder.bag.coefficients)
                    tensor2temps.update(tensor2temp)
                    for init in inits:
                        insns.append(init)

                # replaces call with tsfc call, which gets linked to tsfc kernel later
                action_insns, action_knl_list, action_builder, events = generate_tsfc_knls_and_calls(action_builder, terminal, tensor2temps, insn)
                all_events += events
                insns += action_insns
                knl_list.update(action_knl_list)
                builder.bag.copy_extra_args(action_builder.bag)

            else:
                if not (isinstance(slate_node, sl.Solve)):
                    # ----- Codepath for matrix-free solves on tensor shells ----
                    # This path handles the inlining of action which don't have a
                    # terminal as the tensor to be acted on

                    # NOTE we kick of a new compilation here since
                    # we need to compile expression within a tensor shell node to gem
                    # and then futher into a loopy kernel
                    old_slate_node = slate_node
                    slate_node = optimise(slate_node, slate_parameters)
                    gem_action_node, gem2slate_actions = slate_to_gem(slate_node, slate_parameters)
                    (action_wrapper_knl, ctx_g2l_action, event), action_output_arg = gem_to_loopy(gem_action_node,
                                                                                           gem2slate_actions,
                                                                                           tsfc_parameters["scalar_type"],
                                                                                           "tensorshell",
                                                                                           insn.assignee_name,
                                                                                           matfree=True)
                    all_events += (event, )

                    # Prepare data structures of builder for a new swipe
                    action_wrapper_knl_name = ctx_g2l_action.kernel_name
                    action_builder = LocalLoopyKernelBuilder(slate_node, builder.tsfc_parameters, action_wrapper_knl_name)
                    
                    original_coeffs, _ = builder.collect_coefficients(expr=old_slate_node.children[0], artificial=False)
                    old_coeffs = builder.bag.coefficients
                    new_coeffs = {}
                    for c in original_coeffs:
                        if c in old_coeffs:
                            new_coeffs[c] = old_coeffs[c]
                        else:
                            new_coeffs[c] = original_coeffs[c]
                    if new_coeffs:
                        action_builder.bag.copy_coefficients(new_coeffs, None)

                    # glue the action coeff to the newly generated kernel
                    # we need this because the new run through the compiler above generated new temps, also for the coefficient,
                    # but we want the kernel for the tensorshell to work on the coefficient as defined in the instruction we currently deal with
                    i = -1 if isinstance(slate_node, sl.Hadamard) else 1
                    old_arg = action_wrapper_knl[action_wrapper_knl_name].args[i]
                    new_var = insn.expression.parameters[1].subscript.aggregate
                    new_arg = old_arg.copy(name=new_var.name)
                    new_args = action_wrapper_knl[action_wrapper_knl_name].args[:i] + [new_arg]
                    if not i == -1:
                        new_args += action_wrapper_knl[action_wrapper_knl_name].args[i+1:]
                    action_wrapper_knl = lp.fix_parameters(action_wrapper_knl, within=None, **{old_arg.name: new_var})
                    action_wrapper_knl.callables_table[action_wrapper_knl_name].subkernel = action_wrapper_knl[action_wrapper_knl_name].copy(args=new_args)
                    action_tensor2temp = {slate_coeff_node: action_wrapper_knl[action_wrapper_knl_name].args[i]}

                    # we need to initialise the action temporaries for kernels
                    # which contain the action of a non terminal tensor on a coefficient
                    # that is because TSFC generates kernels with an output like A = A + ...
                    if tensor_shell_node not in tensor2temps.keys():
                        # gem terminal node corresponding to the output value of the kernel called
                        gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                        inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: slate_node}, None)
                        tensor2temps.update(tensor2temp)
                        for init in inits:
                            # preconditioner actions are called multiple times but we don't
                            # need to init them twice because one is hand coded in matfree kernel
                            if init.id not in [insn.id for insn in insns]:
                                insns.append(init)

                    if isinstance(slate_node, sl.Hadamard):
                        terminal, coeff = slate_node.children
                        tensor = (terminal.children[0]
                                  if isinstance(terminal, sl.Inverse)
                                  else terminal)
                        terminal = sl.Tensor(tensor.children[0].form, diagonal=tensor.children[0].diagonal)

                        if terminal not in tensor2temps.keys():
                            diagT_arg = action_wrapper_knl[action_wrapper_knl_name].args[1]
                        names = {coeff: (diagT_arg.name, slate_node.shape)}
                        action_tensor2temp.update({terminal: diagT_arg})

                        if terminal in tensor2temps.keys():
                            old_arg = action_wrapper_knl[action_wrapper_knl_name].args[-2]
                            new_arg = old_arg.copy(name=diagT_arg.name)
                            new_args = action_wrapper_knl[action_wrapper_knl_name].args[:-2] + [new_arg] + action_wrapper_knl[action_wrapper_knl_name].args[-1:]
                            action_wrapper_knl = lp.fix_parameters(action_wrapper_knl, within=None, **{old_arg.name: pym.Variable(diagT_arg.name)})
                            action_wrapper_knl.callables_table[action_wrapper_knl_name].subkernel = action_wrapper_knl[action_wrapper_knl_name].copy(args=new_args)

                        if terminal not in tensor2temps.keys():
                            # separate action and non-action coefficients, needed because
                            # figuring out which coefficients needs to be in the kernel data
                            # is different for original coefficients and action coefficients

                            # gem terminal node corresponding to lhs of the instructions
                            gem_inlined_node = Variable(diagT_arg.name, gem_action_node.shape)
                            inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: terminal}, None)
                            tensor2temps.update(tensor2temp)
                            for init in inits:
                                if init.id not in [insn.id for insn in insns]:
                                    insns.append(init)

                            # replaces call with tsfc call, which gets linked to tsfc kernel later
                            ac_temp = builder.bag.action_coefficients
                            builder.bag.action_coefficients = None
                            tsfc_insns, tsfc_knls, modified_action_builder, events = generate_tsfc_knls_and_calls(builder, terminal,
                                                                                                          action_tensor2temp,
                                                                                                          insn,
                                                                                                          pym.Variable(gem_inlined_node.name))
                            all_events += events

                            builder.bag.action_coefficients = ac_temp
                            insns += tsfc_insns
                            knl_list.update(tsfc_knls)

                else:
                    # ----- Codepath for matrix-free solves on terminal tensors ----
                    action_builder = LocalLoopyKernelBuilder(slate_node, builder.tsfc_parameters, None)

                    # Generate matfree solve call and knl
                    new_coeffs = builder.bag.coefficients
                    action_builder.bag.copy_coefficients(new_coeffs)
                    builder.bag.copy_extra_args(action_builder.bag)

                    (action_insn, (action_wrapper_knl_name, action_wrapper_knl),
                     action_output_arg, ctx_g2l, loopy_rhs, event) = action_builder.generate_matfsolve_call(ctx_g2l, insn, gem_action_node)
                    all_events += (event, )

                    # Prepare data structures of builder for a new swipe
                    # in particular the tensor2temp dict needs to hold the rhs of the matrix-solve in Slate and in loopy
                    action_builder.slate_loopy_name = action_wrapper_knl_name
                    action_tensor2temp = {slate_coeff_node: loopy_rhs}
                    gem2slate_actions = gem2slate
                    ctx_g2l_action = ctx_g2l
                    ctx_g2l_action.kernel_name = action_wrapper_knl_name

                    # we don't need inits for this codepath because
                    # the kernel builder generates the matfree solve kernel as A = ...

                # Repeat for the actions which might be in the action wrapper kernel
                builder.bag.copy_extra_args(action_builder.bag)
                _, modified_action_builder, action_wrapper_knl, kernel_args, events = assemble_when_needed(action_builder,
                                                                                      gem2slate_actions,
                                                                                      action_wrapper_knl,
                                                                                      slate_node,
                                                                                      ctx_g2l_action,
                                                                                      tsfc_parameters,
                                                                                      slate_parameters,
                                                                                      init_temporaries=False,
                                                                                      tensor2temp=action_tensor2temp,
                                                                                      output_arg=action_output_arg,
                                                                                      matshell=isinstance(tensor_shell_node, sl.TensorShell))
                
                all_events += events

                # For updating the wrapper kernel args later we want to add all extra args needed in any of the subkernels
                builder.bag.copy_extra_args(modified_action_builder.bag)

                # Modify action wrapper kernel args and params in the call for this insn
                # based on what the tsfc kernels inside need
                action_insn, action_wrapper_knl, builder = update_kernel_call_and_knl(insn,
                                                                                      action_wrapper_knl,
                                                                                      action_wrapper_knl_name,
                                                                                      builder)

                # Update with new insn and its knl
                insns.append(action_insn.copy(expression=pym.Call(action_insn.expression.function,
                                                                  action_insn.expression.parameters)))
                knl_list[action_builder.slate_loopy_name] = action_wrapper_knl

    if init_temporaries:
        # We need to do initialise the temporaries at the end, when we collected all the ones we need
        builder, tensor2temps, inits, init_knls = initialise_temps(builder, gem2slate, tensor2temps)
        for i in inits:
            insns.insert(0, i)
            knl_list.update(init_knls)

    slate_loopy, kernel_args = update_wrapper_kernel(builder, insns, output_arg, tensor2temps, knl_list, slate_loopy)
    return tensor2temps, builder, slate_loopy, kernel_args, all_events


def generate_tsfc_knls_and_calls(builder, terminal, tensor2temps, insn, var=None):
    insns = []
    knl_list = {}

    # local assembly kernel for the action
    tsfc_calls, tsfc_knls, events = zip(*builder.generate_tsfc_calls(terminal, tensor2temps[terminal]))

    # FIXME we need to cover a case for explicit solves I think
    # when do we something mixed - matrix-free and matrix-explicit
    var = var if var else insn.assignees[0].subscript.aggregate

    if tsfc_calls[0] and tsfc_knls[0]:
        # substitute action call with the generated tsfc call for that action
        # but keep the lhs so that the following instructions still act on the right temporaries
        for (i, tsfc_call), ((knl_name, knl), ) in zip(enumerate(tsfc_calls), (t.items() for t in tsfc_knls)):
            # wi = frozenset(i for i in itertools.chain(insn.within_inames, tsfc_call.within_inames))
            name = builder.slate_loopy_name
            idx = tsfc_call.assignees[0].subscript.index
            lhs = lp.symbolic.SubArrayRef(idx, pym.Subscript(var, idx))
            inames = [i.name for i in insn.assignees[0].subscript.index]
            wi = frozenset(i for i in itertools.chain(insn.within_inames, tsfc_call.within_inames))
            expr = pym.Call(tsfc_call.expression.function, (lhs,) + tsfc_call.expression.parameters[1:])
            insns.append(lp.kernel.instruction.CallInstruction((lhs,),
                                                               expr,
                                                               id=str(insn.id) + "_" + name[name.rfind("_")+1:]+"_"+str(i),
                                                               within_inames=wi,
                                                               predicates=tsfc_call.predicates))
            knl_list[knl_name] = knl
    else:
        # This code path covers the case that the tsfc compiler doesn't give a kernel back.
        # This happens when the local assembly call just returns zeros.
        # Since we want to reuse the tensor potentially we need to initialise it anyways.
        rhs = insn.expression.parameters[1]
        var = insn.assignees[0].subscript.aggregate
        lhs = pym.Subscript(var, insn.assignees[0].subscript.index)
        rhs = pym.Subscript(rhs.subscript.aggregate, insn.assignees[0].subscript.index)
        inames = [i.name for i in insn.assignees[0].subscript.index]
        wi = frozenset(i for i in itertools.chain(insn.within_inames, inames))
        insns.append(lp.kernel.instruction.Assignment(lhs, 0., id=insn.id+"_whatsthis", within_inames=wi))
    return insns, knl_list, builder, events


def initialise_temps(builder, gem2slate, tensor2temps):
    # Initialise the very first temporaries
    # (with coefficients from the original ufl form)
    # For that we need to get the temporary which
    # links to the same coefficient as the rhs of this node and init it
    init_coeffs, _ = builder.collect_coefficients(artificial=False)
    gem2slate_vectors = {v: t for (v, t) in gem2slate.items()
                         for cv, ct in init_coeffs.items()
                         if (isinstance(t, sl.AssembledVector) and t._function == cv)
                         or (isinstance(t, sl.BlockAssembledVector) and cv == t._original_function)}

    builder.bag.coefficients = OrderedDict(builder.bag.coefficients)
    for k, v in init_coeffs.items():
        builder.bag.coefficients[k] = v
        builder.bag.coefficients.move_to_end(k)

    inits, tensor2temp = builder.initialise_terminals(gem2slate_vectors, init_coeffs)
    tensor2temps.update(tensor2temp)

    # Get all coeffs into the wrapper kernel bag
    # so that we can generate the right wrapper kernel args of it
    gem2slate_diags = {v: t for (v, t) in gem2slate.items()
                       if t.diagonal and t.terminal and isinstance(t, sl.Tensor)}

    knl_list = {}
    tsfc_inits = []
    for variable, terminal in gem2slate_diags.items():
        ac_temp = builder.bag.action_coefficients
        builder.bag.action_coefficients = None
        # builder.bag.coefficients = None
        if terminal not in tensor2temps.keys():
            # gem terminal node corresponding to lhs of the instructions
            gem_inlined_node = Variable(variable.name, variable.shape)
            inits_diag, tensor2temp = builder.initialise_terminals({gem_inlined_node: terminal}, None)
            tensor2temps.update(tensor2temp)
            for init in inits_diag:
                if init.id not in [insn.id for insn in inits]:
                    inits.append(init)

            # replaces call with tsfc call, which gets linked to tsfc kernel later
            tsfc_insns, tsfc_knls, events = zip(*builder.generate_tsfc_calls(terminal, tensor2temps[terminal]))
            tsfc_inits += tsfc_insns
            for tk in tsfc_knls:
                knl_list.update(tk)
        builder.bag.action_coefficients = ac_temp

    return builder, tensor2temps, tsfc_inits+inits, knl_list

# A note on the following helper functions:
# There are two update functions.
# One is updating the args and so on of the inner kernel, e.g. the matrix-free solve in a slate loopy kernel.
# The other is updating the call to that inner kernel inside the outer loopy, e.g. in the slate loopy kernel.


def update_wrapper_kernel(builder, insns, output_arg, tensor2temps, knl_list, slate_loopy):
    # 1) Prepare the wrapper kernel: scheduling of instructions
    # We remove all existing dependencies and make them sequential instead
    # also help scheduling by setting within_inames_is_final on everything.
    # The problem here is that some of the actions in the kernel get replaced by multiple tsfc calls.
    # So we need to introduce new ids on those calls to keep them unique.
    # But some the dependencies in the local matfree kernel are hand written and depend on the
    # original action id. At this point all the instructions should be ensured to be sorted, so
    # we remove all existing dependencies and make them sequential instead
    # also help scheduling by setting within_inames_is_final on everything
    new_insns = []
    last_id = None
    for i, insn in enumerate(insns):
        if insn:
            kwargs = {}
            if i != 0:
                kwargs["depends_on"] = frozenset({last_id})
            kwargs["priority"] = len(insns)-i
            kwargs["within_inames_is_final"] = True

            kwargs["within_inames"] = insn.within_inames
            new_insns.append(insn.copy(**kwargs))
            last_id = insn.id

    # 2) Prepare the wrapper kernel: in particular args and tvs so that they match the new instructions,
    # which contain the calls to the action, solve and tensorshell kernels
    not_loopy_args, tmp_args = builder.generate_wrapper_kernel_args(tensor2temps.values())
    kernel_args = [output_arg] + not_loopy_args
    args = [a.loopy_arg for a in not_loopy_args]
    new_args = [output_arg.loopy_arg] + args + tmp_args
    global_args = []
    local_args = slate_loopy[builder.slate_loopy_name].temporary_variables
    for n in new_args:
        if isinstance(n, ValueArg) or n.address_space == lp.AddressSpace.GLOBAL:
            global_args += [n]
        else:
            local_args.update({n.name: n})

    # 3) Prepare the wrapper kernel: generate domains for indices used in the new instructions
    new_domains = slate_loopy[builder.slate_loopy_name].domains
    new_domains += builder.bag.index_creator.domains

    # Final: Adjusted wrapper kernel
    copy_args = {"args": global_args, "domains": new_domains, "temporary_variables": local_args, "instructions": new_insns}
    slate_loopy.callables_table[builder.slate_loopy_name].subkernel = slate_loopy.callables_table[builder.slate_loopy_name].subkernel.copy(**copy_args)

    # Link action/matfree/tensorshell kernels to wrapper kernel
    # Match tsfc kernel args to the wrapper kernel callinstruction args,
    # because tsfc kernels have flattened indices
    for name, knl in knl_list.items():
        slate_loopy = lp.merge([slate_loopy, knl])
        slate_loopy = _match_caller_callee_argument_dimension_(slate_loopy, name)
    return slate_loopy, tuple(kernel_args)


def update_kernel_call_and_knl(insn, action_wrapper_knl, action_wrapper_knl_name, builder):
    """
        This function is updating the args of the call to the inner kernel.
        An example: if tfsc produces a local assembly kernel for an action,
        then this might need some extra information when the assembly kernel is generated for a facet integral.
        Depending on which args the kernel generated from tsfc take, we generate a CallInstruction
        which takes the same arguments in the calling kernel.
    """
    knl = action_wrapper_knl[action_wrapper_knl_name]

    # Generate args for the kernel and reads for the call instruction
    # FIXME something similar is reappearing in lot of places
    # so maybe this function should go in the kernel builder
    def make_reads(a):
        if not isinstance(a, ValueArg):
            var_reads = pym.Variable(a.name)
            idx_reads = builder.bag.index_creator(a.shape)
            return SubArrayRef(idx_reads, pym.Subscript(var_reads, idx_reads))
        else:
            return pym.Variable(a.name)

    # Generate reads form kernel args
    reads = [make_reads(a) for a in knl.args]

    action_insn = insn.copy(expression=pym.Call(pym.Variable(action_wrapper_knl_name), tuple(reads)))
    return action_insn, action_wrapper_knl, builder
