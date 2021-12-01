from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Delta, Index, Division,
                 Action)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from firedrake.parameters import target
import itertools

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

    mapper, var2terminal = slate2gem(expression, options)
    return mapper, var2terminal


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
@_slate2gem.register(sl.BlockAssembledVector)
@_slate2gem.register(sl.TensorShell)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    assert expr not in self.var2terminal.values()
    var = Variable(None, shape)
    self.var2terminal[var] = expr
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
        return ComponentTensor(Product(Indexed(A, (i, i)), Delta(i, j)), (i, j))
    else:
        raise NotImplementedError("Diagonals on Slate expressions are \
                                   not implemented in a matrix-free manner yet.")


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    tensor, = expr.children
    if expr.diagonal:
        # optimise inverse on diagonal tensor by translating to
        # matrix which contains the reciprocal values of the diagonal tensor
        A, = map(self, expr.children)
        i, j = (Index(extent=s) for s in A.shape)
        return ComponentTensor(Product(Division(Literal(1), Indexed(A, (i, i))),
                                       Delta(i, j)), (i, j))
    else:
        return Inverse(self(tensor))


@_slate2gem.register(sl.Reciprocal)
def _slate2gem_reciprocal(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Division(Literal(1.), Indexed(child, indices)), indices)


@_slate2gem.register(sl.Action)
def _slate2gem_action(expr, self):
    assert expr not in self.gem2slate.values()
    children = list(map(self, expr.children))
    var = Action(*children, expr.pick_op)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    if expr.matfree:
        assert expr not in self.gem2slate.values()
        var = Solve(*map(self, expr.children), expr.matfree, self(expr.Aonx), self(expr.Aonp))
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


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A


def slate2gem(expression, options):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    mapper.gem2slate = OrderedDict()
    mapper.matfree = options["replace_mul"]
    m = mapper(expression)
    # WIP actually make use of the fact that we do have two different dicts now
    mapper.var2terminal.update(mapper.gem2slate)
    return m, mapper.var2terminal


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


def merge_loopy(slate_loopy, output_arg, builder, var2terminal, wrapper_name, ctx_g2l, strategy="terminals_first", slate_expr=None, tsfc_parameters=None, slate_parameters=None):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""

    if strategy == _AssemblyStrategy.TERMINALS_FIRST:
        slate_loopy_prg = slate_loopy
        slate_loopy = slate_loopy[builder.slate_loopy_name]
        tensor2temp, tsfc_kernels, insns, builder = assemble_terminals_first(builder, var2terminal, slate_loopy)
        # Construct args
        args, tmp_args = builder.generate_wrapper_kernel_args(tensor2temp)
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
                                         seq_dependencies=True, target=lp.CTarget(),
                                         silenced_warnings=["single_writer_after_creation", "unused_inames"])

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
        return slate_wrapper, tuple(kernel_args)

    elif strategy == _AssemblyStrategy.WHEN_NEEDED:
        tensor2temp, builder, slate_loopy = assemble_when_needed(builder, gem2slate,
                                                                 slate_loopy, slate_expr,
                                                                 ctx_g2l, tsfc_parameters,
                                                                 slate_parameters, True, {}, output_arg)
        return slate_loopy


def assemble_terminals_first(builder, var2terminal, slate_loopy):
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs, _ = builder.collect_coefficients(artificial=False)
    builder.bag = SlateWrapperBag(coeffs, name=slate_loopy.name)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), var2terminal.values()))
    tsfc_calls, tsfc_kernels = zip(*itertools.chain.from_iterable(
                                   (builder.generate_tsfc_calls(terminal, tensor2temp[terminal])
                                    for terminal in terminal_tensors)))

    # Add profiling for inits
    inits, slate_init_event, preamble_init = profile_insns("inits_"+name, inits, PETSc.Log.isActive())

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))
    
    return tensor2temp, tsfc_kernels, insns, builder

def assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr, ctx_g2l, tsfc_parameters, slate_parameters, init_temporaries=True, tensor2temp={}, output_arg=None, matshell=False):
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
                # so that gem node cannot be found in var2terminal even though it is there
                # so we save solve node by name for now
                # [Update: the reason for that is
                # that the gem preprocessing in the loopy kernel generation is rewriting some of
                # the gem expression and the gem action node here is the preprocessed]
                slate_node = var2terminal[insn.assignee_name]
            else:
                slate_node = var2terminal[gem_action_node]

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
                original_coeffs, action_coeffs = builder.collect_coefficients(expr=terminal, names=names)
                builder.bag.copy_coefficients(original_coeffs, action_coeffs)

                # temporaries that have calls assigned, which get inlined later,[
                # need to be initialised, so e.g. the lhs of an action
                # that is because TSFC generates kernels with an output like A = A + ...]
                if terminal not in tensor2temps.keys():
                    # gem terminal node corresponding to lhs of the instructions
                    gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                    inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: terminal}, builder.bag.coefficients)
                    tensor2temps.update(tensor2temp)
                    for init in inits:
                        insns.append(init)

                # replaces call with tsfc call, which gets linked to tsfc kernel later
                action_insns, action_knl_list, builder = generate_tsfc_knls_and_calls(builder, terminal, tensor2temps, insn)
                insns += action_insns
                knl_list.update(action_knl_list)

            else:
                if not (isinstance(slate_node, sl.Solve)):
                    # ----- Codepath for matrix-free solves on tensor shells ----
                    # This path handles the inlining of action which don't have a
                    # terminal as the tensor to be acted on

                    # NOTE we kick of a new compilation here since
                    # we need to compile expression within a tensor shell node to gem
                    # and then futher into a loopy kernel
                    slate_node = optimise(slate_node, slate_parameters)
                    gem_action_node, var2terminal_actions = slate_to_gem(slate_node, slate_parameters)
                    (action_wrapper_knl, ctx_g2l_action), action_output_arg = gem_to_loopy(gem_action_node,
                                                                                           var2terminal_actions,
                                                                                           tsfc_parameters["scalar_type"],
                                                                                           "tensorshell",
                                                                                           insn.assignee_name,
                                                                                           matfree=True)

                    # Prepare data structures of builder for a new swipe
                    action_wrapper_knl_name = ctx_g2l_action.kernel_name
                    action_builder = LocalLoopyKernelBuilder(slate_node, builder.tsfc_parameters, action_wrapper_knl_name)

                    # glue the action coeff to the newly generated kernel
                    # we need this because the new run through the compiler above generated new temps, also for the coefficient,
                    # but we want the kernel for the tensorshell to work on the coefficient as defined in the instruction we currently deal with
                    old_arg = action_wrapper_knl[action_wrapper_knl_name].args[1]
                    new_var = insn.expression.parameters[1].subscript.aggregate
                    new_arg = old_arg.copy(name=new_var.name)
                    new_args = [action_wrapper_knl[action_wrapper_knl_name].args[0], new_arg] + action_wrapper_knl[action_wrapper_knl_name].args[2:]
                    action_wrapper_knl = lp.fix_parameters(action_wrapper_knl, within=None, **{old_arg.name: new_var})
                    action_wrapper_knl.callables_table[action_wrapper_knl_name].subkernel = action_wrapper_knl[action_wrapper_knl_name].copy(args=new_args)
                    action_tensor2temp = {slate_coeff_node: action_wrapper_knl[action_wrapper_knl_name].args[1]}

                    # we need to initialise the action temporaries for kernels
                    # which contain the action of a non terminal tensor on a coefficient
                    # that is because TSFC generates kernels with an output like A = A + ...
                    if tensor_shell_node not in tensor2temps.keys():
                        # gem terminal node corresponding to the output value of the kernel called
                        gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                        inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: slate_node}, None)
                        tensor2temps.update(tensor2temp)
                        for init in inits:
                            insns.append(init)
                else:
                    # ----- Codepath for matrix-free solves on terminal tensors ----

                    # Generate matfree solve call and knl
                    (action_insn, (action_wrapper_knl_name, action_wrapper_knl),
                     action_output_arg, ctx_g2l, loopy_rhs) = builder.generate_matfsolve_call(ctx_g2l, insn, gem_action_node)

                    # Prepare data structures of builder for a new swipe
                    # in particular the tensor2temp dict needs to hold the rhs of the matrix-solve in Slate and in loopy
                    action_builder = LocalLoopyKernelBuilder(slate_node, builder.tsfc_parameters, action_wrapper_knl_name)
                    action_tensor2temp = {slate_coeff_node: loopy_rhs}
                    var2terminal_actions = var2terminal
                    ctx_g2l_action = ctx_g2l
                    ctx_g2l_action.kernel_name = action_wrapper_knl_name

                    # we don't need inits for this codepath because
                    # the kernel builder generates the matfree solve kernel as A = ...

                # Repeat for the actions which might be in the action wrapper kernel
                _, modified_action_builder, action_wrapper_knl = assemble_when_needed(action_builder,
                                                                                      var2terminal_actions,
                                                                                      action_wrapper_knl,
                                                                                      slate_node,
                                                                                      ctx_g2l_action,
                                                                                      tsfc_parameters,
                                                                                      slate_parameters,
                                                                                      init_temporaries=False,
                                                                                      tensor2temp=action_tensor2temp,
                                                                                      output_arg=action_output_arg,
                                                                                      matshell=isinstance(tensor_shell_node, sl.TensorShell))

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
        builder, tensor2temps, inits = initialise_temps(builder, var2terminal, tensor2temps)
        for i in inits:
            insns.insert(0, i)

    slate_loopy = update_wrapper_kernel(builder, insns, output_arg, tensor2temps, knl_list, slate_loopy)
    return tensor2temps, builder, slate_loopy


def generate_tsfc_knls_and_calls(builder, terminal, tensor2temps, insn):
    insns = []
    knl_list = {}

    # local assembly kernel for the action
    tsfc_calls, tsfc_knls = zip(*builder.generate_tsfc_calls(terminal, tensor2temps[terminal]))

    # FIXME we need to cover a case for explicit solves I think
    # when do we something mixed - matrix-free and matrix-explicit

    if tsfc_calls[0] and tsfc_knls[0]:
        # substitute action call with the generated tsfc call for that action
        # but keep the lhs so that the following instructions still act on the right temporaries
        for (i, tsfc_call), ((knl_name, knl), ) in zip(enumerate(tsfc_calls), (t.items() for t in tsfc_knls)):
            wi = frozenset(i for i in itertools.chain(insn.within_inames, tsfc_call.within_inames))
            insns.append(lp.kernel.instruction.CallInstruction(insn.assignees,
                                                               tsfc_call.expression,
                                                               id=insn.id+"_"+str(i),
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
    return insns, knl_list, builder


def initialise_temps(builder, var2terminal, tensor2temps):
    # Initialise the very first temporaries
    # (with coefficients from the original ufl form)
    # For that we need to get the temporary which
    # links to the same coefficient as the rhs of this node and init it
    init_coeffs, _ = builder.collect_coefficients(artificial=False)
    var2terminal_vectors = {v: t for (v, t) in var2terminal.items()
                            for cv, ct in init_coeffs.items()
                            if isinstance(t, sl.AssembledVector)
                            and t._function == cv}

    inits, tensor2temp = builder.initialise_terminals(var2terminal_vectors, init_coeffs)
    tensor2temps.update(tensor2temp)

    # Get all coeffs into the wrapper kernel bag
    # so that we can generate the right wrapper kernel args of it
    builder.bag.copy_coefficients(coeffs=init_coeffs)
    return builder, tensor2temps, inits

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
    for i, insn in enumerate(insns):
        if insn:
            kwargs = {}
            if i != 0:
                kwargs["depends_on"] = frozenset({last_id})
            kwargs["priority"] = len(insns)-i
            kwargs["within_inames_is_final"]=True
            new_insns.append(insn.copy(**kwargs))
            last_id=insn.id

    # 2) Prepare the wrapper kernel: in particular args and tvs so that they match the new instructions,
    # which contain the calls to the action, solve and tensorshell kernels
    new_args = [output_arg] + builder.generate_wrapper_kernel_args(tensor2temps)
    global_args = []
    local_args = slate_loopy[builder.slate_loopy_name].temporary_variables
    for n in new_args:
        if n.address_space == lp.AddressSpace.GLOBAL:
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
    return slate_loopy


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
    def make_reads(shape, name):
        var_reads = pym.Variable(name)
        idx_reads = builder.bag.index_creator(shape)
        return SubArrayRef(idx_reads, pym.Subscript(var_reads, idx_reads))

    # Generate reads form kernel args
    reads = [make_reads(a.shape, a.name) for a in knl.args]

    action_insn = insn.copy(expression=pym.Call(pym.Variable(action_wrapper_knl_name), tuple(reads)))
    return action_insn, action_wrapper_knl, builder
