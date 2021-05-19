from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Action)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import itertools

from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
import loopy as lp
import pymbolic.primitives as pym
from loopy.symbolic import SubArrayRef

# FIXME Move all slac loopy in separate file

def visualise(dag, how = None):
    """
        Visualises a slate dag. Can for example used to show the original expression
        vs the optimised slate expression.

        :arg: a dag with nodes that have shape information
    """
    import tsensor
    from collections import OrderedDict

    # Add tensors as locals to this frame.
    # It's how tsensor acesses shape information and so forth
    from firedrake.slate.slac.utils import traverse_dags
    tensors = OrderedDict()
    for node in traverse_dags([dag]):
        tensors[str(node)] = node
    locals().update(tensors)
    

    code = str(dag)
    # plot expr
    if how == "tree":
        g = tsensor.astviz(code)
        g.view()
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        tsensor.pyviz(code, ax=ax)
        plt.show()


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


def slate_to_gem(expression):
    """Convert a slate expression to gem.

    :arg expression: A slate expression.
    :returns: A singleton list of gem expressions and a mapping from
        gem variables to UFL "terminal" forms.
    """

    mapper, var2terminal = slate2gem(expression)
    return mapper, var2terminal


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
@_slate2gem.register(sl.TensorShell)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    name = f"T{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Variable(name, shape)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Block)
def _slate2gem_block(expr, self):
    child, = map(self, expr.children)
    child_shapes = expr.children[0].shapes
    offsets = tuple(sum(shape[:idx]) for shape, (idx, *_)
                    in zip(child_shapes.values(), expr._indices))
    return view(child, *(slice(idx, idx+extent) for idx, extent in zip(offsets, expr.shape)))


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    return Inverse(*map(self, expr.children))

@_slate2gem.register(sl.Action)
def _slate2gem_action(expr, self):
    name = f"A{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Action(*map(self, expr.children), name, expr.pick_op)
    self.var2terminal[var] = expr
    return var

@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    if expr.is_matfree():
        name = f"S{len(self.var2terminal)}"
        assert expr not in self.var2terminal.values()
        var = Solve(*map(self, expr.children), name, expr.is_matfree(), self(expr._Aonx), self(expr._Aonp))
        self.var2terminal[var] = expr
        # FIXME something is happening to the solve action node hash
        # so that gem node cannot be found in var2terminal even though it is there
        # so we save solve node by name for now
        self.var2terminal[name] = expr
        return var
    else:
        return Solve(*map(self, expr.children))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Product(Literal(-1),
                           Indexed(child, indices)),
                           indices)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.children)
    indices = tuple(make_indices(len(A.shape)))
    var = ComponentTensor(Sum(Indexed(A, indices),
                           Indexed(B, indices)),
                           indices)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.children)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                    Indexed(B, tuple([k] + j)))
    var = ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A



def slate2gem(expression):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    return mapper(expression), mapper.var2terminal


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


def merge_loopy(slate_loopy, output_arg, builder, var2terminal,  wrapper_name, ctx_g2l, strategy="terminals_first", slate_expr = None, tsfc_parameters=None):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""

    if strategy == "terminals_first":
        slate_loopy_prg = slate_loopy
        slate_loopy = slate_loopy[builder.slate_loopy_name]
        tensor2temp, tsfc_kernels, insns, builder = assemble_terminals_first(builder, var2terminal, slate_loopy)
        all_kernels = itertools.chain([slate_loopy], tsfc_kernels)
        # Construct args
        args = [output_arg] + builder.generate_wrapper_kernel_args(tensor2temp.values(), list(all_kernels))
        for a in slate_loopy.args:
            if a.name not in [arg.name for arg in args] and a.name.startswith("S"):
                ac = a.copy(address_space=lp.AddressSpace.LOCAL)
                args.append(ac)

        # Inames come from initialisations + loopyfying kernel args and lhs
        domains = slate_loopy.domains + builder.bag.index_creator.domains

        # The problem here is that some of the actions in the kernel get replaced by multiple tsfc calls.
        # So we need to introduce new ids on those calls to keep them unique.
        # But some the dependencies in the local matfree kernel are hand written and depend on the
        # original action id. At this point all the instructions should be ensured to be sorted, so
        # we remove all existing dependencies and make them sequential instead
        # also help scheduling by setting within_inames_is_final on everything
        insns_new = []
        for i, insn in enumerate(insns):
            if insn:
                insns_new.append(insn.copy(depends_on=frozenset({}),
                priority=len(insns)-i,
                within_inames_is_final=True))

        # Generates the loopy wrapper kernel
        slate_wrapper = lp.make_function(domains, insns_new, args, name=wrapper_name,
                                        seq_dependencies=True, target=lp.CTarget())

        # Prevent loopy interchange by loopy
        slate_wrapper = lp.prioritize_loops(slate_wrapper, ",".join(builder.bag.index_creator.inames.keys()))

        # Register kernels
        loop = itertools.chain([k.items() for k in tsfc_kernels], [{slate_loopy.name:slate_loopy_prg}.items()])
        for l in loop:
            (name, knl), = tuple(l)
            if knl:
                slate_wrapper = lp.merge([slate_wrapper, knl])
                slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)
        return slate_wrapper

    elif strategy == "when_needed":
        tensor2temp, builder, slate_loopy = assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr,
                                                    ctx_g2l, tsfc_parameters, True, {}, output_arg)
        return slate_loopy


def assemble_terminals_first(builder, var2terminal, slate_loopy):
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs, _ = builder.collect_coefficients()
    builder.bag = SlateWrapperBag(coeffs, name=slate_loopy.name)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), var2terminal.values()))
    tsfc_calls, tsfc_kernels = zip(*itertools.chain.from_iterable(
                                   (builder.generate_tsfc_calls(terminal, tensor2temp[terminal])
                                    for terminal in terminal_tensors)))

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))

    return tensor2temp, tsfc_kernels, insns, builder


def assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr, ctx_g2l, tsfc_parameters, init_temporaries=True, tensor2temp={}, output_arg=None):
    insns = []
    tensor2temps = tensor2temp
    knl_list = {}
    gem2pym = ctx_g2l.gem_to_pymbolic

    # Keeping track off all coefficients upfront
    # saves us the effort of one of those ugly dict comparisons
    coeffs = {}  # all coefficients including the ones for the action
    new_coeffs = {}  # coeffs coming from action
    old_coeffs = {}  # only old coeffs minus the ones replaced by the action coefficients

    # invert dict
    pyms = [pyms.name if isinstance(pyms, pym.Variable) else pyms.assignee_name for pyms in gem2pym.values()]
    pym2gem = OrderedDict(zip(pyms, gem2pym.keys()))
    c = 0 
    slate_loopy_name = builder.slate_loopy_name
    for insn in slate_loopy[slate_loopy_name].instructions:
        if isinstance(insn, lp.kernel.instruction.CallInstruction):
            if (insn.expression.function.name.startswith("action") or
                insn.expression.function.name.startswith("mtf")):
                c += 1

            # slate node corresponding to current instructions
            if isinstance(gem_action_node, Solve):
                # FIXME something is happening to the solve action node hash
                # so that gem node cannot be found in var2terminal even though it is there
                # so we save solve node by name for now
                gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                slate_node = var2terminal[insn.assignee_name]
            else:
                slate_node = var2terminal[gem_action_node]
                gem_inlined_node = Variable(insn.assignee_name, gem_action_node.shape)
                coeff_name = insn.expression.parameters[1].subscript.aggregate.name

                def link_action_coeff(builder, coeffs=None, names=None, terminals=None):
                    new_coeffs = {}
                    old_coeffs = {}
                    for coeff, name, terminal in zip(coeffs, names, terminals):  
                        old_coeff, new_coeff = builder.collect_coefficients(coeff,
                                                                            names=name,
                                                                            action_node=terminal)
                        new_coeffs.update(new_coeff)
                        old_coeffs.update(old_coeff)

                    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
                    if not builder.bag:
                        builder.bag = SlateWrapperBag(old_coeffs, "k_"+str(c), new_coeff, builder.slate_loopy_name)
                        builder.bag.call_name_generator("k_"+str(c))
                    else:
                        builder.bag.update_coefficients(old_coeffs, "k_"+str(c), new_coeff)
                    return builder, new_coeffs, old_coeffs

                def get_coeff(slate_nodes, coeff_names):
                    for slate_node, coeff_name in zip(slate_nodes, coeff_names):
                        terminal = slate_node.action()
                        coeff = slate_node.ufl_coefficient
                        names = {coeff._ufl_function_space:coeff_name}
                        yield terminal, coeff, names
                 
                if isinstance(slate_node, sl.Action):
                    terminal, coeff, names = tuple(*get_coeff([slate_node], [coeff_name]))
                else:
                    # Generate a kernel for the statements the action is supposed to be inlined with
                    slate_wrapper_bag = builder.bag

                    if not isinstance(slate_node, sl.Solve):
                        # This path handles the inlining of action which don't have a 
                        # terminal as the tensor to be acted on

                        # FIXME This still need to be updated to the new loopy
                        from firedrake.slate.slac.compiler import gem_to_loopy
                        action_wrapper_knl_name = "wrap_" + insn.expression.function.name
                        var2terminal_actions = {g:var2terminal[g] for p,g in pym2gem.items() if p in reads}
                        (action_wrapper_knl, action_gem2pym), action_output_arg = gem_to_loopy(gem_action_node,
                                                                                               var2terminal,
                                                                                               tsfc_parameters["scalar_type"],
                                                                                               action_wrapper_knl_name,
                                                                                               "wrap_"+insn.assignee_name)
                        
                        # Generate an instruction which call the action wrapper kernel
                        action_insn = insn.copy(expression=pym.Call(pym.Variable(builder.slate_loopy_name),
                                                                    insn.expression.parameters))

                        # Prepare data structures for a new swipe
                        slate_wrapper_bag = builder.bag
                        builder.slate_loopy_name = action_wrapper_knl_name 
                        builder.bag = builder.bag.copy(action_wrapper_knl_name+"_", action_wrapper_knl_name)
                        ctx2gl.gem_to_pymbolic = action_gem2pym
                    else:
                        # Generate matfree solve call and knl
                        action_insn, (action_wrapper_knl_name, action_wrapper_knl), action_output_arg = builder.generate_matfsolve_call(ctx_g2l, insn, gem_action_node)

                        # Prepare data structures for a new swipe
                        slate_wrapper_bag = builder.bag
                        builder.slate_loopy_name = action_wrapper_knl_name
                        builder.bag = builder.bag.copy("j_",
                                                        action_wrapper_knl_name)

                    child1, child2 = slate_node.children
                    action_tensor2temp = {child2:action_wrapper_knl.callables_table[builder.slate_loopy_name].subkernel.args[-1]}

                    
                    # Repeat for the actions which might be in the action wrapper kernel
                    action_tensor2temps, builder, action_wrapper_knl = assemble_when_needed(builder,
                                                                        var2terminal,
                                                                        action_wrapper_knl,
                                                                        slate_node,
                                                                        ctx_g2l,
                                                                        tsfc_parameters,
                                                                        init_temporaries=False,
                                                                        tensor2temp=action_tensor2temp,
                                                                        output_arg=action_output_arg)

                    # Update data structure before next instruction is processed
                    knl_list[builder.slate_loopy_name] = action_wrapper_knl 
                    builder.bag = slate_wrapper_bag
                    builder.slate_loopy_name = slate_loopy_name
                    tvs = slate_loopy.callables_table[slate_loopy_name].subkernel.temporary_variables
                    tensor2temps[slate_node] = tvs[insn.assignee_name].copy(target=lp.CTarget())
                    droppedAparams = action_insn.expression.parameters[1:]
                    insns.append(action_insn.copy(expression=pym.Call(action_insn.expression.function,
                                                                    droppedAparams)))
                    continue

                builder, new_coeff, old_coeff = link_action_coeff(builder, [coeff], [names], [terminal])
                new_coeffs.update(new_coeff)
                old_coeffs.update(old_coeff)

                if terminal not in tensor2temps.keys():
                    inits, tensor2temp = builder.initialise_terminals({gem_inlined_node: terminal}, builder.bag.coefficients)
                    tensor2temps.update(tensor2temp)

                    # temporaries that are filled with calls, which get inlined later,
                    # need to be initialised
                    for init in inits:
                        insns.append(init)
                
                # local assembly of the action or the matrix for the solve
                tsfc_calls, tsfc_knls = zip(*builder.generate_tsfc_calls(terminal, tensor2temps[terminal]))

                #FIXME we need to cover a case for explicit solves I think
                if tsfc_calls[0] and tsfc_knls[0]:
                    knl_list.update(tsfc_knls[0])
                    # substitute action call with the generated tsfc call for that action
                    # but keep the lhs so that the following instructions still act on the right temporaries
                    for (i, tsfc_call),((knl_name,knl),) in zip(enumerate(tsfc_calls), (t.items() for t in tsfc_knls)):
                        insns.append(lp.kernel.instruction.CallInstruction(insn.assignees,
                                                                           tsfc_call.expression,
                                                                           id=insn.id,
                                                                           within_inames=insn.within_inames,
                                                                           predicates=tsfc_call.predicates))
                else:
                    # This code path covers the case that the tsfc compiler doesn't give a kernel back
                    # I don't quite know yet what the cases are where it does not
                    # maybe when the kernel would just be an identity operation?
                    rhs = insn.expression.parameters[1]
                    var = insn.assignees[0].subscript.aggregate
                    lhs = pym.Subscript(var, var.index)
                    rhs = pym.Subscript(rhs.subscript.aggregate, var.index)
                    insns.append(lp.kernel.instruction.Assignment(lhs, rhs, 
                                                                  id=insn.id+"_whatsthis"))

        else:
            insns.append(insn)

    if init_temporaries:
        # Initialise the very first temporary
        # For that we need to get the temporary which
        # links to the same coefficient as the rhs of this node and init it              
        init_coeffs,_ = builder.collect_coefficients()
        var2terminal_vectors = {v:t for (v,t) in var2terminal.items()
                                    for (cv,ct) in init_coeffs.items()
                                    if isinstance(t, sl.AssembledVector)
                                    and t._function==cv}
        inits, tensor2temp = builder.initialise_terminals(var2terminal_vectors, init_coeffs)            
        tensor2temps.update(tensor2temp)
        for i in inits:
            insns.insert(0, i)

        # Get all coeffs into the wrapper kernel
        # so that we can generate the right wrapper kernel args of it
        builder.bag.update_coefficients(init_coeffs, "_"+str(c), new_coeffs)

    # FIXME Refactor into generate wrapper kernel function
    # 1) Prepare the wrapper kernel: scheduling of instructions
    # We remove all existing dependencies and make them sequential instead
    # also help scheduling by setting within_inames_is_final on everything
    new_insns = []
    for i, insn in enumerate(insns):
        if insn:
            if i == 0:
                last_id=insns[0].id
                new_insns.append(insn.copy(priority=len(insns)-i,
                within_inames_is_final=True))
            else:
                new_insns.append(insn.copy(depends_on=frozenset({last_id}),
                priority=len(insns)-i,
                within_inames_is_final=True))
                last_id=insn.id
    
    # 2) Prepare the wrapper kernel: in particular args and tvs so that they match the new instructions
    new_args = [output_arg] + builder.generate_wrapper_kernel_args(tensor2temps.values(), list(knl_list.values()))
    # new_args = [a.copy(target=lp.CTarget()) for a in old_new_args]
    global_args = []
    local_args = slate_loopy.callables_table[builder.slate_loopy_name].subkernel.temporary_variables
    for n in new_args:
        if n.address_space==lp.AddressSpace.GLOBAL:
            global_args += [n]
        else:
            local_args.update({n.name: n})

    # 3) Prepare the wrapper kernel: generate domains for indices used in the CallInstructions
    new_domains = slate_loopy[builder.slate_loopy_name].domains 
    new_domains += builder.bag.index_creator.domains

    # Final: Adjusted wrapper kernel
    copy_args = {"args": global_args, "domains":new_domains, "temporary_variables":local_args, "instructions":new_insns}
    slate_loopy.callables_table[builder.slate_loopy_name].subkernel = slate_loopy.callables_table[builder.slate_loopy_name].subkernel.copy(**copy_args)

    # Link action/matfree kernels to wrapper kernel
    # Match tsfc kernel args to the wrapper kernel callinstruction args,
    # because tsfc kernels have flattened indices
    for name, knl in knl_list.items():
        slate_loopy = lp.merge([slate_loopy, knl])
        print(slate_loopy)
        if not name.startswith("mtf"):
            slate_loopy = _match_caller_callee_argument_dimension_(slate_loopy, name)

    return tensor2temps, builder, slate_loopy