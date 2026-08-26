from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import numbers
import functools

import pymbolic as pym
from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr

from pyop3.axis_tree.tree import (
    UNIT_AXIS_TREE,
    IndexedAxisTree,
)
from pyop3 import mpi, utils
from pyop3.buffer import IndexedBuffer
from pyop3.insn.base import Intent, assignment_type_as_intent
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE

from pyop3.insn.base import (
    AbstractAssignment,
    AssignmentType,
    Exscan,
    InstructionList,
    Loop,
    NonEmptyArrayAssignment,
    NullInstruction,
    StandaloneCalledFunction,
    assignment_type_as_intent,
)


class CodegenContext(ABC):
    """
    Abstract base class for code generation contexts.
    
    Abstract methods required for auto-generating based on _compile_static in codegen.py
    """
    
    def __init__(self, *, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        self.propagate_negatives = propagate_negatives
        self.mask_array_accesses = mask_array_accesses

        self._domains = [] 
        self._instructions = []
        self._arguments = []
        self._subkernels = []
        self._last_insn_id = None # determine dependence

        self._name_generator = utils.UniqueNameGenerator()

        # (buffer, nest_indices) -> name in kernel
        self.kernel_names = {}

        # buffer name -> buffer
        self.buffer_intents = {}

        # assignee name -> indirection expression
        self._assignees = {}

    @property
    def domains(self) -> Tuple:
        return tuple(self._domains)

    @property
    def instructions(self) -> Tuple:
        return tuple(self._instructions)

    @property
    def arguments(self) -> Tuple:
        return tuple(sorted(self._arguments, key=lambda arg: getattr(arg, 'name', '')))

    @property
    def subkernels(self) -> Tuple:
        return tuple(self._subkernels)

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._instructions.append(insn)
        self._last_insn_id = insn.id

    # {{{ abstract methods


    @abstractmethod
    def var(self, iname: str, *args) -> str | pym.primitives.Variable:
        """
        Implementation to represent symbolic variable for respective IR
        """
        pass


    @abstractmethod
    def add_domain(self, iname: str, *args) -> None:
        pass

    @abstractmethod
    def add_assignment(self, assignee, expression, prefix: str = "insn") -> None:
        pass

    @abstractmethod
    def add_function_call(self, assignees, expression, prefix: str = "insn") -> None:
        pass
    
    @abstractmethod
    def add_buffer(self, buffer_view: IndexedBuffer, intent: Intent | None = None) -> str:
        pass 

    @abstractmethod
    def add_subkernel(self, subkernel) -> None:
        pass

    @abstractmethod
    def set_temporary_shapes(self, shapes) -> None:
        pass

    ''' Lowering passes for respective codegen context '''
    @abstractmethod
    def lower_expr(self, expr, iname_maps, loop_indices, 
                   intent: Intent | None = None, paths = None):
        """
        Lower a PyOP3 expression to the target's IR representation.
        
        Returns:
            - pymbolic for Loopy
            - xDSL for MLIR
        """
        pass

    @abstractmethod
    def lower_buffer_access(
        self, 
        buffer: IndexedBuffer, 
        layouts, 
        iname_maps, 
        loop_indices, 
        *,
        intent
    ):
        pass

    @abstractmethod
    def add_leaf_assignment(self, assignment, paths, iname_maps, loop_indices):
        pass

    @abstractmethod
    def register_extent(self, obj: Any, inames, loop_indices):
        pass

    @abstractmethod
    def compile_standalone_function(self, call, loop_indices):
        """
        Compiling standalone functions i.e. LACallable for target IR representations 
        """
        pass

    @abstractmethod
    def compile_petsc_mat(self, assignment, loop_indices):
        """
        """
        pass

    @abstractmethod
    def compile_exscan(self, call, loop_indices):
        """
        """
        pass

    # }}}


    # {{{ general implementations


    def compile_array_assignment(
            self,
            assignment, 
            loop_indices, 
            axis_trees, 
            *,
            iname_replace_maps=None, 
            # TODO document these under "Other Parameters"
            axis_tree=None, 
            paths=None
    ):
        if paths is None:
            paths = []
        if iname_replace_maps is None: 
            iname_replace_maps = []

        if axis_tree is None:
            axis_tree, *axis_trees = axis_trees

            paths += [idict()]
            iname_replace_maps += [idict()]
            
            if axis_tree.is_empty or axis_tree is UNIT_AXIS_TREE or isinstance(axis_tree, IndexedAxisTree):
                if axis_trees: 
                    raise NotImplementedError("Refactor needed")

                self.add_leaf_assignment(
                    assignment, 
                    paths, 
                    iname_replace_maps, 
                    loop_indices
                )
                return

        axis = axis_tree.node_map[paths[-1]]
        for component in axis.components:
            new_paths = paths.copy()
            new_paths[-1] = paths[-1] | {axis.label: component.label}
            
            if axis_tree.linearize(new_paths[-1], partial=True).size == 0: 
                continue
            
            if component.local_size != 1:
                iname = self.unique_name("i")
                ext = self.register_extent(
                    component.size, 
                    iname_replace_maps[-1], 
                    loop_indices
                )
                self.add_domain(iname, ext)
                new_maps = iname_replace_maps.copy()
                new_maps[-1] = iname_replace_maps[-1] | {axis.label: self.var(iname)}
                within_inames = {iname}
            else:
                new_maps = iname_replace_maps.copy()
                new_maps[-1] = iname_replace_maps[-1] | {axis.label: 0}
                within_inames = set()

            with self.within_inames(within_inames):
                if axis_tree.node_map[new_paths[-1]]:
                    self.compile_array_assignment(
                        assignment, 
                        loop_indices, 
                        axis_trees, 
                        iname_replace_maps=new_maps, 
                        axis_tree=axis_tree, 
                        paths=new_paths
                    )
                elif axis_trees:
                    self.compile_array_assignment(
                        assignment, 
                        loop_indices, 
                        axis_trees, 
                        iname_replace_maps=new_maps, 
                        axis_tree=None, 
                        paths=new_paths
                    )
                else:
                    self.add_leaf_assignment(
                        assignment, 
                        new_paths, 
                        new_maps, 
                        loop_indices
                    )

    def parse_loop_properly_this_time(
            self,
            loop,
            axis_tree,
            loop_indices,
            *,
            axis=None,
            path=None,
            iname_map=None,
    ) -> None:
        if axis_tree is UNIT_AXIS_TREE:
            # NOTE: might need an expression here sometimes
            for statement in loop.statements:
                _compile(
                    statement,
                    # loop_indices | dict(loop_exprs),
                    loop_indices,
                    self,
                )
            return

        if utils.strictly_all(x is None for x in {axis, path, iname_map}):
            axis = axis_tree.root
            path = idict()
            iname_map = idict()

        for component in axis.components:
            path_ = path | {axis.label: component.label}

            if axis_tree.linearize(path_, partial=True).size == 0:
                continue
            elif component.size != 1:
                iname = self.unique_name("i")
                domain_var = self.register_extent(
                    component.size,
                    iname_map,
                    loop_indices
                )
                self.add_domain(iname, domain_var)
                iname_replace_map_ = iname_map | {axis.label: self.var(iname)}
                within_inames = frozenset({iname})
            else:
                iname_replace_map_ = iname_map | {axis.label: 0}
                within_inames = set()

            with self.within_inames(within_inames):
                if subaxis := axis_tree.node_map[path_]:
                    self.parse_loop_properly_this_time(
                        loop,
                        axis_tree,
                        loop_indices,
                        axis=subaxis,
                        path=path_,
                        iname_map=iname_replace_map_,
                    )
                else:
                    loop_indices |= idict({
                        (loop.index.id, axis_label): iname
                        for axis_label, iname in iname_replace_map_.items()
                    })
                    for statement in loop.statements:
                        _compile(
                            statement,
                            loop_indices,
                            self,
                        )
    # }}} 

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def __str__(self) -> str:
        ctx = f"Domain: {str(self.domains)}\n\n"
        ctx += f"Instructions: {str(self.instructions)}\n\n"
        ctx += f"Arguments: {str(self.arguments)}\n\n"
        ctx += f"Subkernels: {str(self.subkernels)}\n\n"
        return ctx 


@functools.singledispatch
def _compile(expr: Any, loop_indices: Dict, codegen_context: CodegenContext) -> None:
    raise TypeError(f"No handler defined for {type(expr).__name__}")

@_compile.register(NullInstruction)
def _(null, *args, **kwargs): 
    pass

@_compile.register(InstructionList)
def _(
    insn_list, 
    loop_indices, 
    codegen_context
) -> None:
    for insn in insn_list: 
        _compile(insn, loop_indices, codegen_context)

@_compile.register(Loop)
def _(
    loop, 
    loop_indices, 
    codegen_context
) -> None:
    codegen_context.parse_loop_properly_this_time(
        loop, 
        loop.index.iterset, 
        loop_indices, 
    )

@_compile.register(StandaloneCalledFunction)
def _(call, loop_indices, codegen_context):
    codegen_context.compile_standalone_function(call, loop_indices)

@_compile.register(NonEmptyArrayAssignment)
def parse_assignment(assignment: NonEmptyArrayAssignment, loop_indices, codegen_context: CodegenContext):
    if any(isinstance(arg, pyop3.expr.MatPetscMatBufferExpression) for arg in assignment.arguments):
        codegen_context.compile_petsc_mat(assignment, loop_indices)
    else:
        codegen_context.compile_array_assignment(
            assignment,
            loop_indices,
            assignment.axis_trees,
        )

@_compile.register(Exscan)
def _(exscan, loop_indices, codegen_context):
    codegen_context.compile_exscan(exscan, loop_indices)


# NOTE: Make this overloaded function into class in transform.py
# Only issue may be loopy-specific standalone_function overloading.
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")

@_collect_temporary_shapes.register(InstructionList)
def _(insn_list):
    return utils.merge_dicts(_collect_temporary_shapes(insn) for insn in insn_list)

@_collect_temporary_shapes.register(Loop)
def _(loop):
    shapes = {}
    for stmt in loop.statements:
        for temp, shape in _collect_temporary_shapes(stmt).items():
            if shape is None:
                continue
            if temp in shapes:
                assert shapes[temp] == shape
            else:
                shapes[temp] = shape
    return shapes

@_collect_temporary_shapes.register(AbstractAssignment)
@_collect_temporary_shapes.register(NullInstruction)
@_collect_temporary_shapes.register(Exscan)
def _(assignment: AbstractAssignment, /) -> idict:
    return idict()

@_collect_temporary_shapes.register
def _(call: StandaloneCalledFunction):
    import loopy as lp # TODO: Remove once StandaloneCalledFunction/similar integrated with MLIR
    return idict(
        {
            arg.buffer: lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
            if isinstance(lp_arg, lp.ArrayArg)
        }
    )
