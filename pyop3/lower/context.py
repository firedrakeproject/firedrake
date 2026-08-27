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
    Base class for code generation backends
    
    Class designed solely for use in codegen.py, as an interface to specific backends.
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
        """
        Determine indexing and lower buffer expression to respective IR
        """
        pass

    @abstractmethod
    def add_leaf_assignment(self, assignment, paths, iname_maps, loop_indices):
        pass

    @abstractmethod
    def register_extent(self, obj: Any, inames, loop_indices):
        pass

    @abstractmethod
    def compile_standalone_function(self, call, loop_indices):
        pass

    @abstractmethod
    def compile_petsc_mat(self, assignment, loop_indices):
        pass

    @abstractmethod
    def compile_exscan(self, call, loop_indices):
        pass

    # }}}


    # {{{ class methods


    def _compile_array_assignment(
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
                    self._compile_array_assignment(
                        assignment, 
                        loop_indices, 
                        axis_trees, 
                        iname_replace_maps=new_maps, 
                        axis_tree=axis_tree, 
                        paths=new_paths
                    )
                elif axis_trees:
                    self._compile_array_assignment(
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

    def _parse_loop_properly_this_time(
            self,
            loop,
            axis_tree,
            loop_indices,
            *,
            axis=None,
            path=None,
            iname_map=None,
    ) -> None:
        from pyop3.lower.codegen import _compile

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
                    self._parse_loop_properly_this_time(
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
        '''
            Display key properties of CodegenContext
        '''
        ctx = f"Domain: {str(self.domains)}\n\n"
        ctx += f"Instructions: {str(self.instructions)}\n\n"
        ctx += f"Arguments: {str(self.arguments)}\n\n"
        ctx += f"Subkernels: {str(self.subkernels)}\n\n"
        return ctx 

