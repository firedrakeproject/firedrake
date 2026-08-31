from pyop3.lower.context import CodegenContext

class MLIRCodegenContext(CodegenContext):
    """ 
    Class produces MLIR kernels from PyOP3 buffers, using xDSL. 
    """

    def __init__(self, *, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        super().__init__(
            propagate_negatives=propagate_negatives,
            mask_array_accesses=mask_array_accesses
        )
    
    @abstractmethod
    def var(self, iname: str, *args) -> str:
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
    def add_leaf_assignment(
            self, 
            assignee,
            expression,
            assignment_type,
            paths, 
            iname_maps, 
            loop_indices
        ):
        pass

    @abstractmethod
    def register_extent(self, obj: Any, inames, loop_indices):
        pass
