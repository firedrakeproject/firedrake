"""The `gpu` dialect attribute and operation that pyop3 needs and xDSL does not define.

The shared memory of an `ItemBatch` is the dynamic shared memory of its
`gpu.launch`, one `gpu.dynamic_shared_memory` that `memref.view` divides
between the shared temporaries. Unlike a static `memref.global`, it can be
larger than 48 KB, up to the opt-in limit of the device.
"""

from __future__ import annotations

from xdsl.dialects.builtin import DYNAMIC_INDEX, MemRefType, i8
from xdsl.dialects.utils import EnumAttribute, StrEnum
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, result_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class AddressSpace(StrEnum):
    GLOBAL = "global"
    WORKGROUP = "workgroup"
    PRIVATE = "private"


@irdl_attr_definition
class AddressSpaceAttr(EnumAttribute[AddressSpace]):
    """`#gpu.address_space<...>`, the memory space that MLIR's GPU lowerings map to NVVM and ROCDL"""

    name = "gpu.address_space"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> AddressSpace:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)


WORKGROUP_ADDRESS_SPACE = AddressSpaceAttr(AddressSpace.WORKGROUP)
"""The memory space of shared (workgroup) memory."""


@irdl_op_definition
class DynamicSharedMemoryOp(IRDLOperation):
    """The dynamic shared memory of the enclosing `gpu.launch`, as bytes."""

    name = "gpu.dynamic_shared_memory"

    result = result_def(MemRefType)

    def __init__(self) -> None:
        super().__init__(result_types=[MemRefType(i8, [DYNAMIC_INDEX], memory_space=WORKGROUP_ADDRESS_SPACE)])
