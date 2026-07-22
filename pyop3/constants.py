import enum


# TODO: rename to just 'DECIDE'
PYOP3_DECIDE = object()
"""Placeholder indicating that a value should be set by pyop3.

This is important in cases where the more traditional `None` is actually
meaningful.

"""


_nothing = object()
"""Sentinel value indicating nothing should be done.

This is useful in cases where `None` holds some meaning.

"""


class Intent(enum.Enum):
    # developer note, MIN_RW and MIN_WRITE are distinct (unlike PyOP2) to avoid
    # passing "requires_zeroed_output_arguments" around, yuck
    READ = "read"
    WRITE = "write"
    RW = "rw"
    INC = "inc"
    MIN_WRITE = "min_write"
    MIN_RW = "min_rw"
    MAX_WRITE = "max_write"
    MAX_RW = "max_rw"


READ = Intent.READ
WRITE = Intent.WRITE
RW = Intent.RW
INC = Intent.INC
MIN_RW = Intent.MIN_RW
MIN_WRITE = Intent.MIN_WRITE
MAX_RW = Intent.MAX_RW
MAX_WRITE = Intent.MAX_WRITE
