PYOP3_DECIDE = object()
"""Placeholder indicating that a value should be set by pyop3.

This is important in cases where the more traditional `None` is actually
meaningful.

"""


_nothing = object()
"""Sentinel value indicating nothing should be done.

This is useful in cases where `None` holds some meaning.

"""
