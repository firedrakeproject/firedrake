from tsfc.exceptions import MismatchingDomainError  # noqa: F401


class ConvergenceError(Exception):
    """Error raised when a solver fails to converge"""
