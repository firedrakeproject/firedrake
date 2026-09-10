import warnings


warnings.warn(
    "PyOP2 has been replaced by pyop3. The majority of the API is different. "
    "Only 'pyop2.compilation' and 'pyop2.mpi' should continue to work as "
    "expected until the next release of Firedrake when they will be removed.",
    FutureWarning,
)
