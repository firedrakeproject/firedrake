
__version_info__ = (0, 12, 0)
__version__ = '.'.join(map(str, __version_info__))


def check():
    from pyop2.version import __version_info__ as pyop2_version_info
    from pyop2.version import __version__ as pyop2_version
    if pyop2_version_info[:2] != __version_info__[:2]:
        raise RuntimeError(
            """Firedrake version %s and PyOP2 version %s are incompatible.""" %
            (__version__, pyop2_version))
