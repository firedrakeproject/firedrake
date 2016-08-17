""" Generates Mesh and FunctionSpace hierarchies that can have different
refinement factors of multiples of 2 """

from __future__ import absolute_import

from firedrake import *  # noqa
from firedrake.mg import *  # noqa
from firedrake.mg.utils import *  # noqa


class GeneralisedMeshHierarchy(object):

    """ Builds a MeshHierarchy with a user defined refinement factor (multiple of 2)

        :param mesh: The coarsest mesh
        :type mesh: :class:`Mesh'

        :param levels: number of levels (multiple of M)
        :type levels: int

        :param M: Refinement factor (multiple of 2)
        :type M: int

    """

    def __init__(self, mesh, levels, M=2):

        # refinement factor
        self.M = M

        # check that M is a multiple of 2
        if (M % 2) != 0:
            raise ValueError('Refinement factor is not a multiple of 2')

        # 'skip parameter'
        self.skip = (M / 2)

        # carry the full mesh hierarchy
        self._full_hierarchy = MeshHierarchy(mesh, levels * self.skip)

        # carry the generalised mesh hierarchy
        self._hierarchy = self._full_hierarchy[::self.skip]

        super(GeneralisedMeshHierarchy, self).__init__()

    def __iter__(self):
        """ Iterate over the hierarchy of meshes from coarsest to finest """
        for m in self._hierarchy:
            yield m

    def __len__(self):
        """ Return the size of hierarchy """
        return len(self._hierarchy)

    def __getitem__(self, idx):
        """ Return a mesh in the hierarchy

            :arg idx: The :func:`~.Mesh` to return

        """
        return self._hierarchy[idx]


class GeneralisedFunctionSpaceHierarchy(object):

    """ Builds a FunctionSpaceHierarchy from a GeneralisedMeshHierarchy

        :param generalisedmeshhierarchy: The GeneralisedMeshHierarchy
        :type generalisedmeshhierarchy: :class:`GeneralisedMeshHierarchy'

        :param family: 'DG' or 'CG'
        :type L: str

        :param degree: degree
        :type degree: int

    """

    def __init__(self, generalised_mesh_hierarchy, family, degree,
                 name=None, vfamily=None, vdegree=None):

        if hasattr(generalised_mesh_hierarchy, '_full_hierarchy') == 0:
            raise AttributeError('Cant build a generalised function space hierarchy ' +
                                 'from a standard mesh hierarchy')

        self._full_hierarchy = FunctionSpaceHierarchy(generalised_mesh_hierarchy._full_hierarchy,
                                                      family, degree, name=name,
                                                      vfamily=vfamily, vdegree=vdegree)

        self._hierarchy = FunctionSpaceHierarchy(generalised_mesh_hierarchy._hierarchy,
                                                 family, degree, name=name,
                                                 vfamily=vfamily, vdegree=vdegree)

        self.dim = 1

        self.M = generalised_mesh_hierarchy.M

        self.skip = generalised_mesh_hierarchy.skip

        # pull along the full hierarchy and the skip parameter to the generalised hierarchy
        setattr(self._hierarchy, '_full_hierarchy', self._full_hierarchy)
        setattr(self._hierarchy, '_skip', self.skip)

        super(GeneralisedFunctionSpaceHierarchy, self).__init__()

    def __len__(self):
        """Return the size of this generalised function space hierarchy"""
        return len(self._hierarchy)

    def __iter__(self):
        """Iterate over the function spaces in this generalised hierarchy (from
        coarse to fine)."""
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        """Return a function space in the generalised hierarchy

        :arg idx: The :class:`~.FunctionSpace` to return"""
        return self._hierarchy[idx]


class GeneralisedMixedFunctionSpaceHierarchy(object):

    """ Builds a MixedFunctionSpaceHierarchy from a GeneralisedMeshHierarchy

        :param generalised_fs_hierarchies: A list of generalised function spaces
        :type generalised_fs_hierarchies: :class:`GeneralisedFunctionSpaceHierarchy`\s

    """

    def __init__(self, generalised_fs_hierarchies):

        if isinstance(generalised_fs_hierarchies, list) == 0:
            raise TypeError('Function Space Hierarchies need to be inside a list [spaces]')

        if hasattr(generalised_fs_hierarchies[0], '_full_hierarchy') == 0:
            raise AttributeError('Cant build a generalised mixed function space hierarchy ' +
                                 'from standard function space hierarchies')

        list_of_full_hierarchies = []
        for i in range(len(generalised_fs_hierarchies)):
            list_of_full_hierarchies.append(generalised_fs_hierarchies[i]._full_hierarchy)

        self._full_hierarchy = MixedFunctionSpaceHierarchy(list_of_full_hierarchies)

        list_of_hierarchies = []
        for i in range(len(generalised_fs_hierarchies)):
            list_of_hierarchies.append(generalised_fs_hierarchies[i]._hierarchy)

        self._hierarchy = MixedFunctionSpaceHierarchy(list_of_hierarchies)

        self.M = generalised_fs_hierarchies[0].M

        self.skip = generalised_fs_hierarchies[0].skip

        # pull along the full hierarchy and the skip parameter to the generalised hierarchy
        setattr(self._hierarchy, '_full_hierarchy', self._full_hierarchy)
        setattr(self._hierarchy, '_skip', self.skip)

        super(GeneralisedMixedFunctionSpaceHierarchy, self).__init__()

    def __len__(self):
        """Return the size of this generalised mixed function space hierarchy"""
        return len(self._hierarchy)

    def __iter__(self):
        """Iterate over the mixed function spaces in this generalised hierarchy (from
        coarse to fine)."""
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        """Return a mixed function space in the generalised hierarchy

        :arg idx: The :class:`~.MixedFunctionSpace` to return"""
        return self._hierarchy[idx]


class GeneralisedVectorFunctionSpaceHierarchy(object):

    """ Builds a VectorFunctionSpaceHierarchy from a GeneralisedMeshHierarchy

        :param generalisedmeshhierarchy: The GeneralisedMeshHierarchy
        :type generalisedmeshhierarchy: :class:`GeneralisedMeshHierarchy'

        :param family: 'DG' or 'CG'
        :type L: str

        :param degree: degree
        :type degree: int

    """

    def __init__(self, generalised_mesh_hierarchy, family, degree,
                 dim=None, name=None, vfamily=None, vdegree=None):

        if hasattr(generalised_mesh_hierarchy, '_full_hierarchy') == 0:
            raise AttributeError('Cant build a generalised vector function space hierarchy ' +
                                 'from a standard mesh hierarchy')

        self._full_hierarchy = (VectorFunctionSpaceHierarchy(generalised_mesh_hierarchy._full_hierarchy,
                                                             family, degree, dim=dim, name=name,
                                                             vfamily=vfamily, vdegree=vdegree))

        self._hierarchy = (VectorFunctionSpaceHierarchy(generalised_mesh_hierarchy._hierarchy,
                                                        family, degree, dim=dim, name=name,
                                                        vfamily=vfamily, vdegree=vdegree))

        self.dim = self._full_hierarchy.dim

        self.M = generalised_mesh_hierarchy.M

        self.skip = generalised_mesh_hierarchy.skip

        # pull along the full hierarchy and the skip parameter to the generalised hierarchy
        setattr(self._hierarchy, '_full_hierarchy', self._full_hierarchy)
        setattr(self._hierarchy, '_skip', self.skip)

        super(GeneralisedVectorFunctionSpaceHierarchy, self).__init__()

    def __len__(self):
        """Return the size of this generalised vector function space hierarchy"""
        return len(self._hierarchy)

    def __iter__(self):
        """Iterate over the vector function spaces in this generalised hierarchy (from
        coarse to fine)."""
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        """Return a vector function space in the generalised hierarchy

        :arg idx: The :class:`~.VectorFunctionSpace` to return"""
        return self._hierarchy[idx]
