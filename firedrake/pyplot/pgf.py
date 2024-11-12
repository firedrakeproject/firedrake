r"""
=======
pgfplot
=======

pgfplots numbering by patch type:
---------------------------------
.. code-block::

   2              2              3-------2        3---6---2
   | \            | \            |       |        |       |
   |   \          5   4          |       |        7   8   5
   |     \        |     \        |       |        |       |
   0------1       0---3--1       0-------1        0---4---1

   triangle    triangle quadr    rectangle       biquadratic

FIAT/FInAT DoF orderings:
-------------------------
UFCTriangle:

.. code-block::

  2              2
  | \            | \
  |   \          4   3
  |     \        |     \
  0------1       0---5--1

    DP1            DP2


UFCTetrahedron:

.. code-block::

   3.             3.        3
   | \            | 4        \    edge 1-3
   |  .2.         7  .2.       5
   |.~   \        |.8   6.       \
   0------1       0---9---1       1

   DP1            DP2


UFCQuadrilateral:

.. code-block::

   1-------3    1---7---4
   |       |    |       |
   |       |    2   8   5
   |       |    |       |
   0-------2    0---6---3

       DQ1          DQ2


UFCHexahedron:

.. code-block::

     3-------7    3-------7            4--22--13      4--22--13
    /.       |   /       /|          7 .       |    7  25  16 |
   1 .       |  1-------5 |        1   5  23  14  1--19--10  14
   | .       |  |       | |        | 8 .       |  |       |17 |
   | 2 . . . 6  |       | 6        2   3 .21 .12  2  20  11  12
   |.       /   |       |/         | 6  24  15    |       |15
   0-------4    0-------4          0--18---9      0--18---9

      DQ1("equispaced")               DQ2("equispaced")

"""
import os
import numpy as np


from firedrake import Function, FunctionSpace, SpatialCoordinate
from firedrake.embedding import get_embedding_dg_element, get_embedding_method_for_checkpointing
from FIAT.reference_element import UFCTriangle, UFCTetrahedron, UFCQuadrilateral, UFCHexahedron
from firedrake.petsc import PETSc


def _pgfplot_make_perms(cell, degree):
    # make DoF permutations: pgfplot DP/DQ DoFs -> UFC DoFs.
    if isinstance(cell, UFCTriangle):
        if degree == 1:
            return np.array([[0, 1, 2]]), "triangle"
        elif degree == 2:
            return np.array([[0, 1, 2, 5, 3, 4]]), "triangle quadr"
        else:
            raise NotImplementedError(f"Not implemented for degree {degree} on {cell}")
    elif isinstance(cell, UFCTetrahedron):
        if degree == 1:
            return np.array([[1, 2, 3],
                             [0, 2, 3],
                             [0, 1, 3],
                             [0, 1, 2]]), "triangle"
        elif degree == 2:
            return np.array([[1, 2, 3, 6, 4, 5],
                             [0, 2, 3, 8, 4, 7],
                             [0, 1, 3, 9, 5, 7],
                             [0, 1, 2, 9, 6, 8]]), "triangle quadr"
        else:
            raise NotImplementedError(f"Not implemented for degree {degree} on {cell}")
    elif isinstance(cell, UFCQuadrilateral):
        if degree == 1:
            return np.array([[0, 2, 3, 1]]), "rectangle"
        elif degree == 2:
            return np.array([[0, 3, 4, 1, 6, 5, 7, 2, 8]]), "biquadratic"
        else:
            raise NotImplementedError(f"Not implemented for degree {degree} on {cell}")
    elif isinstance(cell, UFCHexahedron):
        if degree == 1:
            return np.array([[0, 2, 3, 1],
                             [4, 6, 7, 5],
                             [0, 4, 5, 1],
                             [2, 6, 7, 3],
                             [0, 4, 6, 2],
                             [1, 5, 7, 3]]), "rectangle"
        elif degree == 2:
            return np.array([[0, 3, 4, 1, 6, 5, 7, 2, 8],
                             [9, 12, 13, 10, 15, 14, 16, 11, 17],
                             [0, 9, 10, 1, 18, 11, 19, 2, 20],
                             [3, 12, 13, 4, 21, 14, 22, 5, 23],
                             [0, 9, 12, 3, 18, 15, 21, 6, 24],
                             [1, 10, 13, 4, 19, 16, 22, 7, 25]]), "biquadratic"
        else:
            raise NotImplementedError(f"Not implemented for degree {degree} on {cell}")
    else:
        raise NotImplementedError(f"Not implemented for cell {cell}")


def _pgfplot_create_patch_arrays(data, cell_node_list, cells, perms):
    a = cell_node_list[cells]
    offsets = np.tile((np.arange(a.shape[0]) * a.shape[1]).reshape(-1, 1), perms.shape[1])
    return data[a.reshape(-1)[offsets + perms].reshape(-1), :]


def _pgfplot_create_patches(f, coords, complex_component):
    V = f.function_space()
    elem = V.ufl_element()
    fiat_cell = V.finat_element.cell
    degree = elem.degree()
    coordV = coords.function_space()
    mesh = V.ufl_domain()
    cdata = coords.dat.data_ro.real
    fdata = f.dat.data_ro.real if complex_component == 'real' else f.dat.data_ro.imag
    map_facet_dofs, patch_type = _pgfplot_make_perms(fiat_cell, degree)
    if isinstance(fiat_cell, UFCTriangle):
        cells = np.arange(mesh.cell_set.size)
        perms = map_facet_dofs
    elif isinstance(fiat_cell, UFCTetrahedron):
        _facets = mesh.exterior_facets
        nfacets = _facets.classes[1]
        cells = _facets.facet_cell[:nfacets, 0]
        perms = map_facet_dofs[_facets.local_facet_dat.data_ro[:nfacets]]
    elif isinstance(fiat_cell, UFCQuadrilateral):
        cells = np.arange(mesh.cell_set.size)
        perms = map_facet_dofs
    elif isinstance(fiat_cell, UFCHexahedron):
        _facets = mesh.exterior_facets
        nfacets = _facets.classes[1]
        cells = _facets.facet_cell[:nfacets, 0]
        perms = map_facet_dofs[_facets.local_facet_dat.data_ro[:nfacets]]
    else:
        raise NotImplementedError(f"Got unsupported FIAT cell: {fiat_cell}")
    patches_c = _pgfplot_create_patch_arrays(cdata, coordV.cell_node_list, cells, perms)
    patches_f = _pgfplot_create_patch_arrays(fdata.reshape(-1, 1), V.cell_node_list, cells, perms)
    patches = np.concatenate([patches_c, patches_f], axis=1)
    return patches, patch_type


def pgfplot(f, filename, degree=1, complex_component='real', print_latex_example=True):
    """Produce a data file for LaTeX tikz plotting in parallel.

    Parameters
    ----------
    f : Function
       `Function` to plot.
    filename : str
        Name of the output file.
    degree : int
        Degree of interpolation for plotting: ``1`` (linear) or ``2`` (quadratic).
    complex_component : str
        Complex component to be plotted: ``"real"`` or ``"imag"``.
    print_latex_example : bool
        Flag indicating whether to print a latex example or not.

    Notes
    -----
    Currently this functionality is only for plotting scalar functions in two- or
    three-dimensional spaces using 2D patches. If the topological dimension of the
    function is two, it outputs values on the cells, while, if the topological
    dimension is three, it outputs values on the exterior facets.

    Do not use this for large functions, or it will take forever to
    compile your LaTeX file.

    For large functions, ``pdflatex`` might fail to compile your document with the
    error message: ``TeX capacity exceeded, sorry [main memory size=5000000].``
    If this happens, you could consider handling this error directly one way or
    another or consider using ``lualatex`` instead, which allocates memory dynamically.

    This function seamlessly works in parallel.

    """
    if degree not in (1, 2):
        raise NotImplementedError(f"degree must be {1, 2}: got {degree}")
    if complex_component not in ('real', 'imag'):
        raise NotImplementedError(f"complex_component must be {'real', 'imag'}: got {complex_component}")
    V = f.function_space()
    elem = V.ufl_element()
    mesh = V.ufl_domain()
    dim = mesh.geometric_dimension()
    if dim not in (2, 3):
        raise NotImplementedError(f"Not yet implemented for functions in spatial dimension {dim}")
    if mesh.extruded:
        raise NotImplementedError("Not yet implemented for functions on extruded meshes")
    if V.value_shape:
        raise NotImplementedError("Currently only implemeted for scalar functions")
    coordelem = get_embedding_dg_element(mesh.coordinates.function_space().ufl_element(), (dim, )).reconstruct(degree=degree, variant="equispaced")
    coordV = FunctionSpace(mesh, coordelem)
    coords = Function(coordV).interpolate(SpatialCoordinate(mesh))
    elemdg = get_embedding_dg_element(elem, V.value_shape).reconstruct(degree=degree, variant="equispaced")
    Vdg = FunctionSpace(mesh, elemdg)
    fdg = Function(Vdg)
    method = get_embedding_method_for_checkpointing(elem)
    getattr(fdg, method)(f)
    patches, patch_type = _pgfplot_create_patches(fdg, coords, complex_component)
    # Output
    size = f.comm.size
    rank = f.comm.rank
    filename_rank = filename + f"_{rank}"
    if os.path.exists(filename_rank):
        raise RuntimeError(f"File already exists: {filename_rank}")
    np.savetxt(filename_rank, patches)
    f.comm.Barrier()
    if rank == 0:
        coordname = {1: 'x ',
                     2: 'x y ',
                     3: 'x y z '}[dim]
        with open(filename, 'w') as outfile:
            outfile.write(coordname + f.name() + "\n")
            for rnk in range(size):
                with open(filename + f"_{rnk}", 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    f.comm.Barrier()
    os.remove(filename_rank)
    if print_latex_example:
        with coords.dat.vec_ro as vec:
            arg_coordslim = ""
            for d in range(dim):
                _, cmax = vec.strideMax(d)
                _, cmin = vec.strideMin(d)
                c = ['x', 'y', 'z'][d]
                arg_coordslim += f"""             {c}min={cmin: .2f}, {c}max={cmax: .2f},\n"""
        table_arg = {1: 'x=x, ',
                     2: 'x=x, y=y, ',
                     3: 'x=x, y=y, z=z, '}[dim]
        table_arg += f'meta={f.name()}'
        fname_arg = '{' + filename + '}'
        texts = f"""
===========================================================================================
% pgfplot_example.tex

\\documentclass{{article}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\usepgfplotslibrary{{patchplots}}
\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
\\begin{{figure}}[ht]
\\begin{{tikzpicture}}
\\begin{{axis}}[title={f.name().replace("_", " ")},
{arg_coordslim[:-1]}
             xlabel={{$x$}},
             ylabel={{$y$}},
             zlabel={{$z$}},
             % xtick={{0, 1, 2}},
             % xticklabels={{0, 1, 2}},
             % ytick={{0, 1}},
             % yticklabels={{0, 1}},
             % ztick={{0, 1}},
             % zticklabels={{0, 1}},
             % axis equal,
             axis equal image,
             colorbar,
             colormap/hot, % hot, cool, bluered, greenyellow, redyellow, violet, blackwhite
             colorbar/width=10pt,
             view={{30}}{{45}},
             width=300pt,
             height=300pt,
             % axis line style={{draw=none}},
             % tick style={{draw=none}},
            ]
\\addplot3[patch,
          patch type={patch_type},
          point meta=explicit,
          shader=faceted interp, % interp, faceted, faceted interp
          opacity=1.,
         ] table[{table_arg}] {fname_arg};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{figure}}
\\end{{document}}
===========================================================================================

Run:

lualatex -shell-escape pgfplot_example.tex

For more details, see:

https://anorien.csc.warwick.ac.uk/mirrors/CTAN/graphics/pgf/contrib/pgfplots/doc/pgfplots.pdf
https://pgfplots.sourceforge.net/gallery.html
https://github.com/pgf-tikz/pgfplots
"""
        PETSc.Sys.Print(texts)
