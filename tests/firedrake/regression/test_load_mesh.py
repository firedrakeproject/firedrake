from math import pi

from firedrake import *
from petsctools import OptionsManager, get_external_packages
import numpy as np
import pytest
from os.path import abspath, dirname, join
from pathlib import Path
from tempfile import TemporaryDirectory


path = join(abspath(dirname(__file__)), '..', 'meshes')


def load_mesh(filename, use_path_object):
    if use_path_object:
        return Mesh(Path(path) / filename)

    return Mesh(join(path, filename))


@pytest.mark.parametrize(
    'filename', [
        'square.msh',
        'square_binary.msh',
    ])
@pytest.mark.parametrize('use_path_object', [False, True])
def test_load_mesh(filename, use_path_object):
    m = load_mesh(filename, use_path_object)
    v = assemble(1*dx(domain=m))
    assert np.allclose(v, 1)


# --- Periodic Gmsh meshes ---


@pytest.fixture(params=["p2d", "p2d_xy"])
def periodic_2d_mesh(request):
    return Mesh(join(path, f"{request.param}.msh"))


def test_periodic_2d_coordinates(periodic_2d_mesh):
    """Mesh uses a DG coordinate element after loading."""
    elem = periodic_2d_mesh.ufl_coordinate_element()
    assert elem.family() == "Discontinuous Lagrange"


@pytest.mark.parallel([1, 2])
def test_periodic_helmholtz_2d_x():
    """Helmholtz on x-periodic rectangle [0,0.6]x[0,0.5].

    Manufactured solution u_exact = cos(2*pi*x/0.6) * y*(0.5 - y).
    Periodic in x with non-constant boundary data, zero on y boundaries.
    """
    mesh = Mesh(join(path, "p2d.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    Lx = 0.6
    u_exact_expr = cos(2 * pi * x[0] / Lx) * x[1] * (0.5 - x[1])

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = a(v, u_exact_expr)

    uh = Function(V)
    bc = DirichletBC(V, Constant(0), [1, 3])
    solve(a == L, uh, bcs=bc, solver_parameters={"ksp_type": "cg"})

    assert errornorm(u_exact_expr, uh, "L2") < 0.005


@pytest.mark.parallel([1, 2])
def test_periodic_2d_xy_solve():
    """Helmholtz on doubly-periodic rectangle [0,0.6]x[0,0.5].

    Trigonometric manufactured solution
    u_exact = cos(2*pi*x/0.6) * cos(2*pi*y/0.5), periodic in both
    x and y with non-constant boundary data.  No boundary conditions
    needed.

    Uses a wider tolerance than the other tests because the
    trigonometric solution requires fine resolution per wavelength.
    """
    mesh = Mesh(join(path, "p2d_xy.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    Lx, Ly = 0.6, 0.5
    u_exact_expr = cos(2 * pi * x[0] / Lx) * cos(2 * pi * x[1] / Ly)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = a(v, u_exact_expr)

    uh = Function(V)
    solve(a == L, uh)

    assert errornorm(u_exact_expr, uh, "L2") / norm(u_exact_expr, "L2") < 0.15


@pytest.mark.parallel([1, 2])
def test_periodic_3d_solve():
    """Helmholtz on x-periodic box [0,1]^3.

    Manufactured polynomial solution
    u_exact = 42 + y*(1-y)*z*(1-z), periodic in x (constant in x),
    zero on y/z boundaries.  CG4 reproduces the polynomial exactly.
    """
    mesh = Mesh(join(path, "p3d.msh"))
    V = FunctionSpace(mesh, "CG", 4)
    x = SpatialCoordinate(mesh)

    u_exact_expr = 42 + x[1] * (1 - x[1]) * x[2] * (1 - x[2])

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = a(v, u_exact_expr)

    uh = Function(V)
    bc = DirichletBC(V, u_exact_expr, [3, 4, 5, 6])
    solve(a == L, uh, bcs=bc, solver_parameters={"ksp_type": "cg"})

    assert errornorm(u_exact_expr, uh, "L2") < 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("use_path_object", [False, True])
def test_gmsh_region_names_are_loaded(use_path_object: bool) -> None:
    with TemporaryDirectory() as directory:
        filename = None
        if COMM_WORLD.rank == 0:
            source = (Path(path) / "square.msh").read_text()
            physical_names = (
                "$PhysicalNames\n3\n"
                "1 1 \"left wall\"\n"
                "1 2 \"firedrake_is_ghost\"\n"
                "2 1 \"left wall\"\n"
                "$EndPhysicalNames\n"
            )
            filename = Path(directory) / "named_square.msh"
            filename.write_text(source.replace("$EndMeshFormat\n", "$EndMeshFormat\n" + physical_names))
        filename = COMM_WORLD.bcast(filename, root=0)
        mesh = Mesh(filename if use_path_object else str(filename))

    assert mesh.topology.region_names == {
        1: {"left wall": [1], "firedrake_is_ghost": [2]},
        2: {"left wall": [1]},
    }
    for marker in ("left wall", "firedrake_is_ghost", 1, 2):
        assert assemble(Constant(1) * ds(marker, domain=mesh)) == pytest.approx(1.0)
    for marker in ("left wall", 1):
        assert assemble(Constant(1) * dx(marker, domain=mesh)) == pytest.approx(1.0)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("parallel_reader", [False, True])
def test_cgns_region_names(parallel_reader: bool) -> None:
    if "cgns" not in get_external_packages():
        pytest.skip("PETSc was built without CGNS support")

    filename = "named_rectangle.cgns" if parallel_reader else "rectangle.cgns"
    options = {"dm_plex_cgns_parallel": parallel_reader}
    with OptionsManager(options, "").inserted_options():
        mesh = Mesh(Path(path) / filename)

    if parallel_reader:
        assert mesh.topology.region_names == {1: {"inlet": [2], "wall": [3]}}
        boundary = ds(domain=mesh)
        for marker, length in (("inlet", 1.0), ("wall", 2.0), (1, 1.0), (2, 1.0), (3, 2.0)):
            assert assemble(1 * boundary(marker)) == pytest.approx(length)
    else:
        assert not mesh.topology.region_names
        assert mesh.topology_dm.hasLabel("zone")

    assert assemble(1 * dx(domain=mesh)) == pytest.approx(2.0)
