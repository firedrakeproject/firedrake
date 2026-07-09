"""Profile mesh-topology construction (cProfile per-kernel) and verify
correctness: cell arrays vs a saved reference, an interior-facet invariant
(jump of a CG field across dS must be ~0), and an end-to-end solve Tmax.

Env: BENCH_N (default 79), BENCH_CHECK_DIR (ref arrays dir).
"""
import os, io, cProfile, pstats
import numpy as np
from firedrake import (BoxMesh, FunctionSpace, Function, TrialFunction,
                       TestFunction, dot, grad, dx, dS, jump, Constant,
                       DirichletBC, solve, SpatialCoordinate, sin, assemble)

N = int(os.environ.get("BENCH_N", "79"))

pr = cProfile.Profile(); pr.enable()
mesh = BoxMesh(N, N, N, 1.0, 1.0, 1.0)
V = FunctionSpace(mesh, "CG", 1)
nl = V.cell_node_list
pr.disable()

s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
print(f"--- per-kernel tottime (N={N}) ---")
for line in s.getvalue().splitlines():
    if any(k in line for k in ["_facets", "cell_closure", "make_cell_node_list",
                               "entity_orientations", "_renumber_entities",
                               "plex_from_cell_list", "seconds"]):
        print(line.strip())

print("--- correctness ---")
chk = os.environ.get("BENCH_CHECK_DIR")
if chk:
    for name, arr in [("cell_closure", mesh.topology.cell_closure),
                      ("entity_orientations", mesh.topology.entity_orientations),
                      ("cell_node_list", nl)]:
        ref = np.load(os.path.join(chk, name + ".npy"))
        print(f"  {name:20s}: {'MATCH' if np.array_equal(ref, np.asarray(arr)) else 'MISMATCH'}")

# interior-facet invariant: jump of a continuous (CG) field across dS is 0
x, y, z = SpatialCoordinate(mesh)
u = Function(V).interpolate(sin(3*x) + 2*y*z)
jval = assemble(jump(u) * dS)
print(f"  interior-facet jump(CG)*dS = {jval:.3e}  (should be ~0)")

# exterior-facet path: DirichletBC solve, check Tmax
w = Function(V)
uu, vv = TrialFunction(V), TestFunction(V)
solve(dot(grad(uu), grad(vv))*dx == Constant(100.0)*vv*dx, w,
      bcs=DirichletBC(V, Constant(25.0), 6),
      solver_parameters={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-6})
print(f"  solve Tmax = {float(w.dat.data_ro.max()):.4f}  (baseline 75.0087)")
