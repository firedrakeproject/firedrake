"""Per-stage timing of unstructured mesh-topology construction on a BoxMesh.

Builds the mesh once and times each cached-property stage on first access, so
we can see which dmcommon kernel each optimisation targets and its share of the
total. Env: BENCH_N (default 79). Optionally saves reference arrays for
correctness checks (BENCH_SAVE_DIR).
"""
import os, time
import numpy as np
from firedrake import BoxMesh, FunctionSpace

N = int(os.environ.get("BENCH_N", "79"))
sd = os.environ.get("BENCH_SAVE_DIR")

t = time.perf_counter()
mesh = BoxMesh(N, N, N, 1.0, 1.0, 1.0)
topo = mesh.topology
t_build = time.perf_counter() - t
print(f"N={N}  DOFs~={(N+1)**3}")

def stage(label, fn):
    s = time.perf_counter()
    r = fn()
    print(f"  {label:22s}: {time.perf_counter() - s:8.3f} s")
    return r

print(f"  {'BoxMesh(...) ctor':22s}: {t_build:8.3f} s")
cc  = stage("cell_closure",        lambda: topo.cell_closure)
eo  = stage("entity_orientations", lambda: topo.entity_orientations)
ef  = stage("exterior_facets",     lambda: topo.exterior_facets)
iff = stage("interior_facets",     lambda: topo.interior_facets)
V   = stage("FunctionSpace CG1",   lambda: FunctionSpace(mesh, "CG", 1))
nl  = stage("cell_node_list",      lambda: V.cell_node_list)

if sd:
    os.makedirs(sd, exist_ok=True)
    np.save(os.path.join(sd, "cell_closure.npy"), cc)
    np.save(os.path.join(sd, "entity_orientations.npy"), eo)
    np.save(os.path.join(sd, "cell_node_list.npy"), nl)
    print("saved reference arrays ->", sd)

chk = os.environ.get("BENCH_CHECK_DIR")
if chk:
    for name, arr in [("cell_closure", cc), ("entity_orientations", eo), ("cell_node_list", nl)]:
        ref = np.load(os.path.join(chk, name + ".npy"))
        ok = np.array_equal(ref, np.asarray(arr))
        print(f"  CHECK {name:22s}: {'MATCH' if ok else 'MISMATCH'}")
