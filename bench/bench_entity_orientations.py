"""Micro-benchmark for firedrake.cython.dmcommon.entity_orientations.

Times the kernel in isolation on a BoxMesh (unstructured tets), so we can
measure the prange/nogil optimisation against the serial baseline and verify
the result is bit-identical.

Env vars:
  BENCH_N        BoxMesh resolution (default 79 -> (N+1)^3 DOFs, ~2.4M tets)
  BENCH_REPS     repeats for a stable timing (default 5)
  OMP_NUM_THREADS threads for the parallel path
  BENCH_SAVE     path to np.save the orientation array (for correctness ref)
  BENCH_CHECK    path to np.load a reference array and assert equality
"""
import os, time
import numpy as np
from firedrake import BoxMesh
from firedrake.cython import dmcommon

N = int(os.environ.get("BENCH_N", "79"))
REPS = int(os.environ.get("BENCH_REPS", "5"))

mesh = BoxMesh(N, N, N, 1.0, 1.0, 1.0)
topo = mesh.topology
cc = topo.cell_closure                      # compute the closure once (not timed)

print(f"N={N}  cells={cc.shape[0]}  closure_width={cc.shape[1]}  DOFs~={(N+1)**3}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")

times, res = [], None
for r in range(REPS):
    t = time.perf_counter()
    res = dmcommon.entity_orientations(topo, cc)
    dt = time.perf_counter() - t
    times.append(dt)
    print(f"  run {r}: {dt:.3f} s")
times.sort()
print(f"RESULT  min={times[0]:.3f}s  median={times[len(times)//2]:.3f}s")

if os.environ.get("BENCH_SAVE"):
    np.save(os.environ["BENCH_SAVE"], res)
    print("saved reference ->", os.environ["BENCH_SAVE"])
if os.environ.get("BENCH_CHECK"):
    ref = np.load(os.environ["BENCH_CHECK"])
    ok = np.array_equal(ref, res)
    print(f"CORRECTNESS vs {os.environ['BENCH_CHECK']}: {'MATCH' if ok else 'MISMATCH (%d diffs)' % int((ref != res).sum())}")
