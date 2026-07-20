# Sum factorization on simplices: status and milestone-2 design

Companion to `PLAN.md`. Everything in "Implemented" is validated by tests.

## Implemented (milestone 1 + jagged-loop infrastructure)

### FIAT (`FIAT/expansions.py`)

* `dubiner_jacobi_parameters(codim, m, variant)` and `dubiner_norm2(d, m, i, variant)`:
  shared helpers extracted from `dubiner_recurrence` (behavior-preserving).
* `principal_functions(n, eta, axis, order, variant)`: 1D tabulations of the
  Karniadakis–Sherwin principal functions `G[m, i](eta) = norm * w(eta)^m * g_{m,i}(eta)`
  with `w = (1 - eta)/2`, as tables `'V'` (values), `'D'` (d/d eta), `'W'`
  (`w^{m-1} g norm`, zeroed at `m = 0`), `'tD'` (`(1+eta)/2 * D`).
* `ExpansionSet.tabulate_duffy(n, eta_pts, order, cell)`: the separable
  tabulation on collapsed points. On the reference simplex, the Dubiner basis
  factorizes exactly as
  `phi_{(i_1..i_d)}(eta) = scale * prod_t G_t[m_t, i_t](eta_t)` with
  `m_t = i_1 + ... + i_{t-1}`. First derivatives use the closed-form chain rule
  `d eta_t / d xi_k = ((1+eta_t)/2)^{[k>t]} * prod_{u>t} 1/w_u` (k >= t), which
  yields exactly `k` separable terms for `d phi / d xi_k`; the affine cell map
  is applied on top. Raises `NotImplementedError` for `order > 1` and for C0
  (`continuity is not None`) expansion sets.

### finat (`finat/point_set.py`, `finat/spectral.py`)

* `CollapsedTensorProductPointSet`: 1D factor point sets in collapsed
  coordinates on `[0, 1]`, mapped to the simplex by the Duffy map
  `x_t = eta_t * prod_{u>t} (1 - eta_u)`.
* `Legendre.duffy_evaluation(order, ps, entity=None)`: returns
  `(multiindex, result)` where `multiindex` enumerates the basis by lattice
  indices `(i_1, ..., i_d)` (a tuple of `gem.JaggedIndex`, see below) and
  `result[alpha]` is a scalar gem expression built from `gem.Literal` 1D tables
  contracted per axis. The weight exponents `m_t` are gem expressions: `0`,
  `i_1`, then `VariableIndex` lookups into clamped uint index tables.

**Zero-padding invariant** (load-bearing for everything below): lattice indices
range over the rectangular box `(degree+1)^d`; all literal index tables are
clamped with `numpy.minimum`, and out-of-lattice entries (`sum(i_t) > degree`)
tabulate to exactly zero. Memory safety and correctness never depend on jagged
loop bounds — jaggedness is purely a flop optimization.

### gem (`gem/gem.py`, `gem/interpreter.py`)

* `JaggedIndex(Index)`: free index with a `parents` tuple; iteration bound
  `0 <= i < extent - sum(parents)`. `.extent` remains the static rectangular
  bound, so every consumer that ignores jaggedness stays correct (via the
  zero-padding invariant). Picklable; exported in `__all__`.
* `gem.interpreter._evaluate_indexed`: repaired the bit-rotted `VariableIndex`
  path, including a gather path for index expressions with free indices.

### tsfc (`tsfc/loopy.py`)

* `LoopyContext.index_parents`: iname -> parent inames, recorded in
  `statement_for` when an `imp.For` loop index is a `JaggedIndex` and all its
  parents are active (i.e. the loop is nested inside them); otherwise the
  rectangular bound is kept (correct by the invariant).
* `create_domains(indices, index_parents=None)`: emits parametrized ISL sets
  `[p] -> {[q] : 0 <= q < E - p}` for jagged inames. Loopy nests these domains
  automatically; no loopy changes were needed (verified experimentally first).
* `ComponentTensor` materialization stays rectangular on purpose: temporaries
  are always fully initialized, so a jagged write can never leave garbage that
  a rectangular read later observes.

### Tests

* `test/FIAT/unit/test_polynomial.py`: `test_tabulate_duffy` (values +
  gradients vs `_tabulate_on_cell`, dims 1–3, variants None/dual, degrees
  0/1/4, points including the collapsed vertex), `test_principal_functions_bubble`,
  `test_morton_tables` (`morton_forward_table`/`morton_inverse_table` agree
  with `morton_index` and are mutual inverses on the simplex lattice).
* `test/finat/test_point_evaluation.py`: `test_duffy_evaluation` vs dense
  `basis_evaluation` through the gem interpreter, checking Morton flat-index
  agreement and exact zeros outside the lattice.
* `tests/tsfc/test_codegen.py::test_jagged_index_codegen`: compiles the 2D
  jagged Morton-gather contraction through `compile_gem` -> `tsfc.loopy.generate`,
  executes the C kernel, checks the parametrized ISL domain exists and the
  numbers match numpy.
* `tests/tsfc/test_pickle_gem.py::test_pickle_jagged_index`.

## Milestone 2: O(p^d)-per-dof matrix-free DG residual — implemented (Route B)

A DG residual action has three phases per cell:

1. **Forward transform:** evaluate `u` (and `grad u`) at quadrature points from
   coefficients `c`. Sum-factorized on collapsed points this is a sequence of d
   per-axis contractions with jagged intermediate temporaries, e.g. in 3D
   `T1[p, q, k] = sum_r C[p, q, r] * G3[p+q, r, k]`,
   `T2[p, j, k] = sum_q T1[p, q, k] * G2[p, q, j]`,
   `u[i, j, k] = sum_p T2[p, j, k] * G1[p, i]` — O(p^{d+1}) total.
2. **Pointwise:** multiply by quadrature weights, geometry, coefficients.
3. **Backward transform:** contract against the test function tables (the
   transpose sweep), scattering into the residual vector.

### (a) Collapsed quadrature rule — implemented

`finat/quadrature.py` now has `CollapsedTensorProductQuadratureRule` (1D
Gauss–Jacobi factor rules in collapsed coordinates; axis `u` carries the Jacobi
weight `(1 - eta_u)^u`, which absorbs the Duffy Jacobian, so the simplex
weights are per-axis products) and a `scheme="collapsed"` branch in
`make_quadrature` building it via `collapsed_gauss_jacobi_quadrature`. Since
`tsfc/fem.py::get_quadrature_rule` passes the UFL measure's scheme metadata
straight to `make_quadrature`, `dx(scheme="collapsed")` already produces the
structured rule with **no** tsfc changes. Tested against FIAT's canonical
collapsed scheme on all monomials up to the requested degree
(`test/finat/test_quadrature.py::test_collapsed_quadrature`).

### (b) fem.py integration — Route B, no driver/kernel-interface changes

The standard path in `tsfc/fem.py` calls `element.basis_evaluation(order, ps,
entity)` and contracts the resulting `(ndof,)`-shaped tables with the
element's flat basis index (`element.get_indices()`), and `translate_argument`
extracts one flat entry with `ctx.argument_multiindices[number]`. The local
element tensor's shape (`element.index_shape` in
`kernel_interface/common.py::prepare_arguments`) and `argument_multiindices`
are flat and ndof-based *everywhere else in the kernel-interface/PyOP2 stack*,
so the original plan of making `argument_multiindices` itself a lattice
multiindex (see the old Route-B write-up below) would have changed the local
tensor's shape and broken that contract. The implementation instead keeps
`element.index_shape` and `argument_multiindices` exactly as they are today,
and confines the lattice multiindex — and all the Duffy/Morton machinery — to
a new `finat/duffy.py` module, reached from `tsfc/fem.py` through ordinary
FInAT element methods rather than through tsfc-side branching:

* `finat.duffy.DuffyElement` is a mixin (`finat.spectral.Legendre` is
  currently its only user) providing `duffy_evaluation` (the lattice-indexed,
  sum-factorized tabulation) plus two dispatch points:
  * **Backward transform / `basis_evaluation` override.** When `ps` is a
    `CollapsedTensorProductPointSet` and the entity is the cell interior (and
    not a macrocell — the only other case `duffy_evaluation` rejects),
    `DuffyElement.basis_evaluation` calls `duffy_evaluation` and scatters the
    lattice tabulation to the standard flat-dof-indexed convention: it
    introduces one *fresh* flat dof index `r`, builds the *inverse* Morton
    table (`FIAT.expansions.morton_inverse_table`, shape `(ndof, d)`) to get
    per-axis lookups `i_t(r)`, and substitutes `multiindex[t] ->
    VariableIndex(inverse_table[:, t][r])` throughout `duffy_evaluation`'s
    expression tree via `gem.node.MemoizerArg(gem.optimise.filtered_replace_indices)`
    (the same substitution mechanism `translate_argument`/`translate_coefficient`
    use for canonical quadrature-point reordering; `filtered_replace_indices`
    recurses into `VariableIndex.expression`, so the nested `m_t` lookups are
    rewritten too). The result, wrapped in `gem.ComponentTensor(..., (r,))`,
    is indistinguishable, from `fiat_to_ufl`/`prepare_arguments`'s point of
    view, from a standard dense `basis_evaluation` tabulation — so
    `translate_argument` in `tsfc/fem.py` needs **no special case at all**: it
    always calls `ctx.basis_evaluation(element, mt, entity_id)`, exactly as
    for any other element. `gem.optimise.contraction` never runs on this
    side (there is no sum to hoist yet at this stage); the sum-factorized
    quadrature contraction happens later, per dof, when
    `vanilla.py`/`spectral.py` process the quadrature `IndexSum` — the
    collapsed quadrature's own per-axis structure is what still delivers the
    O(p) win per axis there.
  * **Forward transform / `duffy_contraction`.** Unlike `basis_evaluation`,
    coefficient contraction has no generic per-element hook to dispatch
    through (it additionally needs the coefficient's dof vector `vec`), so
    `translate_coefficient` in `tsfc/fem.py` keeps a small
    `_use_duffy_contraction(element, ctx)` guard (`isinstance(element,
    DuffyElement)`, `ctx.point_set` a `CollapsedTensorProductPointSet`, cell
    interior, `ctx.unsummed_coefficient_indices` empty) before calling
    `element.duffy_contraction(mt.local_derivatives, ctx.point_set, entity,
    vec, ctx.epsilon)`. `duffy_contraction` builds a forward Morton lookup
    table (`FIAT.expansions.morton_forward_table`, shape `(degree+1,)^d`,
    clamped to a valid dof so out-of-lattice reads are merely wasted, never
    out of bounds — they always multiply a zero tabulation), gathers
    `vec[VariableIndex(table[multiindex])]`, and hands
    `IndexSum(Product(duffy[alpha], vec_r), multiindex)` to
    `gem.optimise.contraction`, exactly as originally planned: the `m_t`
    `VariableIndex` couplings inside `duffy_evaluation`'s own expression make
    the per-axis free-index sets nested (`{i_1} ⊂ {i_1, i_2} ⊂ ...`), so
    `contraction` finds the innermost-axis-first Karniadakis–Sherwin sweep by
    itself — no bespoke sum-factorization code needed. The result is wrapped
    back into a `gem.ComponentTensor` over `element.get_value_indices()`
    (empty for the scalar `Legendre` element), so it slots into `fiat_to_ufl`
    exactly like a standard dense tabulation would.

Both dispatch points were validated to ~1e-13/1e-14 against FIAT's dense
`tabulate()`, via compiled-and-executed loopy kernels, for values and first
derivatives on triangles and tetrahedra
(`tests/tsfc/test_codegen.py::test_duffy_scatter_and_contract`), and end to
end through `firedrake.assemble` (residuals and matrices, `dx` vs
`dx(scheme="collapsed")`, on triangle and tetrahedron meshes, degrees 1 and 3:
`tests/firedrake/regression/test_quadrature.py::test_collapsed_quadrature_sum_factorisation`).

### (c) Basis-index integration route — Route B chosen, Route C's dof reorder adopted underneath it

The element's flat basis index was originally Morton-ordered
(`FIAT.expansions.morton_index`, using `morton_index2`/`morton_index3` =
total-degree-major); it is now lattice-lexicographic (`FIAT.expansions.
lexicographic_permutation`, `FIAT.hierarchical.LegendreDual`), while the
factorization is indexed by the lattice multiindex. Routes considered:

* **Route A — layer-wise `Concatenate`.** Reuse tsfc/spectral.py's
  `Concatenate`/`unconcatenate` machinery by splitting the basis into
  contiguous Morton layers of fixed total degree `s`. Rejected: the layer
  decomposition does not align with the per-axis contraction structure (the
  sweeps contract one lattice axis at a time, not one total-degree layer at a
  time), so it buys the wrong factorization.

* **Route B — gather/scatter via `VariableIndex` (chosen; see (b)).**
  Keeps FIAT's dof ordering, `element.index_shape`, and
  `argument_multiindices` completely untouched; the flat-index arithmetic
  lives entirely inside `finat/duffy.py` (`DuffyElement.basis_evaluation` and
  `DuffyElement.duffy_contraction`). No `driver.py` or
  `kernel_interface/*.py` changes were needed at all, and — after the
  fem.py-integration refactor above — no bespoke branching in `tsfc/fem.py`
  either for the argument side; the lattice multiindex never escapes
  `finat/duffy.py`. The indirection costs one uint load per accumulation
  (forward) or one uint load per dof (backward), negligible against the O(p)
  inner contraction.

* **Route C — reorder FIAT dofs p-major (adopted, underneath Route B).**
  `Legendre` (variant="integral") now uses lattice-lexicographic dof order
  (`FIAT.hierarchical.LegendreDual` permuted via `FIAT.expansions.
  lexicographic_permutation`), so the flat index is `offsets[i_1, .., i_{d-1}]
  + i_d` — affine in the innermost coordinate for fixed outer coordinates,
  unlike Morton's `(p+q)(p+q+1)/2 + q`, which mixes every coordinate
  non-separably. This didn't eliminate Route B's `VariableIndex` gather/
  scatter (see "Deferred" below for why), but it did let `finat/duffy.py`'s
  index arithmetic (`_flat_index_expr`/`_inverse_lex_index_exprs`) shrink
  from multi-stage triangular/tetrahedral-number arithmetic to one small
  `(degree+1,)`-ish table lookup plus a bounded subtraction, both directions.
  Reordering FIAT's own dof numbering is externally visible (checkpoints,
  hand-written index hacks), judged an acceptable, narrowly-scoped cost
  specifically because `Legendre`/`variant="integral"` is a niche element
  family — ordinary `Lagrange`/`variant=None` elements are untouched.

## Deferred

* **Fully eliminating Route B's `VariableIndex` gather/scatter by making
  `element.get_indices()` return the lattice multiindex directly.** With the
  lattice-lexicographic dof order in place, this was investigated in depth
  and found to be blocked by more than "dof ordering is externally visible":
  `get_indices()` takes no arguments and is cached once per kernel, so it
  cannot distinguish a cell-interior kernel (where the lattice multiindex
  applies) from a facet-integral kernel (where `DuffyElement._duffy_applies`
  is always `False` and tabulation always uses the dense, flat-`(ndof,)`
  FIAT path) — returning a lattice tuple unconditionally would break
  `translate_argument` for every facet integral on a DG element (SIPG,
  upwinding), not just an edge case. A real fix is possible (pad the dense
  facet tabulation to the same `(degree+1,)**d` bounding-box shape at
  TSFC-compile time, and give `tsfc/kernel_interface/common.py`'s
  `prepare_arguments`/`prepare_coefficient` a bespoke jagged
  `gem.FlexiblyIndexed` view — `ComponentTensor(FlexiblyIndexed(flat_var,
  ((offset(p), ((q, 1),)),)), (p, q))` — instead of the current pure-
  rectangular `gem.reshape`) but that code is shared by every Firedrake
  kernel; confirmed via `test_collapsed_quadrature_sum_factorisation`'s
  mass-matrix case that it must also handle *two* Duffy arguments combined
  in one "A" tensor. Deliberately deferred as a separate, higher-risk
  follow-up rather than folded into the dof-reorder work above.
* **CG / C0 basis (milestones 3–4):** the C0 recombination makes each basis
  function a sum of <= 3 separable members (Sherwin–Karniadakis vertex/edge/face
  recombination); `tabulate_duffy` currently raises `NotImplementedError` for
  `continuity is not None`. The factored-term representation
  (`alpha -> [(coeff, factors), ...]`) was chosen so C0 can extend it by
  returning more terms per basis function.
* **Derivative order > 1:** raises `NotImplementedError`.
* **Macro cells** (`is_macrocell()`): raises `NotImplementedError` in
  `duffy_evaluation`.
