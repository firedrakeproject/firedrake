# Firedrake Adjoint

Two properties are treated as defining features of Firedrake, not optional extras: **composability**
(function spaces, forms, boundary conditions, solvers, and preconditioners are expected to combine
freely, without special-casing particular combinations) and **differentiability** (via `pyadjoint`,
essentially any computation built from Firedrake's own operations — assembly, interpolation,
variational solves, boundary condition application — can be taped and differentiated end-to-end).
Differentiability is expected to fall out of composability: a new feature built from already-annotated
Firedrake operations should be differentiable for free, with no extra work. A feature that instead
reaches past those operations into a lower-level, unannotated API silently breaks this guarantee —
forward runs stay numerically correct, but the adjoint quietly goes wrong, and nothing fails until
pyadjoint is specifically exercised.

## Coding Style And Conventions

* **Keep Annotation Out Of Plain Modules:** pyadjoint bookkeeping never appears inside a method body in
  `firedrake/*.py`. Differentiable types instead have a `*Mixin` in `firedrake/adjoint_utils/` exposing
  one `_ad_annotate_<name>` decorator per method that needs taping, applied where the method is defined
  (`@SomeMixin._ad_annotate_foo` on `def foo`). The decorator wraps the whole method, so the real
  implementation stays pyadjoint-agnostic. Add a new decorator to the `Mixin` rather than calling `_ad_*`
  from `firedrake/*.py` directly.

## Design And Debugging Method

Guiding principles for building and debugging taped operations with `firedrake.adjoint`, in the
order they should be applied:

* **Adjoints come from composition, not re-derivation.** If implementing a block's
  `evaluate_adj_component` has you calling `derivative`/`adjoint`/`action` on an operation
  Firedrake already tapes (assembly, interpolation, projection, a solve), the tape structure is
  wrong, not incomplete: make the block depend on that operation's *output*, and the operation's
  own block supplies the adjoint, TLM, Hessian, and recompute. A block's dependency is the value
  its operation actually consumes, never the raw user input that value was derived from.
* **Tape derived state when its consumer is taped, not when it is computed.** A value lazily
  re-derived from a mutable input must be re-annotated at the moment the consuming block is
  recorded, so the dependency edge points at the input's *current* block variable; for a
  `FloatingType`, override `_ad_will_add_as_dependency` to refresh (and thereby tape) the value
  before `super()` tapes the block. Object reuse in a time loop then records one correct chain per
  step with no extra bookkeeping. Conversely, run internal updates at construction/setter time
  under `stop_annotating()`: taping work nothing depends on leaves dangling blocks that recompute
  pays for.
* **Prove the linchpin primitive in isolation before restructuring around it.** Check in a few
  lines that the symbolic machinery you plan to rely on (e.g.
  `assemble(action(adjoint(derivative(form, c)), cof))`) accepts every input class you must
  handle. This surfaces hard limits early, and often shows that existing special-case code is
  subsumed by the general path — or was already dead.
* **Fix the block class that actually runs, not just the base.** Solves taped through solver
  objects execute `NonlinearVariationalSolveBlock`, which overrides several `GenericSolveBlock`
  methods (`prepare_evaluate_adj`, `evaluate_adj_component` with its own `_dFdm_cache`, and
  `_ad_assign_map`, which refreshes its cached cloned solvers by matching coefficients across
  clones via `.count()`); a case added only to the generic method silently never executes there.
  Print `type(block)` from the tape and grep the subclass for overrides before editing the base.
* **Verify the structure before the numbers.** Print each block in
  `get_working_tape().get_blocks()` with the identities of its `.get_dependencies()` and
  `.get_outputs()`, and check the DAG is exactly the chain you designed — no stale or dangling
  block variables — for both freshly-created and reused objects. Only then run `taylor_test` with
  a genuinely nonlinear control: rate ≈ 2 is a pass; residuals all ~1e-16 mean the functional is
  accidentally linear in the control (square it); rate ≈ 1 means a stale value reached the tape.
  A zero gradient is named by the warning `Adjoint value is None, is the functional independent of
  the control variable?` — any later `ZeroDivisionError`/`nan` is just its echo. In a
  `taylor_to_dict` check, rates that start correct then collapse with residuals stuck at a small
  floor mean a small absolute error in the highest-order term supplied (a nearly-right Hessian or
  gradient); attribute it by re-running the same Taylor test over a matrix of feature toggles
  (one new argument at a time) against a known-good baseline.
* **Dependency discovery is structural, not conceptual.** `form.coefficients()` finds `Function`s
  (including on `"R"` spaces) but not `firedrake.Constant`, so a bare `Constant` silently records
  no dependency — represent differentiable scalars as `Function`s on an `"R"` space.
* **A cache reused across recomputes must resync every axis that varies between reuses.** A cached
  solver refreshes some of its inputs automatically (form coefficients) but not others (its
  problem's boundary conditions); each missed axis silently reuses stale data under a perturbed
  control.
