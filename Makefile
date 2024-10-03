.PHONY: all
all: modules

.PHONY: modules
modules:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace > build.log 2>&1 || cat build.log

# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format='::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
else
	FLAKE8_FORMAT=
endif

.PHONY: lint
lint:
	@echo "    Linting firedrake codebase"
	@python -m flake8 $(FLAKE8_FORMAT) firedrake
	@echo "    Linting firedrake test suite"
	@python -m flake8 $(FLAKE8_FORMAT) tests
	@echo "    Linting firedrake scripts"
	@python -m flake8 $(FLAKE8_FORMAT) scripts --filename=*

.PHONY: actionlint
actionlint:
	@echo "    Pull latest actionlint image"
	@docker pull rhysd/actionlint:latest
	@docker run --rm -v $$(pwd):/repo --workdir /repo rhysd/actionlint -color

.PHONY: dockerlint
dockerlint:
	@echo "    Pull latest hadolint image"
	@docker pull hadolint/hadolint:latest
	@for DOCKERFILE in docker/Dockerfile.*; \
		do \
		echo "    Linting $$DOCKERFILE"; \
		docker run --rm \
			-e HADOLINT_IGNORE=DL3005,DL3007,DL3008,DL3015,DL3059 \
			-i hadolint/hadolint \
			< $$DOCKERFILE \
			|| exit 1; \
	done

.PHONY: clean
clean:
	@echo "    Cleaning extension modules"
	@python setup.py clean > /dev/null 2>&1
	@echo "    RM firedrake/cython/dmplex.*.so"
	-@rm -f firedrake/cython/dmplex.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/dmplex.c"
	-@rm -f firedrake/cython/dmplex.c > /dev/null 2>&1
	@echo "    RM firedrake/cython/extrusion_numbering.*.so"
	-@rm -f firedrake/cython/extrusion_numbering.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/extrusion_numbering.c"
	-@rm -f firedrake/cython/extrusion_numbering.c > /dev/null 2>&1
	@echo "    RM firedrake/cython/hdf5interface.*.so"
	-@rm -f firedrake/cython/hdf5interface.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/hdf5interface.c"
	-@rm -f firedrake/cython/hdf5interface.c > /dev/null 2>&1
	@echo "    RM firedrake/cython/spatialindex.*.so"
	-@rm -f firedrake/cython/spatialindex.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/spatialindex.c"
	-@rm -f firedrake/cython/spatialindex.c > /dev/null 2>&1
	@echo "    RM firedrake/cython/supermeshimpl.*.so"
	-@rm -f firedrake/cython/supermeshimpl.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/supermeshimpl.c"
	-@rm -f firedrake/cython/supermeshimpl.c > /dev/null 2>&1
	@echo "    RM firedrake/cython/mg/impl.*.so"
	-@rm -f firedrake/cython/mg/impl.so > /dev/null 2>&1
	@echo "    RM firedrake/cython/mg/impl.c"
	-@rm -f firedrake/cython/mg/impl.c > /dev/null 2>&1
	@echo "    RM tinyasm/*.so"
	-@rm -f tinyasm/*.so

# This is the minimum required to run the test suite
NPROCS=8
# On CI we add additional `pytest` args to spit out more information
PYTEST_ARGS=
# specifically --durations=200 --timeout=1800 --timeout-method=thread -o faulthandler_timeout=1860 -s
# and --cov firedreake once the performance regression is fixed!

# Requires pytest and pytest-mpi only
.PHONY: test_serial
test_serial:
	@echo "    Running all tests in serial"
	@python -m pytest -v tests

# Requires pytest and pytest-mpi only
.PHONY: test_smoke
test_smoke:
	@echo "    Running the bare minimum smoke tests"
	@python -m pytest -k "poisson_strong or stokes_mini or dg_advection" -v tests/regression/

.PHONY: _test_serial_tests
_test_serial_tests:
	@echo "    Running serial tests over $(NPROCS) threads"
	@python -m pytest \
		-n $(NPROCS) --dist worksteal \
		$(PYTEST_ARGS) \
		-m "not parallel" \
		-v tests

.PHONY: _test_small_world_tests
_test_small_world_tests:
	@echo "    Running parallel tests over $(NPROCS) ranks"
	@for N in 2 3 4 ; do \
		echo "    COMM_WORLD=$$N ranks"; \
		mpispawn -nU $(NPROCS) -nW $$N --propagate-errcodes python -m pytest \
			--splitting-algorithm least_duration \
			--splits \$$MPISPAWN_NUM_TASKS \
			--group \$$MPISPAWN_TASK_ID1 \
			$(PYTEST_ARGS) \
			-m "parallel[\$$MPISPAWN_WORLD_SIZE] and not broken" \
			-v tests; \
	done

.PHONY: _test_large_world_test
_test_large_world_tests:
	@echo "    Running parallel tests over $(NPROCS) ranks"
	@for N in 6 7 8 ; do \
		echo "    COMM_WORLD=$$N ranks"; \
		mpiexec -n $$N python -m pytest \
			$(PYTEST_ARGS) \
			-m "parallel[$$N] and not broken" \
            -v tests; \
	done

.PHONY: test_adjoint
test_adjoint:
	@echo "    Running adjoint tests over $(NPROCS) threads"
	@python -m pytest \
		$(PYTEST_ARGS) \
		-v $(VIRTUAL_ENV)/src/pyadjoint/tests/firedrake-adjoint

# Additionally requires pytest-xdist, mpispawn and pytest-split
.PHONY: test
test: _test_serial_tests _test_small_world_tests _test_large_world_tests

.PHONY: test_ci
test_ci: test test_adjoint

.PHONY: test_generate_timings
test_generate_timings:
	@echo "    Generate timings to optimise pytest-split"
	for N in 2 3 4 ; do \
		@mpiexec \
			-n 1 pytest --store-durations -m "parallel[$$N] and not broken" -v tests : \
			-n $$(( $$N - 1 )) pytest -m "parallel[$$N] and not broken" -q tests; \
	done

.PHONY: alltest
alltest: modules lint test_serial
