all: modules

modules:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace > build.log 2>&1 || cat build.log

lint:
	@echo "    Linting firedrake codebase"
	@flake8 firedrake
	@echo "    Linting firedrake test suite"
	@flake8 tests


THREADS=1
ifeq ($(THREADS), 1)
	PYTEST_ARGS=
else
	PYTEST_ARGS=-n $(THREADS)
endif

test_regression: modules
	@echo "    Running non-extruded regression tests"
	@py.test tests/regression $(PYTEST_ARGS)

test_extrusion: modules
	@echo "    Running extruded regression tests"
	@py.test tests/extrusion $(PYTEST_ARGS)

test: modules
	@echo "    Running all regression tests"
	@py.test tests $(PYTEST_ARGS)

alltest: modules lint test

shorttest: modules lint
	@echo "    Running short regression tests"
	@py.test --short tests $(PYTEST_ARGS)
