all: modules

modules:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace > build.log 2>&1 || cat build.log

lint:
	@echo "    Linting firedrake codebase"
	@flake8 firedrake
	@echo "    Linting firedrake test suite"
	@flake8 tests
	@echo "    Linting firedrake install script"
	@flake8 scripts/firedrake-install

clean:
	@echo "    Cleaning extension modules"
	@python setup.py clean > /dev/null 2>&1
	@echo "    RM firedrake/dmplex.so"
	-@rm -f firedrake/dmplex.so > /dev/null 2>&1
	@echo "    RM firedrake/dmplex.c"
	-@rm -f firedrake/dmplex.c > /dev/null 2>&1
	@echo "    RM firedrake/hdf5interface.so"
	-@rm -f firedrake/hdf5interface.so > /dev/null 2>&1
	@echo "    RM firedrake/hdf5interface.c"
	-@rm -f firedrake/hdf5interface.c > /dev/null 2>&1
	@echo "    RM firedrake/spatialindex.so"
	-@rm -f firedrake/spatialindex.so > /dev/null 2>&1
	@echo "    RM firedrake/spatialindex.cpp"
	-@rm -f firedrake/spatialindex.cpp > /dev/null 2>&1
	@echo "    RM firedrake/mg/impl.so"
	-@rm -f firedrake/mg/impl.so > /dev/null 2>&1
	@echo "    RM firedrake/mg/impl.c"
	-@rm -f firedrake/mg/impl.c > /dev/null 2>&1


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
