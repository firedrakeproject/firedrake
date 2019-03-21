all: modules

modules:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace > build.log 2>&1 || cat build.log

lint:
	@echo "    Linting firedrake codebase"
	@python -m flake8 firedrake
	@echo "    Linting firedrake test suite"
	@python -m flake8 tests
	@echo "    Linting firedrake scripts"
	@python -m flake8 scripts --filename=*

clean:
	@echo "    Cleaning extension modules"
	@python setup.py clean > /dev/null 2>&1
	@echo "    RM firedrake/dmplex.*.so"
	-@rm -f firedrake/dmplex.so > /dev/null 2>&1
	@echo "    RM firedrake/dmplex.c"
	-@rm -f firedrake/dmplex.c > /dev/null 2>&1
	@echo "    RM firedrake/extrusion_numbering.*.so"
	-@rm -f firedrake/extrusion_numbering.so > /dev/null 2>&1
	@echo "    RM firedrake/extrusion_numbering.c"
	-@rm -f firedrake/extrusion_numbering.c > /dev/null 2>&1
	@echo "    RM firedrake/hdf5interface.*.so"
	-@rm -f firedrake/hdf5interface.so > /dev/null 2>&1
	@echo "    RM firedrake/hdf5interface.c"
	-@rm -f firedrake/hdf5interface.c > /dev/null 2>&1
	@echo "    RM firedrake/spatialindex.*.so"
	-@rm -f firedrake/spatialindex.so > /dev/null 2>&1
	@echo "    RM firedrake/spatialindex.c"
	-@rm -f firedrake/spatialindex.c > /dev/null 2>&1
	@echo "    RM firedrake/supermeshimpl.*.so"
	-@rm -f firedrake/supermeshimpl.so > /dev/null 2>&1
	@echo "    RM firedrake/supermeshimpl.c"
	-@rm -f firedrake/supermeshimpl.c > /dev/null 2>&1
	@echo "    RM firedrake/mg/impl.*.so"
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
	@python -m pytest tests/regression $(PYTEST_ARGS)

test_extrusion: modules
	@echo "    Running extruded regression tests"
	@python -m pytest tests/extrusion $(PYTEST_ARGS)

test_demos: modules
	@echo "    Running test of demos"
	@python -m pytest tests/demos $(PYTEST_ARGS)

test: modules
	@echo "    Running all regression tests"
	@python -m pytest tests $(PYTEST_ARGS)

alltest: modules lint test

shorttest: modules lint
	@echo "    Running short regression tests"
	@python -m pytest --short tests $(PYTEST_ARGS)
