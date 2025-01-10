all: modules

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

lint:
	@echo "    Linting firedrake"
	@python -m flake8 $(FLAKE8_FORMAT) firedrake
	@echo "    Linting firedrake scripts"
	@python -m flake8 $(FLAKE8_FORMAT) firedrake/scripts --filename=*
	@python -m flake8 $(FLAKE8_FORMAT) scripts --filename=*
	@echo "    Linting firedrake tests"
	@python -m flake8 $(FLAKE8_FORMAT) tests
	@echo "    Linting PyOP2"
	@python -m flake8 $(FLAKE8_FORMAT) pyop2
	@echo "    Linting PyOP2 scripts"
	@python -m flake8 $(FLAKE8_FORMAT) pyop2/scripts --filename=*
	@echo "    Linting TSFC"
	@python -m flake8 $(FLAKE8_FORMAT) tsfc

actionlint:
	@echo "    Pull latest actionlint image"
	@docker pull rhysd/actionlint:latest
	@docker run --rm -v $$(pwd):/repo --workdir /repo rhysd/actionlint -color

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
	@echo "    RM pyop2/*.so"
	-@rm -f pyop2/*.so > /dev/null 2>&1
	@echo "    RM tinyasm/*.so"
	-@rm -f tinyasm/*.so > /dev/null 2>&1


THREADS=1
ifeq ($(THREADS), 1)
	PYTEST_ARGS=
else
	PYTEST_ARGS=-n $(THREADS)
endif

test_regression: modules
	@echo "    Running non-extruded regression tests"
	@python -m pytest tests/firedrake/regression $(PYTEST_ARGS)

test_extrusion: modules
	@echo "    Running extruded regression tests"
	@python -m pytest tests/firedrake/extrusion $(PYTEST_ARGS)

test_demos: modules
	@echo "    Running test of demos"
	@python -m pytest tests/firedrake/demos $(PYTEST_ARGS)

test: modules
	@echo "    Running all regression tests"
	@python -m pytest tests $(PYTEST_ARGS)

alltest: modules lint test

shorttest: modules lint
	@echo "    Running short regression tests"
	@python -m pytest --short tests/firedrake $(PYTEST_ARGS)
