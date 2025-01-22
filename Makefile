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
	@echo "    RM pyop2/*.so"
	-@rm -f pyop2/*.so > /dev/null 2>&1
	@echo "    RM tinyasm/*.so"
	-@rm -f tinyasm/*.so > /dev/null 2>&1

.PHONY: check
check:
	@echo "    Running serial smoke tests"
	@python -m pytest --quiet --no-summary \
		tests/firedrake/regression/test_stokes_mini.py::test_stokes_mini \
		tests/firedrake/regression/test_locate_cell.py `# spatialindex` \
		tests/firedrake/supermesh/test_assemble_mixed_mass_matrix.py::test_assemble_mixed_mass_matrix[2-CG-CG-0-0]  `# supermesh`
	@echo "    Serial tests passed"
	@echo "    Running parallel smoke tests"
	@mpiexec -n 3 python -m pytest --quiet --no-summary -m parallel[3] \
		tests/firedrake/regression/test_dg_advection.py::test_dg_advection_icosahedral_sphere
	@echo "    Parallel tests passed"

.PHONY: durations
durations:
	@echo "    Generate timings to optimise pytest-split"
	python -m pytest --store-durations -m "parallel[1] or not parallel" tests/
	for nprocs in 2 3 4 6 7 8; do
		mpiexec -n 1 \
			python -m pytest --store-durations -m parallel[$${nprocs}] tests/ : \
			-n $$(( $${nprocs} - 1 )) pytest -m parallel[$${nprocs}] -q tests/
	done
