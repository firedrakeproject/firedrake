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
	@echo "    Linting firedrake codebase"
	@python -m flake8 $(FLAKE8_FORMAT) firedrake
	@echo "    Linting firedrake test suite"
	@python -m flake8 $(FLAKE8_FORMAT) tests
	@echo "    Linting firedrake scripts"
	@python -m flake8 $(FLAKE8_FORMAT) scripts --filename=*

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

# PyOP2 Makefile
# PYTEST = py.test
#
# TEST_BASE_DIR = test
#
# UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit
#
# SPHINX_DIR = doc/sphinx
# SPHINX_BUILD_DIR = $(SPHINX_DIR)/build
# SPHINX_TARGET = html
# SPHINX_TARGET_DIR = $(SPHINX_BUILD_DIR)/$(SPHINX_TARGET)
# SPHINXOPTS = -a
#
# PORT = 8000
#
# MESHES_DIR = demo/meshes
#
# GIT_REV = $(shell git rev-parse --verify --short HEAD)
#
# all: ext
#
# .PHONY : help test lint unit doc update_docs ext ext_clean meshes
#
# help:
# 	@echo "make COMMAND with COMMAND one of:"
# 	@echo "  test               : run lint and unit tests"
# 	@echo "  lint               : run flake8 code linter"
# 	@echo "  unit               : run unit tests"
# 	@echo "  unit_BACKEND       : run unit tests for BACKEND"
# 	@echo "  doc                : build sphinx documentation"
# 	@echo "  serve              : launch local web server to serve up documentation"
# 	@echo "  update_docs        : build sphinx documentation and push to GitHub"
# 	@echo "  ext                : rebuild Cython extension"
# 	@echo "  ext_clean          : delete generated extension"
# 	@echo "  meshes             : download demo meshes"
# 	@echo
# 	@echo "Available OpenCL contexts: $(OPENCL_CTXS)"
#
# test: lint unit
#
# lint:
# 	@flake8
#
# unit:
# 	cd $(TEST_BASE_DIR); $(PYTEST) unit
#
# doc:
# 	make -C $(SPHINX_DIR) $(SPHINX_TARGET) SPHINXOPTS=$(SPHINXOPTS)
#
# serve:
# 	make -C $(SPHINX_DIR) livehtml
#
# update_docs:
# 	if [ ! -d $(SPHINX_TARGET_DIR)/.git ]; then \
# 	mkdir -p $(SPHINX_BUILD_DIR); \
# 	cd $(SPHINX_BUILD_DIR); git clone `git config --get remote.origin.url` $(SPHINX_TARGET); \
# fi
# 	cd $(SPHINX_TARGET_DIR); git fetch -p; git checkout -f gh-pages; git reset --hard origin/gh-pages
# 	make -C $(SPHINX_DIR) $(SPHINX_TARGET) SPHINXOPTS=$(SPHINXOPTS)
# 	cd $(SPHINX_TARGET_DIR); git add .; git commit -am "Update documentation to revision $(GIT_REV)"; git push origin gh-pages
#
# ext: ext_clean
# 	python setup.py build_ext -i
#
# ext_clean:
# 	rm -rf build pyop2/compute_ind.c pyop2/compute_ind.so pyop2/sparsity.c pyop2/sparsity.so
#
# meshes:
# 	make -C $(MESHES_DIR) meshes
