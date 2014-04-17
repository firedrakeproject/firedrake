PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

BACKENDS ?= sequential opencl openmp cuda
OPENCL_ALL_CTXS := $(shell scripts/detect_opencl_devices)
OPENCL_CTXS ?= $(OPENCL_ALL_CTXS)

SPHINX_DIR = doc/sphinx
SPHINX_BUILD_DIR = $(SPHINX_DIR)/build
SPHINX_TARGET = html
SPHINX_TARGET_DIR = $(SPHINX_BUILD_DIR)/$(SPHINX_TARGET)
SPHINXOPTS = -a

PORT = 8000

MESHES_DIR = demo/meshes

GIT_REV = $(shell git rev-parse --verify --short HEAD)

all: ext

.PHONY : help test lint unit doc update_docs ext ext_clean meshes

help:
	@echo "make COMMAND with COMMAND one of:"
	@echo "  test               : run lint and unit tests"
	@echo "  lint               : run flake8 code linter"
	@echo "  unit               : run unit tests"
	@echo "  unit_BACKEND       : run unit tests for BACKEND"
	@echo "  doc                : build sphinx documentation"
	@echo "  serve              : launch local web server to serve up documentation"
	@echo "  update_docs        : build sphinx documentation and push to GitHub"
	@echo "  ext                : rebuild Cython extension"
	@echo "  ext_clean          : delete generated extension"
	@echo "  meshes             : download demo meshes"
	@echo
	@echo "Available OpenCL contexts: $(OPENCL_CTXS)"

test: lint unit

lint:
	@flake8

unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_%:
	cd $(UNIT_TEST_DIR); $(PYTEST) --backend=$*

unit_opencl:
	cd $(UNIT_TEST_DIR); for c in $(OPENCL_CTXS); do PYOPENCL_CTX=$$c $(PYTEST) --backend=opencl; done

doc:
	make -C $(SPHINX_DIR) $(SPHINX_TARGET) SPHINXOPTS=$(SPHINXOPTS)

serve:
	make -C $(SPHINX_DIR) livehtml

update_docs:
	if [ ! -d $(SPHINX_TARGET_DIR)/.git ]; then \
	mkdir -p $(SPHINX_BUILD_DIR); \
	cd $(SPHINX_BUILD_DIR); git clone `git config --get remote.origin.url` $(SPHINX_TARGET); \
fi
	cd $(SPHINX_TARGET_DIR); git fetch -p; git checkout -f gh-pages; git reset --hard origin/gh-pages
	make -C $(SPHINX_DIR) $(SPHINX_TARGET) SPHINXOPTS=$(SPHINXOPTS)
	cd $(SPHINX_TARGET_DIR); git add .; git commit -am "Update documentation to revision $(GIT_REV)"; git push origin gh-pages

ext: ext_clean
	python setup.py build_ext -i

ext_clean:
	rm -rf build pyop2/compute_ind.c pyop2/compute_ind.so pyop2/plan.c pyop2/plan.so pyop2/sparsity.c pyop2/sparsity.so

meshes:
	make -C $(MESHES_DIR) meshes
