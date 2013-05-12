PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

REGRESSION_TEST_DIR = $(TEST_BASE_DIR)/regression

TESTHARNESS = $(REGRESSION_TEST_DIR)/testharness.py
BACKENDS ?= sequential opencl openmp cuda mpi_sequential
OPENCL_ALL_CTXS := $(shell python detect_opencl_devices.py)
OPENCL_CTXS ?= $(OPENCL_ALL_CTXS)

SPHINX_DIR = doc/sphinx
SPHINX_BUILD_DIR = $(SPHINX_DIR)/build
SPHINX_TARGET = html
SPHINX_TARGET_DIR = $(SPHINX_BUILD_DIR)/$(SPHINX_TARGET)

all: ext

.PHONY : help test unit regression doc update_docs ext ext_clean

help:
	@echo "make COMMAND with COMMAND one of:"
	@echo "  test               : run unit and regression tests"
	@echo "  unit               : run unit tests"
	@echo "  unit_BACKEND       : run unit tests for BACKEND"
	@echo "  regression         : run regression tests"
	@echo "  regression_BACKEND : run regression tests for BACKEND"
	@echo "  doc                : build sphinx documentation"
	@echo "  update_docs        : build sphinx documentation and push to GitHub"
	@echo "  ext                : rebuild Cython extension"
	@echo "  ext_clean          : delete generated extension"
	@echo
	@echo "Available OpenCL contexts: $(OPENCL_CTXS)"

test: unit regression

unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_mpi_%:
	@echo Not implemented

unit_%:
	cd $(UNIT_TEST_DIR); $(PYTEST) --backend=$*

unit_opencl:
	cd $(UNIT_TEST_DIR); for c in $(OPENCL_CTXS); do PYOPENCL_CTX=$$c $(PYTEST) --backend=opencl; done

regression: $(foreach backend,$(BACKENDS), regression_$(backend))

regression_mpi_%:
	$(TESTHARNESS) -p parallel --backend=$*

regression_%:
	$(TESTHARNESS) --backend=$*

regression_opencl:
	for c in $(OPENCL_CTXS); do PYOPENCL_CTX=$$c $(TESTHARNESS) --backend=opencl; done

doc:
	make -C $(SPHINX_DIR) $(SPHINX_TARGET)

update_docs:
	if [ ! -d $(SPHINX_TARGET_DIR)/.git ]; then \
	mkdir -p $(SPHINX_BUILD_DIR); \
	cd $(SPHINX_BUILD_DIR); git clone `git config --get remote.origin.url` $(SPHINX_TARGET); \
fi
	cd $(SPHINX_TARGET_DIR); git fetch -p; git checkout -f gh-pages; git reset --hard origin/gh-pages
	make -C $(SPHINX_DIR) $(SPHINX_TARGET)
	cd $(SPHINX_TARGET_DIR); git commit -am "Update documentation"; git push origin gh-pages

ext: ext_clean
	python setup.py build_ext -i

ext_clean:
	rm -rf build pyop2/op_lib_core.c pyop2/op_lib_core.so
