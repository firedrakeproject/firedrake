PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

REGRESSION_TEST_DIR = $(TEST_BASE_DIR)/regression

TESTHARNESS = $(REGRESSION_TEST_DIR)/testharness.py
BACKENDS ?= sequential opencl openmp cuda
OPENCL_ALL_CTXS := $(shell python detect_opencl_devices.py)
OPENCL_CTXS ?= $(OPENCL_ALL_CTXS)

SPHINX_DIR = doc/sphinx
SPHINX_BUILD_DIR = $(SPHINX_DIR)/build
SPHINX_TARGET = html
SPHINX_TARGET_DIR = $(SPHINX_BUILD_DIR)/$(SPHINX_TARGET)

.PHONY : help test unit regression doc update_docs

help:
	@echo "make COMMAND with COMMAND one of:"
	@echo "  test               : run unit and regression tests"
	@echo "  unit               : run unit tests"
	@echo "  unit_BACKEND       : run unit tests for BACKEND"
	@echo "  regression         : run regression tests"
	@echo "  regression_BACKEND : run regression tests for BACKEND"
	@echo "  doc                : build sphinx documentation"
	@echo "  update_docs        : build sphinx documentation and push to GitHub"
	@echo
	@echo "Available OpenCL contexts: $(OPENCL_CTXS)"

test: unit regression

unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_%:
	$(PYTEST) $(UNIT_TEST_DIR) --backend=$*

unit_opencl:
	for c in $(OPENCL_CTXS); do PYOPENCL_CTX=$$c $(PYTEST) $(UNIT_TEST_DIR) --backend=opencl; done

regression: $(foreach backend,$(BACKENDS), regression_$(backend))

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
