PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

REGRESSION_TEST_DIR = $(TEST_BASE_DIR)/regression

TESTHARNESS = $(REGRESSION_TEST_DIR)/testharness.py
BACKENDS ?= sequential opencl cuda

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

test: unit regression

unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_%:
	$(PYTEST) $(UNIT_TEST_DIR) --backend=$*

regression: $(foreach backend,$(BACKENDS), regression_$(backend))

regression_%:
	$(TESTHARNESS) --backend=$*

doc:
	make -C doc/sphinx html

update_docs:
	git submodule update --init -f
	git submodule foreach 'git checkout -f gh-pages; git fetch; git reset --hard origin/gh-pages'
	make -C doc/sphinx html
	git submodule foreach 'git commit -am "Update documentation"; git push origin gh-pages'
