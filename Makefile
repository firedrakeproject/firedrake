PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

REGRESSION_TEST_DIR = $(TEST_BASE_DIR)/regression

TESTHARNESS = $(REGRESSION_TEST_DIR)/testharness.py
BACKENDS ?= sequential opencl

help:
	@echo "make COMMAND with COMMAND one of:"
	@echo "  test               : run unit and regression tests"
	@echo "  unit               : run unit tests"
	@echo "  unit_BACKEND       : run unit tests for BACKEND"
	@echo "  regression         : run regression tests"
	@echo "  regression_BACKEND : run regression tests for BACKEND"

test: unit regression

unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_%:
	$(PYTEST) $(UNIT_TEST_DIR) --backend=$*

regression: $(foreach backend,$(BACKENDS), regression_$(backend))

regression_%:
	$(TESTHARNESS) --backend=$*
