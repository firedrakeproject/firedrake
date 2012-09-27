PYTEST = py.test

TEST_BASE_DIR = test

UNIT_TEST_DIR = $(TEST_BASE_DIR)/unit

REGRESSION_TEST_DIR = $(TEST_BASE_DIR)/regression

TESTHARNESS = $(REGRESSION_TEST_DIR)/testharness.py
BACKENDS ?= sequential opencl

all: test
test: unit regression
unit: $(foreach backend,$(BACKENDS), unit_$(backend))

unit_%:
	$(PYTEST) $(UNIT_TEST_DIR) --backend=$*

regression: $(foreach backend,$(BACKENDS), regression_$(backend))

regression_%:
	$(TESTHARNESS) --backend=$*
