all: python_build

python_build:
	@echo "    Build python modules"
	@cd python; python setup.py build_ext --inplace > build.log 2>&1 || cat build.log
