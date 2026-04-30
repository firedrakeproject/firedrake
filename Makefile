.PHONY: ext
ext:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace 2>&1 | tee cat build.log

.PHONY: extforce
extforce:
	@echo "    Force building all extension modules"
	@python setup.py build_ext --inplace --force 2>&1 | tee cat build.log

.PHONY: lint
lint: srclint actionlint dockerlint

# Adds file annotations to Github Actions (only useful on CI)
GITHUB_ACTIONS_FORMATTING=0
ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	FLAKE8_FORMAT=--format='::error file=%(path)s,line=%(row)d,col=%(col)d,title=%(code)s::%(path)s:%(row)d:%(col)d: %(code)s %(text)s'
else
	FLAKE8_FORMAT=
endif

.PHONY: srclint
srclint:
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
	@# Exclude SC2046 so it doesn't complain about unquoted $ characters (the
	@# quoting can prevent proper parsing)
	@docker run -e SHELLCHECK_OPTS='--exclude=SC2046,SC2078,SC2143' --rm -v $$(pwd):/repo --workdir /repo rhysd/actionlint -color

.PHONY: dockerlint
dockerlint:
	@echo "    Pull latest hadolint image"
	@docker pull hadolint/hadolint:latest
	@for DOCKERFILE in docker/Dockerfile.*; \
		do \
		echo "    Linting $$DOCKERFILE"; \
		docker run --rm \
			-e HADOLINT_IGNORE=DL3003,DL3004,DL3005,DL3007,DL3008,DL3013,DL3015,DL3042,DL3059,SC2103,SC2046,SC2086 \
			-i hadolint/hadolint \
			< $$DOCKERFILE \
			|| exit 1; \
	done

# NOTE: It is recommended to run this command from inside the 'firedrake'
# Docker image to reduce the likelihood of test failures.
.PHONY: test_durations
test_durations:
	@echo "    Regenerating test durations"
	@echo "    Removing old durations file"
	rm -f tests/test_durations.json
	python3 -m pytest --store-durations --durations-path=tests/test_durations.json -m parallel[1] tests/ || true
	# use ':' to ensure that only rank 0 writes to the durations file
	for nprocs in 2 3 4 6 7 8; do \
		mpiexec -n 1 python3 -m pytest --store-durations --durations-path=tests/test_durations.json -m parallel[$${nprocs}] tests/ \
		: -n $$(( $${nprocs} - 1 )) python3 -m pytest -m parallel[$${nprocs}] -q tests/ || true ; \
	done
	@echo "    Test durations regenerated"
