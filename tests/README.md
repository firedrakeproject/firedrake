### Test durations

For better load balancing of the test suite it is useful to occasionally regenerate the `test_durations.json` file. The steps to do that are as follows:

1. Delete the existing `test_durations.json` file.
2. Run the test suite locally passing the flag `--store-durations` (needs pytest-split to be installed). For the serial tests run:
    ```
    $ python -m pytest -m parallel[1] tests/ --store-durations --durations-path=tests/test_durations.json
    ```
    and for the parallel tests run:
    ```
    $ mpiexec -n 1 python -m pytest -m parallel[*NPROCS*] tests/ --store-durations  --durations-path=tests/test_durations.json\
      : -n *NPROCS-1* python -m pytest -m parallel[*NPROCS*] -q tests/
    ```
    The complicated invocation here is needed to make sure that only rank 1 writes to the file.

    Note that is is fine if some tests fail - what matters is having the majority of the test suite covered.
3. Submit the new `test_durations.json` file via a pull request.
