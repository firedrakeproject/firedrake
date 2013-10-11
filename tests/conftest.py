"""Global test configuration."""


def pytest_addoption(parser):
    parser.addoption("--short", action="store_true", default=False,
                     help="Skip long tests")
