from __future__ import absolute_import, print_function, division


def test_config_exist():
    import firedrake_configuration
    config = firedrake_configuration.get_config()
    assert config is not None
