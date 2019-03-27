def test_config_exist():
    import firedrake_configuration
    config = firedrake_configuration.get_config()
    assert config is not None
