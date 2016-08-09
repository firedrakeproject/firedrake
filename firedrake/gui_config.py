"""A GUI for setting parameters"""

from __future__ import absolute_import

__all__ = ['show_config_gui']


def show_config_gui(parameters):
    from firedrake import Parameters

    if not isinstance(parameters, Parameters):
        raise TypeError("Expected Type: Parameters")

    pass


def export_params_to_json(parameters, filename):
    import json

    output_file = open(filename, 'w')
    output_file.write(json.dumps(parameters))
    output_file.close()


def import_params_from_json(filename):
    import json
    from firedrake import Parameters

    input_file = open(filename, 'r')
    dictionary = json.loads(input_file.read())
    params = Parameters(**dictionary)
    input_file.close()
    return params
