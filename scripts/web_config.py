from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder="../templates")

__all__ = ["web_config"]


def web_config(parameters):
    """Start a web server for configuring the Parameters on port 5000

    :arg parameters: a :class `firedrake.parameters.Parameters: class to be
        configured
    """

    def format_dict(parameters):
        """Format the parameters to a dictionary for rendering"""
        ret = []
        for k, v in parameters.iteritems():
            if not isinstance(v, dict):
                ret.append({"key": k,
                            "type": str(parameters.get_key(k).type),
                            "value": v,
                            "depends": k.depends if k.depends is not None else '',
                            "visible_level": k.visible_level,
                            "help": k.help})
            else:
                ret.append({"key": k,
                            "type": "dict",
                            "depends": k.depends if k.depends is not None else '',
                            "visible_level": k.visible_level,
                            "summary": render_docstring(v.summary),
                            "value": format_dict(v)})
        return ret

    @app.route('/', methods=["GET", "POST"])
    def index():
        """Entry point of webpage

        Show a form with inputs. If a JSON file is posted, the data in JSON
        will be loaded into the current Parameters instance
        """
        err = []
        if request.method == "POST":
            import json
            json_file = request.files['json']
            try:
                dictionary = json.loads(json_file.read())
                json_file.close()
                from firedrake.gui_config import load_from_dict
                err.extend(validate_input(parameters, dictionary))
                if err == []:
                    load_from_dict(parameters, dictionary)
                json.dump(dictionary, open(config_file, 'w'))
            except:
                pass
        params = format_dict(parameters.unwrapped_dict(-1))
        return render_template('index.html', parameters=params, err=err,
                               summary=render_docstring(parameters.summary))

    def validate_input(parameters, dictionary):
        """Validate inputs using the validation information in Parameters"""
        from firedrake.parameters import Parameters
        err = []
        for k in parameters.keys():
            if (k not in dictionary.keys()):
                continue
            if isinstance(parameters[k], Parameters):
                err.extend(validate_input(parameters[k], dictionary[k]))
            else:
                if not parameters.get_key(k).validate(dictionary[k]):
                    err.append("Invalid value for %s" % k)
        return err

    @app.route('/validate', methods=["GET", "POST"])
    def validate():
        """Validate inputs posted in JSON format"""
        import json
        dictionary = json.loads(request.form['parameters'])
        validate_result = validate_input(parameters, dictionary)
        if validate_result == []:
            return jsonify(successful=True)
        else:
            return jsonify(successful=False, err=validate_result), 400

    @app.route('/save', methods=["GET", "POST"])
    def save():
        """Save inputs posted in JSON format into Parameters"""
        import json
        dictionary = json.loads(request.form['parameters'])
        validate_result = validate_input(parameters, dictionary)
        if validate_result != []:
            return jsonify(successful=False, err=validate_result), 400
        from firedrake.gui_config import load_from_dict
        load_from_dict(parameters, dictionary)
        json.dump(dictionary, open(config_file, 'w'))
        return jsonify(successful=True)

    @app.route('/fetch')
    def fetch():
        """Fetch current Parameter setting"""
        return jsonify(**parameters.unwrapped_dict(-1))

    def render_docstring(docstring):
        if docstring == "":
            return ""
        from docrepr.sphinxify import sphinxify
        import tempfile
        srcdir = tempfile.mkdtemp(dir=tempfile.gettempdir())
        return sphinxify(docstring, srcdir)

    app.run(host="0.0.0.0")


if __name__ == '__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(
        description="Run a web server to configure the module")
    parser.add_argument("module_name", help="The name of the module, it must \
                        contain an attribute of type `firedrake.parameters.Parameters` \
                        of name `parameters`")
    parser.add_argument("--config_file", default="parameters.json",
                        help="The path of file storing the configuration file")
    args = parser.parse_args()
    module_name = args.module_name
    config_file = args.config_file
    module = importlib.import_module(module_name)
    web_config(module.parameters)
