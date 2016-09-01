from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


def web_config(parameters):

    def format_dict(parameters):
        ret = []
        for k, v in parameters.iteritems():
            if not isinstance(v, dict):
                ret.append({"key": k,
                            "type": str(parameters.get_key(k).type),
                            "value": v})
            else:
                ret.append({"key": k,
                            "type": "dict",
                            "value": format_dict(v)})
        return ret

    @app.route('/', methods=["GET", "POST"])
    def index():
        err = []
        if request.method == "POST":
            import json
            json_file = request.files['json']
            dictionary = json.loads(json_file.read())
            json_file.close()
            from firedrake.gui_config import load_from_dict
            err.extend(validate_input(parameters, dictionary))
            if err == []:
                load_from_dict(parameters, dictionary)
        params = format_dict(parameters)
        return render_template('index.html', parameters=params, err=err)

    def validate_input(parameters, dictionary):
        from firedrake.parameters import Parameters
        err = []
        for k in parameters.keys():
            if isinstance(parameters[k], Parameters):
                err.extend(validate_input(parameters[k], dictionary[k]))
            else:
                if not parameters.get_key(k).validate(dictionary[k]):
                    err.append("Invalid value for %s" % k)
        return err

    @app.route('/validate', methods=["GET", "POST"])
    def validate():
        import json
        dictionary = json.loads(request.form['parameters'])
        validate_result = validate_input(parameters, dictionary)
        if validate_result == []:
            return jsonify(successful=True)
        else:
            return jsonify(successful=False, err=validate_result), 400

    @app.route('/save', methods=["GET", "POST"])
    def save():
        import json
        dictionary = json.loads(request.form['parameters'])
        from firedrake.gui_config import load_from_dict
        try:
            load_from_dict(parameters, dictionary)
        except Exception as e:
            return jsonify(successful=False, errmsg=e.message), 400
        return jsonify(successful=True)

    @app.route('/fetch')
    def fetch():
        return jsonify(**parameters.unwrapped_dict)

    app.run()
