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

    @app.route('/')
    def index():
        params = format_dict(parameters)
        return render_template('index.html', parameters=params)

    @app.route('/save', methods=["GET", "POST"])
    def save():
        import json
        dictionary = json.loads(request.form['parameters'])
        from firedrake.gui_config import load_from_dict
        try:
            load_from_dict(parameters, dictionary)
        except Exception as e:
            return e.message, 400
        return jsonify(successful=True)

    @app.route('/fetch')
    def fetch():
        return jsonify(**parameters.unwrapped_dict)

    app.run()
