from flask import Flask, render_template

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

    @app.route('/validate')
    def validate():
        pass

    app.run()
