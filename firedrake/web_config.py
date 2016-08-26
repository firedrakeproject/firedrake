from flask import Flask, render_template

app = Flask(__name__)


def web_config(parameters):

    @app.route('/')
    def index():
        params = [{"key": k,
                   "type": "text",
                   "value": v} for k, v in parameters.iteritems()]
        return render_template('index.html', parameters=params)

    @app.route('/validate')
    def validate():
        pass

    app.run()
