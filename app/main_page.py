from flask import Flask, render_template
from bokeh.client import pull_session
from bokeh.embed import server_session

interlingua_url = "http://localhost:5200/myapp"
decoders_url = "http://localhost:5201/decoder_app"

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/interlingua')
def interlingua():
    with pull_session(url=interlingua_url) as session:
        script = server_session(session_id=session.id, url=interlingua_url)

        return render_template("interlingua.html", script=script)


@app.route('/decoders')
def decoders():
    with pull_session(url=decoders_url) as session:
        script = server_session(session_id=session.id, url=decoders_url)

        return render_template("decoders.html", script=script)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
