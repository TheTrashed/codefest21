import os

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from chatbot import get_response

app = Flask(__name__)
CORS(app)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)


@app.route("/", methods=['GET'])
def index():
    return render_template("base.html")


@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {'answer': response}
    return jsonify(message)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # socketio.run(app, port=int(os.environ.get('PORT', '5000')))
