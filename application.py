import os

from flask import Flask, flash, jsonify, redirect, render_template, request, \
    session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

# Configure application
app = Flask(__name__)
port = 5100

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# @app.after_request
# def after_request(reponse):
#    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#    response.header["Expires"] = 0
#    response.header["Pragma"] = "no-cache"
#    return response

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def index():
    greeting = "Hi there, how can I help you?"
    return render_template("index.html", greeting=greeting)

if __name__ == '__main__':
    app.run(port=port)
