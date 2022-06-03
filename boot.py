from flask import Flask
import flask

app = Flask(__name__)

@app.route("/api/filters/apply", methods=['POST'])
def applyFilters():
    raw_data = flask.request.get_data(False, False, False)
    
    return raw_data
