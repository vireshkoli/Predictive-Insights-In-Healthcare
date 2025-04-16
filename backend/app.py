from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin  # Import the CORS module
# from ensemble import ensemble_model
import os
import sys

sys.path.insert(0, f"{os.getcwd()}/Diabetes")
sys.path.insert(0, f"{os.getcwd()}/LungCancer")

from diabetes_ensemble import ensemble_model_diabetes
from lung_ensemble import ensemble_model_lung


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    print(request.method)
    if request.method == 'GET':
        # Handle GET request
        data = {'message': 'Hello from Flask!'}
        return jsonify(data)
    elif request.method == 'POST':
        # Handle POST request
        data = request.get_json()
        return jsonify(data)
    
@app.route('/api/diabetes', methods=['POST'])
def diabetesEnsemble():
    data = request.get_json()
    print(data['diabetesData'])
    Diabetes = ensemble_model_diabetes(data['diabetesData'])
    # print(percenteDiabetes)
    return jsonify(Diabetes)

@app.route('/api/lungcancer', methods=['POST'])
def lungEnsemble():
    data = request.get_json()
    lungs_data = ensemble_model_lung(data["lungCancer"])
    return jsonify(lungs_data)

if __name__ == '__main__':
    print("working directory: " + os.getcwd())
    app.run(debug=True)
