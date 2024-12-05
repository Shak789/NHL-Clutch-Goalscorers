from flask import Flask, request, jsonify
import joblib
from NHL_Clutch_Goalscoring_Model import create_clutch_rankings  # Import the function from the converted script

model = joblib.load('ridge_cv_model.pkl')

app = Flask(__name__)

@app.route('/predict_and_rank', methods=['POST'])
def predict_and_rank():
    data = request.get_json()

    rankings = create_clutch_rankings(data)  # Call the function

    return jsonify(rankings)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
