from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model = joblib.load("titanic_survival_model.pkl")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            pclass = data.get('pclass')
            sex = data.get('sex')
            age = data.get('age')
        
            sex = 0 if sex.lower() == 'male' else 1

            input_data = pd.DataFrame([[pclass, sex, age]], columns=['pclass', 'sex', 'age'])

            survival_proba = model.predict_proba(input_data)[0][1]
            survived = model.predict(input_data)[0]
            return jsonify({
                'survival_probability': survival_proba,
                'prediction': 'Survived' if survived == 1 else 'Did not survive'
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({'message': 'This is a GET response'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)