from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Load the trained model and scaler
fraud_model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Get the request from Dialogflow
    req = request.get_json()
    intent_name = req.get("queryResult").get("intent").get("displayName")

    # Handle intents based on Dialogflow
    if intent_name == "Upload CSV Intent":
        return handle_csv_upload(req)
    elif intent_name == "Results Query Intent":
        return handle_results_query(req)
    elif intent_name == "Help Intent":
        return jsonify({"fulfillmentText": "You can upload a CSV file, and I will detect fraud or anomalies in it."})
    else:
        return jsonify({"fulfillmentText": "Sorry, I don't understand that."})

# Handle CSV upload intent
def handle_csv_upload(req):
    try:
        # If the file is uploaded via an API, you can extract the URL or other details.
        # In this case, we will simulate CSV loading for simplicity
        data = pd.read_csv("example.csv")  # Replace with the actual file upload logic

        # Preprocess the data
        features = data.drop('is_fraud', axis=1)  # Drop the target column if it's included
        X_scaled = scaler.transform(features)  # Apply the scaler to the features
        
        # Get fraud predictions
        fraud_predictions = fraud_model.predict(X_scaled)
        data['Fraud_Prediction'] = fraud_predictions  # Add predictions to the dataframe

        # Save the predictions (optional)
        data.to_csv("predictions.csv", index=False)

        # Respond back to Dialogflow with the results
        return jsonify({"fulfillmentText": "Your file has been processed. Fraud predictions are ready. Ask me for the results!"})

    except Exception as e:
        return jsonify({"fulfillmentText": f"An error occurred: {str(e)}"})

# Handle results query intent
def handle_results_query(req):
    try:
        # Load previously saved predictions
        data = pd.read_csv("predictions.csv")

        # Summarize results
        fraud_count = data['Fraud_Prediction'].sum()
        total = len(data)
        return jsonify({"fulfillmentText": f"I found {fraud_count} fraudulent transactions out of {total} total transactions."})
    except Exception as e:
        return jsonify({"fulfillmentText": f"Could not fetch results: {str(e)}"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
