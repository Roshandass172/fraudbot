import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(file_path):
    """
    Load CSV file containing transaction data for fraud detection and anomaly detection.
    :param file_path: Path to the CSV file.
    :return: Loaded data as a Pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {data.shape}")
    return data

# Step 4.2: Preprocess the Data
def preprocess_data(data):
    """
    Preprocess the data: handle missing values, encode categorical features, and scale the features.
    :param data: Raw transaction data.
    :return: Preprocessed features and target, and scaling objects.
    """
    # Fill missing values (example: with mean)
    data.fillna(data.mean(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save encoder for reverse lookup

    # Feature scaling
    scaler = StandardScaler()
    features = data.drop('is_fraud', axis=1)  # 'is_fraud' is the target variable
    target = data['is_fraud']

    features_scaled = scaler.fit_transform(features)
    return features_scaled, target, scaler, label_encoders

# Step 4.3: Train the Fraud Detection Model
def train_fraud_model(features, target):
    """
    Train a RandomForestClassifier to detect fraud based on transaction features.
    :param features: Processed features (scaled).
    :param target: Fraud detection target labels (0 or 1).
    :return: Trained model.
    """
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Fraud Detection Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model

# Step 4.4: Train the Anomaly Detection Model
def train_anomaly_model(features):
    """
    Train an Isolation Forest model for anomaly detection in transaction data.
    :param features: Processed features (scaled).
    :return: Trained Isolation Forest model.
    """
    # Train an Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    isolation_forest.fit(features)

    # Predict anomalies (-1 for anomalies, 1 for normal points)
    anomaly_scores = isolation_forest.predict(features)
    print(f"Anomalies Detected: {(anomaly_scores == -1).sum()} out of {len(anomaly_scores)}")
    return isolation_forest

# Step 4.5: Save the Models
def save_models(fraud_model, anomaly_model, scaler, label_encoders):
    """
    Save the trained models and preprocessing tools using joblib.
    :param fraud_model: Trained fraud detection model.
    :param anomaly_model: Trained anomaly detection model.
    :param scaler: Scaler used to scale features.
    :param label_encoders: Label encoders used for categorical features.
    """
    joblib.dump(fraud_model, "fraud_model.pkl")
    joblib.dump(anomaly_model, "anomaly_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    print("Models and preprocessing tools saved successfully!")

# Step 4.6: Main Function to Integrate Everything
if __name__ == "__main__":
    # Load the data
    file_path = "path_to_your_dataset.csv"  # Replace with the actual path to your dataset
    data = load_data(file_path)

    # Preprocess the data
    features_scaled, target, scaler, label_encoders = preprocess_data(data)

    # Train fraud detection model
    fraud_model = train_fraud_model(features_scaled, target)

    # Train anomaly detection model
    anomaly_model = train_anomaly_model(features_scaled)

    # Save the models
    save_models(fraud_model, anomaly_model, scaler, label_encoders)
