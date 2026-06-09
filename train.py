
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

if __name__ == "__main__":
    # Handle SageMaker Paths
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # For local testing
    if not os.path.exists(train_dir):
        train_dir = "data"
    if not os.path.exists(model_dir):
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)

    # Load Data
    train_path = os.path.join(train_dir, "train.csv")
    df = pd.read_csv(train_path)
    print(f"Loaded training data with shape: {df.shape}")

    # Identify Target and Features
    target = "price" if "price" in df.columns else df.columns[-1]

    # Select only numeric columns for features
    X = df.drop(columns=[target]).select_dtypes(include=['number'])
    y = df[target]

    print(f"Training with {len(X.columns)} features: {list(X.columns)}")
    print(f"Target: {target}")

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Calculate training metrics
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Training MSE: {mse:.2f}")
    print(f"Training R2: {r2:.4f}")

    # Save Model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

# Inference function for SageMaker deployment
def model_fn(model_dir):
    """Load the model from the model_dir"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
