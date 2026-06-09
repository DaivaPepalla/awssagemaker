# Complete Local MLOps Workflow - No SageMaker Session Required
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def setup_environment():
    """Setup environment for local execution"""
    print("🔧 Setting up local MLOps environment...")
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    print("✅ Environment setup complete")

def load_dataset_locally():
    """Load dataset with local fallback"""
    print("📊 Loading dataset...")
    
    try:
        from datasets import load_dataset
        print("Attempting to load from Hugging Face...")
        ds = load_dataset("Saathwik56/houseprice")
        df = ds["train"].to_pandas()
        print(f"✅ Dataset loaded from Hugging Face: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️  Hugging Face loading failed: {e}")
        print("🔄 Creating sample house price data...")
        
        # Create realistic sample data
        np.random.seed(42)
        n_samples = 2000
        
        # Generate correlated features for more realistic data
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        sqft_living = np.random.randint(500, 5000, n_samples)
        sqft_lot = sqft_living + np.random.randint(200, 3000, n_samples)
        floors = np.random.randint(1, 4, n_samples)
        
        # Price correlation with features
        base_price = (
            bedrooms * 50000 + 
            bathrooms * 30000 + 
            sqft_living * 150 + 
            floors * 20000
        )
        price = base_price + np.random.normal(0, 100000, n_samples)
        price = np.clip(price, 100000, 3000000)  # Realistic price range
        
        sample_data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'view': np.random.randint(0, 5, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(6, 13, n_samples),
            'yr_built': np.random.randint(1950, 2023, n_samples),
            'yr_renovated': np.random.choice([0] + list(range(1980, 2023)), n_samples, p=[0.8] + [0.2/43]*43),
            'price': price.astype(int)
        }
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Sample dataset created: {df.shape}")
        return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("🧹 Preprocessing data...")
    
    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Handle missing values
    df = df.dropna()
    
    # Convert string numbers to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='ignore')
    
    print(f"✅ Data preprocessed: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    return df

def split_and_save_data(df):
    """Split data and save to files"""
    print("✂️  Splitting and saving data...")
    
    target = 'price'
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=pd.qcut(df[target], q=5, duplicates='drop'))
    
    # Save datasets
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"✅ Data saved:")
    print(f"   📁 Training set: {train_df.shape[0]} samples")
    print(f"   📁 Test set: {test_df.shape[0]} samples")
    
    return train_df, test_df

def train_model(train_df):
    """Train the machine learning model"""
    print("🤖 Training model...")
    
    target = 'price'
    X = train_df.drop(columns=[target]).select_dtypes(include=['number'])
    y = train_df[target]
    
    print(f"📊 Features ({len(X.columns)}): {list(X.columns)}")
    print(f"🎯 Target: {target}")
    
    # Train model with better parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Training metrics
    y_pred_train = model.predict(X)
    train_mse = mean_squared_error(y, y_pred_train)
    train_r2 = r2_score(y, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    
    print(f"📈 Training Results:")
    print(f"   MSE: ${train_mse:,.0f}")
    print(f"   RMSE: ${train_rmse:,.0f}")
    print(f"   R² Score: {train_r2:.4f}")
    
    # Save model
    model_path = 'model/house_price_model.joblib'
    joblib.dump(model, model_path)
    print(f"💾 Model saved: {model_path}")
    
    return model

def evaluate_model(model, test_df):
    """Evaluate model on test data"""
    print("📊 Evaluating model...")
    
    target = 'price'
    X_test = test_df.drop(columns=[target]).select_dtypes(include=['number'])
    y_test = test_df[target]
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    
    print(f"🎯 Test Results:")
    print(f"   MSE: ${test_mse:,.0f}")
    print(f"   RMSE: ${test_rmse:,.0f}")
    print(f"   R² Score: {test_r2:.4f}")
    
    # Sample predictions
    print(f"\n🔍 Sample Predictions:")
    comparison = pd.DataFrame({
        'Actual': y_test.iloc[:10].values,
        'Predicted': y_pred[:10].astype(int),
        'Difference': (y_test.iloc[:10].values - y_pred[:10]).astype(int),
        'Error_%': ((y_test.iloc[:10].values - y_pred[:10]) / y_test.iloc[:10].values * 100).round(1)
    })
    print(comparison.to_string(index=False))
    
    return test_mse, test_r2, test_rmse

def main():
    """Main MLOps workflow"""
    print("🚀 Starting Local MLOps Workflow")
    print("=" * 50)
    
    # Step 1: Setup
    setup_environment()
    
    # Step 2: Load data
    df = load_dataset_locally()
    
    # Step 3: Preprocess
    df = preprocess_data(df)
    
    # Step 4: Split and save
    train_df, test_df = split_and_save_data(df)
    
    # Step 5: Train model
    model = train_model(train_df)
    
    # Step 6: Evaluate
    test_mse, test_r2, test_rmse = evaluate_model(model, test_df)
    
    print("\n" + "=" * 50)
    print("🎉 MLOps Workflow Complete!")
    print(f"📁 Files created:")
    print(f"   - data/train.csv")
    print(f"   - data/test.csv") 
    print(f"   - model/house_price_model.joblib")
    print(f"📊 Final Model Performance:")
    print(f"   - Test RMSE: ${test_rmse:,.0f}")
    print(f"   - Test R²: {test_r2:.4f}")

if __name__ == "__main__":
    main()
