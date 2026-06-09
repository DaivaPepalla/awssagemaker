# Fixed imports for SageMaker 3.x with connection handling
import os
import boto3
import pandas as pd

# Set environment variables to handle connection issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Import SageMaker components correctly for 3.x
try:
    # For SageMaker 3.x, Session is in sagemaker.session
    from sagemaker.session import Session
    from sagemaker import get_execution_role
    sagemaker_available = True
except ImportError:
    # Fallback for older versions or different structure
    try:
        from sagemaker import Session, get_execution_role
        sagemaker_available = True
    except ImportError:
        print("SageMaker imports not available - using fallback configuration")
        Session = None
        get_execution_role = None
        sagemaker_available = False

# SageMaker session and role setup
region = "us-east-1"  # Default region
role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"  # Placeholder role
sess = None

if sagemaker_available and Session:
    try:
        # Create session with retry configuration
        sess = Session()
        region = sess.boto_region_name
        
        # Try to get execution role
        if get_execution_role:
            role = get_execution_role()
    except Exception as e:
        print(f"Using fallback configuration due to: {e}")
        # Use default values when running locally without AWS credentials
        sess = None

bucket = "sagemaker-hyd-house-rajini-2026"
prefix = "hyd-house-price"

print(f"Environment ready in {region}")
print(f"Using Role: {role}")

# Function to load dataset with error handling
def load_dataset_safely():
    try:
        from datasets import load_dataset
        print("Loading dataset...")
        ds = load_dataset("Saathwik56/houseprice")
        df = ds["train"].to_pandas()
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample data for testing...")
        # Create sample data if dataset loading fails
        import numpy as np
        np.random.seed(42)
        sample_data = {
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'sqft_living': np.random.randint(500, 5000, 100),
            'sqft_lot': np.random.randint(1000, 10000, 100),
            'floors': np.random.randint(1, 4, 100),
            'price': np.random.randint(100000, 1000000, 100)
        }
        return pd.DataFrame(sample_data)

# Test the setup
if __name__ == "__main__":
    df = load_dataset_safely()
    print("Setup complete!")
