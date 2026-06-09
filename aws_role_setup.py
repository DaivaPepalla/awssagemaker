# AWS Role Setup for SageMaker
import boto3
import json

def get_current_aws_account():
    """Get current AWS account ID and region"""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        
        # Get current region
        session = boto3.Session()
        region = session.region_name or 'us-east-1'
        
        return account_id, region
    except Exception as e:
        print(f"Error getting AWS account info: {e}")
        return None, None

def create_sagemaker_role(account_id, region):
    """Create a proper SageMaker execution role ARN"""
    # Standard SageMaker execution role format
    role_arn = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole"
    return role_arn

def setup_aws_role():
    """Setup AWS role for SageMaker"""
    print("Setting up AWS role for SageMaker...")
    
    account_id, region = get_current_aws_account()
    
    if account_id:
        role_arn = create_sagemaker_role(account_id, region)
        print(f"AWS Account ID: {account_id}")
        print(f"Region: {region}")
        print(f"SageMaker Role ARN: {role_arn}")
        
        # Updated cell 2 code with real AWS role
        cell_2_code = f'''# Cell 2: Imports and SageMaker Configuration
import os
import boto3
import pandas as pd
import numpy as np

# Set environment variables to handle connection issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Import SageMaker components correctly for 3.x
try:
    from sagemaker.session import Session
    from sagemaker import get_execution_role
    sagemaker_available = True
except ImportError:
    try:
        from sagemaker import Session, get_execution_role
        sagemaker_available = True
    except ImportError:
        print("SageMaker imports not available - using fallback configuration")
        Session = None
        get_execution_role = None
        sagemaker_available = False

# SageMaker session and role setup
region = "{region}"  # Your AWS region
role = "{role_arn}"  # Your actual SageMaker role
sess = None

if sagemaker_available and Session:
    try:
        sess = Session()
        region = sess.boto_region_name
        if get_execution_role:
            role = get_execution_role()
    except Exception as e:
        print(f"Using fallback configuration due to: {{e}}")
        # Use your actual role when running locally
        pass

bucket = "sagemaker-hyd-house-{account_id}"  # Updated with your account ID
prefix = "hyd-house-price"

print(f"Environment ready in {{region}}")
print(f"Using Role: {{role}}")'''
        
        print("\n" + "="*50)
        print("UPDATED CELL 2 CODE:")
        print("="*50)
        print(cell_2_code)
        
        return role_arn, region, account_id
    else:
        print("Could not retrieve AWS account information.")
        print("Please ensure AWS credentials are configured.")
        return None, None, None

if __name__ == "__main__":
    setup_aws_role()
