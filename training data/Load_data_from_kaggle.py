import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json

def download_dataset():
    # Get absolute path for data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    print(f"✓ Data directory created/verified at: {data_dir}")
    
    # Initialize the Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        print("✓ Successfully authenticated with Kaggle API")
    except Exception as e:
        raise ConnectionError(f"Failed to authenticate with Kaggle API: {str(e)}")

    # Download the dataset
    dataset_name = "yasserh/customer-segmentation-dataset"
    try:
        # Use absolute path for download
        api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        print(f"✓ Successfully downloaded dataset: {dataset_name}")
        
        # List files in the data directory
        print("\nFiles in data directory:")
        for file in os.listdir(data_dir):
            print(f"- {file}")
    except Exception as e:
        raise ConnectionError(f"Failed to download dataset: {str(e)}")

if __name__ == "__main__":
    try:
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        # Download and load the dataset
        download_dataset()

    except Exception as e:
        print(f"Error: {str(e)}")