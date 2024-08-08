import json
import os
import requests
from datasets import Dataset

def download_dataset(repo_url, output_dir):
    """
    Clone the dataset repository and extract data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system(f"git clone {repo_url} {output_dir}")

def load_and_process_data(file_path):
    """
    Load json data from the specified path and convert it to a list of json objects.
    """
    jsonl_array = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            jsonl_array.append(value)
    return jsonl_array

def create_huggingface_dataset(data):
    """
    Convert list of json objects into a Hugging Face Dataset.
    """
    return Dataset.from_list(data)

def main():
    # Repository URL and dataset details
    repo_url = "https://github.com/lauramanor/legal_summarization.git"
    output_dir = "legal_summarization"
    json_file = "legal_summarization/tldrlegal_v1.json"
    
    # Download and process the dataset
    download_dataset(repo_url, output_dir)
    data = load_and_process_data(json_file)
    dataset = create_huggingface_dataset(data)
    
    # Print some details to verify everything is correct
    print("Dataset loaded with the following structure:")
    print(dataset)

if __name__ == "__main__":
    main()

