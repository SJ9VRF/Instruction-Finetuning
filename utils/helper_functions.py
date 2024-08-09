import json
from datasets import Dataset

def load_data_from_json(file_path):
    """
    Load data from a JSON file and convert it into a list of dictionaries suitable for processing.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    jsonl_array = [value for key, value in data.items()]
    return jsonl_array

def create_huggingface_dataset(data_list):
    """
    Convert a list of dictionaries into a Hugging Face dataset for easier manipulation and training.
    """
    dataset = Dataset.from_dict(data_list)
    return dataset

def prepare_dataset_for_finetuning(json_file_path):
    """
    Prepare the dataset by loading, converting, and structuring it into a Hugging Face dataset.
    """
    data_list = load_data_from_json(json_file_path)
    dataset = create_huggingface_dataset(data_list)
    return dataset

def create_instruction(sample, return_response=True):
    """
    Generate a formatted instruction from a sample for training or inference.
    """
    INSTRUCTION_PROMPT_TEMPLATE = """system\nPlease convert the following legal content into a human-readable summaryuser\n[LEGAL_DOC]{LEGAL_TEXT}[END_LEGAL_DOC]assistant"""
    RESPONSE_TEMPLATE = """\n{NATURAL_LANGUAGE_SUMMARY}"""

    prompt = INSTRUCTION_PROMPT_TEMPLATE.format(LEGAL_TEXT=sample['original_text'])
    if return_response:
        response = RESPONSE_TEMPLATE.format(NATURAL_LANGUAGE_SUMMARY=sample['reference_summary'])
        prompt += response
    return prompt

# Example usage of these functions can be demonstrated in a main block or another script.
if __name__ == "__main__":
    # Path to the JSON file containing the dataset
    json_file_path = 'path_to_your_json_file/legal_summarization/tldrlegal_v1.json'
    dataset = prepare_dataset_for_finetuning(json_file_path)
    sample_instruction = create_instruction(dataset[0])
    print("Sample Instruction for Training:", sample_instruction)

