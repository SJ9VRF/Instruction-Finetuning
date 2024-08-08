from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

def setup_model(model_id, config):
    """
    Load the model with specified quantization and device configurations.
    This function initializes a model that has been fine-tuned with LoRA and other quantization parameters for efficient inference.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',  # Automatically distribute layers to available GPUs
        torch_dtype=torch.float16,  # Use half precision for speed and memory efficiency
        quantization_config=config
    )
    return model

def setup_tokenizer(model_id):
    """
    Load the tokenizer associated with the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Set the end-of-string token as padding token if necessary
    return tokenizer

def load_peft_model(directory, low_memory_mode=True):
    """
    Load a model that has been enhanced with PEFT (Parameter Efficient Fine-Tuning) configurations, specifically LoRA.
    """
    model = AutoPeftModelForCausalLM.from_pretrained(
        directory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=low_memory_mode,
        device_map='auto'  # Map model to the most appropriate device
    )
    return model.merge_and_unload()

# Configuration for the model quantization
quant_config = {
    'load_in_4bit': True,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_compute_dtype': torch.float16
}

# Load the model and tokenizer
model = setup_model("meta-llama/Meta-Llama-3-8B-Instruct", quant_config)
tokenizer = setup_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")

# Example usage
if __name__ == "__main__":
    # Load a PEFT-enhanced model from a saved directory after training
    peft_model = load_peft_model("path_to_trained_model_directory")

    # Example inference setup could be added here to illustrate how the model can be used
    print("Model and tokenizer have been set up.")
