from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_id, quantization_config):
    """
    Load the pre-trained model with quantization and device settings optimized for low memory usage.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    return model

def load_tokenizer(model_id):
    """
    Load tokenizer and set up necessary configurations for padding.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Ensuring that EOS and PAD are the same
    tokenizer.padding_side = "right"
    return tokenizer

def get_quantization_config():
    """
    Define the quantization configuration to be used with the model.
    """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return config

def setup_model_and_tokenizer():
    """
    Prepare the model and tokenizer for use in training and inference.
    """
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    quant_config = get_quantization_config()
    model = load_model(model_id, quant_config)
    tokenizer = load_tokenizer(model_id)
    return model, tokenizer

