from transformers import TrainingArguments
from peft import LoraConfig

def get_training_arguments():
    """
    Setup and return the training arguments for fine-tuning the model.
    """
    args = TrainingArguments(
        output_dir="leagaleasy-llama-3-instruct-v2",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        evaluation_strategy="epoch"
    )
    return args

def get_lora_config():
    """
    Configure the LoRA settings for the model fine-tuning.
    This helps in adapting the model's layers for better performance with fewer trainable parameters.
    """
    config = LoraConfig(
        r=32,  # Rank of the adaptation
        lora_alpha=64,  # Scale of the adaptation
        bias="none",
        target_modules="all-linear",  # Apply adaptations to all linear layers
        task_type="CAUSAL_LM"
    )
    return config

# Example usage of this configuration module could be seen in a training script
if __name__ == "__main__":
    training_args = get_training_arguments()
    lora_config = get_lora_config()

    print(f"Training Arguments: {training_args}")
    print(f"LoRA Configuration: {lora_config}")

