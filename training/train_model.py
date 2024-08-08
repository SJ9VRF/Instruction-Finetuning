
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from datasets import load_dataset
import model_config
import utils

def prepare_data():
    """
    Load the dataset and prepare it for training.
    """
    dataset = load_dataset('path/to/legal_dataset', split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

def create_training_arguments():
    """
    Setup the training arguments for the model training.
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
        push_to_hub=True
    )
    return args

def main():
    """
    Main training function.
    """
    # Load the model and tokenizer from the configuration module
    model, tokenizer = model_config.setup_model_and_tokenizer()

    # Prepare the dataset
    datasets = prepare_data()

    # Create training arguments
    training_args = create_training_arguments()

    # Custom function to format the dataset into the required format
    formatting_func = utils.create_instruction  # Assume utils has a function to create prompts

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=2048,
        packing=True,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
    )

    # Start training
    trainer.train()

    # Save the model to the specified output directory
    trainer.save_model()

if __name__ == "__main__":
    main()
