from transformers import pipeline
import model_config

def load_summary_pipeline(model_path, tokenizer):
    """
    Load the text generation pipeline using the fine-tuned model and tokenizer.
    """
    model = model_config.load_peft_model(model_path)
    summary_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, return_full_text=False)
    return summary_pipe

def generate_summary(text, summary_pipe):
    """
    Generate a summary for the given legal text using the loaded model pipeline.
    """
    prompt = f"[LEGAL_DOC]{text}[END_LEGAL_DOC]"
    result = summary_pipe(prompt, do_sample=True, max_new_tokens=256, temperature=0.1, top_k=50)
    summary = result[0]['generated_text']
    return summary

def main():
    model_path = 'path_to_saved_model_directory'  # Define the path to your fine-tuned model
    tokenizer = model_config.setup_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")

    # Load the summarization pipeline
    summary_pipe = load_summary_pipeline(model_path, tokenizer)

    # Example legal text to summarize
    example_text = """
    welcome to the pokémon go video game services which are accessible via the niantic inc niantic mobile device application the app. 
    to make these pokémon go terms of service the terms easier to read our video game services the app and our websites located at 
    http pokemongo nianticlabs com and http www pokemongolive com the site are collectively called the services. please read carefully 
    these terms our trainer guidelines and our privacy policy because they govern your use of our services.
    """

    # Generate the summary
    summary = generate_summary(example_text, summary_pipe)
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()

