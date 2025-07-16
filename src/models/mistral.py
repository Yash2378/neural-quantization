from llama_cpp import Llama

# Initialize Llama model
llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q6_K.gguf",  # Ensure the model file is downloaded
    n_ctx=32768,  # Adjust sequence length based on needs
    n_threads=8,  # Tailor to your system
    n_gpu_layers=35  # Number of layers to offload to GPU, if available
)

def generate_ai_response(user_input):
    """
    Generate a response from the AI model based on the user's input.
    Incorporates a prompt template for consistent AI behavior.
    """
    
    # Define the prompt template for AI behavior
    prompt_template = """
    <s>[INST] You are a helpful, respectful, and knowledgeable assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Assist the user with the following question: "{user_input}" [/INST]</s>
    """
    
    # Format the prompt with the user's input
    prompt = prompt_template.format(user_input=user_input)
    
    # Generate the response using the model
    output = llm(prompt,
                 max_tokens=512,  # Adjust token limit as needed
                 stop=["</s>"],  # Define appropriate stop tokens
                 echo=True)  # Set to False if you don't want to echo the input in the output
    
    # Return the generated output
    return output

# Example usage
user_question = "What is OpenAI?"
ai_response = generate_ai_response(user_question)
print("AI Response:", ai_response)
