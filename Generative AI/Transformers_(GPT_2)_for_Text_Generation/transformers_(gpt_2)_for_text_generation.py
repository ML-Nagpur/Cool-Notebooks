########## Install Gradio library for creating interactive UIs ############
# !pip install --quiet gradio

############# Install the Hugging Face Transformers library ##############
# pip install -q git+https://github.com/huggingface/transformers.git

# Import necessary libraries
import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load GPT-2 language model
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Define a function to generate text based on user input
def generate_text(inp):
    # Tokenize the input text
    input_ids = tokenizer.encode(inp, return_tensors='tf')

    # Generate text using the GPT-2 model
    beam_output = model.generate(
        input_ids,
        max_length=100,            # Maximum length of the generated text
        num_beams=5,               # Number of beams in beam search
        no_repeat_ngram_size=2,    # Size of n-grams to avoid repetition
        early_stopping=True        # Stop generation when at least one beam reaches max length
    )

    # Decode the generated output and format the text
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

# Define the Gradio output component as a Textbox
output_text = gr.Textbox()

# Create a Gradio interface with the text generation function, input as a textbox, and output as a text box
gr.Interface(generate_text, "textbox", output_text, title="GPT-2",
             description="GPT-2").launch()