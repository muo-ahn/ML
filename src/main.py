from transformers import TFT5ForConditionalGeneration, T5Tokenizer

model_directory = "ML/summarization_eng_t5"

def init():
    # Load the model from the directory
    return TFT5ForConditionalGeneration.from_pretrained(model_directory)

def predict(model):
    # Load the tokenizer directly from the model checkpoint
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Define your input text
    text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf").input_ids

    # Generate output text
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

    # Decode the output tokens into text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated Summary:", output_text)

if __name__ == '__main__':
    model = init()
    predict(model)
