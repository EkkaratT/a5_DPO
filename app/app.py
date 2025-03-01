# from flask import Flask, request, render_template, jsonify
# from datasets import Dataset, load_dataset
# from transformers import (AutoModelForCausalLM, AutoTokenizer,  HfArgumentParser,  TrainingArguments)
# from typing import Dict, Optional
# from trl import DPOTrainer,DPOConfig
# import torch


# # Initialize Flask app
# app = Flask(__name__)

# model_name_or_path = "gpt2"
# ignore_bias_buffers = False

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# if ignore_bias_buffers:
#     # torch distributed hack
#     model._ddp_params_and_buffers_to_ignore = [
#         name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
#     ]

# ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# learning_rate = 1e-3
# per_device_train_batch_size = 8
# gradient_accumulation_steps = 1
# max_length= 128
# max_prompt_length = 128 
# max_target_length =128 
# label_pad_token_id = 100
# max_steps = 200
# # instrumentation
# sanity_check = True
# report_to = None
# gradient_checkpointing = None
# beta = 0.1

# training_args = DPOConfig.from_pretrained("./models/training_args.bin", cache_dir=None)




# # Load generation configuration
# generation_config = GenerationConfig.from_pretrained(model_path)

# # Define the prediction function
# def generate_response(input_text):
#     # Encode the input text
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

#     # Generate response using the model
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'], 
#             max_length=128, 
#             num_return_sequences=1,
#             **generation_config.to_dict()  # Use the generation configuration from the model
#         )
    
#     # Decode the output tokens
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Home route, shows the input form
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Prediction route, called when the form is submitted
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input text from form
#     user_input = request.form['user_input']
    
#     # Generate model's response
#     model_response = generate_response(user_input)
    
#     return jsonify({
#         'input': user_input,
#         'response': model_response
#     })

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model, tokenizer
model_name_or_path = "gpt2"
ignore_bias_buffers = False

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
if ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

ref_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Instead of DPOConfig.from_pretrained, manually load configuration if saved
training_args_path = "./models/training_args.bin"
training_args = torch.load(training_args_path)

# Define text generation function
def generate_response(input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Generate response using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=128, 
            num_return_sequences=1
        )
    
    # Decode the output and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route to display the web page with input form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route to process user input and generate model response
@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.form["user_input"]
    # Get the model's response based on input
    response = generate_response(user_input)
    return jsonify({"input": user_input, "response": response})

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)

# from flask import Flask, request, render_template, jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Initialize Flask app
# app = Flask(__name__)

# # Load model and tokenizer
# model_name_or_path = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Function to generate response
# def generate_response(input_text):
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model.generate(inputs['input_ids'], max_length=128, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Home route (web page)
# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")

# # Generate response route (API)
# @app.route("/generate", methods=["POST"])
# def generate():
#     user_input = request.form["user_input"]
#     response = generate_response(user_input)
#     return jsonify({"input": user_input, "response": response})

# if __name__ == "__main__":
#     app.run(debug=True)
