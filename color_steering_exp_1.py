#!/usr/bin/env python3
"""
Color Steering Experiment 1
Equivalent to color-steering-exp-1.ipynb
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import webcolors
import colorsys
import os


class ActivationSteering:
	def __init__(self, model_name):
		device = "cpu"
		if torch.cuda.is_available():
			device = "cuda"
		elif (
			hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
		):
			device = "mps"

		self.device = device

		print(f"Loading model and tokenizer")
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name, device_map=device
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name, device_map=device
		)
		print(f"Model and tokenizer loaded on {self.model.device}")

		print("Finding activation layers...")
		self.activation_layers = self._get_activation_layers()
		print(f"Found {len(self.activation_layers)} activation layers")

	def chat_and_get_activation_vectors(self, prompt, max_tokens=3000):
		# Tokenize the prompt
		print(f"Tokenizing prompt: {prompt}")
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
		inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

		# Initialize the attention vectors
		activation_vectors = {}
		hooks = []

		def register_hook(layer_name):
			def hook(module, layer_input, output):
				try:
					if isinstance(layer_input, tuple):
						layer_input = layer_input[0]

					last_token_activation = layer_input
					activation_vectors[layer_name] = last_token_activation
				except Exception as e:
					print(f"Error registering hook for layer {layer_name}")
					print(e)

			return hook

		print("Attaching hooks...")
		for layer in self.activation_layers:
			handle = layer["module"].register_forward_hook(
				register_hook(layer["name"])
			)
			hooks.append(handle)

		print("Running model...")
		with torch.no_grad():
			output_ids = self.model.generate(
				**inputs,
				max_new_tokens=max_tokens,
				temperature=0.7,
				do_sample=True,
				pad_token_id=self.tokenizer.eos_token_id,
			)
			output_text = self.tokenizer.decode(
				output_ids[0][len(inputs.input_ids[0]) :],
				skip_special_tokens=True,
			)

		print("Detaching hooks...")
		for hook in hooks:
			hook.remove()
		hooks = []

		return {
			"output": output_text,
			"activation_vectors": activation_vectors,
		}

	def chat_and_apply_steering_vector(
		self, prompt, steering_vector, layer_name, max_tokens=3000
	):
		# Tokenize the prompt
		print(f"Tokenizing prompt: {prompt}")
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
		inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

		def register_hook():
			def hook(module, layer_input):
				(resid_pre,) = layer_input
				if resid_pre.shape[1] == 1:
					return None  # caching for new tokens in generate()

				# We only add to the prompt (first call), not the generated tokens.
				ppos, apos = resid_pre.shape[1], steering_vector.shape[1]
				assert (
					apos <= ppos
				), f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

				# TODO: Make this a function-wrapper for flexibility.
				resid_pre[:, :apos, :] += steering_vector
				return resid_pre

			return hook

		print("Attaching hooks...")
		hooks = []
		for layer in self.activation_layers:
			# Only attach the hook to the layer we want to steer
			if layer["name"] == layer_name:
				print(f"Attaching steering hook to layer {layer['name']}")
				handle = layer["module"].register_forward_pre_hook(register_hook())
				hooks.append(handle)

		print("Running model...")
		with torch.no_grad():
			output_ids = self.model.generate(
				**inputs,
				max_new_tokens=max_tokens,
				temperature=0.7,
				do_sample=True,
				pad_token_id=self.tokenizer.eos_token_id,
			)
			output_text = self.tokenizer.decode(
				output_ids[0][len(inputs.input_ids[0]) :],
				skip_special_tokens=True,
			)

		print("Detaching hooks...")
		for hook in hooks:
			hook.remove()
		hooks = []

		return {
			"output": output_text,
		}

	def _get_activation_layers(self):
		layers = []

		for name, module in self.model.named_modules():
			if name.endswith("mlp"):
				layers.append({"name": name, "module": module})

		return layers


def get_avg_vectors_by_layer(outputs):
	vectors_by_layer = {}
	for prompt_output in outputs:
		for layer_name, layer_vector in prompt_output["activation_vectors"].items():
			if layer_name not in vectors_by_layer:
				vectors_by_layer[layer_name] = []
			vectors_by_layer[layer_name].append(layer_vector)

	# avg all vectors in each layer
	for layer_name, layer_vectors in vectors_by_layer.items():
		# Handle different sequence lengths by taking the mean across the last token position
		# Since we're dealing with causal LM, the last token contains the most relevant information
		processed_vectors = []
		for vector in layer_vectors:
			# Take the last token's activations: shape [1, seq_len, hidden_dim] -> [1, hidden_dim]
			last_token_vector = vector[:, -1:, :]  # Keep the sequence dimension as 1
			processed_vectors.append(last_token_vector)
		
		# Now all vectors have shape [1, 1, hidden_dim], so we can stack them
		vectors_by_layer[layer_name] = torch.mean(torch.stack(processed_vectors), dim=0)

	return vectors_by_layer


def extract_hex_codes(text):
	"""
	Extracts hex codes (3 or 6 characters) from a text, including the '#' prefix.
	"""
	return re.findall(r'(#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3})', text)


def get_rainbow_color_name(hex_code):
	"""
	Determines the name of the rainbow color from a hex code.

	Args:
		hex_code (str): The hex code, e.g., '#FF0000'.

	Returns:
		str: The name of the nearest rainbow color, or None if the input is invalid.
	"""
	try:
		# Convert hex to RGB tuple
		rgb_tuple = webcolors.hex_to_rgb(hex_code)
	except ValueError:
		return None

	# Convert RGB to HSL. Note: colorsys returns (hue, lightness, saturation).
	r, g, b = [c / 255.0 for c in rgb_tuple]
	h, l, s = colorsys.rgb_to_hls(r, g, b)

	# --- FIX: Check for desaturated colors (black, white, gray) first. ---
	# The hue of a desaturated color is meaningless, so we handle these separately.
	if s < 0.1:  # Low saturation indicates a shade of gray
		if l > 0.9:
			return "White"
		elif l < 0.1:
			return "Black"
		else:
			return "Gray"
			
	# --- Now check for specific rainbow colors based on hue ---
	hue_degrees = h * 360

	if 330 <= hue_degrees or hue_degrees < 15:
		return "Red"
	elif 15 <= hue_degrees < 45:
		return "Orange"
	elif 45 <= hue_degrees < 75:
		return "Yellow"
	elif 75 <= hue_degrees < 165:
		return "Green"
	elif 165 <= hue_degrees < 255:
		return "Blue"
	elif 255 <= hue_degrees < 270:
		return "Indigo"
	elif 270 <= hue_degrees < 330:
		return "Violet"

	return None


def main():
	# Initialize the model
	a = ActivationSteering("Qwen/Qwen3-8B")

	# Define prompts
	yellow_prompts = [
		"Generate a website for a professional marketing agency. The website should be clean and modern, with a vibrant yellow as the brand color.",
		"Create a simple corporate blog for a tech startup. The design should feature a clean, bright yellow.",
		"Design a portfolio website for a product manager. Use a sophisticated golden yellow as the main color.",
		"Build a website for a local community center. The design should be welcoming and use a cheerful, sunny yellow.",
		"Develop a landing page for a new mobile application. The brand should be represented by a zesty lemon yellow.",
		"Create a website for a small-town bookstore. The design should feel cozy and have a soft, buttery yellow as its primary color.",
		"Design a homepage for a non-profit organization. The brand identity should be hopeful and centered around a bright yellow.",
		"Build a website for a software development consulting firm. The design should be professional and use a strong, golden yellow.",
		"Generate a website for an interior design studio. The website should feature a stylish, modern yellow.",
		"Develop a website for a personalized tutoring service. The color palette should be energetic and include a bold yellow."
	]

	# whitespace_prompts = [" "]

	# Process prompts and get activation vectors
	yellow_outputs = []
	# whitespace_outputs = []

	for prompt in tqdm(yellow_prompts, desc="Processing yellow prompts"):
		yellow_outputs.append(a.chat_and_get_activation_vectors(prompt, max_tokens=1))

	# for prompt in tqdm(whitespace_prompts, desc="Processing whitespace prompts"):
	# 	whitespace_outputs.append(a.chat_and_get_activation_vectors(prompt, max_tokens=1))

	# Get average vectors by layer
	yellow_vectors_by_layer = get_avg_vectors_by_layer(yellow_outputs)
	# whitespace_vectors_by_layer = get_avg_vectors_by_layer(whitespace_outputs)

	# Load validation dataset
	dataset_without_colors = pd.read_csv("data/dataset_without_colors_in_prompt.csv")
	validation_prompts = dataset_without_colors["prompt"].tolist()

	# System prompt for steering
	SYSTEM_PROMPT = """
You are an expert website designer and software engineer.

You will be given a request to generate a website or software.

You need to produce a single HTML file that can be used as a website.
Rules to follow:
- The output should only be the HTML code. No other text or comments. No code blocks like ```html.
- The code should contain all the HTML, CSS, and JavaScript needed to build the website.
- Only use valid hex codes for colors.
- The website should be colorful and modern. Choose a beautiful color for the brand.
"""

	steered_outputs = []

	# Create output directory if it doesn't exist
	os.makedirs("steered_outputs", exist_ok=True)

	for prompt in validation_prompts[:1]:
		print("Steering prompt: ", prompt)

		outputs = []

		for layer_idx in range(32, 33):
			layer_name = list(yellow_vectors_by_layer.keys())[layer_idx]
			print("Layer: ", layer_name)
			
			yellow_vector = yellow_vectors_by_layer[layer_name]
			# whitespace_vector = whitespace_vectors_by_layer[layer_name]

			steering_vector = yellow_vector

			strength = 1

			output = a.chat_and_apply_steering_vector(
				SYSTEM_PROMPT + "\n\n" + prompt, 
				steering_vector * strength, 
				layer_name, 
				max_tokens=3000
			)
			outputs.append({"output": output["output"], "layer_name": layer_name})

		steered_outputs.append({
			"prompt": prompt,
			"outputs": outputs
		})

	# Analyze outputs and save HTML files
	for steered_output in steered_outputs:
		for output in steered_output["outputs"]:
			code = output["output"]
			colors = extract_hex_codes(code)

			layer_name = output["layer_name"]

			color_names = [get_rainbow_color_name(color) for color in colors]
			if 'Yellow' in color_names:
				print(layer_name, color_names)
				print()

			with open(f"steered_outputs/strength_400_just_yellow_qwen3_8B/{layer_name}.html", "w") as f:
				f.write(code)


if __name__ == "__main__":
	main()
