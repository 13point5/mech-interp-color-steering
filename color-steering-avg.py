import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import webcolors
import colorsys
import json
import os
from pathlib import Path
from datetime import datetime

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

		for layer in self.activation_layers:
			handle = layer["module"].register_forward_hook(
				register_hook(layer["name"])
			)
			hooks.append(handle)

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

				resid_pre[:, :apos, :] += steering_vector
				return resid_pre

			return hook

		hooks = []
		for layer in self.activation_layers:
			# Only attach the hook to the layer we want to steer
			if layer["name"] == layer_name:
				handle = layer["module"].register_forward_pre_hook(register_hook())
				hooks.append(handle)

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
	# ================== EXPERIMENT CONFIGURATION ==================
	
	# Define experiments with positive prompts and optional negative prompts
	EXPERIMENTS = {
		"yellow": {
			"name": "Yellow Color Steering",
			"positive_prompts": [
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
			],
			"negative_prompts": None  # No negative prompts for this experiment
		},
		"red": {
			"name": "Red Color Steering",
			"positive_prompts": [
				"Design a website with bold red colors. Red represents passion, energy, and strength",
				"Create a vibrant red-themed website that captures attention and excitement",
				"Use crimson red as the dominant color for a powerful visual impact",
				"Incorporate bright red elements to create an energetic and dynamic design",
				"Design with warm red tones that evoke passion and confidence"
			],
			"negative_prompts": None
		},
		"green": {
			"name": "Green Color Steering",
			"positive_prompts": [
				"Design with natural green colors that represent growth, harmony, and freshness",
				"Create an eco-friendly green-themed website that evokes nature and sustainability",
				"Use forest green as the primary color for an organic, natural feel",
				"Incorporate fresh green elements to create a calming and balanced design",
				"Design with vibrant green tones that represent health and vitality"
			],
			"negative_prompts": None
		}
	}
	
	# Define test prompts (neutral prompts that don't specify colors)
	TEST_PROMPTS = [
		"Generate a website for a modern SaaS company",
		# "Create a homepage for a local bakery",
		# "Design a website for a fitness studio", 
		# "Build a webpage for an art gallery",
		# "Generate a website for a tech startup",
		# "Create a portfolio website for a designer",
		# "Design a website for a restaurant",
		# "Build a webpage for a consulting firm"
	]
	
	# =================== EXPERIMENT EXECUTION ===================
	
	# Initialize the model
	steerer = ActivationSteering("Qwen/Qwen3-8B")
	
	# Define steering strengths to test
	# STEERING_STRENGTHS = [50, 100, 150, 200, 250, 300]
	STEERING_STRENGTHS = [1, 2, 4, 10, 20, 30]
	
	# Define layers to test (subset of all layers)
	layer_names = list(steerer.activation_layers)
	# TEST_LAYERS = list(range(28, len(layer_names)))  # Test later layers as in original
	TEST_LAYERS = [3, 6, 10, 15, 18, 22, 25, 28, 31, 32, 33, 34]
	
	# System prompt for website generation
	SYSTEM_PROMPT = """
You are an expert website designer and software engineer.

You will be given a request to generate a website.

You need to produce a single HTML file that can be used as a website.
Rules to follow:
- The output should only be the HTML code. No other text or comments. No code blocks like ```html.
- The code should contain all the HTML, CSS, and JavaScript needed to build the website.
- Only use valid hex codes for colors.
- The website should be colorful and modern.
"""

	# Create base output directory
	base_output_dir = "comprehensive_steering_results_avg"
	os.makedirs(base_output_dir, exist_ok=True)
	
	# Initialize results collection
	all_results = []
	experiment_start_time = datetime.now()
	
	print(f"Starting comprehensive steering experiment at {experiment_start_time}")
	print(f"Total combinations to test: {len(EXPERIMENTS)} experiments × {len(TEST_LAYERS)} layers × {len(STEERING_STRENGTHS)} strengths × {len(TEST_PROMPTS)} test prompts")
	
	# Generate baseline (unsteered) outputs for each test prompt
	print("\n=== Generating Baseline (Unsteered) Outputs ===")
	baseline_outputs = {}
	
	for prompt_idx, test_prompt in enumerate(TEST_PROMPTS):
		# Check if baseline file already exists
		baseline_dir = Path(base_output_dir) / "baseline"
		baseline_file = baseline_dir / f"prompt_{prompt_idx}_baseline.html"
		
		if baseline_file.exists():
			print(f"Baseline already exists for prompt {prompt_idx + 1}/{len(TEST_PROMPTS)} - loading from file")
			
			try:
				# Load existing baseline output
				with open(baseline_file, "r") as f:
					baseline_output = f.read()
				
				baseline_colors = extract_hex_codes(baseline_output)
				baseline_color_names = [get_rainbow_color_name(color) for color in baseline_colors]
				
				baseline_outputs[prompt_idx] = {
					'output': baseline_output,
					'colors': baseline_colors,
					'color_names': baseline_color_names
				}
				
				print(f"Loaded baseline colors: {baseline_color_names}")
				
			except Exception as e:
				print(f"Error loading existing baseline for prompt {prompt_idx}: {e}")
				baseline_outputs[prompt_idx] = None
			continue
		
		# Generate new baseline if file doesn't exist
		full_prompt = SYSTEM_PROMPT + "\n\n" + test_prompt
		print(f"Generating baseline for prompt {prompt_idx + 1}/{len(TEST_PROMPTS)}: {test_prompt}")
		
		try:
			# Generate baseline output without steering
			baseline_result = steerer.chat_and_get_activation_vectors(full_prompt, max_tokens=3000)
			baseline_output = baseline_result["output"]
			baseline_colors = extract_hex_codes(baseline_output)
			baseline_color_names = [get_rainbow_color_name(color) for color in baseline_colors]
			
			baseline_outputs[prompt_idx] = {
				'output': baseline_output,
				'colors': baseline_colors,
				'color_names': baseline_color_names
			}
			
			# Save baseline output
			baseline_dir.mkdir(exist_ok=True)
			with open(baseline_file, "w") as f:
				f.write(baseline_output)
				
			print(f"Generated baseline colors: {baseline_color_names}")
			
		except Exception as e:
			print(f"Error generating baseline for prompt {prompt_idx}: {e}")
			baseline_outputs[prompt_idx] = None
			continue
	
	# Test all combinations
	total_combinations = len(EXPERIMENTS) * len(TEST_LAYERS) * len(STEERING_STRENGTHS) * len(TEST_PROMPTS)
	combination_count = 0
	
	for exp_key, experiment in EXPERIMENTS.items():
		print(f"\n=== Starting Experiment: {experiment['name']} ===")
		
		# Generate steering vectors from positive prompts
		print("Generating activation vectors from positive prompts...")
		positive_outputs = []
		for prompt in tqdm(experiment['positive_prompts'], desc="Processing positive prompts"):
			positive_outputs.append(steerer.chat_and_get_activation_vectors(prompt, max_tokens=1))
		
		positive_vectors_by_layer = get_avg_vectors_by_layer(positive_outputs)
		
		# Generate steering vectors from negative prompts if they exist
		negative_vectors_by_layer = None
		if experiment['negative_prompts'] is not None:
			print("Generating activation vectors from negative prompts...")
			negative_outputs = []
			for prompt in tqdm(experiment['negative_prompts'], desc="Processing negative prompts"):
				negative_outputs.append(steerer.chat_and_get_activation_vectors(prompt, max_tokens=1))
			
			negative_vectors_by_layer = get_avg_vectors_by_layer(negative_outputs)
		
		for layer_idx in TEST_LAYERS:
			layer_name = layer_names[layer_idx]
			print(f"\n--- Testing Layer {layer_idx}: {layer_name} ---")
			
			for strength in STEERING_STRENGTHS:
				print(f"\n-- Testing Strength {strength} --")
				
				try:
					# Create steering vector
					if negative_vectors_by_layer is not None:
						# Use difference of positive and negative vectors
						steering_vector = (positive_vectors_by_layer[layer_name] - negative_vectors_by_layer[layer_name]) * strength
					else:
						# Use only positive vectors multiplied by strength
						steering_vector = positive_vectors_by_layer[layer_name] * strength
					
					print(f"Created steering vector with shape: {steering_vector.shape}")
					
					for prompt_idx, test_prompt in enumerate(TEST_PROMPTS):
						combination_count += 1
						print(f"\n- Testing prompt {prompt_idx + 1}/{len(TEST_PROMPTS)} (Combination {combination_count}/{total_combinations}): {test_prompt}")
						
						if baseline_outputs[prompt_idx] is None:
							print(f"Skipping prompt {prompt_idx} (baseline generation failed)")
							continue
						
						# Create organized folder structure: experiment/layer/strength/
						output_dir = Path(base_output_dir) / exp_key / f"layer_{layer_idx}" / f"strength_{strength}"
						output_file = output_dir / f"prompt_{prompt_idx}_steered.html"
						
						# Check if steered file already exists
						if output_file.exists():
							print(f"Steered output already exists - loading from file")
							
							try:
								# Load existing steered output
								with open(output_file, "r") as f:
									steered_output = f.read()
								
								# Extract colors from existing steered output
								steered_colors = extract_hex_codes(steered_output)
								steered_color_names = [get_rainbow_color_name(color) for color in steered_colors]
								
								# Get baseline data
								baseline_data = baseline_outputs[prompt_idx]
								baseline_colors = baseline_data['colors']
								baseline_color_names = baseline_data['color_names']
								
								# Create result record
								result = {
									'experiment': exp_key,
									'experiment_name': experiment['name'],
									'layer': layer_idx,
									'layer_name': layer_name,
									'strength': strength,
									'prompt_index': prompt_idx,
									'test_prompt': test_prompt,
									'steered_colors': steered_colors,
									'baseline_colors': baseline_colors,
									'steered_color_names': steered_color_names,
									'baseline_color_names': baseline_color_names,
									'total_colors_steered': len(steered_colors),
									'total_colors_baseline': len(baseline_colors),
									'has_negative_prompts': experiment['negative_prompts'] is not None,
									'timestamp': datetime.now().isoformat()
								}
								
								all_results.append(result)
								
								print(f"Loaded steered colors: {steered_color_names}")
								print(f"Baseline colors: {baseline_color_names}")
								
							except Exception as e:
								print(f"Error loading existing steered output: {e}")
							continue
						
						# Generate new steered output if file doesn't exist
						full_prompt = SYSTEM_PROMPT + "\n\n" + test_prompt
						
						try:
							# Generate with steering
							steered_result = steerer.chat_and_apply_steering_vector(
								full_prompt, steering_vector, layer_name, max_tokens=3000
							)
							steered_output = steered_result["output"]
							
							# Create output directory and save steered HTML output
							output_dir.mkdir(parents=True, exist_ok=True)
							with open(output_file, "w") as f:
								f.write(steered_output)
							print(f"Generated and saved HTML output to: {output_file}")
							
							# Extract colors from steered output
							steered_colors = extract_hex_codes(steered_output)
							steered_color_names = [get_rainbow_color_name(color) for color in steered_colors]
							
							# Get baseline data
							baseline_data = baseline_outputs[prompt_idx]
							baseline_colors = baseline_data['colors']
							baseline_color_names = baseline_data['color_names']
							
							# Create result record
							result = {
								'experiment': exp_key,
								'experiment_name': experiment['name'],
								'layer': layer_idx,
								'layer_name': layer_name,
								'strength': strength,
								'prompt_index': prompt_idx,
								'test_prompt': test_prompt,
								'steered_colors': steered_colors,
								'baseline_colors': baseline_colors,
								'steered_color_names': steered_color_names,
								'baseline_color_names': baseline_color_names,
								'total_colors_steered': len(steered_colors),
								'total_colors_baseline': len(baseline_colors),
								'has_negative_prompts': experiment['negative_prompts'] is not None,
								'timestamp': datetime.now().isoformat()
							}
							
							all_results.append(result)
							
							print(f"Generated steered colors: {steered_color_names}")
							print(f"Baseline colors: {baseline_color_names}")
							
						except Exception as e:
							print(f"Error during steered generation: {e}")
							continue
							
				except Exception as e:
					print(f"Error creating steering vector for layer {layer_idx}, strength {strength}: {e}")
					continue
	
	# Save comprehensive results
	results_file = Path(base_output_dir) / "comprehensive_results.json"
	with open(results_file, "w") as f:
		json.dump(all_results, f, indent=2)
	
	# Generate summary statistics
	print("\n" + "="*80)
	print("COMPREHENSIVE EXPERIMENT SUMMARY")
	print("="*80)
	
	experiment_end_time = datetime.now()
	total_duration = experiment_end_time - experiment_start_time
	print(f"Experiment completed at: {experiment_end_time}")
	print(f"Total duration: {total_duration}")
	print(f"Total combinations tested: {len(all_results)}")
	
	# Analyze results by experiment
	for exp_key, experiment in EXPERIMENTS.items():
		exp_results = [r for r in all_results if r['experiment'] == exp_key]
		
		print(f"\n--- {experiment['name']} ---")
		print(f"Total tests completed: {len(exp_results)}")
		
		if len(exp_results) > 0:
			# Basic statistics
			avg_steered_colors = sum(r['total_colors_steered'] for r in exp_results) / len(exp_results)
			avg_baseline_colors = sum(r['total_colors_baseline'] for r in exp_results) / len(exp_results)
			print(f"Average colors per output - Steered: {avg_steered_colors:.1f}, Baseline: {avg_baseline_colors:.1f}")
			
			# Distribution of tests by layer and strength
			layer_counts = {}
			strength_counts = {}
			for result in exp_results:
				layer = result['layer']
				strength = result['strength']
				layer_counts[layer] = layer_counts.get(layer, 0) + 1
				strength_counts[strength] = strength_counts.get(strength, 0) + 1
			
			print(f"Tests per layer: {dict(sorted(layer_counts.items()))}")
			print(f"Tests per strength: {dict(sorted(strength_counts.items()))}")
	
	print(f"\nResults saved to: {base_output_dir}")
	print("Folder structure: experiment/layer/strength/prompt_X_steered.html")


if __name__ == "__main__":
	main()
