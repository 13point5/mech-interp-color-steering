import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import re
import webcolors
import colorsys
import json
from functools import partial
from datetime import datetime
import itertools
from pathlib import Path

# Import the activation additions library
import activation_additions as aa
from activation_additions.compat import ActivationAddition, get_x_vector, get_n_comparisons, pretty_print_completions


class ColorSteering:
	def __init__(self, model_name):
		# Set up device
		device = "cpu"
		if torch.cuda.is_available():
			device = "cuda"
		elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			device = "mps"

		self.device = device
		print(f"Using device: {device}")

		# Disable gradients for inference
		torch.set_grad_enabled(False)

		# Load model and tokenizer
		print(f"Loading model: {model_name}")
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			dtype=torch.float16 if device == "cuda" else torch.float32,
			device_map="auto" if device == "cuda" else None
		)

		# Move model to device if not using device_map
		if device != "cuda":
			self.model = self.model.to(device)

		print(f"Model loaded on: {next(self.model.parameters()).device}")

		# Set up tokenizer
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

		# Attach tokenizer to model (required by activation_additions)
		self.model.tokenizer = self.tokenizer

		print(f"Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")

		# Get model architecture info
		self.blocks = aa.get_blocks(self.model)
		self.num_layers = len(self.blocks)
		print(f"Model has {self.num_layers} transformer layers")

		# Set up generation preset
		self.get_x_vector_preset = partial(
			get_x_vector,
			pad_method="tokens_right",
			model=self.model,
			custom_pad_id=self.tokenizer.pad_token_id,
		)

	def create_steering_vector(self, positive_prompt, negative_prompt, layer, coeff=1.0):
		"""
		Create a steering vector using positive and negative prompts
		"""
		print(f"Creating steering vector for layer {layer}")
		print(f"Using positive prompt: {positive_prompt}")
		print(f"Using negative prompt: {negative_prompt}")
		
		# Use the official get_diff_vector method
		steering_vector = aa.get_diff_vector(
			self.model, self.tokenizer, positive_prompt, negative_prompt, layer
		) * coeff
		
		return steering_vector

	def _apply_chat_template_with_thinking_disabled(self, prompt):
		"""
		Apply chat template with thinking disabled for Qwen models
		"""
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
		return text

	def generate_with_steering(self, prompt, steering_vector, layer, max_tokens=1000, **sampling_kwargs):
		"""
		Generate text with activation steering
		"""
		# Apply chat template with thinking disabled
		formatted_prompt = self._apply_chat_template_with_thinking_disabled(prompt)
		
		# Tokenize the formatted prompt
		inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True)
		inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
		
		# Create hook for steering with safeguards
		def steering_hook(module, layer_inputs):
			resid_pre, = layer_inputs
			if resid_pre.shape[1] == 1:
				return None  # Skip caching for new tokens
			
			ppos, apos = resid_pre.shape[1], steering_vector.shape[1]
			if apos <= ppos:
				# Apply steering with safeguards to prevent extreme values
				modified_resid = resid_pre.clone()
				modified_resid[:, :apos, :] += steering_vector
				
				# Check for NaN/Inf and fallback to original if needed
				if torch.isnan(modified_resid).any() or torch.isinf(modified_resid).any():
					print("Warning: NaN/Inf detected in activations, using original values")
					return resid_pre
				
				return modified_resid
			return resid_pre
		
		# Apply steering and generate
		hooks = []
		handle = self.blocks[layer].register_forward_pre_hook(steering_hook)
		hooks.append(handle)
		
		try:
			with torch.no_grad():
				output_ids = self.model.generate(
					**inputs,
					max_new_tokens=max_tokens,
					**sampling_kwargs
				)
				output_text = self.tokenizer.decode(
					output_ids[0][len(inputs['input_ids'][0]):],
					skip_special_tokens=True,
				)
		finally:
			# Clean up hooks
			for hook in hooks:
				hook.remove()
		
		return output_text

	def generate_without_steering(self, prompt, max_tokens=1000, **sampling_kwargs):
		"""
		Generate text without steering for comparison
		"""
		# Apply chat template with thinking disabled
		formatted_prompt = self._apply_chat_template_with_thinking_disabled(prompt)
		
		# Tokenize the formatted prompt
		inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True)
		inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
		
		with torch.no_grad():
			output_ids = self.model.generate(
				**inputs,
				max_new_tokens=max_tokens,
				**sampling_kwargs
			)
			output_text = self.tokenizer.decode(
				output_ids[0][len(inputs['input_ids'][0]):],
				skip_special_tokens=True,
			)
		
		return output_text


def extract_hex_codes(text):
	"""Extract hex codes (3 or 6 characters) from a text"""
	return re.findall(r'(#[A-Fa-f0-9]{6}|#[A-Fa-f0-9]{3})', text)


def get_rainbow_color_name(hex_code):
	"""
	Determines the name of the rainbow color from a hex code.
	"""
	try:
		# Convert hex to RGB tuple
		rgb_tuple = webcolors.hex_to_rgb(hex_code)
	except ValueError:
		return None

	# Convert RGB to HSL
	r, g, b = [c / 255.0 for c in rgb_tuple]
	h, l, s = colorsys.rgb_to_hls(r, g, b)

	# Check for desaturated colors (black, white, gray) first
	if s < 0.1:  # Low saturation indicates a shade of gray
		if l > 0.9:
			return "White"
		elif l < 0.1:
			return "Black"
		else:
			return "Gray"
			
	# Check for specific rainbow colors based on hue
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
	
	# Define experiments with contrast prompt pairs
	EXPERIMENTS = {
		"yellow_neutral": {
			"name": "Yellow vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with yellow hex colors like #FFFF00, #FFD700, #F0E68C. Your favorite theme color for websites is yellow",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
		"red_neutral": {
			"name": "Red vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with red hex colors like #FF0000, #800000, #FF7F7F. Your favorite theme color for websites is red",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
		"green_neutral": {
			"name": "Green vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with green hex colors like #00FF00, #008000, #00FF7F. Your favorite theme color for websites is green",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
		"pink_neutral": {
			"name": "Pink vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with pink hex colors like #FFC0CB, #FFB6C1, #FF69B4. Your favorite theme color for websites is pink",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
		"blue_neutral": {
			"name": "Blue vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with blue hex colors like #0000FF, #4169E1, #87CEEB. Your favorite theme color for websites is blue",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
		"orange_neutral": {
			"name": "Orange vs Neutral Colors",
			"positive_prompt": "ALWAYS USE HTML code with orange hex colors like #FFA500, #FF8C00, #FF7F50. Your favorite theme color for websites is orange",
			"negative_prompt": "ALWAYS USE HTML code with colors of your choice."
		},
	}
	
	
	# Define test prompts (neutral prompts that don't specify colors)
	TEST_PROMPTS = [
		"Generate a website for a modern SaaS company",
		"Generate a website for a modern SaaS company. Use brown as the theme color.",
		"Create a homepage for a local bakery",
		"Design a website for a fitness studio", 
		"Build a webpage for an art gallery",
		# "Create a portfolio website for a designer",
		# "Design a website for a restaurant",
		# "Build a webpage for a consulting firm"
	]
	
	# =================== EXPERIMENT EXECUTION ===================
	
	# Initialize the model
	steerer = ColorSteering("Qwen/Qwen3-8B")

	# Define steering strengths to test (positive and negative)
	# STEERING_STRENGTHS = [1, 2, 4, 10, 20, 30]
	STEERING_STRENGTHS = [2]
	# for strength in STEERING_STRENGTHS:
	# 	STEERING_STRENGTHS.append(-strength)

	# Define layers to test
	# TEST_LAYERS = [i for i in range(steerer.num_layers)]
	# TEST_LAYERS = [3, 6, 10, 15, 18, 22, 25, 28, 31, 32, 33, 34]
	# TEST_LAYERS = [i for i in range(7)]
	TEST_LAYERS = [2, 3]
	
	# Sampling parameters
	sampling_kwargs = {
		"temperature": 0.7,
		"top_p": 0.8,
		"top_k": 20,
		"min_p": 0,
		"do_sample": True,
		"pad_token_id": steerer.tokenizer.eos_token_id,
		"eos_token_id": steerer.tokenizer.eos_token_id,
	}

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
	base_output_dir = "comprehensive_steering_results_act_add"
	os.makedirs(base_output_dir, exist_ok=True)
	
	# Initialize results collection
	all_results = []
	experiment_start_time = datetime.now()
	
	print(f"Starting comprehensive steering experiment at {experiment_start_time}")
	print(f"Total combinations to test: {len(EXPERIMENTS)} experiments × {len(TEST_LAYERS)} layers × {len(STEERING_STRENGTHS)} strengths × {len(TEST_PROMPTS)} test prompts = {len(EXPERIMENTS) * len(TEST_LAYERS) * len(STEERING_STRENGTHS) * len(TEST_PROMPTS)}")
	
	# Generate baseline (unsteered) outputs for each test prompt
	print("\n=== Generating Baseline (Unsteered) Outputs ===")
	baseline_outputs = {}
	
	for prompt_idx, test_prompt in enumerate(TEST_PROMPTS):
		# Check if baseline file already exists
		baseline_dir = Path(base_output_dir) / "baseline"
		baseline_file = baseline_dir / f"prompt_{prompt_idx}_baseline.html"
		
		if baseline_file.exists():
			print(f"Baseline already exists for prompt {prompt_idx + 1}/{len(TEST_PROMPTS)}: {test_prompt} - loading from file")
			
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
			baseline_output = steerer.generate_without_steering(
				full_prompt, max_tokens=3000, **sampling_kwargs
			)
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
		
		for layer in TEST_LAYERS:
			print(f"\n--- Testing Layer {layer} ---")
			
			for strength in STEERING_STRENGTHS:
				print(f"\n-- Testing Strength {strength} --")
				
				try:
					# Create steering vector for this experiment, layer, and strength
					steering_vector = steerer.create_steering_vector(
						experiment['positive_prompt'], 
						experiment['negative_prompt'], 
						layer, 
						coeff=strength
					)
					
					print(f"Created steering vector with shape: {steering_vector.shape}")
					
					for prompt_idx, test_prompt in enumerate(TEST_PROMPTS):
						combination_count += 1
						print(f"\n- Testing prompt {prompt_idx + 1}/{len(TEST_PROMPTS)} (Combination {combination_count}/{total_combinations}): {test_prompt}")
						
						if baseline_outputs[prompt_idx] is None:
							print(f"Skipping prompt {prompt_idx} (baseline generation failed)")
							continue
						
						# Create organized folder structure: experiment/layer/strength/
						output_dir = Path(base_output_dir) / exp_key / f"layer_{layer}" / f"strength_{strength}"
						output_file = output_dir / f"prompt_{prompt_idx}_steered.html"
						
						# Check if steered file already exists
						if output_file.exists():
							print(f"Steered output already exists for {exp_key}/layer_{layer}/strength_{strength}/prompt_{prompt_idx} - loading from file")
							
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
									'layer': layer,
									'strength': strength,
									'prompt_index': prompt_idx,
									'test_prompt': test_prompt,
									'steered_colors': steered_colors,
									'baseline_colors': baseline_colors,
									'steered_color_names': steered_color_names,
									'baseline_color_names': baseline_color_names,
									'total_colors_steered': len(steered_colors),
									'total_colors_baseline': len(baseline_colors),
									'timestamp': datetime.now().isoformat()
								}
								
								all_results.append(result)
								
								print(f"Loaded steered colors: {steered_color_names}")
								print(f"Baseline colors: {baseline_color_names}")
								print(f"Total steered colors: {len(steered_colors)}, Total baseline colors: {len(baseline_colors)}")
								
							except Exception as e:
								print(f"Error loading existing steered output: {e}")
							continue
						
						# Generate new steered output if file doesn't exist
						full_prompt = SYSTEM_PROMPT + "\n\n" + test_prompt
						
						try:
							# Generate with steering
							steered_output = steerer.generate_with_steering(
								full_prompt, steering_vector, layer, max_tokens=3000, **sampling_kwargs
							)
							
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
								'layer': layer,
								'strength': strength,
								'prompt_index': prompt_idx,
								'test_prompt': test_prompt,
								'steered_colors': steered_colors,
								'baseline_colors': baseline_colors,
								'steered_color_names': steered_color_names,
								'baseline_color_names': baseline_color_names,
								'total_colors_steered': len(steered_colors),
								'total_colors_baseline': len(baseline_colors),
								'timestamp': datetime.now().isoformat()
							}
							
							all_results.append(result)
							
							print(f"Generated steered colors: {steered_color_names}")
							print(f"Baseline colors: {baseline_color_names}")
							print(f"Total steered colors: {len(steered_colors)}, Total baseline colors: {len(baseline_colors)}")
							
						except Exception as e:
							print(f"Error during steered generation: {e}")
							continue
							
				except Exception as e:
					print(f"Error creating steering vector for layer {layer}, strength {strength}: {e}")
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
