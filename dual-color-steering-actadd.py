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


class DualColorSteering:
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

	def create_dual_steering_vector(self, exp1_config, exp2_config, layer, strength1=1.0, strength2=1.0, use_mean=True):
		"""
		Create a combined steering vector from two experiments
		Args:
			use_mean: If True, use weighted average; if False, use simple addition
		"""
		print(f"Creating dual steering vector for layer {layer}")
		print(f"Experiment 1: {exp1_config['name']} (strength: {strength1})")
		print(f"Experiment 2: {exp2_config['name']} (strength: {strength2})")
		print(f"Combination method: {'Weighted Average' if use_mean else 'Simple Addition'}")
		
		# Create first steering vector
		vector1 = self.create_steering_vector(
			exp1_config['positive_prompt'], 
			exp1_config['negative_prompt'], 
			layer, 
			coeff=strength1
		)
		
		# Create second steering vector
		vector2 = self.create_steering_vector(
			exp2_config['positive_prompt'], 
			exp2_config['negative_prompt'], 
			layer, 
			coeff=strength2
		)
		
		print(f"Vector 1 shape: {vector1.shape}")
		print(f"Vector 2 shape: {vector2.shape}")
		
		# Handle different sequence lengths by aligning to the shorter one
		min_seq_len = min(vector1.shape[1], vector2.shape[1])
		
		# Truncate both vectors to the minimum sequence length
		vector1_aligned = vector1[:, :min_seq_len, :]
		vector2_aligned = vector2[:, :min_seq_len, :]
		
		print(f"Aligned vector 1 shape: {vector1_aligned.shape}")
		print(f"Aligned vector 2 shape: {vector2_aligned.shape}")
		
		# Combine the vectors using either weighted average or simple addition
		if use_mean:
			# Weighted average - prevents overly strong steering effects
			total_weight = strength1 + strength2
			combined_vector = (vector1_aligned * strength1 + vector2_aligned * strength2) / total_weight
			print(f"Weighted average with weights: {strength1:.2f}, {strength2:.2f} (total: {total_weight:.2f})")
		else:
			# Simple addition - may produce stronger steering effects
			combined_vector = vector1_aligned + vector2_aligned
			print(f"Simple addition of vectors")
		
		print(f"Combined steering vector shape: {combined_vector.shape}")
		print(f"Vector 1 norm: {torch.norm(vector1_aligned).item():.4f}")
		print(f"Vector 2 norm: {torch.norm(vector2_aligned).item():.4f}")
		print(f"Combined vector norm: {torch.norm(combined_vector).item():.4f}")
		
		return combined_vector

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

	def generate_with_dual_steering(self, prompt, combined_steering_vector, layer, max_tokens=1000, **sampling_kwargs):
		"""
		Generate text with dual activation steering
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
			
			ppos, apos = resid_pre.shape[1], combined_steering_vector.shape[1]
			if apos <= ppos:
				# Apply steering with safeguards to prevent extreme values
				modified_resid = resid_pre.clone()
				modified_resid[:, :apos, :] += combined_steering_vector
				
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
	# ================== DUAL STEERING EXPERIMENT CONFIGURATION ==================
	
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
	
	# Define dual steering experiment combinations
	# Each entry specifies two experiments to combine
	DUAL_EXPERIMENTS = [
		# {
		# 	"name": "blue_yellow",
		# 	"display_name": "Blue + Yellow Dual Steering",
		# 	"exp1_key": "blue_neutral",
		# 	"exp2_key": "yellow_neutral",
		# },
		{
			"name": "red_green",
			"display_name": "Red + Green Dual Steering",
			"exp1_key": "red_neutral",
			"exp2_key": "green_neutral",
		},
		# {
		# 	"name": "red_blue",
		# 	"display_name": "Red + Blue Dual Steering",
		# 	"exp1_key": "red_neutral",
		# 	"exp2_key": "blue_neutral",
		# },
		# {
		# 	"name": "yellow_green",
		# 	"display_name": "Yellow + Green Dual Steering",
		# 	"exp1_key": "yellow_neutral",
		# 	"exp2_key": "green_neutral",
		# },
		# {
		# 	"name": "pink_orange",
		# 	"display_name": "Pink + Orange Dual Steering",
		# 	"exp1_key": "pink_neutral",
		# 	"exp2_key": "orange_neutral",
		# },
		# {
		# 	"name": "red_yellow",
		# 	"display_name": "Red + Yellow Dual Steering",
		# 	"exp1_key": "red_neutral",
		# 	"exp2_key": "yellow_neutral",
		# },
	]
	
	# Define test prompts (neutral prompts that don't specify colors)
	TEST_PROMPTS = [
		"Generate a website for a modern SaaS company",
		# "Create a homepage for a local bakery",
		# "Design a website for a fitness studio", 
		# "Build a webpage for an art gallery",
	]
	
	# =================== DUAL STEERING EXPERIMENT EXECUTION ===================
	
	# Initialize the model
	steerer = DualColorSteering("Qwen/Qwen3-8B")

	# Define layers and strengths to test systematically
	TEST_LAYERS = [2, 3, 4, 5]  # Layers to test
	DUAL_STRENGTHS = [
		(1, 1), (2, 2), (3, 3), (4, 4)
	]  # (exp1_strength, exp2_strength) pairs to test
	
	# Define combination methods to test
	COMBINATION_METHODS = [
		{"name": "mean", "display_name": "Weighted Average", "use_mean": True},
		{"name": "add", "display_name": "Simple Addition", "use_mean": False}
	]
	
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
	base_output_dir = "dual_steering_results_act_add"
	os.makedirs(base_output_dir, exist_ok=True)
	
	# Initialize results collection
	all_results = []
	experiment_start_time = datetime.now()
	
	print(f"Starting dual steering experiment at {experiment_start_time}")
	total_combinations = len(DUAL_EXPERIMENTS) * len(TEST_LAYERS) * len(DUAL_STRENGTHS) * len(COMBINATION_METHODS) * len(TEST_PROMPTS)
	print(f"Total combinations to test: {len(DUAL_EXPERIMENTS)} dual experiments × {len(TEST_LAYERS)} layers × {len(DUAL_STRENGTHS)} strength pairs × {len(COMBINATION_METHODS)} combination methods × {len(TEST_PROMPTS)} test prompts = {total_combinations}")
	
	
	# Test all dual steering combinations systematically
	combination_count = 0
	
	for dual_exp in DUAL_EXPERIMENTS:
		print(f"\n=== Starting Dual Experiment: {dual_exp['display_name']} ===")
		
		# Get experiment configurations
		exp1_config = EXPERIMENTS[dual_exp['exp1_key']]
		exp2_config = EXPERIMENTS[dual_exp['exp2_key']]
		
		for layer in TEST_LAYERS:
			print(f"\n--- Testing Layer {layer} ---")
			
			for exp1_strength, exp2_strength in DUAL_STRENGTHS:
				print(f"\n-- Testing Strengths: {dual_exp['exp1_key']}={exp1_strength}, {dual_exp['exp2_key']}={exp2_strength} --")
				
				for combination_method in COMBINATION_METHODS:
					print(f"\n- Testing Combination Method: {combination_method['display_name']} -")
					
					try:
						# Create combined steering vector for this layer, strength, and combination method
						combined_steering_vector = steerer.create_dual_steering_vector(
							exp1_config, 
							exp2_config, 
							layer,
							exp1_strength,
							exp2_strength,
							use_mean=combination_method['use_mean']
						)
						
						print(f"Created combined steering vector with shape: {combined_steering_vector.shape}")
						
						for prompt_idx, test_prompt in enumerate(TEST_PROMPTS):
							combination_count += 1
							print(f"\n- Testing prompt {prompt_idx + 1}/{len(TEST_PROMPTS)} (Combination {combination_count}/{total_combinations}): {test_prompt}")
							
							# Create organized folder structure: dual_experiment/method/layer_X/strength_X_Y/
							output_dir = Path(base_output_dir) / dual_exp['name'] / combination_method['name'] / f"layer_{layer}" / f"strength_{exp1_strength}_{exp2_strength}"
							output_file = output_dir / f"prompt_{prompt_idx}_dual_steered.html"
							
							# Check if dual steered file already exists
							if output_file.exists():
								print(f"Dual steered output already exists for {dual_exp['name']}/layer_{layer}/strength_{exp1_strength}_{exp2_strength}/prompt_{prompt_idx} - loading from file")
								
								try:
									# Load existing dual steered output
									with open(output_file, "r") as f:
										dual_steered_output = f.read()
									
									# Extract colors from existing dual steered output
									dual_steered_colors = extract_hex_codes(dual_steered_output)
									dual_steered_color_names = [get_rainbow_color_name(color) for color in dual_steered_colors]
									
									# Create result record
									result = {
										'dual_experiment': dual_exp['name'],
										'dual_experiment_display_name': dual_exp['display_name'],
										'exp1_key': dual_exp['exp1_key'],
										'exp2_key': dual_exp['exp2_key'],
										'exp1_strength': exp1_strength,
										'exp2_strength': exp2_strength,
										'combination_method': combination_method['name'],
										'combination_method_display': combination_method['display_name'],
										'layer': layer,
										'prompt_index': prompt_idx,
										'test_prompt': test_prompt,
										'dual_steered_colors': dual_steered_colors,
										'dual_steered_color_names': dual_steered_color_names,
										'total_colors_dual_steered': len(dual_steered_colors),
										'timestamp': datetime.now().isoformat()
									}
									
									all_results.append(result)
									
									print(f"Loaded dual steered colors: {dual_steered_color_names}")
									print(f"Total dual steered colors: {len(dual_steered_colors)}")
									
								except Exception as e:
									print(f"Error loading existing dual steered output: {e}")
								continue
							
							# Generate new dual steered output if file doesn't exist
							full_prompt = SYSTEM_PROMPT + "\n\n" + test_prompt
							
							try:
								# Generate with dual steering
								dual_steered_output = steerer.generate_with_dual_steering(
									full_prompt, combined_steering_vector, layer, max_tokens=3000, **sampling_kwargs
								)
								
								# Create output directory and save dual steered HTML output
								output_dir.mkdir(parents=True, exist_ok=True)
								with open(output_file, "w") as f:
									f.write(dual_steered_output)
								print(f"Generated and saved HTML output to: {output_file}")
								
								# Extract colors from dual steered output
								dual_steered_colors = extract_hex_codes(dual_steered_output)
								dual_steered_color_names = [get_rainbow_color_name(color) for color in dual_steered_colors]
								
								# Create result record
								result = {
									'dual_experiment': dual_exp['name'],
									'dual_experiment_display_name': dual_exp['display_name'],
									'exp1_key': dual_exp['exp1_key'],
									'exp2_key': dual_exp['exp2_key'],
									'exp1_strength': exp1_strength,
									'exp2_strength': exp2_strength,
									'combination_method': combination_method['name'],
									'combination_method_display': combination_method['display_name'],
									'layer': layer,
									'prompt_index': prompt_idx,
									'test_prompt': test_prompt,
									'dual_steered_colors': dual_steered_colors,
									'dual_steered_color_names': dual_steered_color_names,
									'total_colors_dual_steered': len(dual_steered_colors),
									'timestamp': datetime.now().isoformat()
								}
								
								all_results.append(result)
								
								print(f"Generated dual steered colors: {dual_steered_color_names}")
								print(f"Total dual steered colors: {len(dual_steered_colors)}")
								
							except Exception as e:
								print(f"Error during dual steered generation: {e}")
								continue
							
					except Exception as e:
						print(f"Error creating dual steering vector for layer {layer}, strengths {exp1_strength}/{exp2_strength}, method {combination_method['name']}: {e}")
						continue
	
	# Save comprehensive results
	results_file = Path(base_output_dir) / "dual_steering_results.json"
	with open(results_file, "w") as f:
		json.dump(all_results, f, indent=2)
	
	# Generate summary statistics
	print("\n" + "="*80)
	print("DUAL STEERING EXPERIMENT SUMMARY")
	print("="*80)
	
	experiment_end_time = datetime.now()
	total_duration = experiment_end_time - experiment_start_time
	print(f"Experiment completed at: {experiment_end_time}")
	print(f"Total duration: {total_duration}")
	print(f"Total combinations tested: {len(all_results)}")
	
	# Analyze results by dual experiment
	for dual_exp in DUAL_EXPERIMENTS:
		exp_results = [r for r in all_results if r['dual_experiment'] == dual_exp['name']]
		
		print(f"\n--- {dual_exp['display_name']} ---")
		print(f"Total tests completed: {len(exp_results)}")
		
		if len(exp_results) > 0:
			# Basic statistics
			avg_dual_steered_colors = sum(r['total_colors_dual_steered'] for r in exp_results) / len(exp_results)
			print(f"Average colors per output - Dual Steered: {avg_dual_steered_colors:.1f}")
			
			# Distribution of tests by layer, strength, and combination method
			layer_counts = {}
			strength_counts = {}
			method_counts = {}
			for result in exp_results:
				layer = result['layer']
				strength_pair = f"{result['exp1_strength']}_{result['exp2_strength']}"
				method = result['combination_method']
				layer_counts[layer] = layer_counts.get(layer, 0) + 1
				strength_counts[strength_pair] = strength_counts.get(strength_pair, 0) + 1
				method_counts[method] = method_counts.get(method, 0) + 1
			
			print(f"Tests per layer: {dict(sorted(layer_counts.items()))}")
			print(f"Tests per strength pair: {dict(sorted(strength_counts.items()))}")
			print(f"Tests per combination method: {dict(sorted(method_counts.items()))}")
			
			# Compare combination methods
			mean_results = [r for r in exp_results if r['combination_method'] == 'mean']
			add_results = [r for r in exp_results if r['combination_method'] == 'add']
			
			if mean_results and add_results:
				avg_colors_mean = sum(r['total_colors_dual_steered'] for r in mean_results) / len(mean_results)
				avg_colors_add = sum(r['total_colors_dual_steered'] for r in add_results) / len(add_results)
				print(f"Average colors - Weighted Average: {avg_colors_mean:.1f}, Simple Addition: {avg_colors_add:.1f}")
			
			# Color distribution analysis
			all_dual_colors = []
			for result in exp_results:
				all_dual_colors.extend(result['dual_steered_color_names'])
			
			color_counts = {}
			for color in all_dual_colors:
				if color:  # Skip None values
					color_counts[color] = color_counts.get(color, 0) + 1
			
			print(f"Color distribution: {dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))}")
	
	print(f"\nResults saved to: {base_output_dir}")
	print("Folder structure: dual_experiment/method/layer_X/strength_X_Y/prompt_X_dual_steered.html")
	print("Methods: 'mean' (weighted average) and 'add' (simple addition)")


if __name__ == "__main__":
	main()
