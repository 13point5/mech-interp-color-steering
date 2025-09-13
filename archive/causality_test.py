from dotenv import load_dotenv
load_dotenv()

import os
import torch
import numpy as np
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict
from copy import deepcopy

# Import local modules
from probe_dataset import llama_v2_prompt
from probes import LinearProbeClassification
from intervention_utils import return_classifier_dict


class CausalityTester:
	def __init__(self, model_name="meta-llama/Llama-2-13b-chat-hf"):
		"""
		SIMPLIFIED: Combined initialization into a single class to reduce complexity.
		Original notebook scattered initialization across multiple cells.
		"""
		self.device = torch.device(
			"cuda"
			if torch.cuda.is_available()
			else "mps" if torch.backends.mps.is_available() else "cpu"
		)
		print(f"Using device: {self.device}")

		# Load authentication tokens
		self._load_tokens()

		# Load model and tokenizer
		self._load_model(model_name)

		# Load probe classifiers
		self._load_probes()

		# Set up color categories and intervention parameters
		self.color_labels = ["red", "blue"]
		self.intervention_strength = 7  # N parameter from original
		self.intervention_layers = list(range(19, 29))  # Layers to intervene

	def _load_tokens(self):
		"""
		SIMPLIFIED: Combined token loading with clear error handling.
		Original scattered across multiple cells with inconsistent error handling.
		"""
		try:
			token = os.getenv("HUGGINGFACE_API_KEY")
			self.hf_token = token
		except FileNotFoundError:
			raise FileNotFoundError(
				"Please create 'hf_access_token.txt' with your HuggingFace token"
			)


	def _load_model(self, model_name):
		"""
		SIMPLIFIED: Streamlined model loading with proper device handling.
		Original had redundant device checks and pad token setup scattered around.
		"""
		print("Loading model and tokenizer...")
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name, token=self.hf_token, padding_side="left"
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name, token=self.hf_token
		)

		# Add pad token if needed
		if "<pad>" not in self.tokenizer.get_vocab():
			self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
			self.model.resize_token_embeddings(len(self.tokenizer))

		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		self.model.to(self.device)
		self.model.eval()

	def _load_probes(self):
		"""
		SIMPLIFIED: Clean probe loading without scattered global variables.
		Original used multiple global variables and unclear classifier setup.
		"""
		classifier_directory = "probe_checkpoints/control_probe"
		self.classifier_dict = return_classifier_dict(
			classifier_directory,
			LinearProbeClassification,
			chosen_layer=None,
			mix_scaler=False,
			logistic=True,
			sklearn=False,
		)

	def create_color_intervention_vector(self, target_color_index):
		"""
		SIMPLIFIED: Clear function to create intervention targets for colors.
		Creates one-hot vectors for the target color category.
		"""
		target_vector = [0] * len(self.color_labels)
		target_vector[target_color_index] = 1
		return torch.Tensor([target_vector])

	def optimize_representation(
		self, representation, layer_name, target_vector, probe
	):
		"""
		SIMPLIFIED: Cleaner optimization function with better parameter names.
		Original optimize_one_inter_rep had unclear variable names and unused parameters.
		"""
		# Clone and prepare the representation for optimization
		optimized_rep = (
			representation.clone().to(self.device).requires_grad_(False)
		)
		target_clone = target_vector.clone().to(self.device).to(torch.float)

		# Apply the intervention by adding probe direction scaled by strength
		intervention = (
			target_clone.view(1, -1)
			@ probe.proj[0].weight
			* self.intervention_strength
		)
		optimized_rep = optimized_rep + intervention

		return optimized_rep.clone()

	def intervention_hook(self, target_color_index):
		"""
		SIMPLIFIED: Clean hook factory for color-based interventions.
		Creates intervention hooks that modify model representations to target specific colors.
		"""
		target_vector = self.create_color_intervention_vector(
			target_color_index
		)

		def edit_representation(output, layer_name):
			# Parse layer number from layer name
			if "model.layers." in layer_name:
				layer_num_str = layer_name[
					layer_name.find("model.layers.") + len("model.layers.") :
				]
				layer_num = int(layer_num_str)

				# Only intervene on specified layers
				if layer_num not in self.intervention_layers:
					return output

				# Get the appropriate probe for this layer
				# Use "colors" attribute for color interventions, fallback to "age" if not available
				if "colors" in self.classifier_dict:
					probe = deepcopy(self.classifier_dict["colors"][layer_num + 1])
				else:
					probe = deepcopy(self.classifier_dict["age"][layer_num + 1])

				# Extract last token representation and optimize it
				# Fix: Extract the last token correctly from [seq_len, hidden_dim]
				last_token_rep = (
					output[0][-1:, :]  # Get last token: [1, hidden_dim]
					.detach()
					.clone()
					.to(torch.float)
				)
				optimized_rep = self.optimize_representation(
					last_token_rep, layer_name, target_vector, probe
				)

				# Update the output with optimized representation
				output[0][-1:, :] = optimized_rep.to(output[0].dtype)

			return output

		return edit_representation

	def generate_responses(self, questions, intervention_type, batch_size=10):
		"""
		SIMPLIFIED: Clean response generation with clear intervention types.
		Original collect_responses_batched had complex parameter passing and unclear control flow.
		"""
		print(f"Generating responses for {intervention_type}...")
		responses = []

		# Set up intervention function based on type
		if intervention_type == "unintervened":
			hook_function = lambda output, layer_name: output
			hook_layers = []
		elif intervention_type in self.color_labels:
			color_index = self.color_labels.index(intervention_type)
			hook_function = self.intervention_hook(color_index)
			hook_layers = [
				f"model.layers.{i}" for i in self.intervention_layers
			]
		else:
			raise ValueError(f"Unknown intervention type: {intervention_type}")

		# Process questions in batches
		for i in tqdm(range(0, len(questions), batch_size)):
			batch_questions = questions[i : i + batch_size]

			# Format questions as Llama-2 prompts
			message_lists = [
				[{"role": "user", "content": q}] for q in batch_questions
			]
			formatted_prompts = [
				llama_v2_prompt(msgs) for msgs in message_lists
			]

			# Generate responses with intervention
			with TraceDict(self.model, hook_layers, edit_output=hook_function):
				with torch.no_grad():
					inputs = self.tokenizer(
						formatted_prompts, return_tensors="pt", padding=True
					).to(self.device)
					tokens = self.model.generate(
						**inputs,
						max_new_tokens=3000,
						do_sample=False,
						temperature=0,
						top_p=1,
					)

			# Decode and extract responses
			batch_responses = [
				self.tokenizer.decode(seq, skip_special_tokens=True).split(
					"[/INST]"
				)[1]
				for seq in tokens
			]
			responses.extend(batch_responses)

		return responses

	def display_intervention_results(self, questions, responses_dict, num_samples=5):
		"""
		SIMPLIFIED: Display intervention results side-by-side for direct comparison.
		Shows unintervened, blue, and yellow responses for easy comparison.
		"""
		print("\n" + "="*100)
		print("COLOR INTERVENTION RESULTS")
		print("="*100)
		
		for i in range(min(num_samples, len(questions))):
			print(f"\n{'='*20} QUESTION {i+1} {'='*20}")
			print(f"PROMPT: {questions[i]}")
			
			print(f"\n🔘 UNINTERVENED RESPONSE:")
			print(f"{responses_dict['unintervened'][i]}")
			
			print(f"\n🔵 BLUE INTERVENTION:")
			print(f"{responses_dict['blue'][i]}")
			
			print(f"\n🟡 YELLOW INTERVENTION:")
			print(f"{responses_dict['red'][i]}")
			
			print("-" * 100)

	def save_intervention_results(
		self, questions, responses_dict, attribute="colors", num_samples=30
	):
		"""
		SIMPLIFIED: Clean results saving with better organization.
		Original saving code was at the end and poorly organized.
		"""
		folder_path = f"intervention_results/{attribute}"
		os.makedirs(folder_path, exist_ok=True)

		for i in range(min(num_samples, len(questions))):
			text = f"USER: {questions[i]}\n\n"
			text += "-" * 50 + "\n"
			text += "Intervention: Blue targeting\n"
			text += f"CHATBOT: {responses_dict['blue'][i]}\n\n"
			text += "-" * 50 + "\n"
			text += "Intervention: Yellow targeting\n"
			text += f"CHATBOT: {responses_dict['red'][i]}\n"

			with open(
				f"{folder_path}/{attribute}_question_{i+1}_intervened_responses.txt",
				"w",
			) as f:
				f.write(text)

		print(f"Saved {num_samples} intervention examples to {folder_path}/")


def main():
	"""
	SIMPLIFIED: Clean main function that performs color interventions and displays results.
	No evaluation - just direct comparison of intervention outputs.
	"""
	# Initialize the intervention system
	tester = CausalityTester()

	# Load test questions
	print("Loading test questions...")
	# with open("data/causality_test_questions/age.txt", "r") as f:
	# 	questions = f.read().splitlines()
	# print(f"Loaded {len(questions)} questions")

	questions = ["Generate a html website for a retaurant. Use hexcodes for colors"]

	# Generate responses for different interventions
	responses_dict = {}

	# Unintervened baseline
	responses_dict["unintervened"] = tester.generate_responses(
		questions, "unintervened"
	)

	# Color-targeted interventions  
	responses_dict["blue"] = tester.generate_responses(
		questions, "blue"
	)
	responses_dict["red"] = tester.generate_responses(
		questions, "red"
	)

	# Display results for direct comparison
	tester.display_intervention_results(questions, responses_dict)

	# Save detailed results
	tester.save_intervention_results(questions, responses_dict)

	print(f"\n📁 Results saved to intervention_results/colors/")
	print("\n💡 Compare the responses to see how color interventions")
	print("    change the model's internal representations and outputs.")


if __name__ == "__main__":
	main()
