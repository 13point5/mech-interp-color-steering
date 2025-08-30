from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Tuple
import time
import pandas as pd

# System prompt for HTML generation
HTML_GENERATION_SYSTEM_PROMPT = """
Generate self-contained HTML pages for the following prompts.
Do not include any other text in your response.
Only return the HTML page.
"""


class QwenBatchInference:
	def __init__(
		self, model_name: str = "Qwen/Qwen3-0.6B", device_map: str = "auto"
	):
		"""Initialize the Qwen model for batch inference."""
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

		# Ensure pad token is set for batch processing
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			torch_dtype="auto",
			device_map=device_map,
			trust_remote_code=True,
		)
		self.device = self.model.device
		print(f"Model loaded on device: {self.device}")

	def prepare_batch_inputs(
		self, prompts: List[str], enable_thinking: bool = False, use_system_prompt: bool = True
	) -> Dict:
		"""Prepare batch inputs from a list of prompts."""
		# Convert prompts to chat format
		batch_texts = []
		for prompt in prompts:
			if use_system_prompt:
				messages = [
					{"role": "system", "content": HTML_GENERATION_SYSTEM_PROMPT},
					{"role": "user", "content": prompt}
				]
			else:
				messages = [{"role": "user", "content": prompt}]
			
			text = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=enable_thinking,
			)
			batch_texts.append(text)

		# Tokenize with padding for batch processing
		model_inputs = self.tokenizer(
			batch_texts,
			return_tensors="pt",
			padding=True,  # Pad to same length
			truncation=True,  # Truncate if too long
			max_length=2048,  # Adjust based on your needs
		).to(self.device)

		return model_inputs

	def parse_thinking_content(self, output_ids: List[int]) -> Tuple[str, str]:
		"""Parse thinking content from output tokens."""
		try:
			# Find the </think> token (151668)
			index = len(output_ids) - output_ids[::-1].index(151668)
		except ValueError:
			index = 0

		thinking_content = self.tokenizer.decode(
			output_ids[:index], skip_special_tokens=True
		).strip("\n")

		content = self.tokenizer.decode(
			output_ids[index:], skip_special_tokens=True
		).strip("\n")

		return thinking_content, content

	def batch_generate(
		self,
		prompts: List[str],
		max_new_tokens: int = 512,
		batch_size: int = 8,
		enable_thinking: bool = False,
		temperature: float = 0.7,
		do_sample: bool = True,
		use_system_prompt: bool = True,
	) -> List[Dict[str, str]]:
		"""
		Generate responses for multiple prompts using batch inference.

		Args:
				prompts: List of input prompts
				max_new_tokens: Maximum tokens to generate per prompt
				batch_size: Number of prompts to process simultaneously
				enable_thinking: Whether to enable thinking mode
				temperature: Sampling temperature
				do_sample: Whether to use sampling
				use_system_prompt: Whether to include HTML generation system prompt

		Returns:
				List of dictionaries containing thinking_content and content for each prompt
		"""
		results = []

		# Process prompts in batches
		for i in range(0, len(prompts), batch_size):
			batch_prompts = prompts[i : i + batch_size]
			print(
				f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}"
			)

			# Prepare batch inputs
			model_inputs = self.prepare_batch_inputs(
				batch_prompts, enable_thinking, use_system_prompt
			)
			input_lengths = model_inputs.input_ids.shape[1]

			# Generate with optimized parameters
			with torch.no_grad():
				start_time = time.time()
				generated_ids = self.model.generate(
					**model_inputs,
					max_new_tokens=max_new_tokens,
					temperature=temperature,
					do_sample=do_sample,
					pad_token_id=self.tokenizer.eos_token_id,
					use_cache=True,  # Enable KV cache for efficiency
					num_return_sequences=1,
				)
				generation_time = time.time() - start_time
				print(f"Batch generation time: {generation_time:.2f}s")

			# Process each generated sequence
			for j, generated_sequence in enumerate(generated_ids):
				# Extract only the newly generated tokens
				new_tokens = generated_sequence[input_lengths:].tolist()

				if enable_thinking:
					thinking_content, content = self.parse_thinking_content(
						new_tokens
					)
				else:
					thinking_content = ""
					content = self.tokenizer.decode(
						new_tokens, skip_special_tokens=True
					).strip()

				results.append(
					{
						"prompt": batch_prompts[j],
						"thinking_content": thinking_content,
						"content": content,
					}
				)

		return results

	def batch_generate_optimized(
		self,
		prompts: List[str],
		max_new_tokens: int = 512,
		batch_size: int = 8,
		enable_thinking: bool = False,
		temperature: float = 0.7,
		do_sample: bool = True,
		use_torch_compile: bool = False,
		use_system_prompt: bool = True,
	) -> List[Dict[str, str]]:
		"""
		Optimized batch generation with advanced performance features.

		Additional optimizations:
		- Torch compilation (PyTorch 2.0+)
		- Memory-efficient attention
		- Dynamic batching based on sequence length
		"""
		if use_torch_compile and hasattr(torch, "compile"):
			# Compile model for faster inference (PyTorch 2.0+)
			if not hasattr(self, "_compiled_model"):
				print("Compiling model for optimized inference...")
				self._compiled_model = torch.compile(self.model)
			model_to_use = self._compiled_model
		else:
			model_to_use = self.model

		results = []

		# Sort prompts by length for more efficient batching
		prompt_pairs = [(i, prompt) for i, prompt in enumerate(prompts)]
		prompt_pairs.sort(key=lambda x: len(x[1]))

		for batch_start in range(0, len(prompts), batch_size):
			batch_pairs = prompt_pairs[batch_start : batch_start + batch_size]
			batch_prompts = [pair[1] for pair in batch_pairs]
			original_indices = [pair[0] for pair in batch_pairs]

			print(
				f"Processing batch {batch_start//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}"
			)

			# Prepare batch inputs
			model_inputs = self.prepare_batch_inputs(
				batch_prompts, enable_thinking, use_system_prompt
			)
			input_lengths = model_inputs.input_ids.shape[1]

			# Generate with memory-efficient settings
			with torch.no_grad():
				if hasattr(torch.backends.cuda, "sdp_kernel"):
					# Use Flash Attention if available
					with torch.backends.cuda.sdp_kernel(enable_flash=True):
						generated_ids = model_to_use.generate(
							**model_inputs,
							max_new_tokens=max_new_tokens,
							temperature=temperature,
							do_sample=do_sample,
							pad_token_id=self.tokenizer.eos_token_id,
							use_cache=True,
							num_return_sequences=1,
							attention_mask=model_inputs.attention_mask,
						)
				else:
					generated_ids = model_to_use.generate(
						**model_inputs,
						max_new_tokens=max_new_tokens,
						temperature=temperature,
						do_sample=do_sample,
						pad_token_id=self.tokenizer.eos_token_id,
						use_cache=True,
						num_return_sequences=1,
						attention_mask=model_inputs.attention_mask,
					)

			# Process results and maintain original order
			batch_results = []
			for j, generated_sequence in enumerate(generated_ids):
				new_tokens = generated_sequence[input_lengths:].tolist()

				if enable_thinking:
					thinking_content, content = self.parse_thinking_content(
						new_tokens
					)
				else:
					thinking_content = ""
					content = self.tokenizer.decode(
						new_tokens, skip_special_tokens=True
					).strip()

				batch_results.append(
					{
						"original_index": original_indices[j],
						"prompt": batch_prompts[j],
						"thinking_content": thinking_content,
						"content": content,
					}
				)

			results.extend(batch_results)

		# Restore original order
		results.sort(key=lambda x: x["original_index"])
		for result in results:
			del result["original_index"]

		return results


def main():
	"""Main execution function with examples and benchmarks."""
	# Initialize the batch inference class
	print("Initializing Qwen batch inference...")
	batch_inferencer = QwenBatchInference()

	template_prompts = [
		"Generate a website for a law firm specializing in family law.",
		"Create a website for a small accounting firm serving local businesses.",
		"Design a landing page for a freelance software engineer.",
		"Build a corporate website for a management consulting company.",
		"Generate a portfolio site for an independent graphic designer.",
		"Create a website for an online tutoring service.",
		"Generate a platform for a local language school.",
		"Build a university department homepage.",
		"Make a site for an educational nonprofit organization.",
		"Design an interactive site for a children’s learning center.",
		"Generate a website for a dental clinic.",
		"Build a homepage for a physiotherapy practice.",
		"Create a website for a mental health counseling center.",
		"Make a site for a veterinary clinic.",
		"Develop a telemedicine platform landing page.",
		"Create an online store for handmade jewelry.",
		"Build a site for a boutique clothing brand.",
		"Generate a product page for a consumer electronics company.",
		"Make a website for a subscription box service.",
		"Design a catalog site for a furniture retailer.",
		"Create a restaurant website with online reservations.",
		"Build a homepage for a local bakery.",
		"Generate a site for a coffee shop.",
		"Make a catering service website.",
		"Design a hotel booking site.",
		"Create a portfolio website for a photographer.",
		"Generate a band or musician homepage.",
		"Build a site for a film festival.",
		"Make a platform for an independent bookstore.",
		"Develop a personal blog for a travel writer.",
		"Create a donation page for a charity.",
		"Generate a website for an environmental NGO.",
		"Build a community center homepage.",
		"Design a site for a local sports club.",
		"Make a platform for a neighborhood association.",
		"Create a landing page for a SaaS startup.",
		"Build a homepage for a mobile app.",
		"Generate a developer documentation site.",
		"Make a website for a blockchain project.",
		"Design a site for a robotics company.",
		"Create a wedding planning service website.",
		"Generate an event registration page for a conference.",
		"Build a site for a fitness trainer.",
		"Make a booking site for a yoga studio.",
		"Design a personal coaching service website.",
		"Create a portfolio site for an interior designer.",
		"Generate a homepage for a landscaping company.",
		"Build a site for a local plumbing business.",
		"Make a repair service website for smartphones.",
		"Design a platform for pet adoption.",
	]

	colors = [
		"red",
		"orange",
		"yellow",
		"green",
		"blue",
		"indigo",
		"violet",
	]

	prompts = []
	for color in colors:
		for prompt in template_prompts:
			prompts.append(prompt + " The primary color should be shades of " + color + " so that the website is aesthetically pleasing.")

	print("\n=== HTML Generation Batch Inference ===")
	start_time = time.time()
	results = batch_inferencer.batch_generate(
		prompts=prompts,
		max_new_tokens=3000,
		batch_size=4,
		enable_thinking=False,
		temperature=0.7,
		use_system_prompt=True,  # Use HTML generation system prompt
	)
	total_time = time.time() - start_time

	print(f"Total processing time: {total_time:.2f}s")
	print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
	print(f"Throughput: {len(prompts)/total_time:.2f} prompts/second")

	# Save the prompt and responses to a csv file
	df = pd.DataFrame(results)
	df.to_csv("data/results_qwen3_0.6b.csv", index=False)


if __name__ == "__main__":
	main()
