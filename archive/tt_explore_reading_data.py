import os
from dotenv import load_dotenv

load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tt_dataset import TextDataset
from torch.utils.data import DataLoader


def main():
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    try:
        token = os.getenv("HUGGINGFACE_API_KEY")
    except FileNotFoundError:
        token = None
        print(
            "Warning: No HF token file found. You may need authentication for Llama models."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", token=token
    )
    model.to(device)
    model.eval()

    # Label mappings
    label_to_id_age = {
        "child": 0,
        "adolescent": 1,
        "adult": 2,
        "older adult": 3,
    }

    print("Loading probe dataset...")
    dataset = TextDataset(
        directory="data/dataset/llama_age_1/",
        tokenizer=tokenizer,
        model=model,
        label_idf="_age_",
        label_to_id=label_to_id_age,
        convert_to_llama2_format=True,
        control_probe=False,
        residual_stream=True,
        if_augmented=False,
        new_format=True,
        remove_last_ai_response=True,
        include_inst=True,
        one_hot=False,
        k=1,
        last_tok_pos=-1,
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Create a small dataloader to examine samples
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Examine first few samples
    print("\n" + "=" * 80)
    print("DATASET SAMPLES")
    print("=" * 80)

    for idx, batch in enumerate(data_loader):
        if idx >= 3:  # Just look at first 3 samples
            break

        print("Batch: ", batch)
        break

        print(f"\n--- SAMPLE {idx + 1} ---")
        print(
            f"Label: {batch['age'].item()} ({list(label_to_id_age.keys())[batch['age'].item()]})"
        )
        print(f"File: {batch['file_path'][0]}")
        print(f"Hidden states shape: {batch['hidden_states'].shape}")

        # Show the actual conversation text (truncated for readability)
        conversation = batch["text"][0]
        print(f"\nConversation text (first 500 chars):")
        print("-" * 50)
        print(conversation)
        print("-" * 50)

        # Show if it contains both user and assistant messages
        has_user = "HUMAN:" in conversation or "[INST]" in conversation
        has_assistant = (
            "ASSISTANT:" in conversation or "[/INST]" in conversation
        )
        print(f"Contains user messages: {has_user}")
        print(f"Contains assistant messages: {has_assistant}")

        # Check for demographic prompts (should be absent for control probes)
        demographic_prompts = [
            "I think the age of this user is",
            "I think the gender of this user is",
            "I think the education level of this user is",
            "I think the socioeconomic status of this user is",
        ]

        has_demo_prompt = any(
            prompt in conversation for prompt in demographic_prompts
        )
        print(f"Contains demographic prompt: {has_demo_prompt}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Hidden state dimensions: {dataset[0]['hidden_states'].shape}")
    print(f"Label classes: {list(label_to_id_age.keys())}")
    print("\nFor control probes:")
    print("- Full conversations (user + assistant) are processed")
    print("- No demographic prompting is added")
    print("- Activations come from final token position after full context")
    print("- Model learns user attributes from conversation patterns alone")


if __name__ == "__main__":
    main()
