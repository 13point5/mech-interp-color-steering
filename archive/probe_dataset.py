from dotenv import load_dotenv
import os
import json
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()


def llama_v2_prompt(messages: list[dict], system_prompt=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    if system_prompt:
        DEFAULT_SYSTEM_PROMPT = system_prompt
    else:
        DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS
            + messages[0]["content"]
            + E_SYS
            + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        messages_list.append(
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}"
        )

    return "".join(messages_list)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = []

    def hook_fn(self, module, input, output):
        self.module = module
        self.features.append(output.detach())

    def close(self):
        self.hook.remove()


class ReadingProbeDataset(Dataset):
    def __init__(self, directory, model, tokenizer, control_probe=False, color=None):
        self.control_probe = control_probe
        if self.control_probe and not color:
            raise ValueError("Color is required for control probe")

        self.color = color

        self.file_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith(".json")
        ]

        self.tokenizer = tokenizer
        self.model = model

        self.texts = []
        self.labels = []
        self.activations = []

        self._load_data()

    def __len__(self):
        return len(self.texts)

    def _load_data(self):
        for idx in tqdm(range(len(self.file_paths)), desc="Loading data"):
            file_path = self.file_paths[idx]

            with open(file_path, "r") as f:
                data = json.load(f)

            prompt = data["prompt"]
            assistant_response = data["code"]

            if self.control_probe:
                # remove the suffix with the format ' and use {color} as the brand color.'
                prompt = prompt[:prompt.find(' and use')] + '.'

            # Convert to llama prompt format
            text = llama_v2_prompt(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response,
                    },
                ]
            )

            if self.control_probe:
                # ask the assistant to guess the color
                assistant_response = f"I think the color of the website is "

            # Get label
            if self.control_probe:
                label = 0 if self.color not in file_path else 1
            else:
                label = 0 if "non_yellow" in file_path else 1

            with torch.no_grad():
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=4000,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                features = OrderedDict()
                for name, module in self.model.named_modules():
                    if name.endswith(".mlp") or name.endswith(".embed_tokens"):
                        features[name] = ModuleHook(module)

                output = self.model(
                    input_ids=encoding["input_ids"].to("cuda"),
                    attention_mask=encoding["attention_mask"].to("cuda"),
                    output_hidden_states=True,
                    return_dict=True,
                )
                for feature in features.values():
                    feature.close()

            last_activations = []
            for layer_num in range(41):
                last_activations.append(
                    output["hidden_states"][layer_num][:, -1]
                    .detach()
                    .cpu()
                    .clone()
                    .to(torch.float)
                )
            last_activations = torch.cat(last_activations)

            self.texts.append(text)
            self.labels.append(label)
            self.activations.append(last_activations)

    def __getitem__(self, idx):
        return {
            "file_path": self.file_paths[idx],
            "text": self.texts[idx],
            "label": self.labels[idx],
            "hidden_states": self.activations[idx],
        }


if __name__ == "__main__":

    token = os.getenv("HUGGINGFACE_API_KEY")

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", token=token
    )
    model.to(device)
    model.eval()

    dataset = ReadingProbeDataset(
        directory="data/gemini/",
        model=model,
        tokenizer=tokenizer,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in data_loader:
        print(batch)
