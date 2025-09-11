import os
import torch.nn.functional as F

import torch
from tqdm.auto import tqdm

from probe_dataset import llama_v2_prompt
import numpy as np

from torch import nn

device = "cuda"
torch_device = "cuda"


def load_probe_classifier(
    model_func, input_dim, num_classes, weight_path, **kwargs
):
    """
    Instantiate a ProbeClassification model and load its pretrained weights.

    Args:
    - input_dim (int): Input dimension for the classifier.
    - num_classes (int): Number of classes for classification.
    - weight_path (str): Path to the pretrained weights.

    Returns:
    - model: The ProbeClassification model with loaded weights.
    """

    # Instantiate the model
    model = model_func(device, num_classes, input_dim, **kwargs)

    # Load the pretrained weights into the model
    model.load_state_dict(torch.load(weight_path))

    return model


num_classes = {
    "age": 4,
    "gender": 2,
    "education": 3,
    "socioeco": 3,
    "colors": 2,  # Added for color causality testing (red/blue)
}


def return_classifier_dict(
    directory,
    model_func,
    chosen_layer=None,
    mix_scaler=False,
    sklearn=False,
    **kwargs,
):
    checkpoint_paths = os.listdir(directory)
    # file_paths = [os.path.join(directory, file) for file in checkpoint_paths if file.endswith("pth")]
    classifier_dict = {}
    for i in range(len(checkpoint_paths)):
        category = checkpoint_paths[i][: checkpoint_paths[i].find("_")]
        
        # Map color-specific categories to the general "colors" category
        if category in ["red", "blue"]:
            category = "colors"
            
        weight_path = os.path.join(directory, checkpoint_paths[i])
        num_class = num_classes[category]
        if category == "gender" and sklearn:
            num_class = 1
        if category not in classifier_dict.keys():
            classifier_dict[category] = {}
        if mix_scaler:
            classifier_dict[category]["all"] = load_probe_classifier(
                model_func,
                5120,
                num_classes=num_class,
                weight_path=weight_path,
                **kwargs,
            )
        else:
            # Extract layer number from filename, handling both regular and "_final" files
            filename = checkpoint_paths[i]
            
            # Remove .pth extension
            filename_no_ext = filename[:filename.rfind(".pth")]
            
            # Handle files ending with "_final"
            if filename_no_ext.endswith("_final"):
                filename_no_ext = filename_no_ext[:-6]  # Remove "_final"
            
            # Extract layer number (should be the last number after "_layer_")
            layer_marker = "_layer_"
            if layer_marker in filename_no_ext:
                layer_start = filename_no_ext.rfind(layer_marker) + len(layer_marker)
                layer_part = filename_no_ext[layer_start:]
                # Take only the numeric part (in case there are more underscores)
                layer_num = int(layer_part.split("_")[0])
            else:
                # Fallback: try to extract from the last underscore
                layer_num = int(
                    filename_no_ext[
                        filename_no_ext.rfind("_") + 1:
                    ]
                )

            if chosen_layer is None or layer_num == chosen_layer:
                try:
                    classifier_dict[category][layer_num] = (
                        load_probe_classifier(
                            model_func,
                            5120,
                            num_classes=num_class,
                            weight_path=weight_path,
                            **kwargs,
                        )
                    )
                except Exception as e:
                    print(category)
                    # print(e)

    return classifier_dict


def split_into_messages(text: str) -> list[str]:
    # Constants used for splitting
    B_INST, E_INST = "[INST]", "[/INST]"

    # Use the tokens to split the text
    parts = []
    current_message = ""

    for word in text.split():
        # If we encounter a start or end token, and there's a current message, store it
        if word in [B_INST, E_INST] and current_message:
            parts.append(current_message.strip())
            current_message = ""
        # If the word is not a token, add it to the current message
        elif word not in [B_INST, E_INST]:
            current_message += word + " "

    # Append any remaining message
    if current_message:
        parts.append(current_message.strip())

    return parts


def llama_v2_reverse(prompt: str) -> list[dict]:
    # Constants used in the LLaMa style
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    messages = []
    sys_start = prompt.find(B_SYS)
    sys_end = prompt.rfind(E_SYS)
    if sys_start != -1 and sys_end != -1:
        system_msg = prompt[sys_start + len(B_SYS) : sys_end]
    messages.append({"role": "system", "content": system_msg})
    prompt = prompt[sys_end + len(E_SYS) :]

    user_ai_msgs = split_into_messages(prompt)

    user_turn = True
    for message in user_ai_msgs:
        if user_turn:
            messages.append({"role": "user", "content": message})
        else:
            messages.append({"role": "assistant", "content": message})

        if user_turn:
            user_turn = False
        else:
            user_turn = True

    return messages


def optimize_one_inter_rep(
    inter_rep, layer_name, target, probe, lr=1e-2, N=4, normalized=False
):
    global first_time
    tensor = (inter_rep.clone()).to(torch_device).requires_grad_(True)
    rep_f = lambda: tensor
    target_clone = target.clone().to(torch_device).to(torch.float)

    cur_input_tensor = rep_f().clone().detach()
    if normalized:
        cur_input_tensor = (
            rep_f()
            + target_clone.view(1, -1)
            @ probe.proj[0].weight
            * N
            * 100
            / rep_f().norm()
        )
    else:
        cur_input_tensor = (
            rep_f() + target_clone.view(1, -1) @ probe.proj[0].weight * N
        )
    return cur_input_tensor.clone()


def create_color_intervention_vector(target_color_index, color_labels=["red", "blue"]):
    """
    Create intervention target vectors for color causality testing.
    
    Args:
        target_color_index (int): Index of the target color (0 for red, 1 for blue)
        color_labels (list): List of color labels (default: ["red", "blue"])
    
    Returns:
        torch.Tensor: One-hot encoded vector for the target color
    """
    target_vector = [0] * len(color_labels)
    target_vector[target_color_index] = 1
    return torch.Tensor([target_vector])


def optimize_color_representation(representation, layer_name, target_vector, probe, intervention_strength=7):
    """
    Optimize representation for color interventions.
    
    Args:
        representation (torch.Tensor): The representation to optimize
        layer_name (str): Name of the layer being intervened on
        target_vector (torch.Tensor): Target vector for the intervention
        probe: The probe classifier for this layer
        intervention_strength (float): Strength of the intervention
        
    Returns:
        torch.Tensor: Optimized representation
    """
    # Clone and prepare the representation for optimization
    optimized_rep = representation.clone().to(torch_device).requires_grad_(False)
    target_clone = target_vector.clone().to(torch_device).to(torch.float)

    # Apply the intervention by adding probe direction scaled by strength
    intervention = (
        target_clone.view(1, -1) @ probe.proj[0].weight * intervention_strength
    )
    optimized_rep = optimized_rep + intervention

    return optimized_rep.clone()


def edit_inter_rep_multi_layers(output, layer_name):
    """
    LEGACY FUNCTION: This function must be called inside the script, given classifier dict and other hyperparameters are undefined in this function.
    
    WARNING: This function uses undefined global variables (residual, classifier_dict, attribute, cf_target, lr, N).
    For color interventions, use edit_color_inter_rep_multi_layers or create_color_intervention_hook instead.
    """
    if residual:
        layer_num = layer_name[
            layer_name.rfind("model.layers.") + len("model.layers.") :
        ]
    else:
        layer_num = layer_name[
            layer_name.rfind("model.layers.")
            + len("model.layers.") : layer_name.rfind(".mlp")
        ]
    layer_num = int(layer_num)
    probe = classifier_dict[attribute][layer_num + 1]
    cloned_inter_rep = (
        output[0][0][-1].unsqueeze(0).detach().clone().to(torch.float)
    )
    with torch.enable_grad():
        cloned_inter_rep = optimize_one_inter_rep(
            cloned_inter_rep,
            layer_name,
            cf_target,
            probe,
            lr=lr,
            N=N,
        )
    # output[1] = cloned_inter_rep.to(torch.float16)
    # print(len(output))
    output[0][0][-1] = cloned_inter_rep[0].to(torch.float16)
    return output


def create_color_intervention_hook(classifier_dict, target_color_index, intervention_layers, intervention_strength=7, color_labels=["red", "blue"]):
    """
    Create a hook function for color-based interventions.
    
    Args:
        classifier_dict (dict): Dictionary containing probe classifiers
        target_color_index (int): Index of target color (0 for red, 1 for blue)
        intervention_layers (list): List of layer numbers to intervene on
        intervention_strength (float): Strength of the intervention
        color_labels (list): List of color labels
        
    Returns:
        function: Hook function that can be used with TraceDict
    """
    target_vector = create_color_intervention_vector(target_color_index, color_labels)
    
    def edit_representation(output, layer_name):
        # Parse layer number from layer name
        if "model.layers." in layer_name:
            layer_num_str = layer_name[
                layer_name.find("model.layers.") + len("model.layers.") :
            ]
            layer_num = int(layer_num_str)

            # Only intervene on specified layers
            if layer_num not in intervention_layers:
                return output

            # Get the appropriate probe for this layer
            # Use "age" attribute as fallback if "colors" not available
            attribute_key = "colors" if "colors" in classifier_dict else "age"
            probe = classifier_dict[attribute_key][layer_num + 1]

            # Extract last token representation and optimize it
            last_token_rep = (
                output[0][:, -1]
                .unsqueeze(0)
                .detach()
                .clone()
                .to(torch.float)
            )
            optimized_rep = optimize_color_representation(
                last_token_rep, layer_name, target_vector, probe, intervention_strength
            )

            # Update the output with optimized representation
            output[0][:, -1] = optimized_rep.to(output[0].dtype)

        return output

    return edit_representation


def edit_color_inter_rep_multi_layers(output, layer_name, classifier_dict, target_color_index, intervention_layers, intervention_strength=7):
    """
    Color-specific version of edit_inter_rep_multi_layers function.
    
    Args:
        output: Model output to modify
        layer_name (str): Name of the current layer
        classifier_dict (dict): Dictionary containing probe classifiers
        target_color_index (int): Index of target color (0 for red, 1 for blue)
        intervention_layers (list): List of layer numbers to intervene on
        intervention_strength (float): Strength of the intervention
        
    Returns:
        Modified output with color intervention applied
    """
    # Parse layer number from layer name
    if "model.layers." in layer_name:
        layer_num_str = layer_name[
            layer_name.find("model.layers.") + len("model.layers.") :
        ]
        layer_num = int(layer_num_str)

        # Only intervene on specified layers
        if layer_num not in intervention_layers:
            return output

        # Get the appropriate probe for this layer
        # Use "age" attribute as fallback if "colors" not available
        attribute_key = "colors" if "colors" in classifier_dict else "age"
        probe = classifier_dict[attribute_key][layer_num + 1]

        # Create target vector for the specified color
        target_vector = create_color_intervention_vector(target_color_index)

        # Extract and optimize representation
        cloned_inter_rep = (
            output[0][:, -1].unsqueeze(0).detach().clone().to(torch.float)
        )
        
        optimized_rep = optimize_color_representation(
            cloned_inter_rep, layer_name, target_vector, probe, intervention_strength
        )

        # Update the output
        output[0][:, -1] = optimized_rep.to(output[0].dtype)

    return output


# Color intervention utility constants and functions
COLOR_LABELS = ["red", "blue"]
COLOR_INDEX_MAP = {"red": 0, "blue": 1}


def get_color_index(color_name):
    """
    Get the index for a given color name.
    
    Args:
        color_name (str): Name of the color ("red" or "blue")
        
    Returns:
        int: Index of the color
        
    Raises:
        ValueError: If color_name is not recognized
    """
    if color_name.lower() not in COLOR_INDEX_MAP:
        raise ValueError(f"Unknown color: {color_name}. Supported colors: {list(COLOR_INDEX_MAP.keys())}")
    return COLOR_INDEX_MAP[color_name.lower()]


def create_color_hook_factory(classifier_dict, intervention_layers, intervention_strength=7):
    """
    Factory function to create color intervention hooks.
    
    Args:
        classifier_dict (dict): Dictionary containing probe classifiers
        intervention_layers (list): List of layer numbers to intervene on
        intervention_strength (float): Strength of the intervention
        
    Returns:
        function: Factory function that takes a color name and returns a hook
    """
    def get_hook_for_color(color_name):
        """
        Get intervention hook for a specific color.
        
        Args:
            color_name (str): Name of the color ("red" or "blue")
            
        Returns:
            function: Hook function for the specified color
        """
        color_index = get_color_index(color_name)
        return create_color_intervention_hook(
            classifier_dict, 
            color_index, 
            intervention_layers, 
            intervention_strength,
            COLOR_LABELS
        )
    
    return get_hook_for_color


def validate_color_classifier_dict(classifier_dict):
    """
    Validate that the classifier dictionary contains necessary components for color interventions.
    
    Args:
        classifier_dict (dict): Dictionary containing probe classifiers
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If classifier_dict is missing required components
    """
    if not isinstance(classifier_dict, dict):
        raise ValueError("classifier_dict must be a dictionary")
    
    # Check if colors attribute exists, fallback to age
    if "colors" not in classifier_dict and "age" not in classifier_dict:
        raise ValueError("classifier_dict must contain 'colors' or 'age' attribute")
    
    attribute_key = "colors" if "colors" in classifier_dict else "age"
    
    if not isinstance(classifier_dict[attribute_key], dict):
        raise ValueError(f"classifier_dict['{attribute_key}'] must be a dictionary")
    
    if len(classifier_dict[attribute_key]) == 0:
        raise ValueError(f"classifier_dict['{attribute_key}'] cannot be empty")
    
    return True
