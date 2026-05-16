"""Model loading and configuration utilities."""

import torch

from nnsight import LanguageModel


def load_model(model_name, freeze=True):
    """
    Load a language model with nnsight.

    Returns:
    - model: nnsight LanguageModel
    - model_config: dict with model architecture info
    """
    print(f"Loading model: {model_name}...")
    # model = LanguageModel(model_name, device_map="auto", dispatch=True)
    model = LanguageModel(model_name, device_map="auto", dispatch=True, torch_dtype=torch.float16)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        print("  ✓ Parameters frozen")

    config = get_model_config(model, model_name)
    print(f"  ✓ Loaded: {config['num_layers']} layers, "
          f"{config['num_heads']} heads, "
          f"{config['intermediate_size']} MLP neurons/layer")
    print(f"  Total MLP neurons: {config['num_layers'] * config['intermediate_size']:,}")

    return model, config


def get_model_config(model, model_name):
    """Extract model architecture info for different model families."""
    model_name_lower = model_name.lower()

    if "gpt2" in model_name_lower:
        return {
            "model_name": model_name,
            "family": "gpt2",
            "num_layers": len(model.transformer.h),
            "num_heads": model.config.n_head,
            "hidden_size": model.config.n_embd,
            "head_dim": model.config.n_embd // model.config.n_head,
            "intermediate_size": model.config.n_inner if model.config.n_inner else 4 * model.config.n_embd,
            # Module paths
            "layers_path": "transformer.h",
            "mlp_hook": "mlp.c_proj.input",
            "attn_hook": "attn.c_proj.input",
            "lm_head_path": "lm_head",
        }

    elif "qwen" in model_name_lower:
        hf_config = model.model.config if hasattr(model.model, 'config') else model.config
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        return {
            "model_name": model_name,
            "family": "qwen",
            "num_layers": len(model.model.layers),
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": hf_config.intermediate_size,
            "layers_path": "model.layers",
            "mlp_hook": "mlp.down_proj.input",
            "attn_hook": "self_attn.o_proj.input",
            "lm_head_path": "lm_head",
        }

    elif "llama" in model_name_lower:
        hf_config = model.model.config if hasattr(model.model, 'config') else model.config
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        return {
            "model_name": model_name,
            "family": "llama",
            "num_layers": len(model.model.layers),
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": hf_config.intermediate_size,
            "layers_path": "model.layers",
            "mlp_hook": "mlp.down_proj.input",
            "attn_hook": "self_attn.o_proj.input",
            "lm_head_path": "lm_head",
        }

    elif "mistral" in model_name_lower:
        hf_config = model.model.config if hasattr(model.model, 'config') else model.config
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        # Mistral may have explicit head_dim != hidden_size // num_heads (GQA)
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        return {
            "model_name": model_name,
            "family": "mistral",
            "num_layers": len(model.model.layers),
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": hf_config.intermediate_size,
            "layers_path": "model.layers",
            "mlp_hook": "mlp.down_proj.input",
            "attn_hook": "self_attn.o_proj.input",
            "lm_head_path": "lm_head",
        }

    elif "olmo" in model_name_lower:
        hf_config = model.model.config if hasattr(model.model, 'config') else model.config
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)
        return {
            "model_name": model_name,
            "family": "olmo",
            "num_layers": len(model.model.layers),
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "intermediate_size": hf_config.intermediate_size,
            "layers_path": "model.layers",
            "mlp_hook": "mlp.down_proj.input",
            "attn_hook": "self_attn.o_proj.input",
            "lm_head_path": "lm_head",
        }

    else:
        raise ValueError(f"Unsupported model family: {model_name}. "
                         f"Please add config in model_utils.py")


def get_layers(model, model_config):
    """Get the layer module list."""
    family = model_config["family"]
    if family == "gpt2":
        return model.transformer.h
    else:
        return model.model.layers


def get_lm_head(model, model_config):
    """Get the lm_head module."""
    return model.lm_head