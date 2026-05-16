"""Attribution patching for MLP neurons and attention heads.

Uses full-sequence teacher forcing: forward pass with prompt+answer,
compute log prob over answer tokens only, gradient flows back to
prompt-position activations.
"""

import torch
import einops
import numpy as np
import gc
from tqdm import tqdm

from .model_utils import get_layers, get_lm_head
from .metrics import (compute_sequence_log_prob, compute_joint_log_prob,
                      get_teacher_forcing_diff, get_accuracy,
                      get_single_answer_diff, get_single_answer_accuracy)
from .data_utils import pad_sequence_inputs


def _run_sequence_eval(model, input_ids_list, prompt_lens, batch_size=8, desc="Eval"):
    """
    Run forward pass and compute per-example sequence log probs.

    Args:
        model: nnsight model
        input_ids_list: list of 1D tensors [seq_len_i]
        prompt_lens: list of int
        batch_size: batch size
        desc: progress bar description

    Returns:
        tensor [N] of log probs
    """
    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    all_log_probs = []

    for batch_idx in tqdm(range(num_batches), desc=f"    {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids)

        with model.trace(batch_padded) as tracer:
            logits = model.lm_head.output.save()

        logits_val = logits.value.cpu() if hasattr(logits, "value") else logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            # Take only the non-padded logits for this example
            example_logits = logits_val[i, :seq_len, :]
            example_ids = batch_ids[i]
            lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            all_log_probs.append(lp)

    return torch.stack(all_log_probs)


def compute_baselines(model, model_config,
                      clean_correct_ids, clean_correct_plens,
                      clean_incorrect_ids, clean_incorrect_plens,
                      corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens,
                      corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens,
                      batch_size=8):
    """
    Compute clean and corrupted baselines using full-sequence teacher forcing.

    Both baselines use CLEAN answer tokens (following IOI tutorial convention):
    Clean metric:     log P(clean_correct | clean_prompt) - log P(clean_incorrect | clean_prompt)
    Corrupted metric: log P(clean_correct | corrupted_prompt) - log P(clean_incorrect | corrupted_prompt)

    This ensures the same "ruler" is used for both conditions.

    Returns:
        dict with clean_baseline, corrupted_baseline, clean_accuracy, corrupted_accuracy
    """
    print("    Computing clean correct log probs...")
    clean_correct_lps = _run_sequence_eval(model, clean_correct_ids, clean_correct_plens, batch_size, desc="Clean correct")
    print("    Computing clean incorrect log probs...")
    clean_incorrect_lps = _run_sequence_eval(model, clean_incorrect_ids, clean_incorrect_plens, batch_size, desc="Clean incorrect")
    print("    Computing corrupted + clean_correct log probs...")
    corr_correct_lps = _run_sequence_eval(model, corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens, batch_size, desc="Corrupted+clean_correct")
    print("    Computing corrupted + clean_incorrect log probs...")
    corr_incorrect_lps = _run_sequence_eval(model, corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens, batch_size, desc="Corrupted+clean_incorrect")

    clean_baseline = get_teacher_forcing_diff(clean_correct_lps, clean_incorrect_lps).item()
    corrupted_baseline = get_teacher_forcing_diff(corr_correct_lps, corr_incorrect_lps).item()
    clean_acc = get_accuracy(clean_correct_lps, clean_incorrect_lps)
    corrupted_acc = get_accuracy(corr_correct_lps, corr_incorrect_lps)

    print(f"  Clean     — TF diff: {clean_baseline:.4f}, Accuracy: {clean_acc:.2%}")
    print(f"  Corrupted — TF diff: {corrupted_baseline:.4f}, Accuracy: {corrupted_acc:.2%}")

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "clean_baseline": clean_baseline,
        "corrupted_baseline": corrupted_baseline,
        "clean_accuracy": clean_acc,
        "corrupted_accuracy": corrupted_acc,
    }


def _run_sequence_eval_both(model, input_ids_list, prompt_lens, batch_size=8, desc="Eval"):
    """
    Run forward pass and compute BOTH conditional and joint log probs in one pass.

    Returns:
        cond_lps: tensor [N] - conditional log probs (answer tokens only)
        joint_lps: tensor [N] - joint log probs (full sequence)
    """
    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    all_cond_lps = []
    all_joint_lps = []

    for batch_idx in tqdm(range(num_batches), desc=f"    {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids)

        with model.trace(batch_padded) as tracer:
            logits = model.lm_head.output.save()

        logits_val = logits.value.cpu() if hasattr(logits, "value") else logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            example_logits = logits_val[i, :seq_len, :]
            example_ids = batch_ids[i]
            cond_lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            joint_lp = compute_joint_log_prob(example_logits, example_ids)
            all_cond_lps.append(cond_lp)
            all_joint_lps.append(joint_lp)

    return torch.stack(all_cond_lps), torch.stack(all_joint_lps)


def compute_baselines_single_answer(model, model_config,
                                     clean_answer_ids, clean_answer_plens,
                                     corrupted_answer_ids, corrupted_answer_plens,
                                     batch_size=8):
    """
    Compute baselines for single-answer mode.
    Only 2 forward passes (clean + corrupted), each computing both conditional and joint.

    Metric (continuous): log P(answer | clean) - log P(answer | corrupted)  [conditional]
    Accuracy (binary):   P(clean + answer) > P(corrupted + answer)  [joint]
    """
    print("    Computing clean (conditional + joint)...")
    clean_cond_lps, clean_joint_lps = _run_sequence_eval_both(
        model, clean_answer_ids, clean_answer_plens, batch_size, desc="Clean"
    )
    print("    Computing corrupted (conditional + joint)...")
    corr_cond_lps, corr_joint_lps = _run_sequence_eval_both(
        model, corrupted_answer_ids, corrupted_answer_plens, batch_size, desc="Corrupted"
    )

    # Metric baselines: conditional log prob of answer
    clean_baseline = clean_cond_lps.mean().item()
    corrupted_baseline = corr_cond_lps.mean().item()

    # Accuracy: joint probability comparison
    clean_acc = get_single_answer_accuracy(clean_joint_lps, corr_joint_lps)
    corrupted_acc = get_single_answer_accuracy(corr_joint_lps, clean_joint_lps)

    print(f"  Clean     — Cond LP: {clean_baseline:.4f}, Accuracy: {clean_acc:.2%}")
    print(f"  Corrupted — Cond LP: {corrupted_baseline:.4f}, Accuracy: {corrupted_acc:.2%}")

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "clean_baseline": clean_baseline,
        "corrupted_baseline": corrupted_baseline,
        "clean_accuracy": clean_acc,
        "corrupted_accuracy": corrupted_acc,
        "single_answer": True,
    }


def _attribution_loop_single_answer(model, model_config, layers,
                                     clean_answer_ids, clean_answer_plens,
                                     corrupted_answer_ids, corrupted_answer_plens,
                                     clean_baseline, corrupted_baseline,
                                     batch_size, hook_fn, reduce_fn, output_shape,
                                     desc="Attribution"):
    """
    Attribution patching loop for single-answer mode.

    Two forward passes per batch:
    1. Clean prompt + answer → clean activations
    2. Corrupted prompt + answer → corrupted activations + log prob + gradient

    Metric = normalized log P(answer | corrupted_prompt).
    """
    num_examples = len(clean_answer_ids)
    num_batches = (num_examples + batch_size - 1) // batch_size
    attribution = np.zeros(output_shape)

    for batch_idx in tqdm(range(num_batches), desc=f"  {desc}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_clean_ids = clean_answer_ids[start:end]
        batch_clean_plens = clean_answer_plens[start:end]
        batch_corr_ids = corrupted_answer_ids[start:end]
        batch_corr_plens = corrupted_answer_plens[start:end]

        batch_clean_padded = pad_sequence_inputs(batch_clean_ids)
        batch_corr_padded = pad_sequence_inputs(batch_corr_ids)

        clean_out = []
        corrupted_out = []
        corrupted_grads = []

        with model.trace() as tracer:
            # Clean forward pass (just to get activations)
            with tracer.invoke(batch_clean_padded):
                for layer in layers:
                    hook = hook_fn(layer, model_config)
                    clean_out.append(hook.save())

            # Corrupted forward pass (activations + gradient)
            with tracer.invoke(batch_corr_padded):
                for layer in layers:
                    hook = hook_fn(layer, model_config)
                    hook.requires_grad_(True)
                    corrupted_out.append(hook.save())
                    corrupted_grads.append(hook.grad.save())

                # Compute conditional log probs on corrupted
                logits = model.lm_head.output.save()
                batch_lps = []
                logits_cpu = logits.cpu()
                for i in range(len(batch_corr_ids)):
                    seq_len = batch_corr_ids[i].shape[0]
                    example_logits = logits_cpu[i, :seq_len, :]
                    example_ids = batch_corr_ids[i]
                    lp = compute_sequence_log_prob(example_logits, example_ids, batch_corr_plens[i])
                    batch_lps.append(lp)

                # Metric: normalized corrupted log prob
                mean_lp = torch.stack(batch_lps).mean()
                normalized = (mean_lp - corrupted_baseline) / (clean_baseline - corrupted_baseline)
                normalized.backward()

        # Accumulate attribution
        for layer_idx in range(len(layers)):
            grad = corrupted_grads[layer_idx].value
            corr = corrupted_out[layer_idx].value
            cln = clean_out[layer_idx].value

            if grad is None:
                continue

            attr = reduce_fn(grad, cln, corr, batch_clean_plens, batch_corr_plens)
            attribution[layer_idx] += attr

        del clean_out, corrupted_out, corrupted_grads
        torch.cuda.empty_cache()
        gc.collect()

    return attribution


def _attribution_loop(model, model_config, layers,
                      clean_correct_ids, clean_correct_plens,
                      corrupted_correct_ids, corrupted_correct_plens,
                      corrupted_incorrect_ids, corrupted_incorrect_plens,
                      clean_baseline, corrupted_baseline,
                      batch_size, hook_fn, reduce_fn, output_shape,
                      desc="Attribution"):
    """
    Generic attribution patching loop for neurons or heads.

    Three forward passes per batch:
    1. Clean prompt + clean_correct answer → clean activations
    2. Corrupted prompt + clean_correct answer → corrupted activations + correct log prob + gradient
    3. Corrupted prompt + clean_incorrect answer → incorrect log prob (no hooks needed)

    Following IOI convention: both correct and incorrect are CLEAN answers,
    evaluated on corrupted prompt. This ensures the same metric is used for both baselines.

    Metric = normalized (correct_lp - incorrect_lp) on corrupted prompt.
    Gradient flows from this diff metric back through the corrupted activations.
    """
    num_examples = len(clean_correct_ids)
    num_batches = (num_examples + batch_size - 1) // batch_size
    attribution = np.zeros(output_shape)

    for batch_idx in tqdm(range(num_batches), desc=f"  {desc}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_clean_ids = clean_correct_ids[start:end]
        batch_clean_plens = clean_correct_plens[start:end]
        batch_corr_correct_ids = corrupted_correct_ids[start:end]
        batch_corr_correct_plens = corrupted_correct_plens[start:end]
        batch_corr_incorrect_ids = corrupted_incorrect_ids[start:end]
        batch_corr_incorrect_plens = corrupted_incorrect_plens[start:end]

        # Pad each set independently
        batch_clean_padded = pad_sequence_inputs(batch_clean_ids)
        batch_corr_correct_padded = pad_sequence_inputs(batch_corr_correct_ids)
        batch_corr_incorrect_padded = pad_sequence_inputs(batch_corr_incorrect_ids)

        # First: run corrupted+incorrect (no hooks needed, just get log probs)
        with model.trace(batch_corr_incorrect_padded) as tracer:
            inc_logits = model.lm_head.output.save()

        inc_logits_val = inc_logits.value.cpu() if hasattr(inc_logits, "value") else inc_logits.cpu()
        batch_incorrect_lps = []
        for i in range(len(batch_corr_incorrect_ids)):
            seq_len = batch_corr_incorrect_ids[i].shape[0]
            example_logits = inc_logits_val[i, :seq_len, :]
            example_ids = batch_corr_incorrect_ids[i]
            lp = compute_sequence_log_prob(example_logits, example_ids, batch_corr_incorrect_plens[i])
            batch_incorrect_lps.append(lp.detach())

        # Now: clean + corrupted correct with hooks and gradient
        clean_out = []
        corrupted_out = []
        corrupted_grads = []

        with model.trace() as tracer:
            # Clean forward pass (just to get activations)
            with tracer.invoke(batch_clean_padded):
                for layer in layers:
                    hook = hook_fn(layer, model_config)
                    clean_out.append(hook.save())

            # Corrupted + correct forward pass (activations + gradient)
            with tracer.invoke(batch_corr_correct_padded):
                for layer in layers:
                    hook = hook_fn(layer, model_config)
                    hook.requires_grad_(True)
                    corrupted_out.append(hook.save())
                    corrupted_grads.append(hook.grad.save())

                # Compute correct log probs
                logits = model.lm_head.output.save()
                batch_correct_lps = []
                logits_cpu = logits.cpu()
                for i in range(len(batch_corr_correct_ids)):
                    seq_len = batch_corr_correct_ids[i].shape[0]
                    example_logits = logits_cpu[i, :seq_len, :]
                    example_ids = batch_corr_correct_ids[i]
                    lp = compute_sequence_log_prob(example_logits, example_ids, batch_corr_correct_plens[i])
                    batch_correct_lps.append(lp)

                # Metric: normalized (correct - incorrect) diff
                correct_lps = torch.stack(batch_correct_lps)
                incorrect_lps = torch.stack(batch_incorrect_lps)
                tf_diff = (correct_lps - incorrect_lps).mean()
                normalized = (tf_diff - corrupted_baseline) / (clean_baseline - corrupted_baseline)
                normalized.backward()

        # Accumulate attribution
        for layer_idx in range(len(layers)):
            grad = corrupted_grads[layer_idx].value
            corr = corrupted_out[layer_idx].value
            cln = clean_out[layer_idx].value

            if grad is None:
                continue

            attr = reduce_fn(grad, cln, corr, batch_clean_plens, batch_corr_correct_plens)
            attribution[layer_idx] += attr

        del clean_out, corrupted_out, corrupted_grads, inc_logits_val
        torch.cuda.empty_cache()
        gc.collect()

    return attribution


def run_neuron_attribution(model, model_config,
                           clean_correct_ids, clean_correct_plens,
                           corrupted_correct_ids, corrupted_correct_plens,
                           corrupted_incorrect_ids, corrupted_incorrect_plens,
                           clean_baseline, corrupted_baseline,
                           batch_size=8):
    """
    Run attribution patching over MLP neurons using full-sequence TF.

    Returns:
        np.ndarray of shape [num_layers, intermediate_size]
    """
    num_layers = model_config["num_layers"]
    intermediate_size = model_config["intermediate_size"]
    layers = get_layers(model, model_config)

    def reduce_fn(grad, cln, corr, clean_plens, corr_plens):
        # For each example, compare activations at the last prompt token
        batch_attr = []
        for i in range(grad.shape[0]):
            # Use last prompt position (before answer starts)
            c_pos = clean_plens[i] - 1
            r_pos = corr_plens[i] - 1
            g = grad[i, r_pos, :]
            c = cln[i, c_pos, :]
            r = corr[i, r_pos, :]
            batch_attr.append(g * (c - r))
        attr = torch.stack(batch_attr).sum(dim=0)
        return attr.float().detach().cpu().numpy()

    return _attribution_loop(
        model, model_config, layers,
        clean_correct_ids, clean_correct_plens,
        corrupted_correct_ids, corrupted_correct_plens,
        corrupted_incorrect_ids, corrupted_incorrect_plens,
        clean_baseline, corrupted_baseline,
        batch_size,
        hook_fn=_get_mlp_hook,
        reduce_fn=reduce_fn,
        output_shape=(num_layers, intermediate_size),
        desc="Neuron attribution",
    )


def run_head_attribution(model, model_config,
                         clean_correct_ids, clean_correct_plens,
                         corrupted_correct_ids, corrupted_correct_plens,
                         corrupted_incorrect_ids, corrupted_incorrect_plens,
                         clean_baseline, corrupted_baseline,
                         batch_size=8):
    """
    Run attribution patching over attention heads using full-sequence TF.

    Returns:
        np.ndarray of shape [num_layers, num_heads]
    """
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    layers = get_layers(model, model_config)

    def reduce_fn(grad, cln, corr, clean_plens, corr_plens):
        batch_attr = []
        for i in range(grad.shape[0]):
            c_pos = clean_plens[i] - 1
            r_pos = corr_plens[i] - 1
            g = grad[i, r_pos, :]
            c = cln[i, c_pos, :]
            r = corr[i, r_pos, :]
            attr_per_dim = g * (c - r)
            # Reshape to [num_heads, head_dim] and sum over head_dim
            attr_per_head = attr_per_dim.reshape(num_heads, head_dim).sum(dim=1)
            batch_attr.append(attr_per_head)
        attr = torch.stack(batch_attr).sum(dim=0)
        return attr.float().detach().cpu().numpy()

    return _attribution_loop(
        model, model_config, layers,
        clean_correct_ids, clean_correct_plens,
        corrupted_correct_ids, corrupted_correct_plens,
        corrupted_incorrect_ids, corrupted_incorrect_plens,
        clean_baseline, corrupted_baseline,
        batch_size,
        hook_fn=_get_attn_hook,
        reduce_fn=reduce_fn,
        output_shape=(num_layers, num_heads),
        desc="Head attribution",
    )


def _get_mlp_hook(layer, model_config):
    """Get the MLP hook point for a layer."""
    family = model_config["family"]
    if family == "gpt2":
        return layer.mlp.c_proj.input
    else:  # qwen, llama, mistral
        return layer.mlp.down_proj.input


def _get_attn_hook(layer, model_config):
    """Get the attention hook point for a layer."""
    family = model_config["family"]
    if family == "gpt2":
        return layer.attn.c_proj.input
    else:  # qwen, llama, mistral
        return layer.self_attn.o_proj.input


# =============================================================================
# Single-answer attribution wrappers
# =============================================================================

def run_neuron_attribution_single(model, model_config,
                                   clean_answer_ids, clean_answer_plens,
                                   corrupted_answer_ids, corrupted_answer_plens,
                                   clean_baseline, corrupted_baseline,
                                   batch_size=8):
    """
    Run attribution patching over MLP neurons for single-answer mode.
    """
    num_layers = model_config["num_layers"]
    intermediate_size = model_config["intermediate_size"]
    layers = get_layers(model, model_config)

    def reduce_fn(grad, cln, corr, clean_plens, corr_plens):
        batch_attr = []
        for i in range(grad.shape[0]):
            c_pos = clean_plens[i] - 1
            r_pos = corr_plens[i] - 1
            g = grad[i, r_pos, :]
            c = cln[i, c_pos, :]
            r = corr[i, r_pos, :]
            batch_attr.append(g * (c - r))
        attr = torch.stack(batch_attr).sum(dim=0)
        return attr.float().detach().cpu().numpy()

    return _attribution_loop_single_answer(
        model, model_config, layers,
        clean_answer_ids, clean_answer_plens,
        corrupted_answer_ids, corrupted_answer_plens,
        clean_baseline, corrupted_baseline,
        batch_size,
        hook_fn=_get_mlp_hook,
        reduce_fn=reduce_fn,
        output_shape=(num_layers, intermediate_size),
        desc="Neuron attribution (single-answer)",
    )


def run_head_attribution_single(model, model_config,
                                 clean_answer_ids, clean_answer_plens,
                                 corrupted_answer_ids, corrupted_answer_plens,
                                 clean_baseline, corrupted_baseline,
                                 batch_size=8):
    """
    Run attribution patching over attention heads for single-answer mode.
    """
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    layers = get_layers(model, model_config)

    def reduce_fn(grad, cln, corr, clean_plens, corr_plens):
        batch_attr = []
        for i in range(grad.shape[0]):
            c_pos = clean_plens[i] - 1
            r_pos = corr_plens[i] - 1
            g = grad[i, r_pos, :]
            c = cln[i, c_pos, :]
            r = corr[i, r_pos, :]
            attr_per_dim = g * (c - r)
            attr_per_head = attr_per_dim.reshape(num_heads, head_dim).sum(dim=1)
            batch_attr.append(attr_per_head)
        attr = torch.stack(batch_attr).sum(dim=0)
        return attr.float().detach().cpu().numpy()

    return _attribution_loop_single_answer(
        model, model_config, layers,
        clean_answer_ids, clean_answer_plens,
        corrupted_answer_ids, corrupted_answer_plens,
        clean_baseline, corrupted_baseline,
        batch_size,
        hook_fn=_get_attn_hook,
        reduce_fn=reduce_fn,
        output_shape=(num_layers, num_heads),
        desc="Head attribution (single-answer)",
    )