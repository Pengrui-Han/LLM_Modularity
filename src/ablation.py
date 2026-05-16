"""Ablation experiments: load attribution scores, ablate, evaluate.

Uses full-sequence teacher forcing: prompt+answer forward pass,
loss computed only over answer tokens.

Supports two ablation modes:
  - "zero": set selected neurons/heads to 0 (original method)
  - "corrupted": replace selected neurons/heads with corrupted activations (Hanna et al. style)
"""

import os
import torch
import json
import numpy as np
import gc
from tqdm import tqdm

from .model_utils import get_layers
from .metrics import (compute_sequence_log_prob, compute_joint_log_prob,
                      get_teacher_forcing_diff, get_accuracy,
                      get_single_answer_accuracy)
from .data_utils import pad_sequence_inputs


# =============================================================================
# Memory-management helper
# =============================================================================
def _force_cleanup(tag=""):
    """Force garbage collection and CUDA cache cleanup.

    Called between evaluation phases to prevent GPU memory accumulation /
    OOM on large-model runs across many ablation batches. `tag` documents
    the call site.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Attribution-score loading and neuron selection
# =============================================================================

def load_attribution_scores(scores_path):
    """
    Load saved attribution scores.

    Returns:
        np.ndarray of shape [num_layers, num_units]
    """
    tensor = torch.load(scores_path, map_location="cpu")
    return tensor.numpy()


def select_neurons(attribution, percentage, sign="positive"):
    """
    Select top neurons by attribution score.

    Args:
        attribution: np.ndarray [num_layers, num_units]
        percentage: float, e.g. 1.0 for top 1%
        sign: 'positive', 'negative', or 'absolute'

    Returns:
        dict: {layer_idx: [neuron_indices]}
        list: [(layer, neuron, score)] sorted by importance
    """
    num_layers, num_units = attribution.shape
    total = num_layers * num_units
    top_k = max(1, int(total * percentage / 100.0))

    all_units = []
    for layer_idx in range(num_layers):
        for unit_idx in range(num_units):
            all_units.append((layer_idx, unit_idx, attribution[layer_idx, unit_idx]))

    if sign == "positive":
        candidates = [(l, n, s) for l, n, s in all_units if s > 0]
        candidates.sort(key=lambda x: x[2], reverse=True)
    elif sign == "negative":
        candidates = [(l, n, s) for l, n, s in all_units if s < 0]
        candidates.sort(key=lambda x: x[2])
    elif sign == "absolute":
        candidates = all_units
        candidates.sort(key=lambda x: abs(x[2]), reverse=True)
    else:
        raise ValueError(f"sign must be 'positive', 'negative', or 'absolute', got '{sign}'")

    selected = candidates[:top_k]

    neurons_dict = {}
    for l, n, s in selected:
        if l not in neurons_dict:
            neurons_dict[l] = []
        neurons_dict[l].append(n)

    return neurons_dict, selected


def get_per_layer_counts(neurons_dict):
    """Get count of neurons per layer."""
    return {l: len(ns) for l, ns in neurons_dict.items()}


def generate_layer_matched_random(per_layer_counts, num_units, seed=None, exclude=None):
    """
    Generate random neurons with same per-layer distribution.

    Args:
        per_layer_counts: dict {layer_idx: count}
        num_units: total units per layer
        seed: optional random seed
        exclude: optional dict {layer_idx: list of indices to exclude}
    """
    if seed is not None:
        np.random.seed(seed)
    random_dict = {}
    for layer_idx, count in per_layer_counts.items():
        if exclude and layer_idx in exclude:
            excluded_set = set(exclude[layer_idx])
            available = [i for i in range(num_units) if i not in excluded_set]
            if len(available) < count:
                random_indices = np.array(available)
            else:
                random_indices = np.random.choice(available, size=count, replace=False)
        else:
            random_indices = np.random.choice(num_units, size=count, replace=False)
        random_dict[layer_idx] = random_indices.tolist()
    return random_dict


def generate_truly_random(num_layers, num_units, total_count, seed=None):
    if seed is not None:
        np.random.seed(seed)
    random_layers = np.random.randint(0, num_layers, size=total_count)
    random_neurons = np.random.randint(0, num_units, size=total_count)
    random_dict = {}
    for l, n in zip(random_layers, random_neurons):
        l, n = int(l), int(n)
        if l not in random_dict:
            random_dict[l] = []
        random_dict[l].append(n)
    return random_dict


# =============================================================================
# Core ablation evaluation functions
# =============================================================================

def _run_ablation_sequence_eval(model, model_config, input_ids_list, prompt_lens,
                                neurons_to_ablate, component="neurons",
                                ablation_type="zero",
                                corrupted_ids_list=None, corrupted_plens=None,
                                batch_size=8, desc=""):
    """
    Run forward pass with ablation on prompt+answer sequences.
    Returns per-example log probs over answer tokens.
    """
    if ablation_type == "corrupted" and neurons_to_ablate:
        assert corrupted_ids_list is not None, "corrupted_ids_list required for corrupted ablation"
        assert corrupted_plens is not None, "corrupted_plens required for corrupted ablation"
        assert len(corrupted_ids_list) == len(input_ids_list), \
            f"Mismatch: {len(corrupted_ids_list)} corrupted vs {len(input_ids_list)} clean examples"

    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    layers = get_layers(model, model_config)
    all_log_probs = []

    for batch_idx in tqdm(range(num_batches), desc=f"  {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids)

        if not neurons_to_ablate:
            # No ablation — just run clean forward pass
            with model.trace(batch_padded) as tracer:
                logits = model.lm_head.output.save()

        elif ablation_type == "zero":
            # ===== ZERO ABLATION (original) =====
            with model.trace(batch_padded) as tracer:
                for layer_idx in sorted(neurons_to_ablate.keys()):
                    layer = layers[layer_idx]
                    indices = neurons_to_ablate[layer_idx]

                    if component == "neurons":
                        layer.mlp.down_proj.input[:, :, indices] = 0.0
                    elif component == "heads":
                        head_dim = model_config["head_dim"]
                        for head_idx in indices:
                            start_dim = head_idx * head_dim
                            end_dim = (head_idx + 1) * head_dim
                            if model_config["family"] == "gpt2":
                                layer.attn.c_proj.input[:, :, start_dim:end_dim] = 0.0
                            else:
                                layer.self_attn.o_proj.input[:, :, start_dim:end_dim] = 0.0

                logits = model.lm_head.output.save()

        elif ablation_type == "corrupted":
            # ===== CORRUPTED ABLATION (Hanna et al. style) =====
            batch_corr_ids = corrupted_ids_list[start:end]
            batch_corr_padded = pad_sequence_inputs(batch_corr_ids)

            # --- Pass 1: Get corrupted activations ---
            corrupted_acts = {}
            with torch.no_grad():
                with model.trace(batch_corr_padded) as tracer:
                    for layer_idx in sorted(neurons_to_ablate.keys()):
                        layer = layers[layer_idx]
                        if component == "neurons":
                            corrupted_acts[layer_idx] = layer.mlp.down_proj.input.save()
                        elif component == "heads":
                            if model_config["family"] == "gpt2":
                                corrupted_acts[layer_idx] = layer.attn.c_proj.input.save()
                            else:
                                corrupted_acts[layer_idx] = layer.self_attn.o_proj.input.save()

            # Move to CPU to save GPU memory, detach
            corrupted_acts_cpu = {}
            for layer_idx, act in corrupted_acts.items():
                val = act.value if hasattr(act, "value") else act
                corrupted_acts_cpu[layer_idx] = val.detach().cpu()
            del corrupted_acts
            torch.cuda.empty_cache()

            # --- Pass 2: Clean forward pass with corrupted replacement ---
            with torch.no_grad():
                with model.trace(batch_padded) as tracer:
                    for layer_idx in sorted(neurons_to_ablate.keys()):
                        layer = layers[layer_idx]
                        indices = neurons_to_ablate[layer_idx]
                        corr_act = corrupted_acts_cpu[layer_idx].to(batch_padded.device)

                        if component == "neurons":
                            layer.mlp.down_proj.input[:, :, indices] = corr_act[:, :, indices]
                        elif component == "heads":
                            head_dim = model_config["head_dim"]
                            for head_idx in indices:
                                start_dim = head_idx * head_dim
                                end_dim = (head_idx + 1) * head_dim
                                if model_config["family"] == "gpt2":
                                    layer.attn.c_proj.input[:, :, start_dim:end_dim] = \
                                        corr_act[:, :, start_dim:end_dim]
                                else:
                                    layer.self_attn.o_proj.input[:, :, start_dim:end_dim] = \
                                        corr_act[:, :, start_dim:end_dim]

                    logits = model.lm_head.output.save()

            del corrupted_acts_cpu
            torch.cuda.empty_cache()

        else:
            raise ValueError(f"ablation_type must be 'zero' or 'corrupted', got '{ablation_type}'")

        logits_val = logits.value.cpu() if hasattr(logits, "value") else logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            example_logits = logits_val[i, :seq_len, :]
            example_ids = batch_ids[i]
            lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            all_log_probs.append(lp)

        # Free per-batch GPU memory to prevent accumulation
        del logits_val, logits
        gc.collect()
        torch.cuda.empty_cache()

    return torch.stack(all_log_probs)


def run_ablation_eval(model, model_config,
                      correct_ids, correct_plens,
                      incorrect_ids, incorrect_plens,
                      normalized_metric,
                      neurons_to_ablate, component="neurons",
                      ablation_type="zero",
                      corrupted_correct_ids=None, corrupted_correct_plens=None,
                      corrupted_incorrect_ids=None, corrupted_incorrect_plens=None,
                      batch_size=8, label=""):
    """
    Run ablation and evaluate using full-sequence teacher forcing.
    Supports both zero and corrupted ablation.
    """
    correct_lps = _run_ablation_sequence_eval(
        model, model_config, correct_ids, correct_plens,
        neurons_to_ablate, component=component,
        ablation_type=ablation_type,
        corrupted_ids_list=corrupted_correct_ids,
        corrupted_plens=corrupted_correct_plens,
        batch_size=batch_size, desc=f"Ablate correct ({label})"
    )

    _force_cleanup(f"between-correct-incorrect ({label})")

    incorrect_lps = _run_ablation_sequence_eval(
        model, model_config, incorrect_ids, incorrect_plens,
        neurons_to_ablate, component=component,
        ablation_type=ablation_type,
        corrupted_ids_list=corrupted_incorrect_ids,
        corrupted_plens=corrupted_incorrect_plens,
        batch_size=batch_size, desc=f"Ablate incorrect ({label})"
    )

    score = normalized_metric(correct_lps, incorrect_lps).item()
    acc = get_accuracy(correct_lps, incorrect_lps)

    return score, acc


def _run_ablation_sequence_eval_both(model, model_config, input_ids_list, prompt_lens,
                                      neurons_to_ablate, component="neurons",
                                      ablation_type="zero",
                                      corrupted_ids_list=None, corrupted_plens=None,
                                      batch_size=8, desc=""):
    """
    Run forward pass with ablation and compute BOTH conditional and joint log probs.
    Supports both zero and corrupted ablation.
    """
    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    layers = get_layers(model, model_config)
    all_cond_lps = []
    all_joint_lps = []

    if ablation_type == "corrupted" and neurons_to_ablate:
        assert corrupted_ids_list is not None, "corrupted_ids_list required for corrupted ablation"

    for batch_idx in tqdm(range(num_batches), desc=f"  {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids)

        if not neurons_to_ablate:
            with model.trace(batch_padded) as tracer:
                logits = model.lm_head.output.save()

        elif ablation_type == "zero":
            with model.trace(batch_padded) as tracer:
                for layer_idx in sorted(neurons_to_ablate.keys()):
                    layer = layers[layer_idx]
                    indices = neurons_to_ablate[layer_idx]

                    if component == "neurons":
                        layer.mlp.down_proj.input[:, :, indices] = 0.0
                    elif component == "heads":
                        head_dim = model_config["head_dim"]
                        for head_idx in indices:
                            start_dim = head_idx * head_dim
                            end_dim = (head_idx + 1) * head_dim
                            if model_config["family"] == "gpt2":
                                layer.attn.c_proj.input[:, :, start_dim:end_dim] = 0.0
                            else:
                                layer.self_attn.o_proj.input[:, :, start_dim:end_dim] = 0.0

                logits = model.lm_head.output.save()

        elif ablation_type == "corrupted":
            batch_corr_ids = corrupted_ids_list[start:end]
            batch_corr_padded = pad_sequence_inputs(batch_corr_ids)

            # Pass 1: Get corrupted activations
            corrupted_acts = {}
            with torch.no_grad():
                with model.trace(batch_corr_padded) as tracer:
                    for layer_idx in sorted(neurons_to_ablate.keys()):
                        layer = layers[layer_idx]
                        if component == "neurons":
                            corrupted_acts[layer_idx] = layer.mlp.down_proj.input.save()
                        elif component == "heads":
                            if model_config["family"] == "gpt2":
                                corrupted_acts[layer_idx] = layer.attn.c_proj.input.save()
                            else:
                                corrupted_acts[layer_idx] = layer.self_attn.o_proj.input.save()

            corrupted_acts_cpu = {}
            for layer_idx, act in corrupted_acts.items():
                val = act.value if hasattr(act, "value") else act
                corrupted_acts_cpu[layer_idx] = val.detach().cpu()
            del corrupted_acts
            torch.cuda.empty_cache()

            # Pass 2: Clean forward with corrupted replacement
            with model.trace(batch_padded) as tracer:
                for layer_idx in sorted(neurons_to_ablate.keys()):
                    layer = layers[layer_idx]
                    indices = neurons_to_ablate[layer_idx]
                    corr_act = corrupted_acts_cpu[layer_idx].to(batch_padded.device)

                    if component == "neurons":
                        layer.mlp.down_proj.input[:, :, indices] = corr_act[:, :, indices]
                    elif component == "heads":
                        head_dim = model_config["head_dim"]
                        for head_idx in indices:
                            start_dim = head_idx * head_dim
                            end_dim = (head_idx + 1) * head_dim
                            if model_config["family"] == "gpt2":
                                layer.attn.c_proj.input[:, :, start_dim:end_dim] = \
                                    corr_act[:, :, start_dim:end_dim]
                            else:
                                layer.self_attn.o_proj.input[:, :, start_dim:end_dim] = \
                                    corr_act[:, :, start_dim:end_dim]

                logits = model.lm_head.output.save()

            del corrupted_acts_cpu
            torch.cuda.empty_cache()

        else:
            raise ValueError(f"ablation_type must be 'zero' or 'corrupted', got '{ablation_type}'")

        logits_val = logits.value.cpu() if hasattr(logits, "value") else logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            example_logits = logits_val[i, :seq_len, :]
            example_ids = batch_ids[i]
            cond_lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            joint_lp = compute_joint_log_prob(example_logits, example_ids)
            all_cond_lps.append(cond_lp)
            all_joint_lps.append(joint_lp)

        # Free per-batch GPU memory to prevent accumulation
        del logits_val, logits
        gc.collect()
        torch.cuda.empty_cache()

    return torch.stack(all_cond_lps), torch.stack(all_joint_lps)


def run_ablation_eval_single(model, model_config,
                              clean_answer_ids, clean_answer_plens,
                              corrupted_answer_ids, corrupted_answer_plens,
                              normalized_metric,
                              neurons_to_ablate, component="neurons",
                              ablation_type="zero",
                              batch_size=8, label=""):
    """
    Run ablation and evaluate for single-answer mode.
    Supports both zero and corrupted ablation.
    """
    clean_cond_lps, clean_joint_lps = _run_ablation_sequence_eval_both(
        model, model_config, clean_answer_ids, clean_answer_plens,
        neurons_to_ablate, component=component,
        ablation_type=ablation_type,
        corrupted_ids_list=corrupted_answer_ids if ablation_type == "corrupted" else None,
        corrupted_plens=corrupted_answer_plens if ablation_type == "corrupted" else None,
        batch_size=batch_size, desc=f"Ablate clean ({label})"
    )

    _force_cleanup(f"between-clean-corrupted-single ({label})")

    corrupted_cond_lps, corrupted_joint_lps = _run_ablation_sequence_eval_both(
        model, model_config, corrupted_answer_ids, corrupted_answer_plens,
        neurons_to_ablate, component=component,
        ablation_type=ablation_type,
        corrupted_ids_list=clean_answer_ids if ablation_type == "corrupted" else None,
        corrupted_plens=clean_answer_plens if ablation_type == "corrupted" else None,
        batch_size=batch_size, desc=f"Ablate corrupted ({label})"
    )

    score = normalized_metric(clean_cond_lps, corrupted_cond_lps).item()
    acc = get_single_answer_accuracy(clean_joint_lps, corrupted_joint_lps)

    return score, acc


def load_random_baselines(random_cache_path):
    """Load cached random baseline results."""
    if os.path.exists(random_cache_path):
        with open(random_cache_path, "r") as f:
            return json.load(f)
    return None


def save_random_baselines(random_cache_path, data):
    """Save random baseline results."""
    os.makedirs(os.path.dirname(random_cache_path), exist_ok=True)
    with open(random_cache_path, "w") as f:
        json.dump(data, f, indent=2)


def run_full_ablation_experiment(model, model_config,
                                normalized_metric,
                                neurons_dict, selected_list,
                                eval_inputs=None,
                                num_random_trials=3, component="neurons",
                                ablation_type="zero",
                                batch_size=8, sign="positive",
                                clean_accuracy=None,
                                truly_random_cache_path=None,
                                layer_matched_cache_path=None,
                                overwrite=False,
                                skip_layer_matched=False):
    """
    Run full ablation experiment: target ablation + random baselines.
    Supports both dual-answer and single-answer modes via eval_inputs dict.
    Supports both zero and corrupted ablation via ablation_type.
    """
    num_layers = model_config["num_layers"]
    if component == "neurons":
        num_units = model_config["intermediate_size"]
    else:
        num_units = model_config["num_heads"]

    per_layer_counts = get_per_layer_counts(neurons_dict)
    total_count = sum(per_layer_counts.values())
    single_answer = eval_inputs.get("single_answer", False)

    print(f"  Ablation type: {ablation_type}")

    def _run_eval(neurons_to_ablate, label):
        """Run ablation eval in either mode."""
        if single_answer:
            return run_ablation_eval_single(
                model, model_config,
                eval_inputs["clean_answer_ids"], eval_inputs["clean_answer_plens"],
                eval_inputs["corrupted_answer_ids"], eval_inputs["corrupted_answer_plens"],
                normalized_metric, neurons_to_ablate,
                component=component, ablation_type=ablation_type,
                batch_size=batch_size, label=label,
            )
        else:
            return run_ablation_eval(
                model, model_config,
                eval_inputs["correct_ids"], eval_inputs["correct_plens"],
                eval_inputs["incorrect_ids"], eval_inputs["incorrect_plens"],
                normalized_metric, neurons_to_ablate,
                component=component, ablation_type=ablation_type,
                corrupted_correct_ids=eval_inputs.get("corrupted_correct_ids"),
                corrupted_correct_plens=eval_inputs.get("corrupted_correct_plens"),
                corrupted_incorrect_ids=eval_inputs.get("corrupted_incorrect_ids"),
                corrupted_incorrect_plens=eval_inputs.get("corrupted_incorrect_plens"),
                batch_size=batch_size, label=label,
            )

    # No ablation: score=1.0 by definition, accuracy=clean_accuracy from baselines
    no_abl_score = 1.0
    no_abl_acc = 1.0  # both_correct subset: by definition all clean examples are correct
    print(f"\n  No ablation (from baselines)...")
    print(f"    Score: {no_abl_score:.4f}, Accuracy: {no_abl_acc:.2%}")

    # Target ablation
    print(f"\n  Target ablation ({sign}, {total_count} {component}, type={ablation_type})...")
    target_score, target_acc = _run_eval(neurons_dict, f"{sign}")
    print(f"    Score: {target_score:.4f}, Accuracy: {target_acc:.2%}")

    _force_cleanup("after-target-ablation")

    # --- Layer-matched random (source-specific cache) ---
    if layer_matched_cache_path and ablation_type == "corrupted":
        base, ext = os.path.splitext(layer_matched_cache_path)
        layer_matched_cache_path = f"{base}_corrupted{ext}"

    if skip_layer_matched:
        print(f"\n  Layer-matched random: SKIPPED")
        lm_scores, lm_accs = [], []
    else:
        _force_cleanup("before-layer-matched")

        lm_cached = load_random_baselines(layer_matched_cache_path) if (layer_matched_cache_path and not overwrite) else None

        if lm_cached:
            lm_scores = lm_cached.get("scores", [])
            lm_accs = lm_cached.get("accuracies", [])
        else:
            lm_scores, lm_accs = [], []

        need_lm = num_random_trials - len(lm_scores)
        if need_lm > 0:
            print(f"\n  Layer-matched random ({need_lm} new trials, {len(lm_scores)} cached)...")
            for trial in range(need_lm):
                random_dict = generate_layer_matched_random(per_layer_counts, num_units, exclude=neurons_dict)
                score, acc = _run_eval(random_dict, f"layer-random-{len(lm_scores)+1}")
                lm_scores.append(score)
                lm_accs.append(acc)
                print(f"    Trial {len(lm_scores)}: Score={score:.4f}, Acc={acc:.2%}")
                _force_cleanup(f"after-layer-matched-trial-{len(lm_scores)}")
        else:
            print(f"\n  Layer-matched random ({num_random_trials} cached, skipping)...")
            for i in range(num_random_trials):
                print(f"    Trial {i+1}: Score={lm_scores[i]:.4f}, Acc={lm_accs[i]:.2%}")

        if layer_matched_cache_path:
            os.makedirs(os.path.dirname(layer_matched_cache_path), exist_ok=True)
            save_random_baselines(layer_matched_cache_path, {
                "scores": lm_scores, "accuracies": lm_accs,
                "metadata": {
                    "total_neurons": total_count,
                    "per_layer_counts": {str(k): v for k, v in per_layer_counts.items()},
                    "ablation_type": ablation_type,
                },
            })

    _force_cleanup("after-layer-matched")

    # --- Truly random (shared across sources, target-specific cache) ---
    if truly_random_cache_path and ablation_type == "corrupted":
        base, ext = os.path.splitext(truly_random_cache_path)
        truly_random_cache_path = f"{base}_corrupted{ext}"

    tr_cached = load_random_baselines(truly_random_cache_path) if (truly_random_cache_path and not overwrite) else None

    if tr_cached:
        tr_scores = tr_cached.get("scores", [])
        tr_accs = tr_cached.get("accuracies", [])
    else:
        tr_scores, tr_accs = [], []

    need_tr = num_random_trials - len(tr_scores)
    if need_tr > 0:
        print(f"\n  Truly random ({need_tr} new trials, {len(tr_scores)} cached)...")
        _force_cleanup("before-truly-random")
        for trial in range(need_tr):
            random_dict = generate_truly_random(num_layers, num_units, total_count)
            score, acc = _run_eval(random_dict, f"truly-random-{len(tr_scores)+1}")
            tr_scores.append(score)
            tr_accs.append(acc)
            print(f"    Trial {len(tr_scores)}: Score={score:.4f}, Acc={acc:.2%}")
            _force_cleanup(f"after-truly-random-trial-{len(tr_scores)}")
    else:
        print(f"\n  Truly random ({num_random_trials} cached, skipping)...")
        for i in range(num_random_trials):
            print(f"    Trial {i+1}: Score={tr_scores[i]:.4f}, Acc={tr_accs[i]:.2%}")

    if truly_random_cache_path:
        os.makedirs(os.path.dirname(truly_random_cache_path), exist_ok=True)
        save_random_baselines(truly_random_cache_path, {
            "scores": tr_scores, "accuracies": tr_accs,
            "metadata": {
                "total_neurons": total_count,
                "ablation_type": ablation_type,
            },
        })

    lm_scores_used = lm_scores[:num_random_trials]
    lm_accs_used = lm_accs[:num_random_trials]
    tr_scores_used = tr_scores[:num_random_trials]
    tr_accs_used = tr_accs[:num_random_trials]

    lm_result = {
        "scores": lm_scores_used, "accuracies": lm_accs_used,
        "mean_score": float(np.mean(lm_scores_used)) if lm_scores_used else None,
        "std_score": float(np.std(lm_scores_used)) if lm_scores_used else None,
        "mean_accuracy": float(np.mean(lm_accs_used)) if lm_accs_used else None,
        "std_accuracy": float(np.std(lm_accs_used)) if lm_accs_used else None,
    }

    return {
        "no_ablation": {"score": no_abl_score, "accuracy": no_abl_acc},
        "target": {"score": target_score, "accuracy": target_acc},
        "layer_matched_random": lm_result,
        "truly_random": {
            "scores": tr_scores_used, "accuracies": tr_accs_used,
            "mean_score": float(np.mean(tr_scores_used)) if tr_scores_used else None,
            "std_score": float(np.std(tr_scores_used)) if tr_scores_used else None,
            "mean_accuracy": float(np.mean(tr_accs_used)) if tr_accs_used else None,
            "std_accuracy": float(np.std(tr_accs_used)) if tr_accs_used else None,
        },
        "ablation_type": ablation_type,
    }
