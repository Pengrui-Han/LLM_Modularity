"""
Run evaluation only: compute baselines (accuracy + TF diff) for tasks.
No attribution or ablation. Uses direct HuggingFace forward pass (no nnsight trace).

Usage:
    # Single task
    python scripts/run_eval.py --model Qwen/Qwen2.5-32B-Instruct --domain MD --task arithmetic_addition

    # All tasks in a domain
    python scripts/run_eval.py --model Qwen/Qwen2.5-32B-Instruct --domain MD

    # All tasks, overwrite cached baselines
    python scripts/run_eval.py --model Qwen/Qwen2.5-32B-Instruct --domain MD --overwrite
"""

import argparse
import os
import sys
import json
import gc
import random

import torch
from tqdm import tqdm

random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_model
from src.data_utils import (
    load_task_config, load_task_data, format_prompts, prepare_sequence_inputs,
    pad_sequence_inputs,
)
from src.metrics import (
    compute_sequence_log_prob, compute_joint_log_prob,
    get_teacher_forcing_diff, get_accuracy,
    get_single_answer_accuracy,
)


def get_model_short_name(model_name):
    return model_name.replace("/", "_").replace(".", "-")


def get_debug_samples(model, tokenizer, input_ids_list, prompt_lens, sample_indices):
    """
    For a list of sample indices, run forward pass and get model's top predicted token
    at the last prompt position.
    Returns list of (index, top_token_str, top_token_logprob).
    """
    hf_model = model._model if hasattr(model, '_model') else model
    results = []

    for idx in sample_indices:
        ids = input_ids_list[idx].unsqueeze(0).cuda()
        plen = prompt_lens[idx]

        with torch.no_grad():
            outputs = hf_model(ids)
            logits = outputs.logits.cpu()

        # Logits at last prompt position (predicts first answer token)
        last_prompt_logits = logits[0, plen - 1, :]
        log_probs = torch.log_softmax(last_prompt_logits, dim=-1)
        top_idx = log_probs.argmax().item()
        top_token = tokenizer.decode([top_idx])
        top_lp = log_probs[top_idx].item()

        del logits, outputs

        results.append((idx, top_token, top_lp))

    torch.cuda.empty_cache()
    return results


# =============================================================================
# Direct forward pass (no nnsight trace)
# =============================================================================

def _forward_log_probs(model, input_ids_list, prompt_lens, batch_size=8, desc="Eval"):
    """
    Run forward pass using direct HuggingFace model (no nnsight trace).
    Returns per-example conditional log probs.
    """
    hf_model = model._model if hasattr(model, '_model') else model

    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    all_log_probs = []

    for batch_idx in tqdm(range(num_batches), desc=f"    {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids).cuda()

        with torch.no_grad():
            outputs = hf_model(batch_padded)
            logits = outputs.logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            example_logits = logits[i, :seq_len, :]
            example_ids = batch_ids[i]
            lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            all_log_probs.append(lp)

        del logits, outputs
        torch.cuda.empty_cache()

    return torch.stack(all_log_probs)


def _forward_log_probs_both(model, input_ids_list, prompt_lens, batch_size=8, desc="Eval"):
    """
    Run forward pass and compute BOTH conditional and joint log probs.
    Direct HuggingFace forward, no nnsight trace.
    """
    hf_model = model._model if hasattr(model, '_model') else model

    num_examples = len(input_ids_list)
    num_batches = (num_examples + batch_size - 1) // batch_size
    all_cond_lps = []
    all_joint_lps = []

    for batch_idx in tqdm(range(num_batches), desc=f"    {desc}", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_examples)

        batch_ids = input_ids_list[start:end]
        batch_plens = prompt_lens[start:end]
        batch_padded = pad_sequence_inputs(batch_ids).cuda()

        with torch.no_grad():
            outputs = hf_model(batch_padded)
            logits = outputs.logits.cpu()

        for i in range(len(batch_ids)):
            seq_len = batch_ids[i].shape[0]
            example_logits = logits[i, :seq_len, :]
            example_ids = batch_ids[i]
            cond_lp = compute_sequence_log_prob(example_logits, example_ids, batch_plens[i])
            joint_lp = compute_joint_log_prob(example_logits, example_ids)
            all_cond_lps.append(cond_lp)
            all_joint_lps.append(joint_lp)

        del logits, outputs
        torch.cuda.empty_cache()

    return torch.stack(all_cond_lps), torch.stack(all_joint_lps)


# =============================================================================
# Baseline computation (direct forward, no trace)
# =============================================================================

def compute_baselines_direct(model, clean_correct_ids, clean_correct_plens,
                              clean_incorrect_ids, clean_incorrect_plens,
                              corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens,
                              corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens,
                              batch_size=8):
    """Compute baselines for dual-answer mode using direct forward pass."""
    print("    Computing clean correct log probs...")
    clean_correct_lps = _forward_log_probs(model, clean_correct_ids, clean_correct_plens, batch_size, "Clean correct")
    print("    Computing clean incorrect log probs...")
    clean_incorrect_lps = _forward_log_probs(model, clean_incorrect_ids, clean_incorrect_plens, batch_size, "Clean incorrect")
    print("    Computing corrupted + clean_correct log probs...")
    corr_correct_lps = _forward_log_probs(model, corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens, batch_size, "Corrupted+correct")
    print("    Computing corrupted + clean_incorrect log probs...")
    corr_incorrect_lps = _forward_log_probs(model, corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens, batch_size, "Corrupted+incorrect")

    clean_baseline = get_teacher_forcing_diff(clean_correct_lps, clean_incorrect_lps).item()
    corrupted_baseline = get_teacher_forcing_diff(corr_correct_lps, corr_incorrect_lps).item()
    clean_acc = get_accuracy(clean_correct_lps, clean_incorrect_lps)
    corrupted_acc = get_accuracy(corr_correct_lps, corr_incorrect_lps)

    # Count examples where clean is correct AND corrupted is incorrect
    clean_correct_mask = clean_correct_lps > clean_incorrect_lps
    corrupted_incorrect_mask = corr_incorrect_lps > corr_correct_lps
    both_mask = clean_correct_mask & corrupted_incorrect_mask
    n_both_correct = both_mask.sum().item()
    n_total = len(clean_correct_lps)

    print(f"  Clean     — TF diff: {clean_baseline:.4f}, Accuracy: {clean_acc:.2%}")
    print(f"  Corrupted — TF diff: {corrupted_baseline:.4f}, Accuracy: {corrupted_acc:.2%}")
    print(f"  Both correct: {n_both_correct}/{n_total} ({n_both_correct/n_total:.2%})")

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "clean_baseline": clean_baseline,
        "corrupted_baseline": corrupted_baseline,
        "clean_accuracy": clean_acc,
        "corrupted_accuracy": corrupted_acc,
        "n_both_correct": n_both_correct,
        "n_total": n_total,
        "both_correct_pct": round(n_both_correct / n_total, 4) if n_total > 0 else 0,
        "avg_clean_correct_lp": clean_correct_lps.mean().item(),
        "avg_clean_incorrect_lp": clean_incorrect_lps.mean().item(),
        "avg_corrupted_correct_lp": corr_correct_lps.mean().item(),
        "avg_corrupted_incorrect_lp": corr_incorrect_lps.mean().item(),
        "_per_example": {
            "clean_correct_lps": clean_correct_lps.tolist(),
            "clean_incorrect_lps": clean_incorrect_lps.tolist(),
            "corrupted_correct_lps": corr_correct_lps.tolist(),
            "corrupted_incorrect_lps": corr_incorrect_lps.tolist(),
        },
    }


def compute_baselines_single_direct(model, clean_answer_ids, clean_answer_plens,
                                     corrupted_answer_ids, corrupted_answer_plens,
                                     batch_size=8):
    """Compute baselines for single-answer mode using direct forward pass."""
    print("    Computing clean (conditional + joint)...")
    clean_cond_lps, clean_joint_lps = _forward_log_probs_both(
        model, clean_answer_ids, clean_answer_plens, batch_size, "Clean"
    )
    print("    Computing corrupted (conditional + joint)...")
    corr_cond_lps, corr_joint_lps = _forward_log_probs_both(
        model, corrupted_answer_ids, corrupted_answer_plens, batch_size, "Corrupted"
    )

    clean_baseline = clean_cond_lps.mean().item()
    corrupted_baseline = corr_cond_lps.mean().item()
    clean_acc = get_single_answer_accuracy(clean_joint_lps, corr_joint_lps)
    corrupted_acc = get_single_answer_accuracy(corr_joint_lps, clean_joint_lps)

    print(f"  Clean     — Cond LP: {clean_baseline:.4f}, Accuracy: {clean_acc:.2%}")
    print(f"  Corrupted — Cond LP: {corrupted_baseline:.4f}, Accuracy: {corrupted_acc:.2%}")

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "clean_baseline": clean_baseline,
        "corrupted_baseline": corrupted_baseline,
        "clean_accuracy": clean_acc,
        "corrupted_accuracy": corrupted_acc,
        "single_answer": True,
        "avg_clean_cond_lp": clean_cond_lps.mean().item(),
        "avg_corrupted_cond_lp": corr_cond_lps.mean().item(),
        "avg_clean_joint_lp": clean_joint_lps.mean().item(),
        "avg_corrupted_joint_lp": corr_joint_lps.mean().item(),
        "_per_example": {
            "clean_cond_lps": clean_cond_lps.tolist(),
            "corrupted_cond_lps": corr_cond_lps.tolist(),
            "clean_joint_lps": clean_joint_lps.tolist(),
            "corrupted_joint_lps": corr_joint_lps.tolist(),
        },
    }


# =============================================================================
# Main eval logic
# =============================================================================

def run_eval_for_task(model, model_config, domain, task_name, task_config,
                      args, results_base):
    """Compute baselines for a single task."""
    model_short = get_model_short_name(args.model)
    results_dir = os.path.join(results_base, model_short, domain, task_name)
    os.makedirs(results_dir, exist_ok=True)
    baselines_path = os.path.join(results_dir, "baselines.json")

    # Check cache
    if os.path.exists(baselines_path) and not args.overwrite:
        print(f"\n  Loading cached baselines from {baselines_path}")
        with open(baselines_path, "r") as f:
            baselines = json.load(f)
        single_answer = baselines.get("single_answer", False)
        if single_answer:
            print(f"    Clean     — Cond LP: {baselines['clean_baseline']:.4f}, Accuracy: {baselines['clean_accuracy']:.2%}")
            print(f"    Corrupted — Cond LP: {baselines['corrupted_baseline']:.4f}, Accuracy: {baselines['corrupted_accuracy']:.2%}")
        else:
            print(f"    Clean     — TF diff: {baselines['clean_baseline']:.4f}, Accuracy: {baselines['clean_accuracy']:.2%}")
            print(f"    Corrupted — TF diff: {baselines['corrupted_baseline']:.4f}, Accuracy: {baselines['corrupted_accuracy']:.2%}")
        return baselines

    # Load data
    print(f"\n  Loading data for {task_name}...")
    data = load_task_data(args.data_dir, domain, task_config)
    print(f"    {len(data)} examples")

    # Format prompts
    formatted = format_prompts(data, task_config, args.model, model.tokenizer)
    single_answer = formatted.get("single_answer", False)

    # Print one full example (exactly what the model sees)
    print(f"\n  === Full Example (item 0) ===")
    if single_answer:
        print(f"  Clean + answer:\n    {formatted['clean_prompts'][0]}{formatted['answers'][0]}")
        print(f"  Corrupted + answer:\n    {formatted['corrupted_prompts'][0]}{formatted['answers'][0]}")
    else:
        print(f"  Clean + correct:\n    {formatted['clean_prompts'][0]}{formatted['clean_correct'][0]}")
        print(f"  Clean + incorrect:\n    {formatted['clean_prompts'][0]}{formatted['clean_incorrect'][0]}")
        print(f"  Corrupted + correct:\n    {formatted['corrupted_prompts'][0]}{formatted['clean_correct'][0]}")
        print(f"  Corrupted + incorrect:\n    {formatted['corrupted_prompts'][0]}{formatted['clean_incorrect'][0]}")
    print(f"  =============================\n")

    if single_answer:
        print(f"  Mode: single-answer")
        clean_answer_ids, clean_answer_plens = prepare_sequence_inputs(
            formatted["clean_prompts"], formatted["answers"], model.tokenizer
        )
        corrupted_answer_ids, corrupted_answer_plens = prepare_sequence_inputs(
            formatted["corrupted_prompts"], formatted["answers"], model.tokenizer
        )
        print(f"    Clean prompt len: {clean_answer_plens[0]}, total len: {clean_answer_ids[0].shape[0]}")

        print(f"\n  Computing baselines (direct forward)...")
        baselines = compute_baselines_single_direct(
            model,
            clean_answer_ids, clean_answer_plens,
            corrupted_answer_ids, corrupted_answer_plens,
            batch_size=args.batch_size,
        )
    else:
        print(f"  Mode: dual-answer")
        clean_correct_ids, clean_correct_plens = prepare_sequence_inputs(
            formatted["clean_prompts"], formatted["clean_correct"], model.tokenizer
        )
        clean_incorrect_ids, clean_incorrect_plens = prepare_sequence_inputs(
            formatted["clean_prompts"], formatted["clean_incorrect"], model.tokenizer
        )
        corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens = prepare_sequence_inputs(
            formatted["corrupted_prompts"], formatted["clean_correct"], model.tokenizer
        )
        corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens = prepare_sequence_inputs(
            formatted["corrupted_prompts"], formatted["clean_incorrect"], model.tokenizer
        )
        print(f"    Clean prompt len: {clean_correct_plens[0]}, total len: {clean_correct_ids[0].shape[0]}")

        print(f"\n  Computing baselines (direct forward)...")
        baselines = compute_baselines_direct(
            model,
            clean_correct_ids, clean_correct_plens,
            clean_incorrect_ids, clean_incorrect_plens,
            corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens,
            corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens,
            batch_size=args.batch_size,
        )

    # Generate debug samples
    num_examples = len(formatted["clean_prompts"])
    n_debug = min(5, num_examples)
    debug_indices = sorted(random.sample(range(num_examples), n_debug))

    print(f"\n  Generating {n_debug} debug samples...")

    if single_answer:
        debug_top = get_debug_samples(
            model, model.tokenizer, clean_answer_ids, clean_answer_plens, debug_indices
        )
        per_ex = baselines.pop("_per_example")
        debug_samples = []
        for (idx, top_token, top_lp) in debug_top:
            debug_samples.append({
                "index": idx,
                "clean_prompt": formatted["clean_prompts"][idx],
                "answer": formatted["answers"][idx],
                "clean_cond_lp": round(per_ex["clean_cond_lps"][idx], 4),
                "corrupted_cond_lp": round(per_ex["corrupted_cond_lps"][idx], 4),
                "model_top_token": top_token,
                "model_top_token_lp": round(top_lp, 4),
            })
        baselines["debug_samples"] = debug_samples
    else:
        debug_top = get_debug_samples(
            model, model.tokenizer, clean_correct_ids, clean_correct_plens, debug_indices
        )
        per_ex = baselines.pop("_per_example")
        debug_samples = []
        for (idx, top_token, top_lp) in debug_top:
            c_correct = per_ex["clean_correct_lps"][idx]
            c_incorrect = per_ex["clean_incorrect_lps"][idx]
            debug_samples.append({
                "index": idx,
                "clean_prompt": formatted["clean_prompts"][idx],
                "correct_answer": formatted["clean_correct"][idx],
                "incorrect_answer": formatted["clean_incorrect"][idx],
                "clean_correct_lp": round(c_correct, 4),
                "clean_incorrect_lp": round(c_incorrect, 4),
                "model_top_token": top_token,
                "model_top_token_lp": round(top_lp, 4),
                "correct": c_correct > c_incorrect,
            })
        baselines["debug_samples"] = debug_samples

        # Save error samples for debugging
        # Type 1: clean is wrong (model doesn't understand the task)
        clean_error_indices = [i for i in range(num_examples)
                              if per_ex["clean_correct_lps"][i] <= per_ex["clean_incorrect_lps"][i]]
        n_ce = min(5, len(clean_error_indices))
        if n_ce > 0:
            ce_samples = sorted(random.sample(clean_error_indices, n_ce))
            ce_top = get_debug_samples(
                model, model.tokenizer, clean_correct_ids, clean_correct_plens, ce_samples
            )
            baselines["clean_error_samples"] = [{
                "index": idx,
                "clean_prompt": formatted["clean_prompts"][idx],
                "correct_answer": formatted["clean_correct"][idx],
                "incorrect_answer": formatted["clean_incorrect"][idx],
                "clean_correct_lp": round(per_ex["clean_correct_lps"][idx], 4),
                "clean_incorrect_lp": round(per_ex["clean_incorrect_lps"][idx], 4),
                "model_top_token": top_token,
                "model_top_token_lp": round(top_lp, 4),
            } for (idx, top_token, top_lp) in ce_top]
            print(f"    Saved {n_ce} clean error samples (out of {len(clean_error_indices)} total)")
        else:
            baselines["clean_error_samples"] = []
            print(f"    No clean errors (100% clean accuracy)")

        # Type 2: clean correct but corrupted also correct (corruption didn't work)
        corrupt_error_indices = [i for i in range(num_examples)
                                if per_ex["clean_correct_lps"][i] > per_ex["clean_incorrect_lps"][i]
                                and per_ex["corrupted_correct_lps"][i] > per_ex["corrupted_incorrect_lps"][i]]
        n_re = min(5, len(corrupt_error_indices))
        if n_re > 0:
            re_samples = sorted(random.sample(corrupt_error_indices, n_re))
            re_top = get_debug_samples(
                model, model.tokenizer, corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens, re_samples
            )
            baselines["corrupted_error_samples"] = [{
                "index": idx,
                "corrupted_prompt": formatted["corrupted_prompts"][idx],
                "correct_answer": formatted["clean_correct"][idx],
                "incorrect_answer": formatted["clean_incorrect"][idx],
                "corrupted_correct_lp": round(per_ex["corrupted_correct_lps"][idx], 4),
                "corrupted_incorrect_lp": round(per_ex["corrupted_incorrect_lps"][idx], 4),
                "model_top_token": top_token,
                "model_top_token_lp": round(top_lp, 4),
            } for (idx, top_token, top_lp) in re_top]
            print(f"    Saved {n_re} corrupted error samples (out of {len(corrupt_error_indices)} total)")
        else:
            baselines["corrupted_error_samples"] = []
            print(f"    No corrupted errors (corruption works on all clean-correct examples)")

    # Save
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"    ✓ Saved: {baselines_path}")

    return baselines


def main():
    parser = argparse.ArgumentParser(description="Run evaluation only (baselines)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--domain", required=True, choices=["MD", "ToM", "phys", "Lan"])
    parser.add_argument("--task", default=None, help="Specific task (default: all in domain)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config-file", default="config.json", help="Config filename (default: config.json)")
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATION ONLY (Direct Forward, No Trace)")
    print("=" * 80)

    model, model_config = load_model(args.model)
    # domain_config = load_task_config(args.config_dir, args.domain)
    domain_config = load_task_config(args.config_dir, args.domain, args.config_file)

    if args.task:
        if args.task not in domain_config:
            print(f"Error: task '{args.task}' not found in {args.domain} config")
            print(f"Available tasks: {list(domain_config.keys())}")
            sys.exit(1)
        tasks = {args.task: domain_config[args.task]}
    else:
        tasks = {}
        for task_name, task_config in domain_config.items():
            data_path = os.path.join(args.data_dir, args.domain, task_config["path"])
            if os.path.exists(data_path):
                tasks[task_name] = task_config
            else:
                print(f"  ⏭ Skipping {task_name} (data not found: {data_path})")
        if not tasks:
            print(f"No tasks with data found for domain {args.domain}")
            sys.exit(1)

    print(f"\nModel: {args.model}")
    print(f"Domain: {args.domain}")
    print(f"Tasks: {list(tasks.keys())}")
    print("=" * 80)

    summary = []

    for task_name, task_config in tasks.items():
        print(f"\n{'='*80}")
        print(f"TASK: {task_name}")
        print(f"{'='*80}")

        baselines = run_eval_for_task(
            model, model_config, args.domain, task_name, task_config,
            args, args.results_dir
        )
        summary.append((task_name, baselines))

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Task':<40} {'Clean Acc':>10} {'Corrupted Acc':>14} {'Both Correct':>16} {'Clean BL':>10} {'Corrupted BL':>12}")
    print(f"  {'-'*102}")
    for task_name, bl in summary:
        n_both = bl.get('n_both_correct', None)
        n_tot = bl.get('n_total', None)
        if n_both is not None and n_tot is not None:
            both_str = f"{n_both}/{n_tot} ({n_both/n_tot:.0%})"
        else:
            both_str = "?"
        print(f"  {task_name:<40} {bl['clean_accuracy']:>10.2%} {bl['corrupted_accuracy']:>14.2%} {both_str:>16} {bl['clean_baseline']:>10.4f} {bl['corrupted_baseline']:>12.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()