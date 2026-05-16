"""
Run attribution patching for a specific task or all tasks in a domain.
Uses full-sequence teacher forcing.

Includes:
  - Token alignment filtering (removes examples where clean/corrupted have different token counts)
  - Both-correct filtering (only uses examples where model is correct on both clean and corrupted)

Usage:
    python scripts/run_attribution.py --model Qwen/Qwen2.5-32B-Instruct --domain phys --task physical_reasoning_newton --component neurons
    python scripts/run_attribution.py --model Qwen/Qwen2.5-32B-Instruct --domain phys --component both
"""

import argparse
import os
import sys
import json
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_model
from src.data_utils import (
    load_task_config, load_task_data, format_prompts, prepare_sequence_inputs,
)
from src.attribution import (
    compute_baselines, compute_baselines_single_answer,
    run_neuron_attribution, run_head_attribution,
    run_neuron_attribution_single, run_head_attribution_single,
)


def get_model_short_name(model_name):
    return model_name.replace("/", "_").replace(".", "-")


# ============================================================
# Cache invalidation helper
# ============================================================
def clear_overlap_cache(results_dir, component):
    """
    Delete sorted index cache files (.npy) and metadata (.json) in results_dir
    when attribution is overwritten, so run_overlap.py will recompute from fresh attribution.

    Args:
        results_dir: task-level results directory (where attribution.pt lives)
        component: 'neurons', 'heads', or 'both'
    """
    if component in ("neurons", "both"):
        for f in glob.glob(os.path.join(results_dir, "sorted_indices_*.npy")):
            if "_heads" not in os.path.basename(f):
                os.remove(f)
                print(f"    [cache cleared] {os.path.basename(f)}")
        meta = os.path.join(results_dir, "attribution_meta.json")
        if os.path.exists(meta):
            os.remove(meta)
            print(f"    [cache cleared] attribution_meta.json")

    if component in ("heads", "both"):
        for f in glob.glob(os.path.join(results_dir, "sorted_indices_*_heads.npy")):
            os.remove(f)
            print(f"    [cache cleared] {os.path.basename(f)}")
        meta = os.path.join(results_dir, "attribution_meta_heads.json")
        if os.path.exists(meta):
            os.remove(meta)
            print(f"    [cache cleared] attribution_meta_heads.json")


def filter_by_token_alignment(formatted, results_dir, model_tokenizer):
    """
    Filter out examples where clean and corrupted prompts have different token counts.
    Reads from token_align.json if available, otherwise computes on the fly.

    Returns:
        misaligned_indices: set of indices to remove
    """
    token_align_path = os.path.join(results_dir, "token_align.json")

    if os.path.exists(token_align_path):
        with open(token_align_path, "r") as f:
            align_info = json.load(f)
        misaligned_indices = set(item["idx"] for item in align_info["misaligned_examples"])
        print(f"    Loaded token alignment from {token_align_path}: {align_info['n_misaligned']} misaligned")
    else:
        # Compute on the fly
        print(f"    Computing token alignment...")
        misaligned_indices = set()
        clean_prompts = formatted["clean_prompts"]
        corrupted_prompts = formatted["corrupted_prompts"]
        for i in range(len(clean_prompts)):
            clean_len = len(model_tokenizer(clean_prompts[i], add_special_tokens=False)["input_ids"])
            corr_len = len(model_tokenizer(corrupted_prompts[i], add_special_tokens=False)["input_ids"])
            if clean_len != corr_len:
                misaligned_indices.add(i)
        print(f"    Found {len(misaligned_indices)} misaligned examples")

    return misaligned_indices


def filter_indices_from_list(lst, keep_indices):
    """Filter a list to keep only specified indices."""
    return [lst[i] for i in keep_indices]


def filter_sequence_inputs(input_ids_list, prompt_lens, keep_indices):
    """Filter sequence inputs (list of tensors + list of ints) by indices."""
    return (
        [input_ids_list[i] for i in keep_indices],
        [prompt_lens[i] for i in keep_indices],
    )


def compute_per_example_correctness_dual(model, model_config,
                                          clean_correct_ids, clean_correct_plens,
                                          clean_incorrect_ids, clean_incorrect_plens,
                                          corrupted_correct_ids, corrupted_correct_plens,
                                          corrupted_incorrect_ids, corrupted_incorrect_plens,
                                          batch_size=8):
    """
    Compute per-example correctness for dual-answer mode.
    Reuses _run_sequence_eval from attribution module to get per-example log probs.

    Returns:
        clean_correct_lps, clean_incorrect_lps, corr_correct_lps, corr_incorrect_lps: tensors [N]
        per_clean_correct: list of bool
        per_corrupted_correct: list of bool
        per_both_correct: list of bool
    """
    from src.attribution import _run_sequence_eval

    print("    Computing clean correct log probs...")
    clean_correct_lps = _run_sequence_eval(model, clean_correct_ids, clean_correct_plens, batch_size)
    print("    Computing clean incorrect log probs...")
    clean_incorrect_lps = _run_sequence_eval(model, clean_incorrect_ids, clean_incorrect_plens, batch_size)
    print("    Computing corrupted + clean_correct log probs...")
    corr_correct_lps = _run_sequence_eval(model, corrupted_correct_ids, corrupted_correct_plens, batch_size)
    print("    Computing corrupted + clean_incorrect log probs...")
    corr_incorrect_lps = _run_sequence_eval(model, corrupted_incorrect_ids, corrupted_incorrect_plens, batch_size)

    # Clean correct: model picks correct answer on clean prompt
    per_clean_correct = (clean_correct_lps > clean_incorrect_lps).tolist()
    # Corrupted correct: corruption worked — model picks WRONG answer on corrupted prompt
    # i.e., clean_incorrect gets higher score than clean_correct under corrupted prompt
    per_corrupted_correct = (corr_incorrect_lps > corr_correct_lps).tolist()
    per_both_correct = [c and r for c, r in zip(per_clean_correct, per_corrupted_correct)]

    return (clean_correct_lps, clean_incorrect_lps, corr_correct_lps, corr_incorrect_lps,
            per_clean_correct, per_corrupted_correct, per_both_correct)


def compute_per_example_correctness_single(model, model_config,
                                            clean_answer_ids, clean_answer_plens,
                                            corrupted_answer_ids, corrupted_answer_plens,
                                            batch_size=8):
    """
    Compute per-example correctness for single-answer mode.
    Correct = joint P(clean+answer) > P(corrupted+answer).

    Returns:
        clean_cond_lps, corr_cond_lps: tensors [N]
        clean_joint_lps, corr_joint_lps: tensors [N]
        per_clean_correct: list of bool
        per_corrupted_correct: list of bool
        per_both_correct: list of bool
    """
    from src.attribution import _run_sequence_eval_both

    print("    Computing clean (conditional + joint)...")
    clean_cond_lps, clean_joint_lps = _run_sequence_eval_both(
        model, clean_answer_ids, clean_answer_plens, batch_size, desc="Clean"
    )
    print("    Computing corrupted (conditional + joint)...")
    corr_cond_lps, corr_joint_lps = _run_sequence_eval_both(
        model, corrupted_answer_ids, corrupted_answer_plens, batch_size, desc="Corrupted"
    )

    per_clean_correct = (clean_joint_lps > corr_joint_lps).tolist()
    per_corrupted_correct = per_clean_correct
    per_both_correct = per_clean_correct

    return (clean_cond_lps, corr_cond_lps, clean_joint_lps, corr_joint_lps,
            per_clean_correct, per_corrupted_correct, per_both_correct)


def run_attribution_for_task(model, model_config, domain, task_name, task_config,
                             args, results_base):
    """Run attribution patching for a single task."""
    model_short = get_model_short_name(args.model)
    results_dir = os.path.join(results_base, model_short, domain, task_name)

    # Check overwrite
    neuron_path = os.path.join(results_dir, "neuron_attribution.pt")
    head_path = os.path.join(results_dir, "head_attribution.pt")

    skip_neurons = (args.component in ["neurons", "both"]
                    and os.path.exists(neuron_path) and not args.overwrite)
    skip_heads = (args.component in ["heads", "both"]
                  and os.path.exists(head_path) and not args.overwrite)

    if skip_neurons and (args.component == "neurons" or
                         (args.component == "both" and skip_heads)):
        print(f"  ⏭ Skipping {task_name} (results exist, use --overwrite to rerun)")
        return

    if skip_heads and args.component == "heads":
        print(f"  ⏭ Skipping {task_name} heads (results exist)")
        return

    # Load data
    print(f"\n  Loading data for {task_name}...")
    data = load_task_data(args.data_dir, domain, task_config)
    n_total = len(data)
    print(f"    {n_total} examples")

    # Format prompts
    formatted = format_prompts(data, task_config, args.model, model.tokenizer)
    single_answer = formatted.get("single_answer", False)

    os.makedirs(results_dir, exist_ok=True)
    baselines_path = os.path.join(results_dir, "baselines.json")

    # =====================================================================
    # Step 1: Token alignment filtering
    # =====================================================================
    misaligned_indices = filter_by_token_alignment(formatted, results_dir, model.tokenizer)
    aligned_indices = sorted(set(range(n_total)) - misaligned_indices)
    n_after_align = len(aligned_indices)
    print(f"    After alignment filter: {n_after_align}/{n_total} examples")

    if n_after_align == 0:
        print(f"  ⏭ Skipping {task_name}: no aligned examples")
        return

    # =====================================================================
    # Step 2: Compute per-example correctness & both-correct filtering
    # =====================================================================
    baselines_valid = False
    if os.path.exists(baselines_path) and not args.overwrite:
        with open(baselines_path, "r") as f:
            baselines = json.load(f)
        if "both_correct_indices" in baselines:
            baselines_valid = True
            print(f"\n  Loading cached baselines from {baselines_path}")
        else:
            print(f"\n  Cached baselines missing 'both_correct_indices', recomputing...")

    if baselines_valid:
        both_correct_indices = baselines["both_correct_indices"]

        print(f"    n_total: {baselines['n_total']}, n_after_alignment: {baselines['n_after_alignment_filter']}")
        print(f"    Clean acc: {baselines['clean_accuracy']:.2%}, Corrupted acc: {baselines['corrupted_accuracy']:.2%}")
        print(f"    Both correct: {baselines['n_both_correct']}/{baselines['n_after_alignment_filter']} ({baselines['both_correct_accuracy']:.2%})")
        print(f"    Clean baseline: {baselines['clean_baseline']:.4f}, Corrupted baseline: {baselines['corrupted_baseline']:.4f}")
    else:
        print(f"\n  Computing baselines on aligned examples...")

        if single_answer:
            print(f"  Mode: single-answer")
            clean_answer_ids, clean_answer_plens = prepare_sequence_inputs(
                filter_indices_from_list(formatted["clean_prompts"], aligned_indices),
                filter_indices_from_list(formatted["answers"], aligned_indices),
                model.tokenizer,
            )
            corrupted_answer_ids, corrupted_answer_plens = prepare_sequence_inputs(
                filter_indices_from_list(formatted["corrupted_prompts"], aligned_indices),
                filter_indices_from_list(formatted["answers"], aligned_indices),
                model.tokenizer,
            )

            (clean_cond_lps, corr_cond_lps, clean_joint_lps, corr_joint_lps,
             per_clean_correct, per_corrupted_correct, per_both_correct) = \
                compute_per_example_correctness_single(
                    model, model_config,
                    clean_answer_ids, clean_answer_plens,
                    corrupted_answer_ids, corrupted_answer_plens,
                    batch_size=args.batch_size,
                )

            both_correct_mask = per_both_correct
            both_correct_indices = [aligned_indices[i] for i, bc in enumerate(both_correct_mask) if bc]
            not_both_correct_indices = [aligned_indices[i] for i, bc in enumerate(both_correct_mask) if not bc]

            clean_acc = sum(per_clean_correct) / len(per_clean_correct)
            corrupted_acc = sum(per_corrupted_correct) / len(per_corrupted_correct)
            both_correct_acc = sum(per_both_correct) / len(per_both_correct)

            bc_local = [i for i, bc in enumerate(both_correct_mask) if bc]
            if len(bc_local) > 0:
                clean_baseline = clean_cond_lps[bc_local].mean().item()
                corrupted_baseline = corr_cond_lps[bc_local].mean().item()
            else:
                clean_baseline = 0.0
                corrupted_baseline = 0.0

        else:
            print(f"  Mode: dual-answer")
            aligned_clean_prompts = filter_indices_from_list(formatted["clean_prompts"], aligned_indices)
            aligned_corrupted_prompts = filter_indices_from_list(formatted["corrupted_prompts"], aligned_indices)
            aligned_clean_correct = filter_indices_from_list(formatted["clean_correct"], aligned_indices)
            aligned_clean_incorrect = filter_indices_from_list(formatted["clean_incorrect"], aligned_indices)

            clean_correct_ids, clean_correct_plens = prepare_sequence_inputs(
                aligned_clean_prompts, aligned_clean_correct, model.tokenizer
            )
            clean_incorrect_ids, clean_incorrect_plens = prepare_sequence_inputs(
                aligned_clean_prompts, aligned_clean_incorrect, model.tokenizer
            )
            corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens = prepare_sequence_inputs(
                aligned_corrupted_prompts, aligned_clean_correct, model.tokenizer
            )
            corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens = prepare_sequence_inputs(
                aligned_corrupted_prompts, aligned_clean_incorrect, model.tokenizer
            )

            (clean_correct_lps, clean_incorrect_lps, corr_correct_lps, corr_incorrect_lps,
             per_clean_correct, per_corrupted_correct, per_both_correct) = \
                compute_per_example_correctness_dual(
                    model, model_config,
                    clean_correct_ids, clean_correct_plens,
                    clean_incorrect_ids, clean_incorrect_plens,
                    corrupted_w_clean_correct_ids, corrupted_w_clean_correct_plens,
                    corrupted_w_clean_incorrect_ids, corrupted_w_clean_incorrect_plens,
                    batch_size=args.batch_size,
                )

            both_correct_mask = per_both_correct
            both_correct_indices = [aligned_indices[i] for i, bc in enumerate(both_correct_mask) if bc]
            not_both_correct_indices = [aligned_indices[i] for i, bc in enumerate(both_correct_mask) if not bc]

            clean_acc = sum(per_clean_correct) / len(per_clean_correct)
            corrupted_acc = sum(per_corrupted_correct) / len(per_corrupted_correct)
            both_correct_acc = sum(per_both_correct) / len(per_both_correct)

            from src.metrics import get_teacher_forcing_diff
            bc_local = [i for i, bc in enumerate(both_correct_mask) if bc]
            if len(bc_local) > 0:
                clean_baseline = get_teacher_forcing_diff(
                    clean_correct_lps[bc_local], clean_incorrect_lps[bc_local]
                ).item()
                corrupted_baseline = get_teacher_forcing_diff(
                    corr_correct_lps[bc_local], corr_incorrect_lps[bc_local]
                ).item()
            else:
                clean_baseline = 0.0
                corrupted_baseline = 0.0

        print(f"    Clean accuracy:     {clean_acc:.2%}")
        print(f"    Corrupted accuracy: {corrupted_acc:.2%}")
        print(f"    Both correct:       {len(both_correct_indices)}/{n_after_align} ({both_correct_acc:.2%})")
        print(f"    Clean baseline (both-correct):     {clean_baseline:.4f}")
        print(f"    Corrupted baseline (both-correct): {corrupted_baseline:.4f}")

        baselines = {
            "n_total": n_total,
            "n_token_misaligned": len(misaligned_indices),
            "token_misaligned_indices": sorted(misaligned_indices),
            "n_after_alignment_filter": n_after_align,
            "clean_accuracy": clean_acc,
            "corrupted_accuracy": corrupted_acc,
            "both_correct_accuracy": both_correct_acc,
            "n_both_correct": len(both_correct_indices),
            "both_correct_indices": both_correct_indices,
            "not_both_correct_indices": not_both_correct_indices,
            "clean_baseline": clean_baseline,
            "corrupted_baseline": corrupted_baseline,
        }
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"    💾 Saved: {baselines_path}")

    # =====================================================================
    # Step 3: Check threshold
    # =====================================================================
    if not args.no_filter and baselines["both_correct_accuracy"] < args.min_accuracy:
        print(f"  ⏭ Skipping {task_name}: both_correct accuracy "
              f"{baselines['both_correct_accuracy']:.2%} < {args.min_accuracy:.0%} threshold")
        return

    # =====================================================================
    # Step 4: Prepare subset for attribution
    # =====================================================================
    if args.no_filter:
        # Use all aligned examples (no both-correct filtering)
        both_correct_indices = baselines.get("aligned_indices", list(range(baselines.get("n_after_alignment_filter", baselines.get("n_total", 0)))))
        if not both_correct_indices:
            both_correct_indices = list(range(len(formatted["clean_prompts"])))
        print(f"\n  Running attribution on ALL {len(both_correct_indices)} aligned examples (--no-filter)...")
    else:
        both_correct_indices = baselines["both_correct_indices"]
        print(f"\n  Running attribution on {len(both_correct_indices)} both-correct examples...")

    if single_answer:
        bc_clean_prompts = filter_indices_from_list(formatted["clean_prompts"], both_correct_indices)
        bc_corrupted_prompts = filter_indices_from_list(formatted["corrupted_prompts"], both_correct_indices)
        bc_answers = filter_indices_from_list(formatted["answers"], both_correct_indices)

        bc_clean_ids, bc_clean_plens = prepare_sequence_inputs(
            bc_clean_prompts, bc_answers, model.tokenizer
        )
        bc_corrupted_ids, bc_corrupted_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_answers, model.tokenizer
        )

        print(f"    Sample clean + answer: {model.tokenizer.decode(bc_clean_ids[0][:50])}...")
        print(f"    Sample corrupted + answer: {model.tokenizer.decode(bc_corrupted_ids[0][:50])}...")
        print(f"    Clean prompt len: {bc_clean_plens[0]}, total len: {bc_clean_ids[0].shape[0]}")

        if args.component in ["neurons", "both"] and not skip_neurons:
            print(f"\n  Running neuron attribution (single-answer)...")
            neuron_attr = run_neuron_attribution_single(
                model, model_config,
                bc_clean_ids, bc_clean_plens,
                bc_corrupted_ids, bc_corrupted_plens,
                baselines["clean_baseline"], baselines["corrupted_baseline"],
                batch_size=args.batch_size,
            )
            torch.save(torch.tensor(neuron_attr), neuron_path)
            print(f"    ✓ Saved: {neuron_path}")
            clear_overlap_cache(results_dir, "neurons")  # invalidate overlap cache

        if args.component in ["heads", "both"] and not skip_heads:
            print(f"\n  Running head attribution (single-answer)...")
            head_attr = run_head_attribution_single(
                model, model_config,
                bc_clean_ids, bc_clean_plens,
                bc_corrupted_ids, bc_corrupted_plens,
                baselines["clean_baseline"], baselines["corrupted_baseline"],
                batch_size=args.batch_size,
            )
            torch.save(torch.tensor(head_attr), head_path)
            print(f"    ✓ Saved: {head_path}")
            clear_overlap_cache(results_dir, "heads")  # invalidate overlap cache

    else:
        bc_clean_prompts = filter_indices_from_list(formatted["clean_prompts"], both_correct_indices)
        bc_corrupted_prompts = filter_indices_from_list(formatted["corrupted_prompts"], both_correct_indices)
        bc_clean_correct = filter_indices_from_list(formatted["clean_correct"], both_correct_indices)
        bc_clean_incorrect = filter_indices_from_list(formatted["clean_incorrect"], both_correct_indices)

        bc_clean_correct_ids, bc_clean_correct_plens = prepare_sequence_inputs(
            bc_clean_prompts, bc_clean_correct, model.tokenizer
        )
        bc_corrupted_correct_ids, bc_corrupted_correct_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_clean_correct, model.tokenizer
        )
        bc_corrupted_incorrect_ids, bc_corrupted_incorrect_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_clean_incorrect, model.tokenizer
        )

        print(f"    Sample clean + correct: {model.tokenizer.decode(bc_clean_correct_ids[0][:50])}...")
        print(f"    Sample corrupted + clean_correct: {model.tokenizer.decode(bc_corrupted_correct_ids[0][:50])}...")
        print(f"    Clean prompt len: {bc_clean_correct_plens[0]}, total len: {bc_clean_correct_ids[0].shape[0]}")

        if args.component in ["neurons", "both"] and not skip_neurons:
            print(f"\n  Running neuron attribution...")
            neuron_attr = run_neuron_attribution(
                model, model_config,
                bc_clean_correct_ids, bc_clean_correct_plens,
                bc_corrupted_correct_ids, bc_corrupted_correct_plens,
                bc_corrupted_incorrect_ids, bc_corrupted_incorrect_plens,
                baselines["clean_baseline"], baselines["corrupted_baseline"],
                batch_size=args.batch_size,
            )
            torch.save(torch.tensor(neuron_attr), neuron_path)
            print(f"    ✓ Saved: {neuron_path}")
            clear_overlap_cache(results_dir, "neurons")  # invalidate overlap cache

        if args.component in ["heads", "both"] and not skip_heads:
            print(f"\n  Running head attribution...")
            head_attr = run_head_attribution(
                model, model_config,
                bc_clean_correct_ids, bc_clean_correct_plens,
                bc_corrupted_correct_ids, bc_corrupted_correct_plens,
                bc_corrupted_incorrect_ids, bc_corrupted_incorrect_plens,
                baselines["clean_baseline"], baselines["corrupted_baseline"],
                batch_size=args.batch_size,
            )
            torch.save(torch.tensor(head_attr), head_path)
            print(f"    ✓ Saved: {head_path}")
            clear_overlap_cache(results_dir, "heads")  # invalidate overlap cache


def main():
    parser = argparse.ArgumentParser(description="Run attribution patching")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--domain", required=True, choices=["MD", "ToM", "phys", "Lan"])
    parser.add_argument("--task", default=None, help="Specific task (default: all in domain)")
    parser.add_argument("--component", default="neurons", choices=["neurons", "heads", "both"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-accuracy", type=float, default=0.70,
                        help="Skip tasks where both-correct accuracy is below this threshold (default: 0.70)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-filter", action="store_true",
                        help="Use all aligned examples, not just both-correct subset (for weak models like GPT-2)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config-file", default="config.json", help="Config filename (default: config.json)")
    args = parser.parse_args()

    print("=" * 80)
    print("ATTRIBUTION PATCHING (Full-Sequence Teacher Forcing)")
    print("=" * 80)

    model, model_config = load_model(args.model)

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

    print(f"\nDomain: {args.domain}")
    print(f"Tasks: {list(tasks.keys())}")
    print(f"Component: {args.component}")
    print("=" * 80)

    for task_name, task_config in tasks.items():
        print(f"\n{'='*80}")
        print(f"TASK: {task_name}")
        print(f"{'='*80}")

        run_attribution_for_task(
            model, model_config, args.domain, task_name, task_config,
            args, args.results_dir
        )

    print(f"\n{'='*80}")
    print("ALL ATTRIBUTION TASKS COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()