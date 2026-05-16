"""
Run ablation experiments with full-sequence teacher forcing.
Evaluates on both-correct subset only (examples where model is correct on both clean and corrupted).

Supports two ablation modes:
  --ablation-type zero       : zero ablation (original, default)
  --ablation-type corrupted  : replace with corrupted activations (Hanna et al. style)

Usage:
    # Zero ablation (original)
    python scripts/run_ablation.py --model Qwen/Qwen2.5-32B-Instruct \
        --source phys/physical_reasoning_newton \
        --target phys/physical_reasoning_newton \
        --component neurons --sign positive --pct 1

    # Corrupted ablation (new)
    python scripts/run_ablation.py --model Qwen/Qwen2.5-32B-Instruct \
        --source phys/physical_reasoning_newton \
        --target phys/physical_reasoning_newton \
        --component neurons --sign positive --pct 1 \
        --ablation-type corrupted

    # Compare both
    python scripts/run_ablation.py --model Qwen/Qwen2.5-32B-Instruct \
        --source phys/physical_reasoning_newton \
        --target phys/physical_reasoning_newton \
        --component neurons --sign positive --pct 1 \
        --ablation-type zero corrupted
"""

import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_model
from src.data_utils import (
    load_task_config, load_task_data, format_prompts, prepare_sequence_inputs,
)
from src.attribution import compute_baselines, compute_baselines_single_answer
from src.ablation import (
    load_attribution_scores, select_neurons,
    run_full_ablation_experiment,
)
from src.metrics import make_normalized_metric, make_normalized_metric_single


def get_model_short_name(model_name):
    return model_name.replace("/", "_").replace(".", "-")


def parse_source_target(s):
    parts = s.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected 'domain/task' format, got '{s}'")
    return parts[0], parts[1]


def filter_indices_from_list(lst, keep_indices):
    """Filter a list to keep only specified indices."""
    return [lst[i] for i in keep_indices]


def run_ablation_for_pair(model, model_config, source_domain, source_task,
                          target_domain, target_task, target_task_config,
                          args, results_base):
    """Run ablation for a single source->target pair."""
    model_short = get_model_short_name(args.model)

    # --- Load source attribution scores ---
    source_results_dir = os.path.join(results_base, model_short, source_domain, source_task)

    if args.component == "neurons":
        scores_path = os.path.join(source_results_dir, "neuron_attribution.pt")
    else:
        scores_path = os.path.join(source_results_dir, "head_attribution.pt")

    if not os.path.exists(scores_path):
        print(f"  ✗ Attribution scores not found: {scores_path}")
        print(f"    Run attribution first: python scripts/run_attribution.py ...")
        return

    attribution = load_attribution_scores(scores_path)
    print(f"  Loaded attribution scores: {attribution.shape}")

    # --- Load target data ---
    print(f"  Loading target data: {target_domain}/{target_task}...")
    target_data = load_task_data(args.data_dir, target_domain, target_task_config)
    n_total = len(target_data)
    print(f"    {n_total} examples")

    formatted = format_prompts(target_data, target_task_config, args.model, model.tokenizer)
    single_answer = formatted.get("single_answer", False)

    # --- Load target baselines (must exist, contains both_correct_indices) ---
    target_results_dir = os.path.join(results_base, model_short, target_domain, target_task)
    baselines_path = os.path.join(target_results_dir, "baselines.json")

    if not os.path.exists(baselines_path):
        print(f"  ✗ Baselines not found: {baselines_path}")
        print(f"    Run attribution first to generate baselines with both-correct info.")
        return

    with open(baselines_path, "r") as f:
        baselines = json.load(f)

    # --- Get both-correct indices ---
    if "both_correct_indices" not in baselines:
        print(f"  ✗ baselines.json missing 'both_correct_indices'. Re-run attribution with updated code.")
        return

    both_correct_indices = baselines["both_correct_indices"]
    n_bc = len(both_correct_indices)
    print(f"  Both-correct subset: {n_bc}/{n_total} examples ({n_bc/n_total:.1%})")

    if single_answer:
        print(f"    Clean     — Cond LP: {baselines['clean_baseline']:.4f}")
        print(f"    Corrupted — Cond LP: {baselines['corrupted_baseline']:.4f}")
    else:
        print(f"    Clean     — TF diff: {baselines['clean_baseline']:.4f}")
        print(f"    Corrupted — TF diff: {baselines['corrupted_baseline']:.4f}")

    # Check threshold
    if baselines["both_correct_accuracy"] < args.min_accuracy:
        print(f"  ⏭ Skipping target {target_domain}/{target_task}: "
              f"both_correct accuracy {baselines['both_correct_accuracy']:.2%} < {args.min_accuracy:.0%} threshold")
        return

    if n_bc == 0:
        print(f"  ⏭ Skipping target {target_domain}/{target_task}: no both-correct examples")
        return

    # --- Filter to both-correct subset ---
    if single_answer:
        # --- Single-answer mode ---
        print(f"  Mode: single-answer")
        bc_clean_prompts = filter_indices_from_list(formatted["clean_prompts"], both_correct_indices)
        bc_corrupted_prompts = filter_indices_from_list(formatted["corrupted_prompts"], both_correct_indices)
        bc_answers = filter_indices_from_list(formatted["answers"], both_correct_indices)

        clean_answer_ids, clean_answer_plens = prepare_sequence_inputs(
            bc_clean_prompts, bc_answers, model.tokenizer
        )
        corrupted_answer_ids, corrupted_answer_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_answers, model.tokenizer
        )

        normalized_metric = make_normalized_metric_single(
            baselines["clean_baseline"], baselines["corrupted_baseline"]
        )

        eval_inputs = {
            "clean_answer_ids": clean_answer_ids,
            "clean_answer_plens": clean_answer_plens,
            "corrupted_answer_ids": corrupted_answer_ids,
            "corrupted_answer_plens": corrupted_answer_plens,
            "single_answer": True,
        }

    else:
        # --- Standard dual-answer mode ---
        print(f"  Mode: dual-answer")
        bc_clean_prompts = filter_indices_from_list(formatted["clean_prompts"], both_correct_indices)
        bc_corrupted_prompts = filter_indices_from_list(formatted["corrupted_prompts"], both_correct_indices)
        bc_clean_correct = filter_indices_from_list(formatted["clean_correct"], both_correct_indices)
        bc_clean_incorrect = filter_indices_from_list(formatted["clean_incorrect"], both_correct_indices)

        # Clean prompt + answers (for evaluation)
        correct_ids, correct_plens = prepare_sequence_inputs(
            bc_clean_prompts, bc_clean_correct, model.tokenizer
        )
        incorrect_ids, incorrect_plens = prepare_sequence_inputs(
            bc_clean_prompts, bc_clean_incorrect, model.tokenizer
        )

        # Corrupted prompt + answers (needed for corrupted ablation)
        corrupted_correct_ids, corrupted_correct_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_clean_correct, model.tokenizer
        )
        corrupted_incorrect_ids, corrupted_incorrect_plens = prepare_sequence_inputs(
            bc_corrupted_prompts, bc_clean_incorrect, model.tokenizer
        )

        normalized_metric = make_normalized_metric(
            baselines["clean_baseline"], baselines["corrupted_baseline"]
        )

        eval_inputs = {
            "correct_ids": correct_ids,
            "correct_plens": correct_plens,
            "incorrect_ids": incorrect_ids,
            "incorrect_plens": incorrect_plens,
            # Corrupted counterparts for corrupted ablation
            "corrupted_correct_ids": corrupted_correct_ids,
            "corrupted_correct_plens": corrupted_correct_plens,
            "corrupted_incorrect_ids": corrupted_incorrect_ids,
            "corrupted_incorrect_plens": corrupted_incorrect_plens,
            "single_answer": False,
        }

    print(f"  Evaluating on {n_bc} both-correct examples")

    # --- Run ablation for each ablation_type, sign, and percentage ---
    for abl_type in args.ablation_type:
      for sign in args.sign:
        for pct in args.pct:
            print(f"\n  {'='*60}")
            print(f"  Ablation: {sign} {pct}% {args.component} (type={abl_type})")
            print(f"  Source: {source_domain}/{source_task}")
            print(f"  Target: {target_domain}/{target_task} ({n_bc} both-correct examples)")
            print(f"  {'='*60}")

            ablation_dir = os.path.join(
                results_base, model_short, "ablation",
                source_domain, source_task,
                f"{args.component}_ablated_on_{target_domain}_{target_task}",
            )
            os.makedirs(ablation_dir, exist_ok=True)

            # Include ablation type in filename to keep zero and corrupted results separate
            if abl_type == "zero":
                result_file = os.path.join(ablation_dir, f"{sign}_{pct}pct.json")
            else:
                result_file = os.path.join(ablation_dir, f"{sign}_{pct}pct_{abl_type}.json")

            if os.path.exists(result_file) and not args.overwrite:
                print(f"  ⏭ Skipping (exists): {result_file}")
                continue

            neurons_dict, selected_list = select_neurons(attribution, pct, sign=sign)
            total_selected = sum(len(v) for v in neurons_dict.values())
            print(f"  Selected {total_selected} {args.component}")

            # Random baselines cache paths
            truly_random_cache_dir = os.path.join(
                results_base, model_short, "random_baselines", "truly_random",
                target_domain, target_task,
            )
            os.makedirs(truly_random_cache_dir, exist_ok=True)
            truly_random_cache_path = os.path.join(
                truly_random_cache_dir, f"{args.component}_{total_selected}units.json"
            )

            layer_matched_cache_dir = os.path.join(
                results_base, model_short, "random_baselines", "layer_matched",
                source_domain, source_task, target_domain, target_task,
            )
            os.makedirs(layer_matched_cache_dir, exist_ok=True)
            layer_matched_cache_path = os.path.join(
                layer_matched_cache_dir, f"{args.component}_{total_selected}units.json"
            )

            results = run_full_ablation_experiment(
                model, model_config,
                normalized_metric,
                neurons_dict, selected_list,
                eval_inputs=eval_inputs,
                num_random_trials=args.num_random_trials,
                component=args.component,
                ablation_type=abl_type,
                batch_size=args.batch_size,
                sign=sign,
                clean_accuracy=baselines["clean_accuracy"],
                truly_random_cache_path=truly_random_cache_path,
                layer_matched_cache_path=layer_matched_cache_path,
                overwrite=args.overwrite,
                skip_layer_matched=args.skip_layer_matched,
            )

            results["metadata"] = {
                "model": args.model,
                "source_domain": source_domain,
                "source_task": source_task,
                "target_domain": target_domain,
                "target_task": target_task,
                "component": args.component,
                "sign": sign,
                "percentage": pct,
                "num_ablated": total_selected,
                "num_random_trials": args.num_random_trials,
                "n_both_correct": n_bc,
                "n_total": n_total,
                "target_baselines": baselines,
                "ablation_type": abl_type,
            }

            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Saved: {result_file}")

            print(f"\n  SUMMARY ({n_bc} both-correct examples, ablation_type={abl_type}):")
            print(f"  {'Condition':<30} {'Score':<10} {'Accuracy'}")
            print(f"  {'-'*55}")
            print(f"  {'No ablation':<30} {results['no_ablation']['score']:<10.4f} {results['no_ablation']['accuracy']:.2%}")
            print(f"  {f'{sign} {pct}% ({abl_type})':<30} {results['target']['score']:<10.4f} {results['target']['accuracy']:.2%}")
            if results['layer_matched_random']['mean_score'] is not None:
                print(f"  {'Layer-matched random':<30} {results['layer_matched_random']['mean_score']:<10.4f} {results['layer_matched_random']['mean_accuracy']:.2%}")
            else:
                print(f"  {'Layer-matched random':<30} {'SKIPPED':>10}")
            if results['truly_random']['mean_score'] is not None:
                print(f"  {'Truly random':<30} {results['truly_random']['mean_score']:<10.4f} {results['truly_random']['mean_accuracy']:.2%}")
            else:
                print(f"  {'Truly random':<30} {'SKIPPED':>10}")


def discover_available_tasks(results_dir, model_short, domain, component="neurons"):
    """
    Discover tasks that have attribution scores in the results directory.
    """
    domain_dir = os.path.join(results_dir, model_short, domain)
    if not os.path.isdir(domain_dir):
        return []

    available = []
    component_singular = component.rstrip("s")
    attr_file = f"{component_singular}_attribution.pt"
    for task_name in sorted(os.listdir(domain_dir)):
        task_dir = os.path.join(domain_dir, task_name)
        if os.path.isdir(task_dir) and os.path.exists(os.path.join(task_dir, attr_file)):
            available.append(task_name)
    return available


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-file", default="config.json", help="Config filename (default: config.json)")

    # Source: either specific task or whole domain
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="Source: 'domain/task'")
    source_group.add_argument("--source-domain", help="All tasks in source domain")

    # Target: either specific task or whole domain
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target", help="Target: 'domain/task'")
    target_group.add_argument("--target-domain", help="All tasks in target domain")

    parser.add_argument("--component", default="neurons", choices=["neurons", "heads"])
    parser.add_argument("--sign", nargs="+", default=["positive"],
                        choices=["positive", "negative", "absolute"],
                        help="Sign(s) to ablate")
    parser.add_argument("--pct", type=float, nargs="+", default=[1.0])
    parser.add_argument("--ablation-type", nargs="+", default=["zero"],
                        choices=["zero", "corrupted"],
                        help="Ablation type(s): 'zero' (original) or 'corrupted' (Hanna et al. style). "
                             "Can specify both to compare: --ablation-type zero corrupted")
    parser.add_argument("--num-random-trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-accuracy", type=float, default=0.70,
                        help="Skip target tasks where both-correct accuracy is below this threshold")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-layer-matched", action="store_true",
                        help="Skip layer-matched random baseline")
    parser.add_argument("--target-attributed-only", action="store_true",
                        help="Only target tasks that have attribution results")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    model_short = get_model_short_name(args.model)

    print("=" * 80)
    print("ABLATION EXPERIMENT (Full-Sequence Teacher Forcing)")
    print(f"Ablation type(s): {args.ablation_type}")
    print("=" * 80)

    # Determine sources
    if args.source:
        source_domain, source_task = parse_source_target(args.source)
        sources = [(source_domain, source_task)]
    else:
        source_domain = args.source_domain
        available = discover_available_tasks(args.results_dir, model_short, source_domain, args.component)
        if not available:
            print(f"  ✗ No attribution results found for domain '{source_domain}'")
            sys.exit(1)
        sources = [(source_domain, t_name) for t_name in available]

    # Determine targets
    if args.target:
        target_domain, target_task = parse_source_target(args.target)
        # target_config = load_task_config(args.config_dir, target_domain)
        target_config = load_task_config(args.config_dir, target_domain, args.config_file)
        if target_task not in target_config:
            print(f"Error: task '{target_task}' not in {target_domain} config")
            sys.exit(1)
        targets = [(target_domain, target_task, target_config[target_task])]
    else:
        target_domain = args.target_domain
        # target_config = load_task_config(args.config_dir, target_domain)
        target_config = load_task_config(args.config_dir, target_domain, args.config_file)

        if args.target_attributed_only:
            attributed_tasks = discover_available_tasks(
                args.results_dir, model_short, target_domain, args.component
            )
            available_targets = []
            for t_name in attributed_tasks:
                if t_name in target_config:
                    available_targets.append((target_domain, t_name, target_config[t_name]))
            if not available_targets:
                print(f"  ✗ No attributed tasks found for domain '{target_domain}'")
                sys.exit(1)
        else:
            available_targets = []
            for t_name, t_conf in target_config.items():
                data_path = os.path.join(args.data_dir, target_domain, t_conf["path"])
                if os.path.exists(data_path):
                    available_targets.append((target_domain, t_name, t_conf))
            if not available_targets:
                print(f"  ✗ No data files found for domain '{target_domain}'")
                sys.exit(1)

        targets = available_targets

    print(f"  Sources ({len(sources)}): {[f'{d}/{t}' for d, t in sources]}")
    print(f"  Targets ({len(targets)}): {[f'{d}/{t}' for d, t, _ in targets]}")
    print(f"  Component: {args.component}, Sign: {args.sign}, Pct: {args.pct}")
    print(f"  Ablation type(s): {args.ablation_type}")
    print("=" * 80)

    # Load model
    model, model_config = load_model(args.model)

    # Run all source × target pairs
    for s_domain, s_task in sources:
        for t_domain, t_task, t_config in targets:
            print(f"\n{'='*80}")
            print(f"SOURCE: {s_domain}/{s_task} → TARGET: {t_domain}/{t_task}")
            print(f"{'='*80}")

            run_ablation_for_pair(
                model, model_config, s_domain, s_task,
                t_domain, t_task, t_config,
                args, args.results_dir
            )

    print(f"\n{'='*80}")
    print("ALL ABLATION EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()