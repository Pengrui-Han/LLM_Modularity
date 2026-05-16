"""Data loading and prompt formatting utilities."""

import json
import os
import torch

from .chat_template import get_chat_template


def load_task_config(config_dir, domain, config_file="config.json"):
    """
    Load config for a domain.

    Args:
        config_dir: path to config root (e.g., 'config/')
        domain: one of 'MD', 'ToM', 'phys', 'Lan'

    Returns:
        dict: {task_name: task_config}
    """
    config_path = os.path.join(config_dir, domain, config_file)
    with open(config_path, "r") as f:
        return json.load(f)


def load_task_data(data_dir, domain, task_config):
    """
    Load data for a specific task.

    Args:
        data_dir: path to data root (e.g., 'data/')
        domain: domain name
        task_config: config dict for this task

    Returns:
        list of dicts
    """
    data_path = os.path.join(data_dir, domain, task_config["path"])
    with open(data_path, "r") as f:
        return json.load(f)


def format_prompts(data, task_config, model_name, tokenizer):
    """
    Format data into clean/corrupted prompts with chat template.

    Supports two modes:
    - Standard (default): data has clean_correct, clean_incorrect, corrupted_correct, corrupted_incorrect
    - Single-answer (single_answer=true in config): data has answer field only

    Prompt construction order: prefix + problem + concatenation + suffix

    Args:
        data: list of dicts with clean_problem, corrupted_problem, etc.
        task_config: config with prefix, suffix, concatenation, etc.
        model_name: model name for chat template
        tokenizer: tokenizer for chat template

    Returns:
        dict with keys depending on mode:
        - Standard: clean_prompts, corrupted_prompts, clean_correct, clean_incorrect
        - Single-answer: clean_prompts, corrupted_prompts, answers, single_answer=True
    """
    prefix = task_config.get("prefix") or ""
    suffix_template = task_config.get("suffix") or ""
    concatenation = task_config.get("concatenation") or ""
    force_base = task_config.get("force_base_evaluation", False)
    single_answer = task_config.get("single_answer", False)

    clean_prompts = []
    corrupted_prompts = []

    for item in data:
        if single_answer:
            clean_msg = prefix + item["clean_problem"]
            if concatenation:
                clean_msg += concatenation
            corrupted_msg = prefix + item["corrupted_problem"]
            if concatenation:
                corrupted_msg += concatenation
        else:
            # Build suffix with answer placeholders filled
            suffix = suffix_template.format(
                correct_answer=str(item["clean_correct"]),
                incorrect_answer=str(item["clean_incorrect"]),
            ) if suffix_template else ""

            clean_msg = prefix + item["clean_problem"]
            if concatenation:
                clean_msg += concatenation
            if suffix:
                clean_msg += " " + suffix

            corrupted_suffix = suffix_template.format(
                correct_answer=str(item["corrupted_correct"]),
                incorrect_answer=str(item["corrupted_incorrect"]),
            ) if suffix_template else ""

            corrupted_msg = prefix + item["corrupted_problem"]
            if concatenation:
                corrupted_msg += concatenation
            if corrupted_suffix:
                corrupted_msg += " " + corrupted_suffix

        # Apply chat template or use raw text
        if force_base:
            clean_prompts.append(clean_msg)
            corrupted_prompts.append(corrupted_msg)
        else:
            clean_prompts.append(get_chat_template(tokenizer, model_name, clean_msg))
            corrupted_prompts.append(get_chat_template(tokenizer, model_name, corrupted_msg))

    if single_answer:
        answers = [item["answer"] for item in data]
        return {
            "clean_prompts": clean_prompts,
            "corrupted_prompts": corrupted_prompts,
            "answers": answers,
            "single_answer": True,
        }
    else:
        clean_correct = [item["clean_correct"] for item in data]
        clean_incorrect = [item["clean_incorrect"] for item in data]
        return {
            "clean_prompts": clean_prompts,
            "corrupted_prompts": corrupted_prompts,
            "clean_correct": clean_correct,
            "clean_incorrect": clean_incorrect,
            "single_answer": False,
        }


def prepare_sequence_inputs(prompts, answers, tokenizer):
    """
    Tokenize prompt+answer sequences for full-sequence teacher forcing.

    Each example: prompt tokens + answer tokens, with tracked prompt_len
    so we can compute loss only on answer tokens.

    Args:
        prompts: list of prompt strings
        answers: list of answer strings

    Returns:
        all_input_ids: list of 1D tensors [seq_len_i] (variable length)
        prompt_lens: list of int
    """
    all_input_ids = []
    prompt_lens = []

    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids

        all_input_ids.append(torch.tensor(full_ids))
        prompt_lens.append(len(prompt_ids))

    return all_input_ids, prompt_lens


def pad_sequence_inputs(input_ids_list):
    """
    Pad a list of variable-length input_ids to the same length.

    Args:
        input_ids_list: list of 1D tensors

    Returns:
        padded: tensor [batch, max_len], padded with 0
    """
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded[i, :ids.shape[0]] = ids
    return padded


def tokenize_prompts(prompts, tokenizer):
    """
    Tokenize a list of prompts with padding.

    Returns:
        input_ids: torch.Tensor [N, seq_len]
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return encoded["input_ids"]


def get_task_list(config_dir, domain):
    """Get list of task names for a domain."""
    config = load_task_config(config_dir, domain)
    return list(config.keys())