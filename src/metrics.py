"""Evaluation metrics for attribution patching and ablation."""

import torch


# =============================================================================
# Full-sequence teacher forcing metrics
# =============================================================================

def compute_sequence_log_prob(logits, input_ids, prompt_len):
    """
    Compute sum of log P(token_i | tokens_<i) over answer tokens only (conditional).

    Args:
        logits: [seq_len, vocab] (single example, no batch dim)
        input_ids: [seq_len] (single example)
        prompt_len: int, number of prompt tokens

    Returns:
        scalar tensor (sum of log probs over answer tokens)
    """
    # shifted: predict token at position i using logits at position i-1
    answer_logits = logits[prompt_len - 1:-1, :]  # [answer_len, vocab]
    answer_labels = input_ids[prompt_len:]  # [answer_len]

    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum()


def compute_joint_log_prob(logits, input_ids):
    """
    Compute sum of log P(token_i | tokens_<i) over ALL tokens (joint probability).

    Args:
        logits: [seq_len, vocab] (single example, no batch dim)
        input_ids: [seq_len] (single example)

    Returns:
        scalar tensor (sum of log probs over all tokens except first)
    """
    # Predict token at position i using logits at position i-1
    shift_logits = logits[:-1, :]  # [seq_len-1, vocab]
    shift_labels = input_ids[1:]   # [seq_len-1]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum()


def compute_batch_sequence_log_probs(all_logits, all_input_ids, all_prompt_lens):
    """
    Compute sequence log probs for a batch of examples.
    Each example can have different answer length.

    Args:
        all_logits: list of [seq_len_i, vocab] tensors
        all_input_ids: list of [seq_len_i] tensors
        all_prompt_lens: list of int

    Returns:
        tensor [batch] of log probs
    """
    log_probs = []
    for logits, input_ids, prompt_len in zip(all_logits, all_input_ids, all_prompt_lens):
        lp = compute_sequence_log_prob(logits, input_ids, prompt_len)
        log_probs.append(lp)
    return torch.stack(log_probs)


def get_teacher_forcing_diff(correct_log_probs, incorrect_log_probs):
    """
    Full-sequence teacher forcing diff: mean of (correct_lp - incorrect_lp).

    Args:
        correct_log_probs: tensor [batch]
        incorrect_log_probs: tensor [batch]

    Returns:
        scalar tensor
    """
    return (correct_log_probs - incorrect_log_probs).mean()


def get_accuracy(correct_log_probs, incorrect_log_probs):
    """
    Accuracy: % of examples where correct_log_prob > incorrect_log_prob.

    Args:
        correct_log_probs: tensor [batch]
        incorrect_log_probs: tensor [batch]

    Returns:
        float
    """
    return (correct_log_probs > incorrect_log_probs).float().mean().item()


def make_normalized_metric(clean_baseline, corrupted_baseline):
    """
    Create a normalized metric function for full-sequence TF diff.
    Returns 1.0 for clean performance, 0.0 for corrupted.
    """
    def metric(correct_lps, incorrect_lps):
        tf_diff = get_teacher_forcing_diff(correct_lps, incorrect_lps)
        return (tf_diff - corrupted_baseline) / (clean_baseline - corrupted_baseline)
    return metric


# =============================================================================
# Single-answer metrics (for BLiMP-style tasks)
# =============================================================================

def get_single_answer_diff(clean_log_probs, corrupted_log_probs):
    """
    Single-answer diff: mean of (clean_lp - corrupted_lp).
    Same answer evaluated under clean vs corrupted prompt.
    """
    return (clean_log_probs - corrupted_log_probs).mean()


def get_single_answer_accuracy(clean_joint_lps, corrupted_joint_lps):
    """
    Single-answer accuracy: % of examples where P(clean+answer) > P(corrupted+answer).
    Uses joint (full-sequence) log probs, not conditional.
    """
    return (clean_joint_lps > corrupted_joint_lps).float().mean().item()


def make_normalized_metric_single(clean_baseline, corrupted_baseline):
    """
    Create a normalized metric for single-answer mode.
    metric = log P(answer|clean) - log P(answer|corrupted), normalized.
    """
    def metric(clean_lps, corrupted_lps):
        diff = get_single_answer_diff(clean_lps, corrupted_lps)
        return (diff - corrupted_baseline) / (clean_baseline - corrupted_baseline)
    return metric