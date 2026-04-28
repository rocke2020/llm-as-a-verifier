"""Tests for scripts/verifier_core.py — SCALE structure, extract_score, select_best."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from verifier_core import SCALE, GRANULARITY, extract_score, _find_tag_logprobs, select_best


# ---------------------------------------------------------------------------
# SCALE inspection
# ---------------------------------------------------------------------------

def test_scale_print():
    """Print SCALE contents for manual inspection."""
    print("\n=== SCALE ===")
    print(f"Granularity: {GRANULARITY}")
    print(f"Score format: {SCALE['score_format']}")
    print(f"\nScale description:\n{SCALE['scale_description']}")
    print(f"\nValid tokens ({len(SCALE['valid_tokens'])} entries):")
    for tok, val in sorted(SCALE["valid_tokens"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  '{tok}' → {val}")


def test_scale_granularity_is_20():
    assert GRANULARITY == 20


def test_scale_has_40_valid_tokens():
    assert len(SCALE["valid_tokens"]) == 40


def test_scale_uppercase_mapping():
    vt = SCALE["valid_tokens"]
    assert vt["A"] == 20.0
    assert vt["B"] == 19.0
    assert vt["S"] == 2.0
    assert vt["T"] == 1.0


def test_scale_lowercase_mirrors_uppercase():
    vt = SCALE["valid_tokens"]
    for i in range(GRANULARITY):
        upper = chr(65 + i)
        lower = chr(97 + i)
        assert vt[upper] == vt[lower], f"'{upper}'={vt[upper]} != '{lower}'={vt[lower]}"


def test_scale_values_range():
    values = set(SCALE["valid_tokens"].values())
    assert min(values) == 1.0
    assert max(values) == 20.0
    assert values == {float(i) for i in range(1, 21)}


def test_scale_score_format():
    assert SCALE["score_format"] == "LETTER_A_TO_T"


# ---------------------------------------------------------------------------
# _find_tag_logprobs
# ---------------------------------------------------------------------------

def test_find_tag_logprobs_basic():
    tokens = ["<score_A>", "B"]
    position_logprobs = [
        [("<score_A>", -0.1)],
        [("B", -0.5), ("C", -1.0)],
    ]
    result = _find_tag_logprobs(tokens, position_logprobs, "<score_A>")
    assert result == [("B", -0.5), ("C", -1.0)]


def test_find_tag_logprobs_split_tokens():
    tokens = ["<score", "_A>", "D"]
    position_logprobs = [
        [("<score", -0.1)],
        [("_A>", -0.2)],
        [("D", -0.3), ("E", -1.5)],
    ]
    result = _find_tag_logprobs(tokens, position_logprobs, "<score_A>")
    assert result == [("D", -0.3), ("E", -1.5)]


def test_find_tag_logprobs_not_found():
    tokens = ["hello", "world"]
    position_logprobs = [[("hello", -0.1)], [("world", -0.2)]]
    assert _find_tag_logprobs(tokens, position_logprobs, "<score_A>") is None


def test_find_tag_logprobs_none_inputs():
    assert _find_tag_logprobs(None, None, "<score_A>") is None
    assert _find_tag_logprobs([], [], "<score_A>") is None


# ---------------------------------------------------------------------------
# extract_score
# ---------------------------------------------------------------------------

def _make_logprobs(tag, token_probs):
    """Helper: build (tokens, position_logprobs) with a tag followed by score logprobs."""
    tokens = [tag, token_probs[0][0]]
    position_logprobs = [
        [(tag, 0.0)],
        [(tok, math.log(p)) for tok, p in token_probs],
    ]
    return tokens, position_logprobs


def test_extract_score_all_A():
    tokens, plp = _make_logprobs("<score_A>", [("A", 1.0)])
    score = extract_score("", tokens, plp, "<score_A>")
    assert abs(score - 1.0) < 1e-6


def test_extract_score_all_T():
    tokens, plp = _make_logprobs("<score_A>", [("T", 1.0)])
    score = extract_score("", tokens, plp, "<score_A>")
    assert abs(score - 0.0) < 1e-6


def test_extract_score_weighted_average():
    tokens, plp = _make_logprobs("<score_A>", [("A", 0.5), ("T", 0.5)])
    score = extract_score("", tokens, plp, "<score_A>")
    # A=20, T=1, expected = (0.5*20 + 0.5*1) / 1.0 = 10.5
    # normalized = (10.5 - 1) / (20 - 1) = 9.5/19 = 0.5
    assert abs(score - 0.5) < 1e-6


def test_extract_score_case_insensitive_logprobs():
    tokens, plp = _make_logprobs("<score_A>", [("a", 0.6), ("b", 0.4)])
    score = extract_score("", tokens, plp, "<score_A>")
    # a=20, b=19, expected = (0.6*20 + 0.4*19) / 1.0 = 19.6
    # normalized = (19.6 - 1) / 19 = 18.6/19 ≈ 0.9789
    assert 0.97 < score < 0.99


def test_extract_score_text_fallback():
    score = extract_score("<score_A> A </score_A>", None, None, "<score_A>")
    assert abs(score - 1.0) < 1e-6


def test_extract_score_text_fallback_low():
    score = extract_score("<score_B> T </score_B>", None, None, "<score_B>")
    assert abs(score - 0.0) < 1e-6


def test_extract_score_no_data_returns_half():
    score = extract_score("", None, None, "<score_A>")
    assert score == 0.5


# ---------------------------------------------------------------------------
# select_best (round-robin tournament)
# ---------------------------------------------------------------------------

def _make_tasks(rewards):
    """Build tasks dict from a list of rewards."""
    return {
        "task1": [
            {"trial_name": f"trial_{i}", "reward": r, "problem": "", "trace": ""}
            for i, r in enumerate(rewards)
        ]
    }


def test_select_best_clear_winner():
    tasks = _make_tasks([0, 1, 0])
    scores = {
        "c1|task1|0,1|0": {"score_i": 0.3, "score_j": 0.9},
        "c1|task1|0,2|0": {"score_i": 0.5, "score_j": 0.4},
        "c1|task1|1,2|0": {"score_i": 0.8, "score_j": 0.2},
    }
    sel = select_best(tasks, ["task1"], scores, ["c1"], n_reps=1, rep_idx=0)
    assert sel["task1"]["idx"] == 1
    assert sel["task1"]["reward"] == 1


def test_select_best_tie_splits():
    tasks = _make_tasks([1, 1])
    scores = {
        "c1|task1|0,1|0": {"score_i": 0.5, "score_j": 0.5},
    }
    sel = select_best(tasks, ["task1"], scores, ["c1"], n_reps=1, rep_idx=0)
    assert sel["task1"]["idx"] in (0, 1)


def test_select_best_multi_criteria():
    tasks = _make_tasks([0, 1, 0])
    scores = {
        "c1|task1|0,1|0": {"score_i": 0.6, "score_j": 0.4},  # c1: 0 beats 1
        "c2|task1|0,1|0": {"score_i": 0.3, "score_j": 0.9},  # c2: 1 beats 0
        "c1|task1|0,2|0": {"score_i": 0.3, "score_j": 0.7},
        "c2|task1|0,2|0": {"score_i": 0.3, "score_j": 0.7},
        "c1|task1|1,2|0": {"score_i": 0.8, "score_j": 0.2},
        "c2|task1|1,2|0": {"score_i": 0.7, "score_j": 0.3},
    }
    sel = select_best(tasks, ["task1"], scores, ["c1", "c2"], n_reps=1, rep_idx=0)
    # pair (0,1): avg c1+c2: i=(0.6+0.3)/2=0.45, j=(0.4+0.9)/2=0.65 → 1 wins
    # pair (0,2): avg: i=(0.3+0.3)/2=0.3, j=(0.7+0.7)/2=0.7 → 2 wins
    # pair (1,2): avg: i=(0.8+0.7)/2=0.75, j=(0.2+0.3)/2=0.25 → 1 wins
    # wins: [0, 2, 1] → best = 1
    assert sel["task1"]["idx"] == 1


if __name__ == "__main__":
    test_scale_print()
