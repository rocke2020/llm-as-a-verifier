# `run_terminal_bench.py` Workflow Explanation

Command:

```bash
python scripts/run_terminal_bench.py --granularity 20 --n-verifications 4 --criteria 3
```

## What This Script Does

This script evaluates agent trajectories on [Terminal-Bench 2](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) tasks using the **LLM-as-a-Verifier** method. Given multiple agent trajectories (trial runs) per task, it uses DeepSeek V4 Flash to score every pair of trajectories across multiple evaluation criteria, then runs a round-robin tournament to select the best trajectory per task and reports the overall success rate.

## CLI Arguments

| Argument | Value | Meaning |
|---|---|---|
| `--granularity` | `20` | Scoring scale uses 20 discrete tokens (letters A-T). Only `20` is supported. |
| `--n-verifications` | `4` | Each (pair, criterion) is scored 4 independent times to reduce noise. |
| `--criteria` | `3` | Use all 3 evaluation criteria (C1: Specification Adherence, C2: Output Match, C3: Error Signal Detection). |

## End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. LOAD TRAJECTORIES                                             │
│    Read JSON files from data/terminal_trajs/<agent>/             │
│    Group trials by task, extract problem + formatted trace       │
├──────────────────────────────────────────────────────────────────┤
│ 2. CLASSIFY TASKS                                                │
│    all-pass: every trial reward=1 (no scoring needed)            │
│    all-fail: every trial reward=0 (no scoring needed)            │
│    swing:    mixed rewards → need LLM scoring to pick best       │
├──────────────────────────────────────────────────────────────────┤
│ 3. PAIRWISE SCORING (only swing tasks)                           │
│    For each pair (i,j) × each criterion × each rep:             │
│      → Build prompt with task + trace_A + trace_B + criterion    │
│      → Call DeepSeek V4 Flash with logprobs                      │
│      → Extract expected score from logprob distribution          │
├──────────────────────────────────────────────────────────────────┤
│ 4. ROUND-ROBIN TOURNAMENT (per swing task)                       │
│    For each pair: average scores across criteria and reps         │
│    Higher average score → 1 win; tie → 0.5 each                 │
│    Trial with most wins is selected                              │
├──────────────────────────────────────────────────────────────────┤
│ 5. EVALUATE & REPORT                                             │
│    Pass@1 (random selection baseline)                            │
│    LLM-as-a-Verifier (tournament-selected, mean ± SE)            │
│    Oracle (best possible selection upper bound)                  │
└──────────────────────────────────────────────────────────────────┘
```

## Detailed Step-by-Step with Code References

### Step 1: Load Trajectories

**File:** `scripts/run_terminal_bench.py`

The script resolves the agent directory and loads all trajectory JSON files:

```python
# run_terminal_bench.py:210-221
agent_dir = os.path.join(TRAJ_DIR, args.trajs)          # data/terminal_trajs/forge_gpt54
tasks = load_all_trials(agent_dir)
```

`load_all_trials()` (`run_terminal_bench.py:150-178`) globs each task subdirectory, reads `*_trajectory.json` files, and builds a dict:

```python
# run_terminal_bench.py:153-173
task_dirs = sorted(globmod.glob(os.path.join(agent_dir, "*/")))
for task_dir in task_dirs:
    traj_files = sorted(globmod.glob(os.path.join(task_dir, "*_trajectory.json")))
    for traj_file in traj_files:
        data = json.load(f)
        trials.append({
            "trial_name": data.get("trial_name", ""),
            "reward": data.get("reward", 0),            # ground-truth: 0 or 1
            "problem": extract_problem(steps, task_name),
            "trace": format_trace(trajectory),
        })
```

**`extract_problem()`** (`run_terminal_bench.py:127-147`) pulls the task description from the first `"user"` step in the trajectory. If missing, falls back to the agent's initial analysis.

**`format_trace()`** (`run_terminal_bench.py:99-124`) converts a trajectory JSON into a readable text log:

```python
# run_terminal_bench.py:104-123
for step in trajectory.get("steps", []):
    if source in ("system", "user"):
        continue                                         # skip system/user steps
    if source == "agent":
        parts.append(f"--- Agent Step {step_id} ---")
        # ... append agent message, commands, and terminal output
```

### Step 2: Classify Tasks

**File:** `scripts/run_terminal_bench.py:225-232`

Tasks are split into three buckets based on their ground-truth rewards:

```python
# run_terminal_bench.py:225-232
for task_name, trials in sorted(tasks.items()):
    rewards = [t["reward"] for t in trials]
    if all(r == 1 for r in rewards):
        all_pass.append(task_name)      # all 5 trials succeeded → always correct
    elif all(r == 0 for r in rewards):
        all_fail.append(task_name)      # all 5 trials failed → always wrong
    else:
        swing.append(task_name)         # mixed → selection matters
```

Only **swing tasks** need LLM scoring. All-pass tasks contribute to every method's score; all-fail tasks contribute to none.

### Step 3: Pairwise LLM Scoring

This is the core of the LLM-as-a-Verifier method. It has several sub-components:

#### 3a. Define Evaluation Criteria

**File:** `scripts/run_terminal_bench.py:44-92`

Three criteria are defined, each with a detailed rubric:

| ID | Name | What it checks |
|---|---|---|
| `specification` | Specification Adherence | Did the agent meet exact task requirements (paths, formats, constraints)? |
| `output_match` | Output Match | Does the final terminal output literally match expected output? |
| `error_signals` | Error Signal Detection | Are there unresolved errors/tracebacks in the later steps? |

The `--criteria 3` flag selects all three:

```python
# run_terminal_bench.py:208
criteria = CRITERIA[: args.criteria]    # CRITERIA[0:3] → all three
```

#### 3b. Generate Scoring Jobs

**File:** `scripts/verifier_core.py:263-285`

`score_all_trials()` enumerates every combination that needs scoring:

```python
# verifier_core.py:275-285
for task_name in swing_tasks:
    n = len(trials)                                      # e.g. 5 trials per task
    for i, j in combinations(range(n), 2):               # C(5,2) = 10 pairs
        for crit in criteria:                            # 3 criteria
            for rep in range(n_reps):                    # 4 repetitions
                key = f"{crit['id']}|{task_name}|{i},{j}|{rep}"
                if key not in cached:
                    jobs.append(...)
```

**Total jobs per swing task:** C(5,2) × 3 × 4 = 10 × 3 × 4 = **120 API calls**.

Results are cached to `cache/cache_terminal_<agent>.json` so re-runs are free.

#### 3c. Build the Pairwise Prompt

**File:** `scripts/verifier_core.py:223-245`

Each prompt asks the LLM to compare two trajectories on a single criterion:

```python
# verifier_core.py:223-245
def create_prompt_for_criterion(problem, trace_a, trace_b, criterion, ground_truth_note):
    return (
        "You are an expert evaluator of AI coding agents. ..."
        f"**Task:**\n{problem}\n\n"
        f"**Trajectory A:**\n{trace_a}\n\n"
        f"**Trajectory B:**\n{trace_b}\n\n"
        f"**Evaluation Guideline — {criterion['name']}:**\n{criterion['description']}\n\n"
        f"**Rating Scale:**\n{SCALE['scale_description']}\n\n"
        "Then output your final scores:\n"
        "<score_A>LETTER_A_TO_T</score_A>\n"
        "<score_B>LETTER_A_TO_T</score_B>\n"
    )
```

A `GROUND_TRUTH_NOTE` is injected to remind the LLM to trust terminal output over agent self-assessment:

```python
# run_terminal_bench.py:38-42
GROUND_TRUTH_NOTE = (
    "**IMPORTANT:** Focus on TERMINAL OUTPUT as ground truth. "
    "Do NOT trust the agent's self-assessment or claims of success. ..."
)
```

#### 3d. Call DeepSeek V4 Flash with Logprobs

**File:** `scripts/verifier_core.py:71-118`

The API call requests `logprobs=True` with `top_logprobs=20`:

```python
# verifier_core.py:95-102
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=4096,
    temperature=1.0,
    logprobs=True,
    top_logprobs=top_logprobs,                           # 20
)
```

Key design decisions:
- **`temperature=1.0`**: full stochasticity so repeated verifications are independent.
- **`logprobs=True`**: instead of just reading the output letter, we read the probability distribution over all 20 score tokens.

The function returns `(text, tokens, position_logprobs)` where `position_logprobs` is a list of `[(token, logprob), ...]` at each output position.

#### 3e. Extract Score from Logprob Distribution

**File:** `scripts/verifier_core.py:125-216`

This is the key insight of the paper. Instead of using the discrete output letter, the score is the **expected value** of the logprob distribution over valid score tokens:

```python
# verifier_core.py:137-196
def extract_score(text, tokens, position_logprobs, tag):
    tag_lp = _find_tag_logprobs(tokens, position_logprobs, tag)  # find logprobs at <score_A>
    probs = {}
    if tag_lp:
        for tok_str, logprob in tag_lp:
            tok = tok_str.strip()
            if tok in valid_tokens:                              # A-T (or a-t)
                val = valid_tokens[tok]                          # A→20, B→19, ... T→1
                p = math.exp(logprob)
                probs[val] = max(probs.get(val, 0.0), p)

    # Compute expected value, normalized to [0,1]
    total_p = sum(probs.values())
    expected = sum(v * p for v, p in probs.items()) / total_p
    return (expected - min_val) / (max_val - min_val)
```

**The g=20 scoring scale** (`verifier_core.py:24-44`):

| Token | Value | Meaning |
|---|---|---|
| A | 20 | Clearly succeeded (best) |
| B-D | 19-17 | Succeeded with minor issues |
| E-G | 16-14 | Above average |
| H-J | 13-11 | Uncertain, leans success |
| K-M | 10-8 | Uncertain, leans failure |
| N-P | 7-5 | Below average |
| Q-S | 4-2 | Failed with partial progress |
| T | 1 | Clearly failed (worst) |

**Fallback** (`verifier_core.py:198-216`): If logprobs are unavailable, the score is parsed from the XML tag in the text output using regex.

#### 3f. Parallel Execution with Caching

**File:** `scripts/verifier_core.py:296-327`

All scoring jobs run concurrently with a thread pool and progress bar:

```python
# verifier_core.py:296-322
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(score_pair_criterion, client, prob, ta, tb, crit,
                        ground_truth_note): key
        for key, prob, ta, tb, crit in jobs
    }
    pbar = tqdm(as_completed(futures), total=len(futures), desc="Scoring")
    # ... collect results, save cache periodically
```

Errors default to `(0.5, 0.5)` — a neutral tie:

```python
# verifier_core.py:313
cached[key] = {"score_i": 0.5, "score_j": 0.5}
```

### Step 4: Round-Robin Tournament

**File:** `scripts/verifier_core.py:330-371`

For each swing task, a round-robin tournament picks the best trial:

```python
# verifier_core.py:336-364
for task_name in swing_tasks:
    wins = [0.0] * n
    for i, j in combinations(range(n), 2):
        # Average scores across all criteria and reps
        si = si_sum / count
        sj = sj_sum / count
        if si > sj:
            wins[i] += 1
        elif sj > si:
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    best_idx = max(range(n), key=lambda t: wins[t])
```

The trial with the most wins is selected (`verifier_core.py:364`). Its ground-truth `reward` (0 or 1) is stored alongside the selection (`verifier_core.py:365-369`):

```python
# verifier_core.py:364-369
best_idx = max(range(n), key=lambda t: wins[t])
selections[task_name] = {
    "idx": best_idx,
    "trial": trials[best_idx]["trial_name"],
    "reward": trials[best_idx]["reward"],       # ground-truth: 0 or 1
}
```

Later, `eval_config()` counts how many selected trials actually solved the task (`verifier_core.py:384`):

```python
# verifier_core.py:382-385
sel = select_best(tasks, swing, scores, criteria_ids, n_reps=1, rep_idx=rep)
sc = sum(1 for s in sel.values() if s["reward"] == 1)   # count correct selections
rep_totals.append(len(all_pass) + sc)
```

The verifier doesn't see `reward` during scoring — it only sees trajectories. The `reward` field is used purely for evaluation: if the verifier picked a trial whose `reward == 1`, the task counts as solved; if it picked `reward == 0`, it doesn't.

**What happens when the verifier picks `reward == 0`:** The task counts as unsolved — it contributes 0 to the final score, even though a passing trial (`reward == 1`) exists among the candidates. This is a missed opportunity: the verifier had a winning trial available but failed to identify it. Each wrong pick on a swing task loses exactly 1 point from the Oracle ceiling.

Using Terminal-Bench as an example (63 all-pass, 9 all-fail, 17 swing):

| Scenario | Swing tasks correct | Total score | Rate |
|---|---|---|---|
| Oracle (perfect selection) | 17/17 | 63 + 17 = 80 | 89.9% |
| LLM-as-a-Verifier | ~16/17 | 63 + 16 = 79 | 88.8% |
| Pass@1 (random selection) | ~9.8/17 (expected) | 63 + 9.8 = 72.8 | 81.8% |
| Worst case (always wrong) | 0/17 | 63 + 0 = 63 | 70.8% |

The `sum(1 for s in sel.values() if s["reward"] == 1)` line (`verifier_core.py:384`) is where this happens — trials with `reward == 0` are simply not counted, so wrong picks silently reduce the total.

### Step 5: Evaluate and Report

**File:** `scripts/verifier_core.py:374-423`

Three metrics are computed and printed:

```python
# verifier_core.py:391-395
# Pass@1: expected score from random trial selection
random_ev = len(all_pass) + sum(
    sum(t["reward"] for t in tasks[tn]) / len(tasks[tn])
    for tn in swing)

# Oracle: upper bound — best possible selection
oracle = len(all_pass) + len(swing)

# LLM-as-a-Verifier: tournament selection, averaged over reps
mean_ens, se_ens = eval_config(crit_ids)
```

The `eval_config()` inner function (`verifier_core.py:379-389`) runs the tournament separately for each rep and computes mean and standard error:

```python
# verifier_core.py:379-389
def eval_config(criteria_ids):
    rep_totals = []
    for rep in range(n_reps):
        sel = select_best(tasks, swing, scores, criteria_ids, n_reps=1, rep_idx=rep)
        sc = sum(1 for s in sel.values() if s["reward"] == 1)
        rep_totals.append(len(all_pass) + sc)
    mean = sum(rep_totals) / len(rep_totals)
    se = (sum((t - mean) ** 2 for t in rep_totals) / len(rep_totals)) ** 0.5 / len(rep_totals) ** 0.5
    return mean, se
```

Results are printed and saved to `results/terminal_<agent>.txt`.

## Example Output

```
Method                                    Score            Rate
-------------------------------------------------------------------
Pass@1                                    72.8/89          81.8%
LLM-as-a-Verifier                        76.9±0.3/89      86.4%
Oracle (Bo5)                              80/89            89.9%
```

## Key Design Decisions

1. **Logprob-based scoring, not discrete labels.** Rather than reading a single letter answer, the system computes the expected value over the full probability distribution. This gives a continuous, fine-grained score from a single API call.

2. **Pairwise comparison, not absolute scoring.** Each LLM call compares two trajectories side-by-side on one criterion. Pairwise comparisons are more calibrated than absolute ratings.

3. **Criteria decomposition.** Breaking evaluation into 3 orthogonal criteria (specification, output match, error signals) lets the LLM focus on one aspect at a time, improving signal quality.

4. **Repeated verification (K=4).** Each pair × criterion is scored 4 times independently (with `temperature=1.0`) and results are averaged, reducing noise from any single stochastic evaluation.

5. **Round-robin tournament for selection.** Instead of averaging raw scores, the system uses pairwise wins to select the best trial — more robust to score scale inconsistencies.

6. **Aggressive caching.** Every API result is cached by `crit_id|task|pair|rep` key, so partial runs can be resumed and re-runs are instant.

## File Map

| File | Role |
|---|---|
| `scripts/run_terminal_bench.py` | Entry point: CLI args, criteria definitions, data loading, orchestration |
| `scripts/verifier_core.py` | Shared engine: DeepSeek client, scoring scale, prompt construction, score extraction, tournament, evaluation |
| `data/terminal_trajs/<agent>/` | Input: trajectory JSON files grouped by task |
| `cache/cache_terminal_<agent>.json` | Persistent cache of all API scoring results |
| `results/terminal_<agent>.txt` | Output: final results table |
