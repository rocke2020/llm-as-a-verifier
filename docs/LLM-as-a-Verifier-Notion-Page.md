---
title: "LLM-as-a-Verifier: A General-Purpose Verification Framework"
source: Notion Blog
url: https://llm-as-a-verifier.notion.site/
fetched: 2026-04-28
---

# LLM-as-a-Verifier: A General-Purpose Verification Framework

[Jacky Kwok](https://www.linkedin.com/in/jackykwok02/)$^{1\dagger}$\*, [Shulu Li](http://linkedin.com/in/shulu-li)$^{2}$\*, [Pranav Atreya](https://pranavatreya.github.io/)$^{2}$, [Yuejiang Liu](https://sites.google.com/view/yuejiangliu/home)$^{1}$, [Marco Pavone](https://research.nvidia.com/person/marco-pavone)$^{13}$

[Ion Stoica](https://people.eecs.berkeley.edu/~istoica/)$^{2§}$, [Azalia Mirhoseini](https://azaliamirhoseini.com/)$^{1§}$ Stanford University$^{1}$ UC Berkeley$^{2}$ NVIDIA$^{3}$

$^{\dagger}$Project Lead · \*Core Contribution · §Equal Advising

> Posted: April 9, 2026

---

## SOTA on Terminal-Bench & SWE-Bench Verified

- We introduce **LLM-as-a-Verifier**, a general-purpose verification framework that provides fine-grained feedback by scaling scoring granularity, repeated verification, and criteria decomposition
- LLM-as-a-Verifier achieves **state-of-the-art performance** on [Terminal-Bench 2](https://www.tbench.ai/leaderboard/terminal-bench/2.0) (86.4%) and [SWE-Bench Verified](https://www.swebench.com/) (77.8%) when used as a **trajectory reward model** for test-time scaling

Try [LLM-as-a-Verifier on GitHub](https://github.com/llm-as-a-verifier/llm-as-a-verifier)

*From Stanford AI Lab & UC Berkeley Sky Computing Lab*

---

## Evaluating LLM-as-a-Verifier

Across challenging long-horizon benchmarks such as Terminal-Bench 2.0 and SWE-Bench Verified, LLM-as-a-Verifier outperforms frontier models including Claude Opus 4.6, GPT 5.4, and Gemini Models. Results are reported from the official Terminal-Bench and SWE-Bench leaderboard.

> **Note:** We use ForgeCode and mini-swe-agent as the scaffolds. For TerminalBench, we sample 5 trajectories from Claude Opus 4.6. For SWE-Bench, we sample 3 trajectories each from Claude Opus 4.6, Gemini 3 Flash, and Claude Opus 4.5. Gemini 2.5 Flash is used as the verifier in our experiments. Our results are fully reproducible and available on [GitHub](https://github.com/llm-as-a-verifier/llm-as-a-verifier).

---

## TL;DR

We find that verification accuracy consistently improves as we scale the scoring granularity, repeated verification, and criteria decomposition. LLM-as-a-Verifier achieves 78.9% pairwise verification accuracy on Terminal-Bench and enhances downstream success rate from 81.8% to 86.4% (SOTA) through test-time scaling and verification.

---

## Motivation

Standard **LLM-as-a-Judge** prompts the model to output a score token (e.g., 1–8) and select the highest-probability token, using it as the final discrete score. However, this approach often suffers from coarse-grained scoring. When comparing complex agent trajectories, standard LLM-as-a-Judge often assigns the same score (e.g., both trajectories receive a score of 4), resulting in a tie and failing to discriminate between them. Coarse scoring leads to **27% ties** on Terminal-Bench.

---

## Methodology

By definition, a **judge** is one who forms an overall opinion and assigns a decision, whereas a **verifier** is one who confirms the truth or correctness of something and requires more detailed evaluations.

To this end, we introduce **LLM-as-a-Verifier**, a general-purpose verification framework that provides fine-grained feedback by scaling:

1. the number of **repeated verifications**
2. the **granularity** of score tokens
3. the **decomposition** of evaluation criteria

Let $V_{\text{score}} = \{v_1, \ldots, v_G\}$ denote an ordered set of tokens representing discrete score levels. Given a task prompt $t$, a language model $p_\theta$, a criterion $c$, and two candidate trajectories $\tau_i$ and $\tau_j$, we construct scoring prompts and obtain their conditional distributions $p_\theta(v \mid t, c, \tau_i)$ and $p_\theta(v \mid t, c, \tau_j)$ by extracting the top logprobs from `<score_A>` and `<score_B>`:

### Example Prompt

```
You are an expert [domain] reviewer. You will see a task description and two trajectories.

Evaluation Criteria: [domain specific criteria]

Task: {task prompt} Trajectory A: {A} Trajectory B: {B}

Carefully analyze each trajectory, then provide your final scores:
<score_A> INTEGER_1_TO_8 </score_A>
<score_B> INTEGER_1_TO_8 </score_B>

Rating Rules:
- Rate correctness on a 1-8 scale (1 = correct, 5 = borderline, 8 = incorrect)
```

> **Note:** We use a letter-based scale instead of digits to enable logprob extraction for granularity scaling.

Rather than reducing each distribution into a single discrete score, we approximate the reward of trajectories as:

$$R(t, \tau) = \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G} p_\theta(v_g \mid t, c, \tau) \, \phi(v_g)$$

**Where:**

- $C$ = number of evaluation criteria
- $K$ = number of repeated verifications
- $G$ = number of score tokens (granularity level)
- $p_\theta(v_g \mid t, c, \tau)$ = probability assigned by model $\theta$ to score token $v_g$
- $\phi(v_g)$ = maps each scoring token to a scalar value
- $V_{\text{score}} = \{v_1, \ldots, v_G\}$ = ordered set of discrete score tokens

To pick the best trajectory among $N$ candidates for a given task, a **round-robin tournament** is conducted. For every pair $(i, j)$ the verifier produces $R(t, \tau_i)$ and $R(t, \tau_j)$ using the formula above. The trajectory with the higher reward receives a win, and the trajectory with the most wins across all $\binom{N}{2}$ pairs is selected.

---

## Results

**LLM-as-a-Verifier** consistently outperforms **LLM-as-a-Judge**, achieving 77.4% verification accuracy while eliminating ties entirely across all repeated verification budgets. Even at k = 16, where repeated verification reduces judge ties, the verifier still maintains 7% higher accuracy.

### Verification Accuracy (higher is better)

| Repeated Verification | LLM-as-a-Verifier | LLM-as-a-Judge (Discrete) |
|---|---|---|
| k=1 | 74.7% | 57.0% |
| k=4 | 77.0% | 67.4% |
| k=16 | 77.4% | 70.2% |

### Tie Rate (lower is better)

| Repeated Verification | LLM-as-a-Verifier | LLM-as-a-Judge (Discrete) |
|---|---|---|
| k=1 | 0% | 26.5% |
| k=4 | 0% | 11.5% |
| k=16 | 0% | 5.4% |

We show that LLM-as-a-Verifier generalizes across different agent harnesses, improving ForgeCode to 86.4%, Terminus-Kira to 79.4%, and Terminus 2 to 71.2%. This demonstrates that our method can be applied in a plug-and-play manner across any agent harness or model.

We find that the verification accuracy consistently improves as we scale both the number of repeated verifications (1 → 16) and score granularities (1 → 20).

Instead of directly estimating the overall quality of a trajectory, we verify three simpler factors:

- **Specification:** whether the trajectory satisfies all task requirements (paths, naming, etc.)
- **Output:** whether the verification output format matches the expected result
- **Errors:** whether the trajectory is free of failure signals

### Takeaway

- Scaling the scoring token granularity reduces quantization error, enabling better approximation of underlying continuous rewards
- Scores from individual verification can be biased or noisy, and an ensemble of repeated verifications helps average out these biases
- Decomposing trajectory verification into complementary factors improves accuracy

---

## What's Next?

LLM-as-a-Verifier opens up a new space for scalable verification. Moving forward, we plan to focus on several key directions:

- **Benchmarking:** Develop more comprehensive evaluation suites and benchmarks for verification.
- **Broader support:** Extend support for process and outcome reward models (PRMs and ORMs).
- **Verification for RL:** Integrate LLM-as-a-Verifier into RL training pipelines.
- …and more to come.

---

## Join us!

We call on the community to join us in this effort, either by providing your feedback or contributing to the project!

Please don't hesitate to get in touch:

- **Github repo:** [llm-as-a-verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier)
- **Email:** llmverifiers@gmail.com
- **Contact:** jackykwok@stanford.edu
- **Slack:** [LLM-as-a-Verifier](https://join.slack.com/t/llm-as-a-verifier/shared_invite/zt-3utx6oe8m-86ACBqtPGfsOnpOoMJQwng)

---

## Citation

If you find LLM-as-a-Verifier useful, please consider citing it:

```bibtex
@misc{kwok2026llmverifier,
  title={LLM-as-a-Verifier: A General-Purpose Verification Framework},
  author={Jacky Kwok and Shulu Li and Pranav Atreya and Yuejiang Liu and Marco Pavone and Ion Stoica and Azalia Mirhoseini},
  year={2026},
  note={Notion Blog}
}
```
