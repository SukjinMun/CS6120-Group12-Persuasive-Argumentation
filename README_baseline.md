# LLM-as-a-Judge Baseline Results

## What We Did

Zero-shot prompted Llama 3.1 8B and Qwen 2.5 7B to judge persuasiveness on 853 CMV test examples. Two settings:

- **Pointwise (single):** Model sees one response at a time, predicts "Yes" (would get Delta) or "No" (wouldn't). Each example gets two predictions — one for chosen, one for rejected.
- **Pairwise:** Model sees both responses, picks which is more persuasive ("A" or "B"). Order is randomized (seed=42).

## Files

| File | Model | Setting |
|---|---|---|
| `llama_single.json` | Llama 3.1 8B | Pointwise |
| `llama_pairwise.json` | Llama 3.1 8B | Pairwise |
| `qwen_single.json` | Qwen 2.5 7B | Pointwise |
| `qwen_pairwise.json` | Qwen 2.5 7B | Pairwise |

## Pointwise Fields

| Field | Meaning |
|---|---|
| `chosen_raw_output` | Model's raw output when shown the chosen (Delta) response |
| `chosen_pred` | Parsed: "Yes" / "No" / "Unknown" |
| `chosen_correct` | `true` if model said "Yes" (chosen should be Yes) |
| `rejected_raw_output` | Model's raw output when shown the rejected (no Delta) response |
| `rejected_pred` | Parsed: "Yes" / "No" / "Unknown" |
| `rejected_correct` | `true` if model said "No" (rejected should be No) |

## Pairwise Fields

| Field | Meaning |
|---|---|
| `chosen_position` | Where chosen was placed: "A" or "B" (randomized) |
| `raw_output` | Model's raw output |
| `pred` | Parsed: "A" / "B" / "Unknown" |
| `correct` | `true` if model picked the chosen response |

## Results

### Pointwise

| Metric | Llama | Qwen |
|---|---|---|
| Chosen Acc (Recall) | 26.3% | 0.2% |
| Rejected Acc (Specificity) | 76.7% | 99.4% |
| Overall Acc | 51.5% | 49.8% |

### Pairwise

| Metric | Llama | Qwen |
|---|---|---|
| Accuracy | 60.8% | 59.3% |

Both models barely beat random chance (50%). Strong negative bias in pointwise — models default to "No".
