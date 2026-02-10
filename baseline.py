"""
LLM-as-a-Judge for CMV Delta Prediction
"""

import json
import argparse
import os
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ========== Config ==========
INPUT_PATH = "/scratch/chen.sijia2/NLP_project/cmv_test.json"
OUTPUT_DIR = "/scratch/chen.sijia2/NLP_project"

MODEL_MAP = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

OUTPUT_MAP = {
    ("llama", "single"): "llama_single.json",
    ("llama", "pairwise"): "llama_pairwise.json",
    ("qwen", "single"): "qwen_single.json",
    ("qwen", "pairwise"): "qwen_pairwise.json",
}

# ========== Prompt Templates ==========
SINGLE_PROMPT = """You are a judge evaluating whether an argument in a debate forum would successfully persuade the original poster to change their view (i.e., receive a Delta award).

## Original Post (View to be Changed):
{prompt}

## Response Argument:
{response}

## Task:
Would this response successfully persuade the original poster to change their view and receive a Delta (Δ)?
Answer with ONLY "Yes" or "No". Do not explain.

Answer:"""

PAIRWISE_PROMPT = """You are a judge evaluating which argument in a debate forum is more persuasive and more likely to change the original poster's view (i.e., receive a Delta award).

## Original Post (View to be Changed):
{prompt}

## Response A:
{response_a}

## Response B:
{response_b}

## Task:
Which response is more likely to successfully persuade the original poster to change their view and receive a Delta (Δ)?
Answer with ONLY "A" or "B". Do not explain.

Answer:"""


def load_model(model_name):
    print(f"Loading model: {model_name}")
    model_id = MODEL_MAP[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(model, tokenizer, prompt_text, model_name):
    if model_name == "llama":
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif model_name == "qwen":
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def parse_yes_no(text):
    """Extract Yes/No from model output."""
    text_lower = text.lower().strip()
    if text_lower.startswith("yes"):
        return "Yes"
    elif text_lower.startswith("no"):
        return "No"
    # Fallback: search in text
    if "yes" in text_lower:
        return "Yes"
    if "no" in text_lower:
        return "No"
    return "Unknown"


def parse_ab(text):
    """Extract A/B from model output."""
    text = text.strip()
    if text.startswith("A"):
        return "A"
    elif text.startswith("B"):
        return "B"
    # Fallback
    match = re.search(r'\b([AB])\b', text)
    if match:
        return match.group(1)
    return "Unknown"


def run_single(model, tokenizer, data, model_name, output_path):
    """Pointwise: judge each response independently."""
    # Load existing results for resume
    results = []
    done_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        for r in results:
            done_indices.add(r["index"])
        print(f"Resuming from {len(results)} completed entries")

    for idx, item in enumerate(tqdm(data, desc=f"{model_name} single")):
        if idx in done_indices:
            continue

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Judge chosen
        chosen_prompt = SINGLE_PROMPT.format(prompt=prompt, response=chosen)
        chosen_raw = generate_response(model, tokenizer, chosen_prompt, model_name)
        chosen_pred = parse_yes_no(chosen_raw)

        # Judge rejected
        rejected_prompt = SINGLE_PROMPT.format(prompt=prompt, response=rejected)
        rejected_raw = generate_response(model, tokenizer, rejected_prompt, model_name)
        rejected_pred = parse_yes_no(rejected_raw)

        result = {
            "index": idx,
            "prompt": prompt,
            "chosen_raw_output": chosen_raw,
            "chosen_pred": chosen_pred,
            "chosen_correct": chosen_pred == "Yes",
            "rejected_raw_output": rejected_raw,
            "rejected_pred": rejected_pred,
            "rejected_correct": rejected_pred == "No",
        }
        results.append(result)

        # Save checkpoint every 10 items
        if (len(results)) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute accuracy
    total = len(results)
    chosen_correct = sum(1 for r in results if r["chosen_correct"])
    rejected_correct = sum(1 for r in results if r["rejected_correct"])
    # Overall: both chosen=Yes AND rejected=No counts as one correct pair
    pair_correct = sum(1 for r in results if r["chosen_correct"] and r["rejected_correct"])

    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Mode: single")
    print(f"Total pairs: {total}")
    print(f"Chosen accuracy (pred=Yes): {chosen_correct}/{total} = {chosen_correct/total:.4f}")
    print(f"Rejected accuracy (pred=No): {rejected_correct}/{total} = {rejected_correct/total:.4f}")
    print(f"Overall accuracy (per-response): {(chosen_correct + rejected_correct)}/{total * 2} = {(chosen_correct + rejected_correct) / (total * 2):.4f}")
    print(f"Pair accuracy (both correct): {pair_correct}/{total} = {pair_correct/total:.4f}")
    print(f"{'='*50}\n")

    return results


def run_pairwise(model, tokenizer, data, model_name, output_path):
    """Pairwise: present both responses, ask which is better."""
    results = []
    done_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        for r in results:
            done_indices.add(r["index"])
        print(f"Resuming from {len(results)} completed entries")

    random.seed(42)

    for idx, item in enumerate(tqdm(data, desc=f"{model_name} pairwise")):
        if idx in done_indices:
            continue

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Randomly assign chosen/rejected to A/B to avoid position bias
        if random.random() < 0.5:
            response_a, response_b = chosen, rejected
            chosen_label = "A"
        else:
            response_a, response_b = rejected, chosen
            chosen_label = "B"

        pair_prompt = PAIRWISE_PROMPT.format(
            prompt=prompt, response_a=response_a, response_b=response_b
        )
        raw_output = generate_response(model, tokenizer, pair_prompt, model_name)
        pred = parse_ab(raw_output)
        correct = pred == chosen_label

        result = {
            "index": idx,
            "prompt": prompt,
            "chosen_position": chosen_label,
            "raw_output": raw_output,
            "pred": pred,
            "correct": correct,
        }
        results.append(result)

        if (len(results)) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute accuracy
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    unknown = sum(1 for r in results if r["pred"] == "Unknown")

    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Mode: pairwise")
    print(f"Total pairs: {total}")
    print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"Unknown/unparseable: {unknown}/{total}")
    print(f"{'='*50}\n")

    return results


def main():
    # Load data
    print(f"Loading data from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    for model_name in ["llama", "qwen"]:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")
        model, tokenizer = load_model(model_name)

        for mode in ["single", "pairwise"]:
            output_filename = OUTPUT_MAP[(model_name, mode)]
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            print(f"\n>>> Running {model_name} - {mode} -> {output_path}")

            if mode == "single":
                run_single(model, tokenizer, data, model_name, output_path)
            else:
                run_pairwise(model, tokenizer, data, model_name, output_path)

        # Free GPU memory before loading next model
        del model, tokenizer
        torch.cuda.empty_cache()

    print("\nAll done!")


if __name__ == "__main__":
    main()