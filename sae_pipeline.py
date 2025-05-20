import json
import os
import datasets
import random
from sae_lens import SAE
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import re
from tqdm import tqdm
from itertools import cycle
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import argparse


# Load role prompts from file
def load_role_prompts(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Load few-shot examples from JSON
def load_eval_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Format CSQA questions to include labeled answer choices
def format_csqa_question(example):
    question = example["question"]
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    labeled_choices = [f"({label.lower()}) {text}" for label, text in zip(labels, choices)]
    return f"{question} Answer Choices: {' '.join(labeled_choices)}"

## Fix spacing before punctuation and apostrophes
def fix_punctuation_spacing(text):
    text = re.sub(r'\s+([,.!?%])', r'\1', text) # Remove space before punctuation
    text = re.sub(r"\s+'\s*", "'", text)  # Remove space before apostrophe
    return text

# Load and preprocess dataset (gsm8k, csqa, or svamp)
def load_and_process_dataset(name):
    if name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        train_questions = [item["question"] for item in dataset["train"]]
        test_questions = [item["question"] for item in dataset["test"]]
        # Extract numeric answers from answer strings using regex
        true_answers = [re.search(r"####\s*([\d,.\-]+)", item["answer"]).group(1).replace(",", "") if re.search(r"####\s*([\d,.\-]+)", item["answer"]) else "N/A"
        for item in dataset["test"]]
    elif name == "csqa":
        dataset = load_dataset("tau/commonsense_qa")
        train_questions = [format_csqa_question(example) for example in dataset["train"]]
        test_questions = [format_csqa_question(example) for example in dataset["validation"]]
        true_answers = [item.lower() for item in dataset["validation"]["answerKey"]]
    elif name == "svamp":
        df = pd.read_csv("data/train.csv")
        train_questions = []
        for idx, row in df.iterrows():
            question = row['Question']
            numbers = str(row['Numbers']).split()
            for i, num in enumerate(numbers):
                try:
                    num_float = float(num)
                    clean_num = str(int(num_float)) if num_float.is_integer() else str(num_float)
                except:
                    clean_num = num
                placeholder = f"number{i}"
                question = question.replace(placeholder, clean_num)
            question = re.sub(r'\s+', ' ', question).strip()
            train_questions.append(fix_punctuation_spacing(question))

        with open("data/SVAMP.json", "r") as f:
            raw_test_data = json.load(f)

        test_questions = []
        true_answers = []
        for item in raw_test_data:
            body = item["Body"].strip()
            question = item["Question"].strip()
            last_sentence = body.split(".")[-1].strip()
            answer = str(int(item["Answer"]))

            if last_sentence.lower().startswith("if"):
                new_question = body + ", " + question.lower()
            else:
                new_question = body + " " + question
            test_questions.append(new_question)
            true_answers.append(answer)

    dataset = {"train": train_questions, "test": test_questions}
    
    return dataset, train_questions, test_questions, true_answers

# Generate prompts with and without role-playing
def prepare_prompts(train_questions, role_prompts, N, seed):
    random.seed(seed)
    random.shuffle(role_prompts)
    role_prompt_cycle = cycle(role_prompts)
    selected_questions = [random.choice(train_questions) for _ in range(N)]
    with_role_prompts = [f"{next(role_prompt_cycle)} {q}" for q in selected_questions]
    without_role_prompts = selected_questions
    return with_role_prompts, without_role_prompts

# Steering vector computation
def compute_steering_shift(model, sae, with_prompts, without_prompts, stopword_set, punctuation_set, hook_point, k, beta, threshold, scaling):
    with_acts, without_acts = [], []

    for i in tqdm(range(len(with_prompts)), desc="Extracting Activations"):
        with torch.no_grad():
            w_prompt = with_prompts[i]
            wo_prompt = without_prompts[i]

            # Run model and cache activations
            w_logits, w_cache = model.run_with_cache(w_prompt, prepend_bos=True)
            wo_logits, wo_cache = model.run_with_cache(wo_prompt, prepend_bos=True)

            # Encode activations using the SAE encoder
            w_acts = sae.encode(w_cache[hook_point].to(sae.cfg.device))
            wo_acts = sae.encode(wo_cache[hook_point].to(sae.cfg.device))

            # Tokenize input and remove stopwords and punctuation
            w_tokens = model.tokenizer.convert_ids_to_tokens(model.to_tokens(w_prompt)[0])
            wo_tokens = model.tokenizer.convert_ids_to_tokens(model.to_tokens(wo_prompt)[0])

            ## Create attention masks for meaningful tokens
            w_mask = torch.tensor([
                (t != '<|begin_of_text|>' and t.lstrip('Ġ').lower() not in stopword_set and t not in punctuation_set)
                for t in w_tokens
            ], dtype=torch.bool, device=w_acts.device)

            wo_mask = torch.tensor([
                (t != '<|begin_of_text|>' and t.lstrip('Ġ').lower() not in stopword_set and t not in punctuation_set)
                for t in wo_tokens
            ], dtype=torch.bool, device=wo_acts.device)

            # Mean activation over valid tokens
            with_acts.append(w_acts[0][w_mask].mean(dim=0))
            without_acts.append(wo_acts[0][wo_mask].mean(dim=0))

            # Free memory
            del w_cache, wo_cache, w_acts, wo_acts, w_logits, wo_logits
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Stack all activations and compute activation strength difference
    with_tensor = torch.stack(with_acts)
    without_tensor = torch.stack(without_acts)
    delta = with_tensor - without_tensor
    mean_delta = delta.mean(dim=0)

    # Compute activation frequency difference
    active_w = (with_tensor > threshold).float().sum(dim=0) / len(with_tensor)
    active_wo = (without_tensor > threshold).float().sum(dim=0) / len(without_tensor)
    activation_diff = active_w - active_wo

    # Compute sensitivity score and select top-k most sensitive features
    sensitivity_score = mean_delta + beta * activation_diff
    top_k_idx = torch.topk(sensitivity_score, k=k).indices
    top_k_features = sensitivity_score[top_k_idx]

    print(f"Top-{k} Role-Play Influenced Features (Indices & Sensitivity Scores):")
    for idx, score in zip(top_k_idx.tolist(), top_k_features.tolist()):
        print(f"Feature {idx}: Sensitivity {score:.4f}")

    # Compute steering vector as weighted sum of decoder vectors
    top_k_acts = with_tensor[:, top_k_idx]
    strengths = top_k_acts.mean(dim=0) * scaling
    steering_vecs = sae.W_dec[top_k_idx]
    steering_shift = (strengths[:, None] * steering_vecs).sum(dim=0)

    return steering_shift

# Steering hook injection
def steering_hook_factory(steering_shift_tensor, steering_on_flag):
    def steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1 or not steering_on_flag:
            return
        position = resid_pre.shape[1] - 1 # Inject only at the final token
        orig_norm = torch.norm(resid_pre[:, position, :], p=2, dim=-1, keepdim=True)
        resid_pre[:, position, :] += steering_shift_tensor.to(resid_pre.device)
        new_norm = torch.norm(resid_pre[:, position, :], p=2, dim=-1, keepdim=True)
        resid_pre[:, position, :] *= (orig_norm / new_norm) # Normalize back to original L2 norm
    return steering_hook

# Few-shot examples selection
def extract_k_shot_examples(dataset, k, seed=42):
    random.seed(seed)
    # Sort questions by length and sample evenly across the range
    question_lengths = [len(item["question"]) for item in dataset]
    pairs = list(zip(dataset, question_lengths))
    pairs.sort(key=lambda x: x[1])
    step = len(pairs) // k
    sampled_items = [pairs[i * step][0] for i in range(k)]
    random.shuffle(sampled_items)
    examples = []
    for item in sampled_items:
        formatted_example = (
            f"Q: {item['question']}\nA: Let's think step by step.\n"
            f"{item['answer'].strip()}\nOutput: {item['output']}"
        )
        examples.append(formatted_example)
    return examples

# Prompt generation
def generate_prompt(k_shot_examples, question):
    return "\n\n".join(k_shot_examples) + f"\n\nQ: {question}\nA: Let's think step by step."

# Model generation with optional hook
def run_generate(model, prompt, layer, hook_fn=None):
    model.reset_hooks()
    hooks = [(f"blocks.{layer}.hook_resid_post", hook_fn)] if hook_fn else []
    with model.hooks(fwd_hooks=hooks):
        tokenized = model.to_tokens([prompt])
        output = model.generate(tokenized, max_new_tokens=150, do_sample=False)
    return model.to_string(output[:, 1:])[0]


# Answer extraction helpers
def extract_answer_numeric(model_response):
    patterns = [
        r"Output:\s*\$?(-?\d[\d,\.]*)", # Capture numeric answers
        r"The answer is\s*\$?(-?\d[\d,\.]*)",
        r"Therefore,?\s*the answer is\s*\$?(-?\d[\d,\.]*)",
        r"final answer is\s*\$?(-?\d[\d,\.]*)",
        r"result is\s*\$?(-?\d[\d,\.]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, model_response)
        if match:
            answer = match.group(1).replace(",", "").rstrip(".")
            if "." in answer and float(answer).is_integer():
                answer = str(int(float(answer)))
            return answer
    return None


def extract_answer_multiple_choice(model_response):
    patterns = [
        r"Output:\s*\(([a-e])\)", # Match (a), (b), etc.          
        r"The answer is\s*\(([a-e])\)",         
        r"Final answer:?\s*\(([a-e])\)",        
        r"Answer:\s*\(([a-e])\)",               
        r"Correct Answer:?\s*\(([a-e])\)",      
    ]
    for pattern in patterns:
        match = re.search(pattern, model_response, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


# Evaluation
def evaluate_model(model, dataset, test_questions, true_answers, steering_shift, k_shot_examples, use_steering, layer):
    correct= 0
    hook_fn = steering_hook_factory(steering_shift, steering_on_flag=use_steering) if use_steering else None

    for i in tqdm(range(len(test_questions)), desc="Generating Responses"):
        question = test_questions[i]
        prompt = generate_prompt(k_shot_examples, question)
        output = run_generate(model, prompt, layer, hook_fn)
        response = output[len(prompt):].strip()

        # Extract prediction depending on dataset type
        if args.dataset_name in ["gsm8k", "svamp"]:
            pred = extract_answer_numeric(response)
        else:
            pred = extract_answer_multiple_choice(response)
        if pred == true_answers[i]:
            correct += 1
        
        print(f"Question {i+1}: {question}")  
        print(f"Response {i+1}: {response}")  
        print(f"Model Answer: {pred} | True Answer: {true_answers[i]}\n")

        # Free memory
        del output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Calculate accuracy
    acc = correct / len(test_questions) * 100
    print(f"\nSteering On: {use_steering}")
    print(f"Accuracy: {acc:.2f}%")
    return acc

    

def main(args):
    nltk.download("stopwords")
    stopword_set = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)

    # Load model and Sparse Autoencoder
    model = HookedTransformer.from_pretrained(args.model_name, n_devices=1, device="cuda")
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=args.sae_release, sae_id=args.sae_id, device="cuda")
    hook_point = sae.cfg.hook_name

    # Load dataset, prompts and few-shot exemplars
    dataset, train_questions, test_questions, true_answers = load_and_process_dataset(args.dataset_name)
    role_prompts = load_role_prompts(args.role_path)
    eval_data = load_eval_data(args.eval_path)

    # Construct contrastive sample pairs and compute steering vector
    with_prompts, without_prompts = prepare_prompts(train_questions, role_prompts, args.N, args.seed)
    steering_shift = compute_steering_shift(
        model, sae, with_prompts, without_prompts,
        stopword_set, punctuation_set, hook_point,
        k=args.k, beta=args.beta, threshold=args.threshold, scaling=args.scaling
    )

    # Generate few-shot examples
    k_shot_examples = extract_k_shot_examples(eval_data, k=args.k_shot)

    # Evaluation
    evaluate_model(model, dataset, test_questions, true_answers, steering_shift, k_shot_examples, use_steering=True, layer=args.layer)
    evaluate_model(model, dataset, test_questions, true_answers, steering_shift, k_shot_examples, use_steering=False, layer=args.layer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-2-2b")
    parser.add_argument("--sae_release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae_id", default="layer_25/width_65k/canonical")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k", "csqa", "svamp"])
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--scaling", type=float, default=8.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", type=str, default="arithmetic", choices=["arithmetic", "commonsense"])
    parser.add_argument("--layer", type=int, default=25)
    args = parser.parse_args()
    args.role_path = f"data/role_prompts_{args.domain}.txt"
    args.eval_path = f"data/few_shot_prompts_{args.domain}.json"
    main(args)
