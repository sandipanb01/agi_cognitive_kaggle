#----------CONFIG-------------
import torch
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TASKS = 10000
OOD_TASKS = 2000

ADAPTIVE_STEPS = 200

MAX_NEW_TOKENS = 512
BATCH_SIZE = 8

SELF_CONSISTENCY_SAMPLES = 3

OUTPUT_DIR = "benchmark_outputs"

MODELS = {
    "gemma": "google/gemma-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}
#----------MODEL LOADER-------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import DEVICE

def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model.eval()

    return tokenizer, model
#----------------------CHAIN-OF-THOUGHT PROMPTING----------------
def build_prompt(question):

    return f"""
You are a careful reasoning system.

Solve the problem step by step.

Question:
{question}

Reasoning:
"""
#-------------INFERENCE-------------------------
import torch
from collections import Counter
from config import MAX_NEW_TOKENS, SELF_CONSISTENCY_SAMPLES


def run_model(prompt, tokenizer, model):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text


def self_consistency(prompt, tokenizer, model, extractor):

    answers = []

    for _ in range(SELF_CONSISTENCY_SAMPLES):

        out = run_model(prompt, tokenizer, model)

        ans = extractor(out)

        answers.append(ans)

    return Counter(answers).most_common(1)[0][0]

#-------------SCORING-------------------------------
import re


def extract_answer(text):

    nums = re.findall(r'-?\d+', text)

    if nums:
        return nums[-1]

    t = text.lower()

    if "yes" in t:
        return "yes"

    if "no" in t:
        return "no"

    words = re.findall(r'[a-z]+', t)

    if words:
        return words[-1]

    return t.strip()


def judge(pred, gold):

    if pred == gold:
        return 1

    if gold in pred:
        return 1

    return 0

#------------------- TASKS -----------------------
import random


def arithmetic_task():

    d = random.randint(1,5)

    a = random.randint(1,10**d)
    b = random.randint(1,10**d)

    return f"What is {a}+{b}?", str(a+b), "arithmetic", d


def logic_task():

    q = "All birds have wings. Penguins are birds. Do penguins have wings?"

    return q, "yes", "logic", 2


def rule_task():

    m = random.randint(2,5)

    f = lambda x: x*m

    prompt = "Discover rule:\n"

    for x in range(1,4):
        prompt += f"{x}->{f(x)}\n"

    t = random.randint(5,8)

    prompt += f"\nWhat does {t} map to?"

    return prompt, str(f(t)), "rule_discovery", 4


TASKS = [
    arithmetic_task,
    logic_task,
    rule_task
]


def generate_tasks(n):

    tasks = []

    for _ in range(n):

        fn = random.choice(TASKS)

        q,a,s,d = fn()

        tasks.append({

            "question": q,
            "answer": a,
            "skill": s,
            "difficulty": d,
            "distribution": "in_distribution"
        })

    return tasks
 #----------------COLLAPSE -----------------------------

 def detect_model_collapse(df):

    collapse = {}

    for m in df["model"].unique():

        sub = df[df["model"]==m]

        curve = sub.groupby("difficulty")["score"].mean()

        drops = []

        for i in range(1,len(curve)):

            drop = curve.iloc[i-1] - curve.iloc[i]

            if drop > 0.3:
                drops.append(drop)

        collapse[m] = sum(drops)

    return collapse

#-----------------REASONING ENTROPY -------------------------

import numpy as np
from collections import Counter


def reasoning_entropy(answers):

    counts = Counter(answers)

    probs = np.array(list(counts.values())) / sum(counts.values())

    entropy = -np.sum(probs * np.log(probs))

    return entropy

#------------------------ ADAPTIVE REASONING --------------------------

import random


def generate_task_at_difficulty(d):

    if d <= 2:
        return "What is 12+7?", "19"

    if d == 3:
        return "If x+3=10 what is x?", "7"

    return "Sequence: 2,4,8,16, ?", "32"

#-------------------------------- PLOTS------------------------------------

import matplotlib.pyplot as plt


def plot_difficulty(df):

    diff = df.groupby(["model","difficulty"])["score"].mean().unstack()

    diff.plot()

    plt.title("Difficulty Scaling")

    plt.savefig("difficulty_curve.png")

    plt.close()

#--------------------FULL PIPELINE---------------------------

import json
import pandas as pd
from tqdm import tqdm

from config import MODELS, NUM_TASKS
from models import load_model
from tasks import generate_tasks
from prompting import build_prompt
from inference import self_consistency
from scoring import extract_answer, judge
from collapse import detect_model_collapse
from plots import plot_difficulty


def main():

    tasks = generate_tasks(NUM_TASKS)

    results = []

    for name, path in MODELS.items():

        tok, model = load_model(path)

        for task in tqdm(tasks):

            prompt = build_prompt(task["question"])

            pred = self_consistency(prompt, tok, model, extract_answer)

            score = judge(pred, task["answer"])

            results.append({

                "model": name,
                "difficulty": task["difficulty"],
                "skill": task["skill"],
                "score": score
            })

    df = pd.DataFrame(results)

    print(df.groupby(["model","skill"])["score"].mean())

    collapse = detect_model_collapse(df)

    print("Model collapse:", collapse)

    plot_difficulty(df)

    with open("results.json","w") as f:
        json.dump(results,f,indent=2)


if __name__ == "__main__":
    main()




