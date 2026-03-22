############################################
# IMPORTS
############################################

import random
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM


############################################
# CONFIG
############################################

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
TOT_BRANCHES = 3
REFLEXION_STEPS = 2
MCTS_SIMULATIONS = 5

TEMPERATURE = 0.7

MODELS = {
    "gemma": "google/gemma-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}


############################################
# MODEL LOADER
############################################

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


############################################
# PROMPT BUILDER
############################################

def build_prompt(question):

    return f"""
You are a careful reasoning system.

Solve the problem step by step.

Question:
{question}

Reasoning:
"""


############################################
# MODEL CALL
############################################

def run_model_batch(prompts, tokenizer, model):

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    return texts, outputs.scores


############################################
# TOKEN ENTROPY
############################################

def token_entropy(scores):

    entropies = []

    for step_scores in scores:

        probs = torch.softmax(step_scores, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        entropies.append(entropy.mean().item())

    return float(np.mean(entropies))


############################################
# ANSWER EXTRACTION
############################################

def extract_answer(text):

    nums = re.findall(r'-?\d+', text)

    if nums:
        return nums[-1]

    t = text.lower()

    if "yes" in t: return "yes"
    if "no" in t: return "no"

    words = re.findall(r'[a-z]+', t)

    if words:
        return words[-1]

    return t.strip()


############################################
# JUDGE
############################################

def judge_answer(pred, gold):

    if pred == gold:
        return 1

    if gold in pred:
        return 1

    return 0


############################################
# SELF CONSISTENCY
############################################

def self_consistency(prompt, tokenizer, model):

    answers = []
    entropies = []

    for _ in range(SELF_CONSISTENCY_SAMPLES):

        out, scores = run_model_batch([prompt], tokenizer, model)

        ans = extract_answer(out[0])

        answers.append(ans)

        entropies.append(token_entropy(scores))

    counts = Counter(answers)

    best = counts.most_common(1)[0][0]

    return best, answers, float(np.mean(entropies))


############################################
# TREE OF THOUGHT
############################################

def tree_of_thought(prompt, tokenizer, model):

    candidates = []

    for _ in range(TOT_BRANCHES):

        out, _ = run_model_batch([prompt], tokenizer, model)

        candidates.append(out[0])

    answers = [extract_answer(c) for c in candidates]

    counts = Counter(answers)

    return counts.most_common(1)[0][0]


############################################
# REFLEXION
############################################

def reflexion(prompt, tokenizer, model):

    reasoning = prompt

    for _ in range(REFLEXION_STEPS):

        out, _ = run_model_batch([reasoning], tokenizer, model)

        text = out[0]

        critique_prompt = f"""
Analyze the reasoning below and fix any mistakes.

{text}

Correct reasoning:
"""

        corrected, _ = run_model_batch([critique_prompt], tokenizer, model)

        reasoning = corrected[0]

    return extract_answer(reasoning)


############################################
# MCTS REASONING
############################################

def mcts_reasoning(prompt, tokenizer, model):

    candidates = []

    for _ in range(MCTS_SIMULATIONS):

        out, _ = run_model_batch([prompt], tokenizer, model)

        candidates.append(out[0])

    answers = [extract_answer(c) for c in candidates]

    return Counter(answers).most_common(1)[0][0]


############################################
# CONFIDENCE ESTIMATION
############################################

def confidence_score(samples):

    counts = Counter(samples)

    best = counts.most_common(1)[0][1]

    return best / len(samples)


############################################
# TASK GENERATORS
############################################

def arithmetic_task():

    d = random.randint(1,5)

    a = random.randint(1,10**d)
    b = random.randint(1,10**d)

    q = f"What is {a}+{b}?"

    return q,str(a+b),"arithmetic",d


def pattern_task():

    start=random.randint(1,10)
    step=random.randint(2,6)

    seq=[start+i*step for i in range(4)]

    q=f"Find pattern: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?"

    return q,str(seq[3]+step),"pattern",2


def logic_task():

    q="All birds have wings. Penguins are birds. Do penguins have wings?"

    return q,"yes","logic",2


def planning_task():

    start=random.randint(1,10)

    goal=start+10

    q=f"You start at {start}. Each move adds 2. How many moves reach {goal}?"

    return q,str((goal-start)//2),"planning",3


############################################
# OOD TASK
############################################

def ood_arithmetic_task():

    a=random.randint(1000,5000)
    b=random.randint(1000,5000)

    q=f"What is {a}+{b}?"

    return q,str(a+b),"ood_arithmetic",5


############################################
# TASK REGISTRY
############################################

TASKS=[
arithmetic_task,
pattern_task,
logic_task,
planning_task
]

OOD_TASK_SET=[
ood_arithmetic_task
]


############################################
# TASK GENERATION
############################################

def generate_tasks(n):

    tasks=[]

    for _ in range(n):

        fn=random.choice(TASKS)

        q,a,skill,d=fn()

        tasks.append({

        "question":q,
        "answer":a,
        "skill":skill,
        "difficulty":d,
        "distribution":"in_distribution"

        })

    return tasks


def generate_ood_tasks(n):

    tasks=[]

    for _ in range(n):

        fn=random.choice(OOD_TASK_SET)

        q,a,skill,d=fn()

        tasks.append({

        "question":q,
        "answer":a,
        "skill":skill,
        "difficulty":d,
        "distribution":"ood"

        })

    return tasks


############################################
# BENCHMARK
############################################

def run_benchmark(tasks):

    results=[]

    for mname,mpath in MODELS.items():

        print("\nLoading",mname)

        tok,model=load_model(mpath)

        for t in tqdm(tasks):

            prompt=build_prompt(t["question"])

            pred,samples,token_ent=self_consistency(prompt,tok,model)

            tot_pred=tree_of_thought(prompt,tok,model)

            reflex_pred=reflexion(prompt,tok,model)

            mcts_pred=mcts_reasoning(prompt,tok,model)

            conf=confidence_score(samples)

            score=judge_answer(pred,t["answer"])

            results.append({

            "model":mname,
            "skill":t["skill"],
            "difficulty":t["difficulty"],
            "distribution":t["distribution"],
            "prediction":pred,
            "tot_prediction":tot_pred,
            "reflexion_prediction":reflex_pred,
            "mcts_prediction":mcts_pred,
            "confidence":conf,
            "gold":t["answer"],
            "score":score,
            "token_entropy":token_ent

            })

    return results


############################################
# ANALYSIS
############################################

def analyze(results):

    df=pd.DataFrame(results)

    print("\nSkill scores\n",df.groupby(["model","skill"])["score"].mean())

    print("\nDifficulty scaling\n",df.groupby(["model","difficulty"])["score"].mean())

    print("\nOOD generalization\n",df[df["distribution"]=="ood"].groupby("model")["score"].mean())

    print("\nConfidence\n",df.groupby("model")["confidence"].mean())

    return df


############################################
# VISUALIZATION
############################################

def plot(df):

    skill=df.groupby(["model","skill"])["score"].mean().unstack()

    ax=skill.plot(kind="bar")

    ax.set_title("Cognitive Skill Accuracy")

    plt.savefig("skill_accuracy.png")

    plt.close()

    diff=df.groupby(["model","difficulty"])["score"].mean().unstack()

    ax=diff.plot()

    ax.set_title("Difficulty Scaling")

    plt.savefig("difficulty_curve.png")

    plt.close()


############################################
# MAIN PIPELINE
############################################

def main():

    print("Generating tasks")

    tasks=generate_tasks(NUM_TASKS)

    ood=generate_ood_tasks(OOD_TASKS)

    dataset=tasks+ood

    with open("tasks.json","w") as f:

        json.dump(dataset,f,indent=2)

    print("Running benchmark")

    results=run_benchmark(dataset)

    with open("results.json","w") as f:

        json.dump(results,f,indent=2)

    df=analyze(results)

    plot(df)

    print("\nBenchmark completed")


############################################

if __name__=="__main__":

    main()
