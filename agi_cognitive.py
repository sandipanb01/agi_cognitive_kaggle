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

from transformers import AutoTokenizer, AutoModelForCausalLM


############################################
# CONFIG
############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TASKS = 10000
OOD_TASKS = 2000
ADAPTIVE_STEPS = 200

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 8

MODELS = {

    "gemma": "google/gemma-2b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"

}


############################################
# MODEL LOADER
############################################

def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    )

    model.eval()

    return tokenizer, model


############################################
# BATCHED MODEL CALL
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
            max_new_tokens=MAX_NEW_TOKENS
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return texts


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
# SCORING
############################################

def judge_answer(pred, gold):

    if pred == gold:
        return 1

    if gold in pred:
        return 1

    return 0


############################################
# TASK GENERATORS
############################################

def arithmetic_task():

    d=random.randint(1,5)

    a=random.randint(1,10**d)
    b=random.randint(1,10**d)

    q=f"What is {a}+{b}?"

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
# ARC-STYLE RULE DISCOVERY TASK
############################################

def rule_discovery_task():

    rule=random.choice(["linear","multiply","square"])

    if rule=="linear":

        a=random.randint(2,4)
        b=random.randint(1,5)

        f=lambda x:a*x+b

    elif rule=="multiply":

        m=random.randint(2,5)

        f=lambda x:x*m

    else:

        f=lambda x:x*x

    prompt="Discover rule:\n"

    for _ in range(3):

        x=random.randint(1,5)

        prompt+=f"{x}->{f(x)}\n"

    t=random.randint(6,10)

    prompt+=f"\nWhat does {t} map to?"

    return prompt,str(f(t)),"rule_discovery",4


############################################
# OOD TASKS
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
planning_task,
rule_discovery_task

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
# BENCHMARK RUN
############################################

def run_benchmark(tasks):

    results=[]

    for mname,mpath in MODELS.items():

        print("\nLoading",mname)

        tok,model=load_model(mpath)

        for i in tqdm(range(0,len(tasks),BATCH_SIZE)):

            batch=tasks[i:i+BATCH_SIZE]

            prompts=[t["question"] for t in batch]

            outputs=run_model_batch(prompts,tok,model)

            for t,raw in zip(batch,outputs):

                pred=extract_answer(raw)

                s=judge_answer(pred,t["answer"])

                results.append({

                "model":mname,
                "skill":t["skill"],
                "difficulty":t["difficulty"],
                "distribution":t["distribution"],
                "prediction":pred,
                "gold":t["answer"],
                "score":s

                })

    return results


############################################
# MODEL COLLAPSE DETECTION
############################################

def detect_model_collapse(df):

    collapse_scores={}

    for model in df["model"].unique():

        sub=df[df["model"]==model]

        curve=sub.groupby("difficulty")["score"].mean()

        drops=[]

        for i in range(1,len(curve)):

            drop=curve.iloc[i-1]-curve.iloc[i]

            if drop>0.3:

                drops.append(drop)

        collapse_scores[model]=sum(drops)

    return collapse_scores


############################################
# ADAPTIVE BENCHMARK
############################################

def generate_task_at_difficulty(d):

    if d<=1:
        fn=logic_task

    elif d==2:
        fn=pattern_task

    elif d==3:
        fn=planning_task

    else:
        fn=rule_discovery_task

    q,a,skill,_=fn()

    return {"question":q,"answer":a,"skill":skill,"difficulty":d}


def run_adaptive_benchmark(tokenizer,model):

    difficulty=1

    history=[]

    for step in range(ADAPTIVE_STEPS):

        task=generate_task_at_difficulty(difficulty)

        output=run_model_batch([task["question"]],tokenizer,model)[0]

        pred=extract_answer(output)

        correct=judge_answer(pred,task["answer"])

        history.append({

        "step":step,
        "difficulty":difficulty,
        "correct":correct

        })

        if correct:
            difficulty=min(5,difficulty+1)

        else:
            difficulty=max(1,difficulty-1)

    return history


############################################
# ANALYSIS
############################################

def analyze(results):

    df=pd.DataFrame(results)

    skill=df.groupby(["model","skill"])["score"].mean()

    diff=df.groupby(["model","difficulty"])["score"].mean()

    ood=df[df["distribution"]=="ood"].groupby("model")["score"].mean()

    print("\nSkill scores\n",skill)

    print("\nDifficulty scaling\n",diff)

    print("\nOOD generalization\n",ood)

    collapse=detect_model_collapse(df)

    print("\nModel collapse metric\n",collapse)

    return df


############################################
# VISUALIZATION
############################################

def plot(df):

    plt.figure()

    skill=df.groupby(["model","skill"])["score"].mean().unstack()

    skill.plot(kind="bar")

    plt.title("Cognitive Skill Accuracy")

    plt.savefig("skill_accuracy.png")

    plt.figure()

    diff=df.groupby(["model","difficulty"])["score"].mean().unstack()

    diff.plot()

    plt.title("Difficulty Scaling")

    plt.savefig("difficulty_curve.png")


############################################
# REASONING FRONTIER
############################################

def plot_frontier(histories):

    plt.figure()

    for model,hist in histories.items():

        diffs=[h["difficulty"] for h in hist]

        plt.plot(diffs,label=model)

    plt.legend()

    plt.title("Reasoning Frontier")

    plt.savefig("reasoning_frontier.png")


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

    print("Running adaptive benchmark")

    histories={}

    for mname,mpath in MODELS.items():

        tok,model=load_model(mpath)

        histories[mname]=run_adaptive_benchmark(tok,model)

    plot_frontier(histories)

    print("\nBenchmark completed")


############################################

if __name__=="__main__":

    main()
