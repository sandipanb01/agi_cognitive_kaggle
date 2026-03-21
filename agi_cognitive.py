import random
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################
# CONFIG
############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TASKS = 10000
MAX_NEW_TOKENS = 150

MODELS = {
    "gemma": "google/gemma-1b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
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

    return tokenizer, model


############################################
# YOUR GEMMA FUNCTION
############################################

def gemma_model(prompt, tokenizer, model):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "answer": text,
        "confidence": 0.5
    }


############################################
# GENERIC MODEL CALL
############################################

def run_model(prompt, tokenizer, model):

    result = gemma_model(prompt, tokenizer, model)

    return result["answer"]


############################################
# TASK GENERATORS
############################################

def arithmetic_task():

    d = random.randint(1,5)

    a = random.randint(1,10**d)
    b = random.randint(1,10**d)

    q = f"What is {a}+{b}? Show reasoning then final answer."

    return q, str(a+b), "arithmetic", d


def logic_task():

    q = "All birds have wings. Penguins are birds. Do penguins have wings? yes or no."

    return q, "yes", "logic", 2


def pattern_task():

    start = random.randint(1,10)
    step = random.randint(2,6)

    seq = [start+i*step for i in range(4)]

    q = f"Find pattern: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?"

    return q, str(seq[3]+step), "pattern", 2


def meta_learning_task():

    m = random.randint(2,5)

    prompt="Learn rule:\n"

    for i in range(3):

        x=random.randint(1,5)

        prompt+=f"{x}->{x*m}\n"

    t=random.randint(6,10)

    prompt+=f"\nWhat does {t} map to?"

    return prompt,str(t*m),"meta_learning",3


def planning_task():

    start=random.randint(1,10)

    goal=start+10

    q=f"You start at {start}. Each move adds 2. How many moves reach {goal}?"

    return q,str((goal-start)//2),"planning",3


def grid_task():

    grid=[[1,0,1],[0,1,0],[1,0,1]]

    q=f"Grid {grid}. Count ones."

    return q,"5","grid_reasoning",4


def analogy_task():

    q="Dog is to puppy as cat is to ?"

    return q,"kitten","analogy",2


def commonsense_task():

    q="If you drop a glass on the floor what happens?"

    return q,"break","commonsense",1


def memory_task():

    q="Remember: apple banana orange. Which fruit was second?"

    return q,"banana","memory",2


def causal_task():

    q="If rain increases water levels what happens after heavy rain?"

    return q,"flood","causal_reasoning",2


TASKS=[
arithmetic_task,
logic_task,
pattern_task,
meta_learning_task,
planning_task,
grid_task,
analogy_task,
commonsense_task,
memory_task,
causal_task
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
        "difficulty":d
        })

    return tasks


############################################
# ANSWER EXTRACTION
############################################

def extract_answer(text):

    nums=re.findall(r'\d+',text)

    if nums:
        return nums[-1]

    t=text.lower()

    if "yes" in t:return "yes"
    if "no" in t:return "no"

    words=re.findall(r'[a-z]+',t)

    if words:
        return words[-1]

    return t.strip()


############################################
# SCORING
############################################

def score(pred,gold):

    return int(pred==gold)


############################################
# LLM AS JUDGE
############################################

def judge_answer(pred,gold):

    if pred==gold:
        return 1

    if gold in pred:
        return 1

    return 0


############################################
# BENCHMARK RUN
############################################

def run_benchmark(tasks):

    results=[]

    for mname,mpath in MODELS.items():

        print("\nLoading",mname)

        tok,model=load_model(mpath)

        for i,t in enumerate(tasks):

            raw=run_model(t["question"],tok,model)

            pred=extract_answer(raw)

            s=judge_answer(pred,t["answer"])

            results.append({

            "model":mname,
            "skill":t["skill"],
            "difficulty":t["difficulty"],
            "prediction":pred,
            "gold":t["answer"],
            "score":s
            })

            if i%200==0:
                print(mname,"processed",i)

    return results


############################################
# ANALYSIS
############################################

def analyze(res):

    df=pd.DataFrame(res)

    skill=df.groupby(["model","skill"])["score"].mean()

    diff=df.groupby(["model","difficulty"])["score"].mean()

    print("\nSkill scores\n",skill)

    print("\nDifficulty curve\n",diff)

    return df


############################################
# PHASE TRANSITION
############################################

def phase_transition(df):

    curve=df.groupby("difficulty")["score"].mean()

    g=np.gradient(curve.values)

    pt=np.argmax(np.abs(g))

    print("\nReasoning phase transition near difficulty:",pt+1)


############################################
# CALIBRATION
############################################

def calibration(df):

    conf=0.5

    err=abs(conf-df["score"].mean())

    print("\nCalibration error:",err)


############################################
# VISUALIZATION
############################################

def plot(df):

    plt.figure()

    skill=df.groupby(["model","skill"])["score"].mean().unstack()

    skill.plot(kind="bar")

    plt.title("Cognitive Skill Accuracy")

    plt.ylabel("accuracy")

    plt.tight_layout()

    plt.savefig("skill_accuracy.png")


    plt.figure()

    diff=df.groupby(["model","difficulty"])["score"].mean().unstack()

    diff.plot()

    plt.title("Difficulty Scaling Curve")

    plt.ylabel("accuracy")

    plt.tight_layout()

    plt.savefig("difficulty_curve.png")


############################################
# MAIN PIPELINE
############################################

def main():

    print("Generating tasks")

    tasks=generate_tasks(NUM_TASKS)

    with open("tasks.json","w") as f:
        json.dump(tasks,f,indent=2)

    print("Running benchmark")

    results=run_benchmark(tasks)

    with open("results.json","w") as f:
        json.dump(results,f,indent=2)

    df=analyze(results)

    phase_transition(df)

    calibration(df)

    plot(df)

    print("\nBenchmark completed")


if __name__=="__main__":
    main()
