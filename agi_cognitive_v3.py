############################################################
# MONOLITHIC FULLY INTEGRATED ARC SOLVER + LLM + JEPA + MuZero + DSL + COGNITIVE REASONING
# Author: SANDIPAN BHATTACHERJEE
############################################################

import os, random, json, pickle, hashlib, itertools, re, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from collections import Counter
from tqdm import tqdm
import heapq

from transformers import AutoTokenizer, AutoModelForCausalLM

############################################################
# GLOBAL CONFIG
############################################################

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 128
BEAM_SIZE = 120
SEARCH_DEPTH = 9
MUTATION_RATE = 0.3
SELF_CONSISTENCY_SAMPLES = 5
TOT_BRANCHES = 4
REFLEXION_STEPS = 2
MCTS_SIMULATIONS = 6
MAX_PROGRAMS = 200
TEMPERATURE = 0.7

MODEL_REGISTRY = {
    "gemma": "google/gemma-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

CACHE_DIR = "cache"
LLM_CACHE = os.path.join(CACHE_DIR, "llm_predictions")
PROGRAM_CACHE = os.path.join(CACHE_DIR, "programs")
os.makedirs(LLM_CACHE, exist_ok=True)
os.makedirs(PROGRAM_CACHE, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############################################################
# HELPER FUNCTIONS
############################################################

def extract_answer(text):
    nums = re.findall(r'-?\d+', text)
    if nums: return nums[-1]
    words = re.findall(r'[a-zA-Z]+', text)
    if words: return words[-1].lower()
    return text.strip()

def canonicalize(grid):
    grid = np.array(grid)
    bg = np.bincount(grid.flatten()).argmax()
    grid = grid - bg
    grid[grid<0] = 0
    return grid

def color_histogram(grid):
    hist = {}
    for v in grid.flatten(): hist[v] = hist.get(v,0)+1
    return hist

def same_shape(a,b): return a.shape==b.shape
def histogram_match(a,b): return color_histogram(a)==color_histogram(b)
def heuristic_score(pred,target):
    score=0
    if same_shape(pred,target): score+=1
    if histogram_match(pred,target): score+=1
    score+=np.sum(pred==target)/target.size
    return score

############################################################
# OBJECT-CENTRIC PERCEPTION
############################################################

def detect_objects(grid):
    grid = np.array(grid)
    objects=[]
    for c in np.unique(grid):
        mask = grid==c
        labeled,n = label(mask)
        for i in range(1,n+1):
            m = labeled==i
            coords=np.argwhere(m)
            objects.append({"color":c,"mask":m,"coords":coords})
    return objects

def object_bbox(obj):
    coords=obj["coords"]
    y=coords[:,0]; x=coords[:,1]
    return y.min(),y.max(),x.min(),x.max()

def translate_object(grid,obj,dy,dx):
    new_grid=grid.copy()
    new_grid[obj["mask"]]=0
    for y,x in obj["coords"]:
        ny,nx=int(y+dy),int(x+dx)
        if 0<=ny<grid.shape[0] and 0<=nx<grid.shape[1]:
            new_grid[ny,nx]=obj["color"]
    return new_grid

def recolor_object(grid,obj,new_color):
    new_grid=grid.copy()
    new_grid[obj["mask"]]=new_color
    return new_grid

############################################################
# DSL OPERATORS
############################################################

def rotate90(g): return np.rot90(g)
def rotate180(g): return np.rot90(g,2)
def flip_horizontal(g): return np.fliplr(g)
def flip_vertical(g): return np.flipud(g)
def mirror_object(g): return np.flipud(np.fliplr(g))
def crop_bbox(g):
    objs=detect_objects(g)
    if not objs: return g
    y1,y2,x1,x2=object_bbox(objs[0])
    return g[y1:y2+1,x1:x2+1]
def duplicate_horizontal(g): return np.concatenate([g,g],axis=1)
def duplicate_vertical(g): return np.concatenate([g,g],axis=0)
def tile_pattern(g): return np.tile(g,(2,2))
def fill_rectangle(g):
    g=g.copy()
    mask=g>0
    ys,xs=np.where(mask)
    if len(ys)==0: return g
    g[ys.min():ys.max()+1,xs.min():xs.max()+1]=g[ys[0],xs[0]]
    return g
def recolor_map(g):
    g=g.copy()
    colors=np.unique(g)
    mapping={c:i for i,c in enumerate(colors)}
    for k,v in mapping.items(): g[g==k]=v
    return g

DSL = [
    rotate90, rotate180, flip_horizontal, flip_vertical,
    mirror_object, crop_bbox, duplicate_horizontal,
    duplicate_vertical, tile_pattern, fill_rectangle, recolor_map
]

DSL_KEYWORDS = {
    "rotate90":["rotate 90","turn right","rotate clockwise"],
    "rotate180":["rotate 180","flip upside down"],
    "flip_horizontal":["flip horizontal","mirror left-right"],
    "flip_vertical":["flip vertical","mirror up-down"],
    "mirror_object":["mirror object","reflect object"],
    "crop_bbox":["crop","bounding box"],
    "duplicate_horizontal":["duplicate horizontal","repeat side"],
    "duplicate_vertical":["duplicate vertical","repeat up-down"],
    "tile_pattern":["tile","repeat grid"],
    "fill_rectangle":["fill rectangle","fill area"],
    "recolor_map":["recolor","map colors","recolor grid"]
}

############################################################
# LLM LOADER
############################################################

def load_llm(model_key="gemma"):
    name = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=dtype,device_map="auto")
    model.eval()
    return tokenizer,model

def predict_operations(prompt,model,tokenizer):
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad(): 
        outputs=model.generate(**inputs,max_new_tokens=60,temperature=TEMPERATURE)
    text=tokenizer.decode(outputs[0]).lower()
    ops=[]
    for op,keywords in DSL_KEYWORDS.items():
        if any(k in text for k in keywords): ops.append(op)
    return ops

def rank_dsl(predicted_ops,dsl_list=None):
    ranked=[]
    dsl=DSL if dsl_list is None else dsl_list
    for op in dsl:
        score=1
        for p in predicted_ops:
            if p in op.__name__: score+=5
        ranked.append((score,op))
    ranked.sort(reverse=True)
    return [o for _,o in ranked]

############################################################
# PROGRAM SEARCH
############################################################

class Program:
    def __init__(self,ops): self.ops=ops
    def run(self,grid):
        g=grid.copy()
        for op in self.ops: g=op(g)
        return g
    def mutate(self):
        if random.random()<MUTATION_RATE: self.ops.append(random.choice(DSL))
    def copy(self): return Program(self.ops.copy())

def score_program(program,train_pairs):
    score=0
    for inp,out in train_pairs:
        try:
            pred=program.run(inp)
            if pred.shape==out.shape: score+=np.sum(pred==out)/pred.size
        except: pass
    return score

def search_program(train_pairs,ranked_dsl,beam_size=BEAM_SIZE,search_depth=SEARCH_DEPTH,task_id=None):
    if task_id:
        cache_file=os.path.join(PROGRAM_CACHE,f"{task_id}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file,"rb") as f: return pickle.load(f)
    population=[Program([])]
    best=None
    best_score=-1
    for _ in range(search_depth):
        new=[]
        for prog in population:
            for op in ranked_dsl[:10]:
                p=prog.copy(); p.ops.append(op)
                s=score_program(p,train_pairs)
                if s>best_score: best_score=s; best=p
                new.append((s,p))
        new.sort(reverse=True,key=lambda x:x[0])
        population=[p for _,p in new[:beam_size]]
        for p in population: p.mutate()
    if task_id:
        with open(os.path.join(PROGRAM_CACHE,f"{task_id}.pkl"),"wb") as f: pickle.dump(best,f)
    return best

############################################################
# COGNITIVE REASONING LOOPS
############################################################

class SearchNode:
    def __init__(self,text): self.text=text; self.visits=0; self.value=0; self.children=[]

def run_model(prompt,tokenizer,model):
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,temperature=TEMPERATURE)
    return tokenizer.decode(outputs[0],skip_special_tokens=True)

def self_consistency(prompt,tokenizer,model):
    answers=[]
    for _ in range(SELF_CONSISTENCY_SAMPLES):
        out=run_model(prompt,tokenizer,model)
        answers.append(extract_answer(out))
    return Counter(answers).most_common(1)[0][0]

def tree_of_thought(prompt,tokenizer,model):
    candidates=[]
    for _ in range(TOT_BRANCHES):
        out=run_model(prompt,tokenizer,model)
        candidates.append(extract_answer(out))
    return Counter(candidates).most_common(1)[0][0]

def reflexion(prompt,tokenizer,model):
    reasoning=prompt
    for _ in range(REFLEXION_STEPS):
        out=run_model(reasoning,tokenizer,model)
        critique=f"Critique and fix errors:\n{out}"
        reasoning=run_model(critique,tokenizer,model)
    return extract_answer(reasoning)

def mcts_reasoning(prompt,tokenizer,model):
    root=SearchNode(prompt)
    for _ in range(MCTS_SIMULATIONS):
        out=run_model(prompt,tokenizer,model)
        root.children.append(SearchNode(out))
    answers=[extract_answer(c.text) for c in root.children]
    return Counter(answers).most_common(1)[0][0]

############################################################
# JEPA WORLD MODEL
############################################################

class JEPA(nn.Module):
    def __init__(self,input_size=16,latent_dim=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(input_size*input_size,input_size*latent_dim),nn.ReLU(),nn.Linear(input_size*latent_dim,latent_dim))
        self.predictor=nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.ReLU(),nn.Linear(latent_dim,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.ReLU(),nn.Linear(latent_dim,input_size*input_size))
        self.input_size=input_size
    def forward(self,x):
        z=self.encoder(x.float())
        z_pred=self.predictor(z)
        out=self.decoder(z_pred).reshape(-1,self.input_size,self.input_size)
        return out

############################################################
# MuZero PLANNER
############################################################

class MuZeroNode:
    def __init__(self,state,prior=1.0):
        self.state=state
        self.prior=prior
        self.value=0
        self.visits=0
        self.children={}

def muzero_plan(init_state,jepa_model,sim_steps=5):
    root=MuZeroNode(init_state)
    for _ in range(sim_steps):
        node=root
        path=[]
        while node.children:
            action,max_prior=max(node.children.items(),key=lambda x:x[1].prior)
            node=node.children[action]
            path.append(node)
        z=torch.tensor(node.state).unsqueeze(0).to(DEVICE)
        pred=jepa_model(z).detach().cpu().numpy()[0]
        node.children[0]=MuZeroNode(pred,prior=random.random())
    best_child=max(root.children.items(),key=lambda x:x[1].value)[1]
    return best_child.state

############################################################
# ARC SOLVER WRAPPER
############################################################

def solve_arc_task(task, model=None, tokenizer=None, beam_size=BEAM_SIZE, search_depth=SEARCH_DEPTH, dsl_list=None):
    train_pairs=[(canonicalize(np.array(x["input"])),canonicalize(np.array(x["output"]))) for x in task["train"]]
    prompt=str(train_pairs)
    task_id=hashlib.md5(str(task["train"]).encode()).hexdigest()
    if model and tokenizer:
        cache_file=os.path.join(LLM_CACHE,f"{hashlib.md5(prompt.encode()).hexdigest()}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file,"rb") as f: predicted_ops=pickle.load(f)
        else:
            predicted_ops=predict_operations(prompt,model,tokenizer)
            with open(cache_file,"wb") as f: pickle.dump(predicted_ops,f)
        ranked_dsl=rank_dsl(predicted_ops,dsl_list)
    else: ranked_dsl=DSL if dsl_list is None else dsl_list
    program=search_program(train_pairs,ranked_dsl,beam_size,search_depth,task_id)
    test_grid=canonicalize(np.array(task["test"][0]["input"]))
    # JEPA + MuZero planning
    jepa_model=JEPA(input_size=test_grid.shape[0]).to(DEVICE)
    planned_grid=muzero_plan(test_grid,jepa_model)
    # LLM reasoning over planned grid
    sc_ans=self_consistency(str(planned_grid),tokenizer,model)
    tot_ans=tree_of_thought(str(planned_grid),tokenizer,model)
    ref_ans=reflexion(str(planned_grid),tokenizer,model)
    mcts_ans=mcts_reasoning(str(planned_grid),tokenizer,model)
    final_grid=program.run(planned_grid)
    return final_grid

############################################################
# KAGGLE SUBMISSION GENERATOR
############################################################

def generate_kaggle_submission(tasks_json_path,model_key="gemma",output_path="submission.json"):
    tokenizer,model=load_llm(model_key)
    with open(tasks_json_path,"r") as f: tasks=json.load(f)
    submission=[]
    for task in tqdm(tasks):
        pred=solve_arc_task(task,model,tokenizer)
        submission.append({"task_id":task["task_id"],"output":pred.tolist()})
    with open(output_path,"w") as f: json.dump(submission,f)
    print(f"Kaggle submission saved to {output_path}")

############################################################
# MAIN
############################################################

if __name__=="__main__":
    # Generate Kaggle submission
    generate_kaggle_submission("tasks.json",model_key="llama",output_path="submission.json")
