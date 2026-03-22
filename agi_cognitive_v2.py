# config.py
import torch
import random
import numpy as np

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline parameters
MAX_NEW_TOKENS = 128
BEAM_SIZE = 120
SEARCH_DEPTH = 9
MUTATION_RATE = 0.3
MAX_PROGRAMS = 200

# LLM models
MODEL_REGISTRY = {
    "gemma": "google/gemma-2b-it",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# models/loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_REGISTRY, DEVICE
import torch

def load_llm(model_key="gemma"):
    name = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model
# utils/grid_utils.py
import numpy as np

def canonicalize(grid):
    grid = np.array(grid)
    bg = np.bincount(grid.flatten()).argmax()
    grid = grid - bg
    grid[grid < 0] = 0
    return grid

def color_histogram(grid):
    hist = {}
    for v in grid.flatten():
        hist[v] = hist.get(v, 0) + 1
    return hist
# utils/heuristics.py
import numpy as np

def same_shape(a, b):
    return a.shape == b.shape

def histogram_match(a, b):
    from utils.grid_utils import color_histogram
    return color_histogram(a) == color_histogram(b)

def heuristic_score(pred, target):
    score = 0
    if same_shape(pred, target):
        score += 1
    if histogram_match(pred, target):
        score += 1
    overlap = np.sum(pred == target)
    score += overlap / target.size
    return score
# utils/answer_extraction.py
import re

def extract_answer(text):
    nums = re.findall(r'-?\d+', text)
    if nums:
        return nums[-1]
    words = re.findall(r'[a-zA-Z]+', text)
    if words:
        return words[-1].lower()
    return text.strip()
# solvers/dsl.py
import numpy as np
from solvers.object_reasoner import detect_objects, object_bbox

# Basic grid transforms
def rotate90(g): return np.rot90(g)
def rotate180(g): return np.rot90(g, 2)
def flip_horizontal(g): return np.fliplr(g)
def flip_vertical(g): return np.flipud(g)
def mirror_object(g): return np.flipud(np.fliplr(g))

# Object transforms
def crop_bbox(g):
    objs = detect_objects(g)
    if not objs: return g
    y1, y2, x1, x2 = object_bbox(objs[0])
    return g[y1:y2+1, x1:x2+1]

def duplicate_horizontal(g): return np.concatenate([g,g],axis=1)
def duplicate_vertical(g): return np.concatenate([g,g],axis=0)
def tile_pattern(g): return np.tile(g,(2,2))
def fill_rectangle(g):
    g = g.copy()
    mask = g > 0
    ys, xs = np.where(mask)
    if len(ys) == 0: return g
    g[ys.min():ys.max()+1, xs.min():xs.max()+1] = g[ys[0], xs[0]]
    return g
def recolor_map(g):
    g = g.copy()
    colors = np.unique(g)
    mapping = {c:i for i,c in enumerate(colors)}
    for k,v in mapping.items():
        g[g==k]=v
    return g

# DSL list
DSL = [
    rotate90, rotate180, flip_horizontal, flip_vertical,
    mirror_object, crop_bbox, duplicate_horizontal,
    duplicate_vertical, tile_pattern, fill_rectangle, recolor_map
]
# solvers/object_reasoner.py
import numpy as np
from scipy.ndimage import label

def detect_objects(grid):
    grid = np.array(grid)
    objects = []
    for c in np.unique(grid):
        mask = grid == c
        labeled, n = label(mask)
        for i in range(1, n+1):
            m = labeled == i
            coords = np.argwhere(m)
            objects.append({"color": c, "mask": m, "coords": coords})
    return objects

def object_bbox(obj):
    coords = obj["coords"]
    y = coords[:,0]; x=coords[:,1]
    return y.min(), y.max(), x.min(), x.max()

def translate_object(grid, obj, dy, dx):
    new_grid = grid.copy()
    new_grid[obj["mask"]] = 0
    for y,x in obj["coords"]:
        ny, nx = int(y+dy), int(x+dx)
        if 0<=ny<grid.shape[0] and 0<=nx<grid.shape[1]:
            new_grid[ny,nx] = obj["color"]
    return new_grid

def recolor_object(grid, obj, new_color):
    new_grid = grid.copy()
    new_grid[obj["mask"]] = new_color
    return new_grid
# solvers/program_search.py
import random
from solvers.dsl import DSL
from utils.heuristics import heuristic_score

MUTATION_RATE = 0.3
BEAM_SIZE = 120
SEARCH_DEPTH = 9

class Program:
    def __init__(self, ops):
        self.ops = ops
    def run(self, grid):
        g = grid.copy()
        for op in self.ops: g = op(g)
        return g
    def mutate(self):
        if random.random() < MUTATION_RATE:
            self.ops.append(random.choice(DSL))
    def copy(self):
        return Program(self.ops.copy())

def score_program(program, train_pairs):
    score = 0
    for inp, out in train_pairs:
        try:
            pred = program.run(inp)
            if pred.shape == out.shape:
                score += np.sum(pred==out)/pred.size
        except: pass
    return score

def search_program(train_pairs, ranked_dsl):
    population=[Program([])]
    best=None
    best_score=-1
    for depth in range(SEARCH_DEPTH):
        new=[]
        for prog in population:
            for op in ranked_dsl[:10]:
                p=prog.copy()
                p.ops.append(op)
                s=score_program(p, train_pairs)
                if s>best_score: best_score=s; best=p
                new.append((s,p))
        new.sort(reverse=True,key=lambda x:x[0])
        population=[p for _,p in new[:BEAM_SIZE]]
        for p in population: p.mutate()
    return best
# solvers/arc_solver.py
import numpy as np
from solvers.dsl import DSL
from solvers.program_search import search_program
from solvers.object_reasoner import detect_objects, translate_object, recolor_object
from utils.grid_utils import canonicalize
from models.neural_dsl_ranker import predict_operations, rank_dsl

def solve_arc(task, model=None, tokenizer=None):
    """
    Solve a single ARC task using:
    - object reasoning
    - neural-guided DSL ranking
    - evolutionary beam search
    """
    # Multi-example train pairs
    train_pairs = [(np.array(x["input"]), np.array(x["output"])) for x in task["train"]]

    # Canonicalize grids
    train_pairs = [(canonicalize(inp), canonicalize(out)) for inp, out in train_pairs]

    prompt = str(train_pairs)

    # Neural-guided ranking if LLM available
    if model and tokenizer:
        predicted_ops = predict_operations(prompt, model, tokenizer)
        ranked_dsl = rank_dsl(predicted_ops)
    else:
        ranked_dsl = DSL

    # Search best program
    program = search_program(train_pairs, ranked_dsl)

    # Apply program to test input
    test_grid = canonicalize(np.array(task["test"][0]["input"]))
    return program.run(test_grid)
# models/neural_dsl_ranker.py
from utils.heuristics import DSL_KEYWORDS
import torch

def predict_operations(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)
    text = tokenizer.decode(outputs[0]).lower()
    ops = []
    for op, keywords in DSL_KEYWORDS.items():
        if any(k in text for k in keywords):
            ops.append(op)
    return ops

def rank_dsl(predicted_ops, dsl_list=None):
    ranked = []
    from solvers.dsl import DSL
    dsl = DSL if dsl_list is None else dsl_list
    for op in dsl:
        score = 1
        for p in predicted_ops:
            if p in op.__name__: score += 5
        ranked.append((score, op))
    ranked.sort(reverse=True)
    return [o for _, o in ranked]
# experiments/run_experiments.py
import os
import json
from solvers.arc_solver import solve_arc
from models.loader import load_llm
import numpy as np

def load_arc_dataset(path):
    tasks = []
    for f in os.listdir(path):
        if f.endswith(".json"):
            with open(os.path.join(path,f)) as j:
                tasks.append(json.load(j))
    return tasks

def evaluate(tasks, solver):
    solved = 0
    for t in tasks:
        try:
            pred = solver(t)
            gold = np.array(t["test"][0]["output"])
            if np.array_equal(pred, gold):
                solved += 1
        except:
            pass
    return solved / len(tasks)

def run_experiment(dataset_path, model_key="gemma"):
    tokenizer, model = load_llm(model_key)
    tasks = load_arc_dataset(dataset_path)
    score = evaluate(tasks, lambda t: solve_arc(t, model, tokenizer))
    print(f"ARC score with {model_key}: {score:.3f}")
    return score
# main.py
from experiments.run_experiments import run_experiment

if __name__ == "__main__":
    DATASET_PATH = "/kaggle/input/arc-prize-2024/"  # Kaggle ARC dataset path
    print("Running full ARC SOTA pipeline...")

    # Run experiments with multiple LLMs
    for model_key in ["gemma", "llama", "mistral"]:
        run_experiment(DATASET_PATH, model_key)
# experiments/parameter_sweep.py
import os
import csv
import itertools
import numpy as np
from solvers.arc_solver import solve_arc
from models.loader import load_llm
from experiments.run_experiments import load_arc_dataset, evaluate

# ------------------------------
# Sweep configuration
# ------------------------------
LLM_MODELS = ["gemma", "llama", "mistral"]
BEAM_SIZES = [50, 100, 150]
SEARCH_DEPTHS = [6, 9, 12]
DSL_SETS = ["default", "expanded"]

DATASET_PATH = "/kaggle/input/arc-prize-2024/"
OUTPUT_CSV = "arc_parameter_sweep_results.csv"

# ------------------------------
# Helper to switch DSL sets
# ------------------------------
def get_dsl_set(name):
    from solvers.dsl import DSL as DEFAULT_DSL
    if name == "default":
        return DEFAULT_DSL[:4]  # first 4 primitives
    elif name == "expanded":
        return DEFAULT_DSL  # all expanded primitives
    else:
        return DEFAULT_DSL

# ------------------------------
# Run sweep
# ------------------------------
def run_parameter_sweep():
    tasks = load_arc_dataset(DATASET_PATH)

    # Prepare CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "llm_model", "beam_size", "search_depth", "dsl_set", "accuracy"
        ])
        writer.writeheader()

        # Iterate over all combinations
        for llm_model, beam, depth, dsl_set_name in itertools.product(
            LLM_MODELS, BEAM_SIZES, SEARCH_DEPTHS, DSL_SETS
        ):
            print(f"\nRunning: LLM={llm_model}, beam={beam}, depth={depth}, DSL={dsl_set_name}")

            # Load LLM
            tokenizer, model = load_llm(llm_model)

            # Override global hyperparameters
            from solvers.program_search import BEAM_SIZE as global_beam, SEARCH_DEPTH as global_depth
            from solvers.arc_solver import DSL as global_dsl

            global_beam = beam
            global_depth = depth
            global_dsl = get_dsl_set(dsl_set_name)

            # Evaluate solver
            def solver(task):
                return solve_arc(task, model, tokenizer)

            accuracy = evaluate(tasks, solver)
            print(f"Accuracy: {accuracy:.3f}")

            # Write to CSV
            writer.writerow({
                "llm_model": llm_model,
                "beam_size": beam,
                "search_depth": depth,
                "dsl_set": dsl_set_name,
                "accuracy": accuracy
            })

if __name__ == "__main__":
    run_parameter_sweep()
            














