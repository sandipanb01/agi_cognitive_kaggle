import torch
import random
import numpy as np
import os

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

# Cache folders
CACHE_DIR = "cache"
LLM_CACHE = os.path.join(CACHE_DIR, "llm_predictions")
PROGRAM_CACHE = os.path.join(CACHE_DIR, "programs")
os.makedirs(LLM_CACHE, exist_ok=True)
os.makedirs(PROGRAM_CACHE, exist_ok=True)

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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

# DSL keywords for LLM-guided ranking
DSL_KEYWORDS = {
    "rotate90": ["rotate 90", "turn right", "rotate clockwise"],
    "rotate180": ["rotate 180", "flip upside down"],
    "flip_horizontal": ["flip horizontal", "mirror left-right"],
    "flip_vertical": ["flip vertical", "mirror up-down"],
    "mirror_object": ["mirror object", "reflect object"],
    "crop_bbox": ["crop", "bounding box"],
    "duplicate_horizontal": ["duplicate horizontal", "repeat side"],
    "duplicate_vertical": ["duplicate vertical", "repeat up-down"],
    "tile_pattern": ["tile", "repeat grid"],
    "fill_rectangle": ["fill rectangle", "fill area"],
    "recolor_map": ["recolor", "map colors", "recolor grid"]
}
import re

def extract_answer(text):
    nums = re.findall(r'-?\d+', text)
    if nums:
        return nums[-1]
    words = re.findall(r'[a-zA-Z]+', text)
    if words:
        return words[-1].lower()
    return text.strip()
import numpy as np
from solvers.object_reasoner import detect_objects, object_bbox

def rotate90(g): return np.rot90(g)
def rotate180(g): return np.rot90(g, 2)
def flip_horizontal(g): return np.fliplr(g)
def flip_vertical(g): return np.flipud(g)
def mirror_object(g): return np.flipud(np.fliplr(g))

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

DSL = [
    rotate90, rotate180, flip_horizontal, flip_vertical,
    mirror_object, crop_bbox, duplicate_horizontal,
    duplicate_vertical, tile_pattern, fill_rectangle, recolor_map
]
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
import random, os, pickle
from solvers.dsl import DSL
from utils.heuristics import heuristic_score
from config import MUTATION_RATE, PROGRAM_CACHE

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

def search_program(train_pairs, ranked_dsl, beam_size=120, search_depth=9, task_id=None):
    # Use cached program if exists
    if task_id:
        cache_file = os.path.join(PROGRAM_CACHE, f"{task_id}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    population = [Program([])]
    best = None
    best_score = -1
    for _ in range(search_depth):
        new = []
        for prog in population:
            for op in ranked_dsl[:10]:
                p = prog.copy()
                p.ops.append(op)
                s = score_program(p, train_pairs)
                if s > best_score:
                    best_score = s
                    best = p
                new.append((s,p))
        new.sort(reverse=True, key=lambda x:x[0])
        population = [p for _,p in new[:beam_size]]
        for p in population: p.mutate()

    if task_id:
        with open(os.path.join(PROGRAM_CACHE, f"{task_id}.pkl"), "wb") as f:
            pickle.dump(best, f)

    return best
import numpy as np
from solvers.dsl import DSL
from solvers.program_search import search_program
from solvers.object_reasoner import detect_objects
from utils.grid_utils import canonicalize
from models.neural_dsl_ranker import predict_operations, rank_dsl
from config import LLM_CACHE
import os, pickle, hashlib

def solve_arc(task, model=None, tokenizer=None, beam_size=120, search_depth=9, dsl_list=None):
    train_pairs = [(np.array(x["input"]), np.array(x["output"])) for x in task["train"]]
    train_pairs = [(canonicalize(inp), canonicalize(out)) for inp, out in train_pairs]

    prompt = str(train_pairs)
    # Hash prompt for caching
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    if model and tokenizer:
        cache_file = os.path.join(LLM_CACHE, f"{prompt_hash}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                predicted_ops = pickle.load(f)
        else:
            predicted_ops = predict_operations(prompt, model, tokenizer)
            with open(cache_file, "wb") as f:
                pickle.dump(predicted_ops, f)
        ranked_dsl = rank_dsl(predicted_ops, dsl_list)
    else:
        ranked_dsl = DSL if dsl_list is None else dsl_list

    # Use program search caching via task ID
    task_id = hashlib.md5(str(task["train"]).encode()).hexdigest()
    program = search_program(train_pairs, ranked_dsl, beam_size, search_depth, task_id=task_id)

    test_grid = canonicalize(np.array(task["test"][0]["input"]))
    return program.run(test_grid)
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
import os, json
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
        except: pass
    return solved / len(tasks)

def run_experiment(dataset_path, model_key="gemma"):
    tokenizer, model = load_llm(model_key)
    tasks = load_arc_dataset(dataset_path)
    score = evaluate(tasks, lambda t: solve_arc(t, model, tokenizer))
    print(f"ARC score with {model_key}: {score:.3f}")
    return score
from experiments.run_experiments import run_experiment

if __name__ == "__main__":
    DATASET_PATH = "/kaggle/input/arc-prize-2024/"
    print("Running full ARC SOTA pipeline with caching...")

    for model_key in ["gemma", "llama", "mistral"]:
        run_experiment(DATASET_PATH, model_key)
import os, csv, itertools
from solvers.arc_solver import solve_arc
from models.loader import load_llm
from experiments.run_experiments import load_arc_dataset, evaluate
from solvers.dsl import DSL as DEFAULT_DSL

LLM_MODELS = ["gemma", "llama", "mistral"]
BEAM_SIZES = [50, 100, 150]
SEARCH_DEPTHS = [6, 9, 12]
DSL_SETS = ["default", "expanded"]

DATASET_PATH = "/kaggle/input/arc-prize-2024/"
OUTPUT_CSV = "arc_parameter_sweep_results.csv"

def get_dsl_set(name):
    if name == "default":
        return DEFAULT_DSL[:4]
    elif name == "expanded":
        return DEFAULT_DSL
    return DEFAULT_DSL

def run_parameter_sweep():
    tasks = load_arc_dataset(DATASET_PATH)
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "llm_model", "beam_size", "search_depth", "dsl_set", "accuracy"
        ])
        writer.writeheader()

        for llm_model, beam, depth, dsl_set_name in itertools.product(
            LLM_MODELS, BEAM_SIZES, SEARCH_DEPTHS, DSL_SETS
        ):
            print(f"\nRunning: LLM={llm_model}, beam={beam}, depth={depth}, DSL={dsl_set_name}")
            tokenizer, model = load_llm(llm_model)
            dsl_list = get_dsl_set(dsl_set_name)

            def solver(task):
                return solve_arc(task, model, tokenizer, beam_size=beam, search_depth=depth, dsl_list=dsl_list)

            accuracy = evaluate(tasks, solver)
            print(f"Accuracy: {accuracy:.3f}")

            writer.writerow({
                "llm_model": llm_model,
                "beam_size": beam,
                "search_depth": depth,
                "dsl_set": dsl_set_name,
                "accuracy": accuracy
            })

if __name__ == "__main__":
    run_parameter_sweep()
                






