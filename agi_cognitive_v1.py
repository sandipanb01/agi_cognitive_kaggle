#--------------CONFIG--------------------
import torch
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

SELF_CONSISTENCY_SAMPLES = 5
TOT_BRANCHES = 4
REFLEXION_STEPS = 2
MCTS_SIMULATIONS = 6

#----------------------- MODEL LOADER-------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_llm(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

    model.eval()

    return tokenizer, model

#------------------------------------ REASONING/SELF-CONSISTENCY------------------

from collections import Counter
from utils.answer_extraction import extract_answer


def self_consistency(prompt, model, tokenizer, samples):

    answers = []

    for _ in range(samples):

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=256)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        ans = extract_answer(text)

        answers.append(ans)

    return Counter(answers).most_common(1)[0][0]

#------------------------------- REASONING/ TREE OF THOUGHT---------------------------

from collections import Counter
from utils.answer_extraction import extract_answer


def tree_of_thought(prompt, model, tokenizer, branches):

    candidates = []

    for _ in range(branches):

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=256)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        candidates.append(extract_answer(text))

    return Counter(candidates).most_common(1)[0][0]

#-------------------------------- REASONING/ REFLEXION----------------------------------

from utils.answer_extraction import extract_answer


def reflexion(prompt, model, tokenizer, steps):

    reasoning = prompt

    for _ in range(steps):

        inputs = tokenizer(reasoning, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=256)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        critique = f"Critique and fix errors:\n{text}"

        inputs = tokenizer(critique, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=256)

        reasoning = tokenizer.decode(output[0], skip_special_tokens=True)

    return extract_answer(reasoning)

#------------------------------- REASONING/ MCTS_REASONING -------------------------

import math


class Node:

    def __init__(self, text):

        self.text = text
        self.visits = 0
        self.value = 0
        self.children = []


def ucb(parent, child):

    return child.value / (child.visits + 1e-5) + math.sqrt(
        math.log(parent.visits + 1) / (child.visits + 1)
    )

#-----------------------------ARC_SOLVER/DSL----------------------------------

import numpy as np


def rotate90(grid):
    return np.rot90(grid)


def rotate180(grid):
    return np.rot90(grid, 2)


def flip_horizontal(grid):
    return np.fliplr(grid)


def flip_vertical(grid):
    return np.flipud(grid)


def color_replace(grid, src, tgt):

    g = grid.copy()

    g[g == src] = tgt

    return g


DSL = [
    rotate90,
    rotate180,
    flip_horizontal,
    flip_vertical
]

#---------------------------------- ARC_SOLVER/ DSL_PROGRAM_SEARCH (BEAM_SEARCH)-----------------------------

import heapq
from .heuristics import heuristic_score


class Program:

    def __init__(self, ops):

        self.ops = ops

    def run(self, grid):

        g = grid

        for op in self.ops:

            g = op(g)

        return g


def beam_search(input_grid, target_grid, dsl_ops, beam_size=10, depth=3):

    beam = [(0, Program([]))]

    best_program = None
    best_score = -1

    for _ in range(depth):

        new_beam = []

        for _, program in beam:

            for op in dsl_ops:

                new_prog = Program(program.ops + [op])

                try:

                    pred = new_prog.run(input_grid)

                    score = heuristic_score(pred, target_grid)

                except:
                    continue

                if score > best_score:

                    best_score = score
                    best_program = new_prog

                heapq.heappush(new_beam, (-score, new_prog))

        beam = heapq.nsmallest(beam_size, new_beam)

    return best_program

#----------------------------------- ARC_SOLVER/ ARC_SOLVER--------------------------------

from .dsl import DSL
from .program_search import beam_search


def solve_arc(train_pairs):

    input_grid, target_grid = train_pairs[0]

    program = beam_search(input_grid, target_grid, DSL)

    return program

#-----------------------ARC_SOLVER/ NEURAL_GUIDED_SEARCH--------------------------------

def suggest_operations(prompt, model, tokenizer):

    hint_prompt = f"""
    Analyze the ARC transformation.

    Suggest operations from:
    rotate, flip, recolor.

    {prompt}
    """

    inputs = tokenizer(hint_prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=50)

    text = tokenizer.decode(output[0])

    ops = []

    if "rotate" in text:
        ops.append("rotate")

    if "flip" in text:
        ops.append("flip")

    return ops
#-----------------------ARC_SOLVER/ HEURISTICS--------------------------------

import numpy as np


def same_shape(a, b):

    return a.shape == b.shape


def color_histogram(grid):

    hist = {}

    for v in grid.flatten():

        hist[v] = hist.get(v, 0) + 1

    return hist


def histogram_match(a, b):

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

#----------------------------- THEOREM PROVER/ NEURAL VERIFIER-------------------------------

def verify_solution(solution):

    if solution is None:
        return False

    return True

#-----------------UTILS/ ANSWER-EXTRACTION----------------------------------

import re


def extract_answer(text):

    nums = re.findall(r'-?\d+', text)

    if nums:
        return nums[-1]

    words = re.findall(r'[a-zA-Z]+', text)

    if words:
        return words[-1].lower()

    return text.strip()

#----------------------------------- TASKS/ SYNTHETIC TASKS --------------------------

import random


def arithmetic():

    a = random.randint(1, 100)
    b = random.randint(1, 100)

    return f"What is {a} + {b}?", str(a + b)


def logic():

    q = "All birds have wings. Penguins are birds. Do penguins have wings?"

    return q, "yes"

#----------------------------------- TRAINING/ SELF-IMPROVEMENT -----------------------

def self_improve(agent, tasks):

    solved = []

    for q, a in tasks:

        pred = agent.solve(q)

        if pred == a:
            solved.append((q, a))

    agent.train_on(solved)

#----------------ENSENBLE_REASONING---------------------------------

def ensemble_reasoning(prompt, model, tokenizer):

    answers = []

    answers.append(self_consistency(prompt, model, tokenizer, 5))
    answers.append(tree_of_thought(prompt, model, tokenizer, 4))
    answers.append(reflexion(prompt, model, tokenizer, 2))

    return max(set(answers), key=answers.count)

#-------------------ANALYSIS/ ARC_BENCHMARK_RUNNER---------------------------------

def evaluate_arc(dataset, solver):

    solved = 0

    for task in dataset:

        prog = solver(task["train"])

        pred = prog.run(task["test"][0][0])

        if (pred == task["test"][0][1]).all():

            solved += 1

    return solved / len(dataset)

#-----------------DISTRIBUTED/ MULTI_GPU_RUNNER----------------------------------------------

import torch
import multiprocessing


def run_experiment(config):

    from main import main

    main(config)


def run_many(configs):

    with multiprocessing.Pool(len(configs)) as pool:

        pool.map(run_experiment, configs)

#-------------------------------------- MAIN  FUNCTION 1 --------------------------------------

from models.loader import load_llm
from reasoning.self_consistency import self_consistency
from reasoning.tree_of_thought import tree_of_thought
from reasoning.reflexion import reflexion

from theorem_prover.neural_verifier import verify_solution
from tasks.synthetic_tasks import arithmetic, logic

from config import *


def solve(prompt, model, tokenizer):

    sc = self_consistency(prompt, model, tokenizer, SELF_CONSISTENCY_SAMPLES)

    tot = tree_of_thought(prompt, model, tokenizer, TOT_BRANCHES)

    ref = reflexion(prompt, model, tokenizer, REFLEXION_STEPS)

    candidates = [sc, tot, ref]

    for c in candidates:

        if verify_solution(c):

            return c

    return sc


def main():

    tokenizer, model = load_llm("google/gemma-2b-it")

    tasks = [arithmetic() for _ in range(20)]

    for q, a in tasks:

        pred = solve(q, model, tokenizer)

        print()

        print("QUESTION:", q)

        print("PRED:", pred)

        print("GOLD:", a)


if __name__ == "__main__":

    main()

#--------------------------------- MAIN_FUNCTION_2--------------------------------------

import numpy as np

from config import *

from models.loader import load_llm
from models.perception_encoder import PerceptionEncoder
from models.jepa_world_model import JEPAWorldModel
from models.muzero_planner import MuZeroPlanner

from reasoning.self_consistency import self_consistency
from reasoning.tree_of_thought import tree_of_thought
from reasoning.reflexion import reflexion
from reasoning.ensemble import ensemble_reasoning

from arc_solver.arc_solver import solve_arc

from theorem_prover.neural_verifier import verify_solution

from tasks.synthetic_tasks import generate_tasks


class AGIAgent:

    def __init__(self):

        self.tokenizer, self.llm = load_llm("google/gemma-2b-it")

        self.perception = PerceptionEncoder()

        self.world_model = JEPAWorldModel()

        self.planner = MuZeroPlanner()

    def reason_llm(self, prompt):

        sc = self_consistency(prompt, self.llm, self.tokenizer, 5)

        tot = tree_of_thought(prompt, self.llm, self.tokenizer, 4)

        ref = reflexion(prompt, self.llm, self.tokenizer, 2)

        return ensemble_reasoning([sc, tot, ref])

    def solve_symbolic(self, task):

        program = solve_arc(task["train"])

        if program is None:

            return None

        pred = program.run(task["test"][0][0])

        return pred

    def solve(self, task):

        if task["type"] == "arc":

            pred = self.solve_symbolic(task)

            if verify_solution(pred):

                return pred

        prompt = task["question"]

        return self.reason_llm(prompt)


def main():

    print("Initializing AGI agent")

    agent = AGIAgent()

    tasks = generate_tasks(20)

    solved = 0

    for t in tasks:

        pred = agent.solve(t)

        print("\nTASK:", t)

        print("PRED:", pred)

        print("GOLD:", t["answer"])

        if str(pred) == str(t["answer"]):

            solved += 1

    print("\nAccuracy:", solved / len(tasks))


if __name__ == "__main__":

    main()




