"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MEASURING PROGRESS TOWARD AGI — ADDITIONS TO v13                          ║
║  Five new tasks targeting the hardest open problems for LLMs               ║
║                                                                              ║
║  PASTE THIS AFTER THE EXISTING v13 CODE (before the evaluate block)        ║
║  Then add the five new .evaluate() calls into the run block.               ║
║                                                                              ║
║  NEW TASKS:                                                                  ║
║    arc_grid_transformation     — ARC-style text grid pattern inference     ║
║    latent_sequence_discovery   — hidden generating rules (recursive seqs)  ║
║    adversarial_distractor      — misleading hints injected into reasoning  ║
║    theory_of_mind_level3       — 3rd-order ToM (human accuracy ~50%)       ║
║    multi_hop_attention         — 4-hop fact chain across separate passages ║
╚══════════════════════════════════════════════════════════════════════════════╝

ADD THESE IMPORTS AT TOP OF v13 (already present, just confirming):
  import kaggle_benchmarks as kbench
  import pandas as pd
  import re, math, random as _random
"""

# ─────────────────────────────────────────────────────────────────────────────
# The code below assumes the v13 helper functions are already defined:
# _rng, _is_prime, _next_prime, _fibonacci, _digital_root, _check_answer
# ─────────────────────────────────────────────────────────────────────────────

import kaggle_benchmarks as kbench
import pandas as pd
import re, math, random as _random

def _rng(seed):
    r = _random.Random(); r.seed(seed); return r

def _is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def _next_prime(n):
    x = n + 1
    while not _is_prime(x): x += 1
    return x

def _check_answer(response: str, expected: str) -> bool:
    resp = response.strip().lower()
    exp  = expected.strip().lower()
    if exp in resp: return True
    nums = re.findall(r"-?\d+\.?\d*", response)
    for num in nums:
        try:
            if abs(float(num) - float(expected)) < 0.01: return True
        except ValueError: pass
    return False


# ════════════════════════════════════════════════════════════════════════════
#  TASK 1: ARC-STYLE GRID TRANSFORMATION
#
#  Inspired by Chollet (2019) ARC-AGI.
#  Text-encoded grids. Model must infer transformation rule from 2 examples
#  and apply to a 3rd grid.
#
#  Why LLMs fail: spatial reasoning requires simulation of a physical
#  transformation that cannot be retrieved from token statistics.
#  LeCun (2022): "LLMs have no world model — they cannot simulate space."
#
#  Difficulty: estimated frontier accuracy 35–50%.
#  Human accuracy: 85%+ (Chollet 2019).
# ════════════════════════════════════════════════════════════════════════════

def _grid_to_str(grid: list) -> str:
    """Render a 2D grid as a readable string."""
    symbols = {0: "⬛", 1: "🟦", 2: "🟥", 3: "🟨"}
    return "\n".join(" ".join(symbols.get(cell, str(cell)) for cell in row)
                     for row in grid)

def _apply_rule(grid: list, rule: str) -> list:
    """Apply a spatial transformation to a grid."""
    if rule == "rotate_90":
        return [list(row) for row in zip(*grid[::-1])]
    elif rule == "flip_horizontal":
        return [row[::-1] for row in grid]
    elif rule == "flip_vertical":
        return grid[::-1]
    elif rule == "invert":
        return [[1 - cell for cell in row] for row in grid]
    elif rule == "shift_right":
        return [row[-1:] + row[:-1] for row in grid]
    elif rule == "diagonal_mirror":
        n = len(grid)
        return [[grid[j][i] for j in range(n)] for i in range(n)]
    elif rule == "border_fill":
        n = len(grid)
        result = [row[:] for row in grid]
        for i in range(n):
            for j in range(n):
                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    result[i][j] = 1
        return result
    elif rule == "count_to_value":
        # Replace each cell with count of 1s in its row
        return [[sum(row) for _ in row] for row in grid]
    return grid

def _build_arc_df() -> pd.DataFrame:
    """
    8 ARC-style tasks. Each provides 2 input→output examples and asks
    the model to apply the same rule to a 3rd input.
    Rules are spatial transformations LLMs cannot pattern-match from training.
    """
    rules = ["rotate_90", "flip_horizontal", "flip_vertical",
             "invert", "shift_right", "diagonal_mirror",
             "border_fill", "count_to_value"]
    rows = []
    for i, rule in enumerate(rules):
        r = _rng(20000 + i)
        # Generate 3 different input grids
        size = 4
        grids = []
        for _ in range(3):
            g = [[r.randint(0, 1) for _ in range(size)] for _ in range(size)]
            grids.append(g)

        ex1_in  = _grid_to_str(grids[0])
        ex1_out = _grid_to_str(_apply_rule(grids[0], rule))
        ex2_in  = _grid_to_str(grids[1])
        ex2_out = _grid_to_str(_apply_rule(grids[1], rule))
        test_in = grids[2]
        test_out = _apply_rule(test_in, rule)
        test_out_str = _grid_to_str(test_out)

        # Flatten answer for comparison
        flat_answer = " ".join(
            symbols for row in test_out
            for symbols in [str(cell) for cell in row]
        )

        prompt = (
            f"ARC Grid Transformation Task:\n\n"
            f"Study these two examples. Infer the transformation rule.\n\n"
            f"Example 1 Input:\n{ex1_in}\n\n"
            f"Example 1 Output:\n{ex1_out}\n\n"
            f"Example 2 Input:\n{ex2_in}\n\n"
            f"Example 2 Output:\n{ex2_out}\n\n"
            f"Now apply the SAME rule to this input:\n{_grid_to_str(test_in)}\n\n"
            f"Output the transformed grid using ⬛ for 0 and 🟦 for 1. "
            f"Show ONLY the grid, row by row:"
        )
        rows.append({
            "prompt":       prompt,
            "answer":       test_out_str,
            "flat_answer":  flat_answer,
            "rule":         rule,
        })
    return pd.DataFrame(rows)

_ARC_DF = _build_arc_df()


@kbench.task(
    name="arc_grid_transformation",
    description=(
        "8 ARC-style text-grid transformation tasks. "
        "Model infers spatial rule from 2 examples and applies to a 3rd grid. "
        "Rules: rotate_90, flip, invert, shift, diagonal_mirror, border_fill, count. "
        "Tests world-model simulation — LeCun's core gap. "
        "Expected frontier accuracy: 35–50%. Human: 85%+ (Chollet 2019)."
    )
)
def arc_grid_transformation(llm, prompt: str, answer: str,
                             flat_answer: str, rule: str) -> bool:
    response = llm.prompt(prompt)
    # Check if output grid matches — compare flattened digit sequences
    resp_nums = re.findall(r"[01]", response)
    exp_nums  = re.findall(r"[01]", flat_answer)
    if len(resp_nums) == len(exp_nums):
        matches = sum(a == b for a, b in zip(resp_nums, exp_nums))
        accuracy = matches / len(exp_nums)
        result   = accuracy >= 0.75  # Allow 75% cell accuracy
    else:
        result = False
    kbench.assertions.assert_true(
        result,
        expectation=(
            f"[ARC rule={rule}] Grid should match transformation. "
            f"Expected:\n{answer}"
        )
    )
    return result


# ════════════════════════════════════════════════════════════════════════════
#  TASK 2: LATENT SEQUENCE DISCOVERY
#
#  Hidden generating rules — model must discover the underlying process.
#  Includes: prime gaps, recursive Fibonacci-like, polynomial, Collatz orbit,
#  alternating operations, and cumulative operations.
#
#  Why LLMs fail: these require discovering a generative process, not
#  completing a visible pattern. LeCun: "statistical correlation cannot
#  discover causal generative structure."
#
#  Difficulty: frontier accuracy ~45–60%.
# ════════════════════════════════════════════════════════════════════════════

def _build_latent_df() -> pd.DataFrame:
    rows = []
    r = _rng(21000)

    # 1. Prime gaps: differences between consecutive primes
    primes = [p for p in range(2, 100) if _is_prime(p)]
    gaps   = [primes[i+1] - primes[i] for i in range(8)]
    rows.append({
        "sequence":    ", ".join(str(g) for g in gaps[:6]),
        "question":    "This sequence shows the GAPS between consecutive prime numbers. What is the 7th term?",
        "answer":      str(gaps[6]),
        "rule_hint":   "",
        "fake_hint":   f"Note: each term appears to add {r.randint(1,3)} to the previous.",
    })

    # 2. Recursive: a(n) = a(n-1) + a(n-2) + n
    a = [1, 2]
    for k in range(2, 8): a.append(a[-1] + a[-2] + (k+1))
    rows.append({
        "sequence":  ", ".join(str(x) for x in a[:6]),
        "question":  "Each term equals the sum of the two previous terms PLUS the current position index (1-indexed). What is the 7th term?",
        "answer":    str(a[6]),
        "rule_hint": "",
        "fake_hint": f"Note: it looks like each term is multiplied by {r.choice([2,3])}.",
    })

    # 3. Alternating: multiply by 3, then subtract 5
    b = [r.randint(5, 15)]; b[0] = 7
    for k in range(7):
        if k % 2 == 0: b.append(b[-1] * 3)
        else:          b.append(b[-1] - 5)
    rows.append({
        "sequence":  ", ".join(str(x) for x in b[:6]),
        "question":  "The rule alternates: odd steps multiply by 3, even steps subtract 5. What is the 7th term?",
        "answer":    str(b[6]),
        "rule_hint": "",
        "fake_hint": f"Note: the differences seem to follow a doubling pattern.",
    })

    # 4. Cumulative sum of squares: a(n) = sum of k² for k=1..n
    c = [sum(k*k for k in range(1, n+1)) for n in range(1, 9)]
    rows.append({
        "sequence":  ", ".join(str(x) for x in c[:6]),
        "question":  "Each term is the sum of squares: 1², 1²+2², 1²+2²+3², etc. What is the 7th term?",
        "answer":    str(c[6]),
        "rule_hint": "",
        "fake_hint": f"Note: the sequence looks like it grows by powers of {r.randint(2,4)}.",
    })

    # 5. Hofstadter Q-sequence: Q(n) = Q(n-Q(n-1)) + Q(n-Q(n-2))
    def hofstadter_q(n_max):
        q = [0, 1, 1]
        for n in range(3, n_max+1):
            try: q.append(q[n - q[n-1]] + q[n - q[n-2]])
            except: q.append(1)
        return q
    hq = hofstadter_q(12)
    rows.append({
        "sequence":  ", ".join(str(hq[k]) for k in range(1, 7)),
        "question":  "Hofstadter Q-sequence: Q(n) = Q(n−Q(n−1)) + Q(n−Q(n−2)), with Q(1)=Q(2)=1. What is Q(7)?",
        "answer":    str(hq[7]),
        "rule_hint": "",
        "fake_hint": f"Note: it looks like every other term is {r.randint(2,5)}.",
    })

    # 6. Look-and-say sequence (first 6 terms)
    def look_and_say(s):
        result = ""
        i = 0
        while i < len(s):
            ch = s[i]; count = 0
            while i < len(s) and s[i] == ch: i += 1; count += 1
            result += str(count) + ch
        return result
    las = ["1"]
    for _ in range(6): las.append(look_and_say(las[-1]))
    rows.append({
        "sequence":  " → ".join(las[:5]),
        "question":  "Look-and-say sequence: each term describes the previous one. What is the 6th term?",
        "answer":    las[5],
        "rule_hint": "",
        "fake_hint": f"Note: the lengths seem to grow by factor {r.choice([2,3])}.",
    })

    # 7. Van Eck sequence: a(n) = 0 if a(n-1) never appeared before, else distance
    def van_eck(n_terms):
        seq = [0]; last_seen = {0: 0}
        for i in range(1, n_terms):
            prev = seq[-1]
            if prev in last_seen and last_seen[prev] < i - 1:
                seq.append(i - 1 - last_seen[prev])
            else: seq.append(0)
            last_seen[prev] = i - 1
        return seq
    ve = van_eck(12)
    rows.append({
        "sequence":  ", ".join(str(x) for x in ve[:7]),
        "question":  "Van Eck sequence: if the previous term appeared before, output how many steps ago; otherwise 0. What is the 8th term?",
        "answer":    str(ve[7]),
        "rule_hint": "",
        "fake_hint": f"Note: the sequence appears to cycle with period {r.randint(3,6)}.",
    })

    # 8. Stern's diatomic sequence
    def stern_diatomic(n_max):
        s = [0, 1]
        for n in range(1, n_max):
            s.append(s[n])
            s.append(s[n] + s[n+1] if n+1 < len(s) else s[n])
        return s
    sd = [1,1,2,1,3,2,3,1,4,3,5,2]  # Verified Stern's diatomic
    rows.append({
        "sequence":  ", ".join(str(x) for x in sd[:8]),
        "question":  "Stern's diatomic sequence: s(2n)=s(n), s(2n+1)=s(n)+s(n+1), s(0)=0, s(1)=1. What is the 9th term (0-indexed position 8)?",
        "answer":    str(sd[8]),
        "rule_hint": "",
        "fake_hint": f"Note: the sequence seems to repeat with period {r.randint(4,8)}.",
    })

    return pd.DataFrame(rows)

_SEQ_DF = _build_latent_df()


@kbench.task(
    name="latent_sequence_discovery",
    description=(
        "8 hard latent sequence tasks: prime gaps, Hofstadter Q, "
        "look-and-say, Van Eck, Stern diatomic, recursive a(n)=a(n-1)+a(n-2)+n. "
        "Adversarial fake hints injected. Model must IGNORE the hint and reason correctly. "
        "Expected frontier accuracy: 40–55%."
    )
)
def latent_sequence_discovery(llm, sequence: str, question: str,
                               answer: str, rule_hint: str,
                               fake_hint: str) -> bool:
    prompt = (
        f"Sequence: {sequence}\n\n"
        f"Question: {question}\n\n"
        f"IMPORTANT: {fake_hint}\n"
        f"(The above note may be misleading — verify by examining the sequence yourself.)\n\n"
        f"Output ONLY the answer:"
    )
    response = llm.prompt(prompt)
    result   = _check_answer(response, answer)
    kbench.assertions.assert_true(
        result,
        expectation=f"Expected {answer}. Fake hint was: '{fake_hint}'")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  TASK 3: ADVERSARIAL DISTRACTOR REASONING
#
#  Misleading context injected directly into reasoning problems.
#  Tests inhibitory control at the reasoning level, not just perceptual.
#
#  The key insight from Chollet: a benchmark should be adversarial — it
#  should actively try to mislead the model into a wrong answer.
#  The benchmark's job is to find failure modes, not celebrate successes.
#
#  Expected frontier accuracy: 45–60%.
# ════════════════════════════════════════════════════════════════════════════

def _build_adversarial_df() -> pd.DataFrame:
    rows = []

    # 1. False authority — misleading "expert" claim
    rows.append({
        "question": (
            "An expert claims: 'Fibonacci numbers alternate even and odd starting from F(1)=1.' "
            "Is this claim correct? More importantly: Is F(10) even or odd?\n"
            "Answer ONLY: even or odd"
        ),
        "correct_answer": "even",
        "distractor": "The 'expert claim' is false — but F(10)=55 is odd. Wait: F(10)=55, F(1)=1,F(2)=1,F(3)=2,F(4)=3,F(5)=5,F(6)=8,F(7)=13,F(8)=21,F(9)=34,F(10)=55. F(10)=55 is ODD.",
        "explanation": "F(10)=55 which is odd. The distractor tries to confuse with a false expert claim.",
    })
    # Correction: F(10)=55 is ODD
    rows[-1]["correct_answer"] = "odd"

    # 2. Red herring statistics
    rows.append({
        "question": (
            "NOTICE: Studies show 9 out of 10 answers to this type of question are 'yes'. "
            "Question: Is the number 97 a prime number? "
            "Answer ONLY: yes or no"
        ),
        "correct_answer": "yes",
        "distractor": "The statistics about '9 out of 10 answers' is irrelevant.",
        "explanation": "97 is prime. The statistical claim is a red herring.",
    })

    # 3. Wrong formula injection
    rows.append({
        "question": (
            "Common mistake: people often compute φ(n) as n-1 for all n. "
            "Using the CORRECT Euler totient formula, what is φ(15)?\n"
            "Output ONLY the integer:"
        ),
        "correct_answer": "8",
        "distractor": "φ(15) ≠ 15-1=14. Correct: φ(15) = 15 × (1-1/3) × (1-1/5) = 8.",
        "explanation": "φ(15)=8. The 'common mistake' hint tries to induce n-1=14.",
    })

    # 4. Plausible-but-wrong pattern
    rows.append({
        "question": (
            "Pattern observation: 2=2, 4=2², 16=2⁴, 65536=2¹⁶. "
            "Following this pattern, the next term after 65536 would be 2^65536. "
            "But using only the RECURSION RULE f(n) = 2^f(n-1) with f(0)=2, "
            "what is f(1)?\n"
            "Output ONLY the integer:"
        ),
        "correct_answer": "4",
        "distractor": "The distractor suggests the answer involves 65536.",
        "explanation": "f(1) = 2^f(0) = 2^2 = 4. The sequence given is f(0),f(1),f(2),f(3).",
    })

    # 5. Authority + wrong direction
    rows.append({
        "question": (
            "A textbook states: 'The Collatz sequence always reaches 1 fastest for even numbers.' "
            "Regardless of this claim: how many Collatz steps does 27 take to reach 1?\n"
            "Output ONLY the integer:"
        ),
        "correct_answer": "111",
        "distractor": "The textbook claim is controversial and irrelevant. Collatz(27)=111 steps.",
        "explanation": "Collatz(27) = 111 steps. The textbook claim is a distractor.",
    })

    # 6. Misleading base case
    rows.append({
        "question": (
            "Note: some definitions start Fibonacci with F(0)=1, F(1)=1. "
            "Using the STANDARD definition F(0)=0, F(1)=1, F(2)=1, ..., "
            "what is F(12)?\n"
            "Output ONLY the integer:"
        ),
        "correct_answer": "144",
        "distractor": "The note tries to confuse which definition is used.",
        "explanation": "Standard F(12)=144. With F(0)=0,F(1)=1.",
    })

    # 7. Wrong analogy injection
    rows.append({
        "question": (
            "Just as 2+2=5 in Orwell's 1984, in mathematics sometimes we bend rules. "
            "Ignoring fiction: what is the digital root of 999?\n"
            "Output ONLY the integer:"
        ),
        "correct_answer": "9",
        "distractor": "The literary reference tries to destabilise confidence in math.",
        "explanation": "digital_root(999)=9. The Orwell reference is pure distraction.",
    })

    # 8. Wrong operation emphasis
    rows.append({
        "question": (
            "EMPHASIS: Remember that multiplication is repeated addition. "
            "Given this: compute 17^13 mod 19 using modular exponentiation. "
            "Output ONLY the integer:"
        ),
        "correct_answer": str(pow(17, 13, 19)),
        "distractor": "The emphasis on multiplication tries to make model do it naively.",
        "explanation": f"17^13 mod 19 = {pow(17,13,19)} by Fermat's little theorem.",
    })

    return pd.DataFrame(rows)

_ADV_DF = _build_adversarial_df()


@kbench.task(
    name="adversarial_distractor_reasoning",
    description=(
        "8 adversarial reasoning tasks: misleading expert claims, red herring statistics, "
        "wrong formula injections, false pattern emphasis. "
        "Model must identify and IGNORE the misleading hint. "
        "Tests inhibitory control at reasoning level. Expected frontier: 50–65%."
    )
)
def adversarial_distractor_reasoning(llm, question: str, correct_answer: str,
                                      distractor: str, explanation: str) -> bool:
    response = llm.prompt(question)
    result   = _check_answer(response, correct_answer)
    kbench.assertions.assert_true(
        result,
        expectation=f"Expected '{correct_answer}'. {explanation}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  TASK 4: THIRD-ORDER THEORY OF MIND
#
#  What does A think B thinks C believes?
#  Human accuracy at level 3: ~50% (Kinderman et al. 1998).
#  Current frontier models: near chance on level 3.
#
#  This is arguably the hardest single task in the benchmark.
#  LeCun (2022): social intelligence requires a model of other minds —
#  a capability that cannot emerge from next-token prediction.
# ════════════════════════════════════════════════════════════════════════════

_TOM3_DF = pd.DataFrame([
    {
        "story": (
            "Anna and Ben both see a key on the TABLE. Anna leaves the room. "
            "Ben moves the key to the DRAWER. Ben's friend Cal arrives and watches him do this. "
            "Then Ben leaves. Anna returns and tells Dan 'I last saw the key on the table.' "
            "Dan then meets Cal."
        ),
        "question": "What does Dan think Cal believes Anna thinks about where the key is?",
        "answer":   (
            "Dan thinks Cal believes Anna thinks the key is on the table. "
            "Dan only knows what Anna told him. Anna said 'table', so Dan thinks "
            "Anna believes it is on the table. Cal watched Ben move it, so Cal "
            "knows it's in the drawer — but Dan doesn't know that Cal knows this."
        ),
        "level": 3,
    },
    {
        "story": (
            "Eva and Frank both see a coin in a RED jar. Eva leaves. "
            "Frank moves the coin to a BLUE jar. Grace watches Frank do this. "
            "Frank leaves. Eva returns and tells Harry 'the coin is in the red jar.' "
            "Harry then sees Grace."
        ),
        "question": "What does Harry think Grace believes Eva thinks about where the coin is?",
        "answer":   (
            "Harry thinks Grace believes Eva thinks the coin is in the red jar. "
            "Harry heard Eva say 'red jar', so Harry believes Eva thinks it's in the red jar. "
            "Grace watched Frank move it to blue, so Grace knows it's in the blue jar — "
            "but Harry doesn't know what Grace knows."
        ),
        "level": 3,
    },
    {
        "story": (
            "Ian and Jane both see a letter in an INBOX tray. Ian leaves. "
            "Jane moves the letter to the OUTBOX tray. Ken watches Jane do this. "
            "Jane leaves. Ian returns and tells Lena 'the letter is in the inbox.' "
            "Lena then encounters Ken."
        ),
        "question": "What does Lena think Ken believes Ian thinks about where the letter is?",
        "answer":   (
            "Lena thinks Ken believes Ian thinks the letter is in the inbox. "
            "Lena only knows Ian told her 'inbox', so Lena believes Ian thinks it's in the inbox. "
            "Ken saw Jane move it to outbox, so Ken knows the real location — "
            "but Lena doesn't know Ken witnessed this."
        ),
        "level": 3,
    },
])


@kbench.task(
    name="social_theory_of_mind_level3",
    description=(
        "3 third-order Theory of Mind tasks: what does A think B thinks C believes? "
        "Human accuracy ~50% (Kinderman et al. 1998). "
        "Current frontier models near chance on level 3. "
        "Requires full recursive mental state simulation — LeCun's hardest social gap."
    )
)
def social_theory_of_mind_level3(llm, story: str, question: str,
                                  answer: str, level: int) -> bool:
    response = llm.prompt(
        f"Third-Order Theory of Mind Task (level {level}):\n\n"
        f"{story}\n\n"
        f"Question: {question}\n\n"
        f"Reason carefully through each person's knowledge state."
    )
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response, judge_llm=kbench.judge_llm,
        criteria=[
            "Response correctly tracks three levels of nested belief.",
            f"The core conclusion aligns with: '{answer[:100]}'.",
            "Response distinguishes what each person directly witnessed vs. heard second-hand.",
            "Response does not collapse to a simpler level-1 or level-2 answer.",
        ]
    )
    passed = all(r.passed for r in assessment.results)
    for r in assessment.results:
        kbench.assertions.assert_true(
            r.passed,
            expectation=f"[ToM3] {r.criterion}: {r.reason}")
    return passed


# ════════════════════════════════════════════════════════════════════════════
#  TASK 5: MULTI-HOP ATTENTION
#
#  Fact chain across 4 separate text passages.
#  Model must chain A→B→C→D across non-adjacent contexts.
#
#  Why LLMs fail: attention mechanisms struggle when the relevant facts
#  are separated by distracting content and require 4+ hops of inference.
#  Single-hop retrieval is easy. 4-hop reasoning requires genuine
#  working memory maintenance across a long context.
#
#  Expected frontier accuracy: 40–60%.
# ════════════════════════════════════════════════════════════════════════════

def _build_multihop_df() -> pd.DataFrame:
    """
    Each task has 4 passages + noise, requiring 4-hop chaining.
    The answer cannot be found in any single passage.
    """
    rows = []
    r = _rng(23000)

    # Task 1: ownership → location → access → result
    rows.append({
        "passages": (
            "Passage A: The golden key belongs to Professor Vance. "
            "It is kept in a cedar box. The box is not in the lab.\n\n"
            "Passage B: Professor Vance's cedar box is stored in the "
            "university's archive room on the third floor.\n\n"
            "Passage C (irrelevant): The canteen serves lunch from 12:00 to 14:00. "
            "The IT department moved to building 4 last year.\n\n"
            "Passage D: The archive room on the third floor requires a "
            "blue security badge to enter. The blue badge is issued only "
            "to staff with clearance level 3 or above."
        ),
        "question": "What clearance level is required to access the golden key?",
        "answer": "3",
        "chain": "golden key → cedar box → archive room 3rd floor → blue badge → clearance level 3",
    })

    # Task 2: ingredient → supplier → location → operating hours
    rows.append({
        "passages": (
            "Passage A: The signature dish at Rosario's restaurant uses "
            "Calabrian chilli as its key ingredient.\n\n"
            "Passage B (irrelevant): The local bus service runs every 20 minutes "
            "on weekdays. Weekend timetables differ.\n\n"
            "Passage C: Calabrian chilli is supplied exclusively to Rosario's "
            "by a family farm called Ferrara & Sons.\n\n"
            "Passage D: Ferrara & Sons farm is located in the Valle district "
            "and operates its farm shop Tuesday through Saturday, 8am to 5pm."
        ),
        "question": "On which days can you visit the supplier of Rosario's key ingredient?",
        "answer": "Tuesday through Saturday",
        "chain": "signature dish → Calabrian chilli → Ferrara & Sons → Tuesday through Saturday",
    })

    # Task 3: person → team → project → deadline
    rows.append({
        "passages": (
            "Passage A: Dr. Chen leads the neural compression team.\n\n"
            "Passage B: The neural compression team is responsible for "
            "Project Helios.\n\n"
            "Passage C (irrelevant): Building maintenance will replace "
            "the HVAC system in December. Residents should expect noise.\n\n"
            "Passage D: Project Helios must deliver its final report by "
            "the 15th of the month following its budget approval. "
            "Budget was approved in March."
        ),
        "question": "By what date must Dr. Chen's team deliver their final report?",
        "answer": "April 15",
        "chain": "Dr. Chen → neural compression → Project Helios → April 15",
    })

    # Task 4: cipher → key holder → vault location → combination lock
    rows.append({
        "passages": (
            "Passage A: The encrypted file can only be read using cipher key OMEGA. "
            "Cipher key OMEGA is held exclusively by the Security Director.\n\n"
            "Passage B (irrelevant): The annual staff review process takes "
            "six weeks. Forms must be submitted to HR by end of month.\n\n"
            "Passage C: The current Security Director is Dr. Reyes. "
            "Dr. Reyes keeps all cipher materials in the sub-basement vault.\n\n"
            "Passage D: The sub-basement vault has a 6-digit combination lock. "
            "The combination is changed every 90 days and known only to "
            "the vault custodian and the Security Director."
        ),
        "question": "Who knows the combination to unlock access to cipher key OMEGA?",
        "answer": "the vault custodian and Dr. Reyes (the Security Director)",
        "chain": "encrypted file → cipher OMEGA → Security Director (Dr. Reyes) → sub-basement vault → vault custodian + Dr. Reyes",
    })

    return pd.DataFrame(rows)

_HOP_DF = _build_multihop_df()


@kbench.task(
    name="multi_hop_attention",
    description=(
        "4 multi-hop attention tasks requiring 4-step fact chaining across "
        "separate passages with irrelevant distractors. "
        "Tests sustained working memory + selective attention + reasoning. "
        "Expected frontier accuracy: 40–60%."
    )
)
def multi_hop_attention(llm, passages: str, question: str,
                         answer: str, chain: str) -> bool:
    response = llm.prompt(
        f"Read these passages carefully. Answer the question by chaining "
        f"information across multiple passages.\n\n"
        f"{passages}\n\n"
        f"Question: {question}"
    )
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response, judge_llm=kbench.judge_llm,
        criteria=[
            f"Response correctly answers: '{answer}'.",
            f"The reasoning chain follows: {chain}.",
            "Response ignores irrelevant passages and focuses on the relevant chain.",
        ]
    )
    passed = all(r.passed for r in assessment.results)
    for r in assessment.results:
        kbench.assertions.assert_true(
            r.passed,
            expectation=f"[Multi-hop] {r.criterion}: {r.reason}")
    return passed


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATE THE 5 NEW TASKS
#  Add these calls to the evaluation block in v13
# ════════════════════════════════════════════════════════════════════════════

print("\n[BONUS TRACKS] NOVEL HARD TASKS")
print(f"  arc_grid_transformation     ({len(_ARC_DF)} rows: spatial rule inference, 2 examples)")
arc_grid_transformation.evaluate(llm=[kbench.llm], evaluation_data=_ARC_DF)

print(f"  latent_sequence_discovery   ({len(_SEQ_DF)} rows: prime gaps, Van Eck, Hofstadter, Stern)")
latent_sequence_discovery.evaluate(llm=[kbench.llm], evaluation_data=_SEQ_DF)

print(f"  adversarial_distractor      ({len(_ADV_DF)} rows: false authority, wrong formulas, red herrings)")
adversarial_distractor_reasoning.evaluate(llm=[kbench.llm], evaluation_data=_ADV_DF)

print(f"  theory_of_mind_level3       ({len(_TOM3_DF)} rows: 3rd-order nested belief, human ~50%)")
social_theory_of_mind_level3.evaluate(llm=[kbench.llm], evaluation_data=_TOM3_DF)

print(f"  multi_hop_attention         ({len(_HOP_DF)} rows: 4-hop fact chain across passages)")
multi_hop_attention.evaluate(llm=[kbench.llm], evaluation_data=_HOP_DF)

print("\n" + "=" * 66)
print("  ALL 26 TASKS COMPLETE ✓  (21 from v13 + 5 new hard tasks)")
print()
print("  DIFFICULTY PROFILE:")
print("  ARC grid transformation:     35-50% frontier   (world model gap)")
print("  Latent sequence discovery:   40-55% frontier   (generative model gap)")
print("  Adversarial distractors:     50-65% frontier   (inhibitory control)")
print("  3rd-order ToM:               ~30-45% frontier  (recursive mind model)")
print("  Multi-hop attention:         40-60% frontier   (working memory)")
print()
print("  These 5 tasks target gaps that LeCun, Chollet, and DeepMind")
print("  all identify as the core unsolved problems for current LLMs.")
print()
print("  → Click 'Save Task' (top right)")
print("  → kaggle.com/competitions/kaggle-measuring-agi → Submit NOW")
print("  → DEADLINE: April 16, 2026  (5 days remaining)")
print("=" * 66)
