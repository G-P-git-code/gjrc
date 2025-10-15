#!/usr/bin/env python3
# AIME 2024 — CoT + Code (single-shot) + Tree-of-Thought (branching) with cosine consensus
# - CoT-ToT: branching beam search over partial solutions; nodes scored by text cosine to problem
# - Code-ToT: branching via plan -> code expansion; predictions aggregated
# - Consensus: instead of majority vote, choose the integer whose vector is closest (cosine) to the centroid

import sys
import re
import io
import math
import fractions
import itertools
import collections
import functools
import traceback
from typing import Dict, Optional, List, Tuple
from contextlib import redirect_stdout
from collections import Counter, defaultdict

from vllm import LLM, SamplingParams

# -------- Load dataset (AIME 2024 only) --------
try:
    from aime_problems import AIME_2024_PROBLEMS  # list of {"question": str, "answer": str}
    print(f"✓ Loaded {len(AIME_2024_PROBLEMS)} real AIME 2024 problems")
except Exception as e:
    print(f"⚠️  WARNING: Could not load real AIME problems ({e}), using fallback dummy problems")
    AIME_2024_PROBLEMS = [
        {"question": "Compute 1+2+...+10.", "answer": "55"},
        {"question": "If 2x+3=17, find x.", "answer": "7"},
        {"question": "Find the remainder when 7^5 is divided by 100.", "answer": "7"},
    ]

# ========================= Utilities =========================

def _coerce_int(x) -> Optional[int]:
    try:
        if isinstance(x, bool): return None
        if isinstance(x, int):  return x
        if isinstance(x, float):
            r = round(x)
            return r if abs(x - r) < 1e-9 else None
        return int(str(x).strip())
    except Exception:
        return None

def _normalize_aime_int(x: Optional[int]) -> Optional[int]:
    """Valid AIME integer [0, 999] or None."""
    if x is None: return None
    try:
        xi = int(x)
        return xi if 0 <= xi <= 999 else None
    except Exception:
        return None

def extract_final_answer(text: str) -> Optional[int]:
    """Prefer a final 'Final Answer:' line; else scan last ~15 lines for a 0-999."""
    patt = [
        r"\*\*Final Answer[:\s]*([0-9]{1,3})\*\*",
        r"^\s*Final Answer[:\s]*([0-9]{1,3})\s*$",
        r"The final answer is[:\s]*([0-9]{1,3})",
        r"Answer[:\s]*([0-9]{1,3})",
    ]
    for p in patt:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            v = _normalize_aime_int(_coerce_int(m.group(1)))
            if v is not None: return v
    tail = "\n".join(text.strip().splitlines()[-15:])
    nums = re.findall(r"[0-9]{1,3}", tail)
    if nums:
        return _normalize_aime_int(_coerce_int(nums[-1]))
    return None

def extract_code_block(text: str) -> Optional[str]:
    """Custom markers preferred; fall back to ```python fences; else heuristic."""
    m = re.search(r"PYCODE:\s*\n(.*?)\nENDPYCODE", text, flags=re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    return collect_codey_lines(text)

def collect_codey_lines(text: str) -> Optional[str]:
    kws = ("import", "def ", "for ", "while ", "if ", "=", "print", "return")
    lines, in_block = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if (not in_block) and any(k in s for k in kws):
            in_block = True
        if in_block and s:
            lines.append(ln)
    code = "\n".join(lines).strip()
    return code if len(code) >= 4 else None

def extract_predicted_output(text: str) -> Optional[int]:
    m = re.search(r"^__PREDICTED_OUTPUT__\s*=\s*([0-9]{1,3})\s*$", text.strip(), flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def extract_cot_commit(text: str) -> Optional[int]:
    m = re.search(r"^__COT_ANS__\s*=\s*([0-9]{1,3})\s*$", text.strip(), flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def extract_direct_sentinel(text: str) -> Optional[int]:
    m = re.search(r"^__ANS__\s*=\s*([0-9]{1,3})\s*$", text, flags=re.MULTILINE)
    return _normalize_aime_int(_coerce_int(m.group(1))) if m else None

def strict_single_sentinel(stdout_text: str) -> Optional[int]:
    lines = [ln.strip() for ln in stdout_text.strip().splitlines() if ln.strip()]
    hits = [ln for ln in lines if re.fullmatch(r"__ANS__\s*=\s*[0-9]{1,3}", ln)]
    if not hits: return None
    return _normalize_aime_int(_coerce_int(hits[-1].split("=")[-1]))

def maybe_execute_for_last_resort(code_src: Optional[str]) -> Optional[int]:
    """Very conservative local exec; used rarely (tie-break)."""
    if not code_src: return None
    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "sum": sum, "max": max, "min": min,
            "abs": abs, "round": round, "pow": pow, "print": print,
            "enumerate": enumerate, "zip": zip, "sorted": sorted,
            "reversed": reversed, "all": all, "any": any,
        },
        "math": math,
        "fractions": fractions,
        "itertools": itertools,
        "collections": collections,
        "functools": functools,
        "__name__": "__main__",
    }
    out_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf):
            exec(code_src, safe_globals, {})
        printed = out_buf.getvalue()
        return strict_single_sentinel(printed)
    except Exception:
        return None

# --------- Cosine utilities (text + answer-consensus) ---------

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def _tf_vector(s: str) -> Dict[str, float]:
    toks = _tokenize(s)
    if not toks: return {}
    c = Counter(toks)
    n = float(sum(c.values()))
    return {t: cnt / n for t, cnt in c.items()}

def cosine_text(a: str, b: str) -> float:
    va, vb = _tf_vector(a), _tf_vector(b)
    if not va or not vb: return 0.0
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v*v for v in va.values()))
    nb = math.sqrt(sum(v*v for v in vb.values()))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

def _answer_vec(n: int) -> List[float]:
    """Stable, low-dim numeric features for cosine consensus."""
    # bias to avoid zero vector + scaled views of the number
    h, t, o = n // 100, (n // 10) % 10, n % 10
    return [1.0, n/999.0, h/9.0, t/9.0, o/9.0]

def _cosine(u: List[float], v: List[float]) -> float:
    dot = sum(x*y for x, y in zip(u, v))
    nu = math.sqrt(sum(x*x for x in u))
    nv = math.sqrt(sum(y*y for y in v))
    return 0.0 if nu == 0.0 or nv == 0.0 else dot/(nu*nv)

def cosine_select_answer(cands: List[int]) -> Optional[int]:
    """Pick the candidate whose vector is most similar (cosine) to the centroid (consensus)."""
    if not cands: return None
    vecs = [_answer_vec(c) for c in cands]
    # centroid
    m = [sum(v[i] for v in vecs)/len(vecs) for i in range(len(vecs[0]))]
    sims = [(_cosine(v, m), c) for v, c in zip(vecs, cands)]
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[0][1]

# ========================= Model Wrapper =========================

class QwenRunner:
    def __init__(self, model_path: str = "openai/gpt-oss-120b"):
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            dtype="bfloat16",
            enforce_eager=False,
            disable_log_stats=True,
            enable_prefix_caching=True,
            max_num_seqs=16,
        )
        # Deterministic single
        self.params_single = SamplingParams(
            temperature=0.0, top_p=1.0, repetition_penalty=1.05,
            n=1, max_tokens=1600,
        )
        print("✓ Model loaded\n")
        try:
            self.tok = self.llm.get_tokenizer()
        except Exception:
            self.tok = None

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.tok is not None and hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # fallback
        prompt = ""
        for m in messages:
            role = m.get("role", "user"); content = m.get("content", "")
            prompt += f"<{role}>\n{content}\n</{role}>\n"
        prompt += "<assistant>\n"
        return prompt

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        out = self.llm.generate([self._build_prompt(messages)], self.params_single)[0]
        return (out.outputs[0].text or "").strip() if out.outputs else ""

    def generate_chat_multi(self, messages: List[Dict[str, str]],
                            n: int, temperature: float, top_p: float,
                            max_tokens: int) -> List[str]:
        params = SamplingParams(
            temperature=temperature, top_p=top_p, repetition_penalty=1.0,
            n=n, max_tokens=max_tokens
        )
        out = self.llm.generate([self._build_prompt(messages)], params)[0]
        return [(o.text or "").strip() for o in out.outputs if (o.text or "").strip()]

# ========================= Prompts =========================

def cot_prompt(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are an AIME solver. Be correct, concise, and deterministic."},
        {"role": "user", "content": f"Solve the AIME problem with compact steps and end with exactly one line: Final Answer: n\n\nProblem:\n{problem}"},
    ]

def tot_cot_expand_prompt(problem: str, scratch: str) -> List[Dict[str, str]]:
    """Ask for either a next step OR a final line; used to branch."""
    return [
        {"role": "system", "content": "You are an AIME solver expanding a partial solution. Keep steps short, precise, symbolic when helpful."},
        {"role": "user", "content": f"""Problem:
{problem}

Current scratch:
{scratch if scratch.strip() else "(none yet)"}

Write ONE of the following:
- A single next step (as a short sentence or equation), or
- If you can finish now, output exactly: Final Answer: n (0–999)

Do NOT add any commentary outside that single step or final line."""}
    ]

def code_plan_prompt(problem: str, scratch: str) -> List[Dict[str, str]]:
    """Branch into concrete strategies for coding; optionally provide a candidate integer."""
    return [
        {"role": "system", "content": "Propose a precise plan to compute the AIME answer; keep it actionable for code."},
        {"role": "user", "content": f"""Problem:
{problem}

Current plan context:
{scratch if scratch.strip() else "(none yet)"}

Respond with:
PLAN: <one-sentence exact strategy including key formulas or search bounds>
(Optional) CAND: <0–999>  # if you are confident

No extra text."""}
    ]

def code_only_prompt_from_plan(problem: str, plan: str) -> List[Dict[str, str]]:
    """Emit final code from the chosen plan using the stricter code scaffold."""
    user_text = f"""Use the plan below to generate code that prints exactly one line "__ANS__=<int>" (0–999).
Plan:
{plan}

Output:
PYCODE:
import math
from fractions import Fraction
from itertools import product, permutations, combinations

def verify(ans):
    return isinstance(ans, int) and 0 <= ans <= 999

def solve():
    ans = 0
    # Implement the plan exactly; use integer/Fraction arithmetic; bounded search if needed
    if not verify(ans):
        best = None
        for x in range(0, 1000):
            cand = x
            if verify(cand):
                best = cand; break
        ans = 0 if best is None else int(best)
    assert verify(ans)
    return ans

answer = solve()
print("__ANS__=" + str(int(answer)))
ENDPYCODE

__PREDICTED_OUTPUT__=<int>  # your simulated program output
"""
    return [
        {"role": "system", "content": "You are an AIME solver. Follow the format strictly; one print line in code."},
        {"role": "user", "content": user_text},
    ]

# ========================= Single-shot Evaluation =========================

def evaluate_single(runner: QwenRunner, prob: Dict, index: int) -> Dict:
    q = prob["question"]; true_ans = int(prob["answer"])
    cot_text = runner.generate_chat(cot_prompt(q))
    cot_pred = extract_final_answer(cot_text)

    # Simple one-shot code generation via plan->code (kept minimal)
    plan = runner.generate_chat(code_plan_prompt(q, scratch=""))
    code_text = runner.generate_chat(code_only_prompt_from_plan(q, plan))
    po = extract_predicted_output(code_text)
    direct = extract_direct_sentinel(code_text)
    code_pred = po if po is not None else direct

    print(f"[TRACE P{index:02d}] (single) CoT={cot_pred}  Code={code_pred}  plan='{plan[:80]}...'")
    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans) if cot_pred is not None else False,
        "code_ok": (code_pred == true_ans) if code_pred is not None else False,
    }

# ========================= ToT (Branching) =========================

def tot_cot_branching(runner: QwenRunner, problem: str, k: int, depth: int) -> Tuple[Optional[int], List[int]]:
    """Beam search over partial CoT steps; return consensus by cosine among predicted integers."""
    problem_txt = problem
    BeamItem = collections.namedtuple("BeamItem", ["scratch", "score"])
    beam: List[BeamItem] = [BeamItem(scratch="", score=0.0)]
    found_answers: List[int] = []

    for d in range(1, depth+1):
        next_items: List[BeamItem] = []
        print(f"    [CoT-Branch] depth {d}/{depth} | beam={len(beam)}")
        for item in beam:
            outs = runner.generate_chat_multi(
                tot_cot_expand_prompt(problem_txt, item.scratch),
                n=k, temperature=0.7, top_p=0.9, max_tokens=200
            )
            for out in outs:
                ans = extract_final_answer(out)
                if ans is not None:
                    found_answers.append(ans)
                else:
                    # extend scratch
                    new_scratch = (item.scratch + ("\n" if item.scratch else "") + out).strip()
                    sc = cosine_text(new_scratch, problem_txt)
                    next_items.append(BeamItem(new_scratch, sc))

        if next_items:
            next_items.sort(key=lambda z: z.score, reverse=True)
            beam = next_items[:k]  # beam width = k
        else:
            # no further steps; break early if only answers came back
            if found_answers:
                break

    if found_answers:
        # Use cosine consensus over integers (no majority vote)
        choice = cosine_select_answer(found_answers)
        return choice, found_answers
    return None, found_answers

def tot_code_branching(runner: QwenRunner, problem: str, k: int, depth: int) -> Tuple[Optional[int], List[int]]:
    """Two-stage branching: (1) expand multiple plans; (2) generate code from top plans; select by cosine consensus."""
    # Stage 1: plan branching
    PlanItem = collections.namedtuple("PlanItem", ["plan", "score", "cand"])
    beam: List[PlanItem] = [PlanItem(plan="", score=0.0, cand=None)]
    for d in range(1, depth+1):
        next_items: List[PlanItem] = []
        print(f"    [Code-Plan] depth {d}/{depth} | beam={len(beam)}")
        for item in beam:
            outs = runner.generate_chat_multi(
                code_plan_prompt(problem, item.plan),
                n=k, temperature=0.65, top_p=0.9, max_tokens=180
            )
            for out in outs:
                # parse
                mplan = re.search(r"PLAN:\s*(.+)", out)
                mcand = re.search(r"CAND:\s*([0-9]{1,3})", out)
                plan = (mplan.group(1).strip() if mplan else out.strip())
                cand = _normalize_aime_int(_coerce_int(mcand.group(1))) if mcand else None
                sc = cosine_text(plan, problem)
                next_items.append(PlanItem(plan=plan, score=sc, cand=cand))
        if next_items:
            next_items.sort(key=lambda z: z.score, reverse=True)
            beam = next_items[:k]
        else:
            break

    # Stage 2: code from each surviving plan (one child per plan to control cost)
    preds: List[int] = []
    for plan_item in beam:
        code_text = runner.generate_chat(code_only_prompt_from_plan(problem, plan_item.plan))
        po = extract_predicted_output(code_text)
        di = extract_direct_sentinel(code_text)
        ans = po if po is not None else di
        if ans is not None:
            preds.append(ans)
        elif plan_item.cand is not None:
            preds.append(plan_item.cand)

    if preds:
        choice = cosine_select_answer(preds)
        return choice, preds
    return None, preds

# ========================= Driver =========================

def evaluate_problem_tot(runner: QwenRunner, prob: Dict, index: int, k: int, depth: int = 3) -> Dict:
    q = prob["question"]; true_ans = int(prob["answer"])

    cot_pred, cot_all = tot_cot_branching(runner, q, k=k, depth=depth)
    code_pred, code_all = tot_code_branching(runner, q, k=k, depth=max(2, depth-1))

    print(f"[TRACE P{index:02d}] (ToT-Branch) COT cands={cot_all} → pick={cot_pred} | CODE cands={code_all} → pick={code_pred}")

    return {
        "true": true_ans,
        "cot_pred": cot_pred,
        "code_pred": code_pred,
        "cot_ok": (cot_pred == true_ans) if cot_pred is not None else False,
        "code_ok": (code_pred == true_ans) if code_pred is not None else False,
    }

def main():
    use_tot = "--tot" in sys.argv
    # k = branching factor and beam width (2..20)
    if "--k" in sys.argv:
        try:
            k = int(sys.argv[sys.argv.index("--k") + 1])
            if k < 2 or k > 20:
                print("K out of range [2,20]; defaulting to 6"); k = 6
        except Exception:
            k = 6
    else:
        k = 6
    # Optional depth
    if "--depth" in sys.argv:
        try:
            depth = int(sys.argv[sys.argv.index("--depth") + 1])
            depth = max(2, min(6, depth))
        except Exception:
            depth = 3
    else:
        depth = 3

    mode = f"ToT-Branching (beam={k}, depth={depth})" if use_tot else "Single-shot"
    print(f"🔥 AIME 2024 — CoT + Code — Mode: {mode}")
    print("=" * 70)

    runner = QwenRunner("openai/gpt-oss-120b")
    problems = AIME_2024_PROBLEMS

    cot_correct = 0
    code_correct = 0
    total = len(problems)

    for i, prob in enumerate(problems, 1):
        if use_tot:
            res = evaluate_problem_tot(runner, prob, i, k=k, depth=depth)
        else:
            res = evaluate_single(runner, prob, i)
        cot_correct += int(res["cot_ok"])
        code_correct += int(res["code_ok"])
        print(f"Problem {i:02d}: "
              f"COT={'✓' if res['cot_ok'] else '✗'}(pred={res['cot_pred']}, true={res['true']})  |  "
              f"CODE={'✓' if res['code_ok'] else '✗'}(pred={res['code_pred']}, true={res['true']})")

    cot_acc = 100.0 * cot_correct / total if total else 0.0
    code_acc = 100.0 * code_correct / total if total else 0.0
    print("\n=== FINAL ACCURACY ===")
    print(f"COT : {cot_correct}/{total}  = {cot_acc:.2f}%")
    print(f"CODE: {code_correct}/{total}  = {code_acc:.2f}%")
    print("======================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
