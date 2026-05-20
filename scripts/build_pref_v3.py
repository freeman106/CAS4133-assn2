"""Build assn2-pref-v3: weak-strong preference data.

chosen   = Qwen2.5-Math-7B-Instruct (~90% GSM8K) correct rollouts
rejected = base Llama-3.2-1B-Instruct (~20-48% GSM8K) wrong rollouts

vLLM 0.19.1 cannot reliably load two models in one process (distributed state
leaks across models). So we split into two phases run as SEPARATE Python
invocations from the yaml; each phase gets a clean CUDA/distributed state.

Phases:
  --phase 1   Qwen-Math-7B → keep correct outputs → save to PHASE1_OUT
  --phase 2   Llama-1B base → keep wrong outputs → pair with phase 1 → save + upload

Outputs:
  data/assn2_pref_dpo.jsonl       (DPO/SimPO training)
  data/assn2_pref_labels.jsonl    (label data, optional)
  HF Hub dataset HF_PREF_REPO_ID  (assn2_pref.jsonl in repo)

Env vars (same defaults across phases):
  HF_TOKEN, HF_PREF_REPO_ID
  ASSN2_QWEN_MODEL, ASSN2_BASE_MODEL
  ASSN2_PREF_MAX_SEED              (default 2000)
  ASSN2_PREF_QWEN_SAMPLES          (default 2)
  ASSN2_PREF_BASE_SAMPLES          (default 5)
  ASSN2_PREF_TEMP                  (default 0.8)
  ASSN2_PREF_MAX_NEW               (default 768)
  ASSN2_PREF_MIN_LEN               (default 80)
  ASSN2_PREF_MAX_LEN               (default 2400)
  ASSN2_PREF_LEN_DELTA_MAX         (default 400)
  ASSN2_PREF_MAX_PAIRS_PER_PROBLEM (default 2)
  ASSN2_MAX_DPO_PAIRS              (default 1000)
"""

import argparse
import gc
import json
import os
import re
import statistics
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from vllm import LLM, SamplingParams


# ===== Config (read once; identical across phases) =====
QWEN_MODEL = os.environ.get("ASSN2_QWEN_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")
BASE_MODEL = os.environ.get("ASSN2_BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

MAX_SEED = int(os.environ.get("ASSN2_PREF_MAX_SEED", "2000"))
NUM_QWEN_SAMPLES = int(os.environ.get("ASSN2_PREF_QWEN_SAMPLES", "2"))
NUM_BASE_SAMPLES = int(os.environ.get("ASSN2_PREF_BASE_SAMPLES", "5"))
GEN_TEMP = float(os.environ.get("ASSN2_PREF_TEMP", "0.8"))
MAX_NEW = int(os.environ.get("ASSN2_PREF_MAX_NEW", "768"))

MIN_LEN = int(os.environ.get("ASSN2_PREF_MIN_LEN", "80"))
MAX_LEN = int(os.environ.get("ASSN2_PREF_MAX_LEN", "2400"))
LEN_DELTA_MAX = int(os.environ.get("ASSN2_PREF_LEN_DELTA_MAX", "400"))
MAX_PAIRS_PER_PROBLEM = int(os.environ.get("ASSN2_PREF_MAX_PAIRS_PER_PROBLEM", "2"))
MAX_DPO_PAIRS = int(os.environ.get("ASSN2_MAX_DPO_PAIRS", "1000"))

VLLM_TP = int(os.environ.get("ASSN2_VLLM_TP", "1"))
VLLM_GPU_MEM = float(os.environ.get("ASSN2_VLLM_GPU_MEM_UTIL", "0.85"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("ASSN2_VLLM_MAX_MODEL_LEN", "4096"))
VLLM_BATCH = int(os.environ.get("ASSN2_PREF_VLLM_BATCH", "32"))

DATA_DIR = Path(os.environ.get("ASSN2_DATA_DIR", "/workspace/code/data"))
OUT_JSONL = DATA_DIR / "assn2_pref_dpo.jsonl"
LABELS_JSONL = DATA_DIR / "assn2_pref_labels.jsonl"

# Phase 1 intermediate output (chosen pool)
PHASE1_OUT = Path(os.environ.get("ASSN2_PHASE1_OUT", "/workspace/code/data/_pref_v3_chosen.json"))


# ===== Reasoning helpers =====
def build_prompt(q: str) -> str:
    return (
        "You are a careful reasoner. Solve step-by-step. "
        "At the end, output the final answer on its own line as: '#### <answer>'\n\n"
        f"Question: {q}\n\nAnswer:\n"
    )


_HASH_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def norm_answer(x):
    if x is None:
        return ""
    try:
        return str(int(round(float(x))))
    except Exception:
        return str(x).strip()


def extract_final(text: str):
    """Extract final answer; supports both '#### N' and '\\boxed{N}' formats."""
    last_hash = None
    for m in _HASH_RE.finditer(text):
        last_hash = m
    if last_hash:
        return last_hash.group(1).strip()
    last_boxed = None
    for m in _BOXED_RE.finditer(text):
        last_boxed = m
    if last_boxed:
        return last_boxed.group(1).strip()
    return None


def ensure_hash_format(text: str, gold: str) -> str:
    """Append '#### N' to text if not already there (so training sees consistent suffix)."""
    if _HASH_RE.search(text):
        return text
    pred = extract_final(text)
    if pred is None:
        return text
    return text.rstrip() + f"\n\n#### {norm_answer(pred)}"


def load_vllm(model_id: str) -> LLM:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=VLLM_TP,
        gpu_memory_utilization=VLLM_GPU_MEM,
        max_model_len=VLLM_MAX_MODEL_LEN,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_log_stats=True,
    )


def generate(llm: LLM, prompts, n_samples: int):
    sp = SamplingParams(n=n_samples, temperature=GEN_TEMP, top_p=0.95, max_tokens=MAX_NEW)
    outs = []
    for start in range(0, len(prompts), VLLM_BATCH):
        batch = prompts[start : start + VLLM_BATCH]
        outs.extend(llm.generate(batch, sp))
    return outs


def load_seed_problems():
    print(f"Loading GSM8K train, MAX_SEED={MAX_SEED} ...", flush=True)
    ds = load_dataset("talzoomanzoo/gsm8k", split="train")
    ds = ds.shuffle(seed=42).select(range(min(MAX_SEED, len(ds))))
    rows = list(ds)
    questions = [r["Question"] for r in rows]
    golds = [norm_answer(r["answer"]) for r in rows]
    prompts = [build_prompt(q) for q in questions]
    print(f"Seeded {len(rows)} problems", flush=True)
    return questions, golds, prompts


# ===== Phase 1: Qwen-Math-7B → chosen =====
def phase1() -> None:
    questions, golds, prompts = load_seed_problems()

    print(f"\n[Phase 1] Loading {QWEN_MODEL} ({NUM_QWEN_SAMPLES} samples/problem) ...", flush=True)
    llm = load_vllm(QWEN_MODEL)
    qwen_outs = generate(llm, prompts, NUM_QWEN_SAMPLES)
    print(f"[Phase 1] generation done.", flush=True)

    positives_per_problem = []
    qwen_correct_count = 0
    for gold, out in zip(golds, qwen_outs):
        comps = [o.text for o in out.outputs]
        pos = []
        for c in comps:
            pred = extract_final(c)
            if pred is not None and norm_answer(pred) == gold:
                pos.append(ensure_hash_format(c, gold))
        if pos:
            qwen_correct_count += 1
        positives_per_problem.append(pos)

    pct = 100 * qwen_correct_count / max(1, len(golds))
    print(f"[Phase 1] Qwen correct on {qwen_correct_count}/{len(golds)} problems ({pct:.1f}%)", flush=True)

    PHASE1_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": questions,
        "golds": golds,
        "prompts": prompts,
        "positives_per_problem": positives_per_problem,
        "qwen_correct_count": qwen_correct_count,
    }
    PHASE1_OUT.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"[Phase 1] saved {PHASE1_OUT} ({PHASE1_OUT.stat().st_size/1e6:.1f} MB)", flush=True)


# ===== Phase 2: Llama-1B → rejected, then pair =====
def phase2() -> None:
    if not PHASE1_OUT.is_file():
        raise FileNotFoundError(f"Phase 1 output missing: {PHASE1_OUT}; run --phase 1 first")
    print(f"\n[Phase 2] Loading phase 1 payload from {PHASE1_OUT} ...", flush=True)
    payload = json.loads(PHASE1_OUT.read_text())
    questions = payload["questions"]
    golds = payload["golds"]
    prompts = payload["prompts"]
    positives_per_problem = payload["positives_per_problem"]
    qwen_correct_count = payload.get("qwen_correct_count", 0)
    print(f"[Phase 2] loaded {len(prompts)} problems; Qwen correct={qwen_correct_count}", flush=True)

    print(f"\n[Phase 2] Loading {BASE_MODEL} ({NUM_BASE_SAMPLES} samples/problem) ...", flush=True)
    llm = load_vllm(BASE_MODEL)
    base_outs = generate(llm, prompts, NUM_BASE_SAMPLES)
    print(f"[Phase 2] generation done.", flush=True)

    negatives_per_problem = []
    base_wrong_count = 0
    for gold, out in zip(golds, base_outs):
        comps = [o.text for o in out.outputs]
        neg = []
        for c in comps:
            pred = extract_final(c)
            if pred is None or norm_answer(pred) != gold:
                neg.append(c)
        if neg:
            base_wrong_count += 1
        negatives_per_problem.append(neg)
    print(f"[Phase 2] Base wrong on {base_wrong_count}/{len(golds)} problems "
          f"({100*base_wrong_count/max(1,len(golds)):.1f}%)", flush=True)

    # ===== Pair =====
    pairs = []
    label_rows = []
    stat_skip_no_pos = 0
    stat_skip_no_neg = 0
    stat_skip_length = 0
    stat_skip_length_delta = 0
    stat_problems_paired = 0

    for prompt, gold, pos, neg in zip(prompts, golds, positives_per_problem, negatives_per_problem):
        if not pos:
            stat_skip_no_pos += 1
            continue
        if not neg:
            stat_skip_no_neg += 1
            continue

        pos = [p for p in pos if MIN_LEN <= len(p) <= MAX_LEN]
        neg = [n for n in neg if MIN_LEN <= len(n) <= MAX_LEN]
        if not pos or not neg:
            stat_skip_length += 1
            continue

        pos.sort(key=len, reverse=True)
        neg.sort(key=len)

        for p in pos:
            label_rows.append({"prompt": prompt, "response": p, "label": True, "answer": gold})
        for n in neg:
            label_rows.append({"prompt": prompt, "response": n, "label": False, "answer": gold})

        kept_for_problem = 0
        for p in pos:
            if kept_for_problem >= MAX_PAIRS_PER_PROBLEM:
                break
            for n in neg:
                if kept_for_problem >= MAX_PAIRS_PER_PROBLEM:
                    break
                if len(pairs) >= MAX_DPO_PAIRS:
                    break
                if len(p) < len(n) - LEN_DELTA_MAX:
                    stat_skip_length_delta += 1
                    continue
                pairs.append({"prompt": prompt, "chosen": p, "rejected": n, "answer": gold})
                kept_for_problem += 1
            if len(pairs) >= MAX_DPO_PAIRS:
                break
        if kept_for_problem > 0:
            stat_problems_paired += 1
        if len(pairs) >= MAX_DPO_PAIRS:
            break

    # ===== Stats =====
    print("\n=== Preference v3 (weak-strong) stats ===", flush=True)
    print(f"  Problems seen:                 {len(prompts)}")
    print(f"  Qwen correct (chosen pool):    {qwen_correct_count}")
    print(f"  Base wrong  (rejected pool):   {base_wrong_count}")
    print(f"  Skipped (no chosen):           {stat_skip_no_pos}")
    print(f"  Skipped (no rejected):         {stat_skip_no_neg}")
    print(f"  Skipped (length window):       {stat_skip_length}")
    print(f"  Pair dropped (length delta):   {stat_skip_length_delta}")
    print(f"  Problems contributing pairs:   {stat_problems_paired}")
    print(f"  DPO pairs kept:                {len(pairs)}")
    print(f"  Label rows kept:               {len(label_rows)}")
    if pairs:
        chos = [len(p["chosen"]) for p in pairs]
        rej = [len(p["rejected"]) for p in pairs]
        deltas = [c - r for c, r in zip(chos, rej)]
        print(f"  chosen   chars  mean={statistics.mean(chos):.0f}  median={statistics.median(chos):.0f}")
        print(f"  rejected chars  mean={statistics.mean(rej):.0f}  median={statistics.median(rej):.0f}")
        print(f"  delta(chosen-rejected) chars  mean={statistics.mean(deltas):+.0f}  median={statistics.median(deltas):+.0f}")
        nshort = sum(1 for d in deltas if d < 0)
        print(f"  pairs where chosen is shorter: {nshort}/{len(deltas)} ({100*nshort/len(deltas):.1f}%)")

    # ===== Save local =====
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving DPO pairs ({len(pairs)}) -> {OUT_JSONL}", flush=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saving label rows ({len(label_rows)}) -> {LABELS_JSONL}", flush=True)
    with LABELS_JSONL.open("w", encoding="utf-8") as f:
        for r in label_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== HF upload =====
    pref_repo = os.environ.get("HF_PREF_REPO_ID")
    if pref_repo:
        print(f"\nUploading to HF dataset {pref_repo} ...", flush=True)
        create_repo(pref_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(
            path_or_fileobj=str(OUT_JSONL),
            path_in_repo="assn2_pref.jsonl",
            repo_id=pref_repo,
            repo_type="dataset",
            commit_message="Add weak-strong pref data (Qwen-Math chosen, Llama-1B rejected)",
        )
        print(f"Uploaded -> https://huggingface.co/datasets/{pref_repo}", flush=True)
    else:
        print("HF_PREF_REPO_ID not set — skip upload.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    args = ap.parse_args()
    if args.phase == 1:
        phase1()
    else:
        phase2()


if __name__ == "__main__":
    main()
