"""Build assn2-pref-v3: weak-strong preference data.

chosen   = Qwen2.5-Math-7B-Instruct (~90% GSM8K) correct rollouts
rejected = base Llama-3.2-1B-Instruct (~20-48% GSM8K) wrong rollouts

Two-phase vLLM generation (one model loaded at a time to fit single 24GB GPU):
  phase 1: Qwen-Math-7B → keep correct outputs → chosen pool per problem
  phase 2: Llama-1B base → keep wrong outputs → rejected pool per problem
  pair: longest correct chosen × shortest wrong rejected, length-delta filter

Output:
  data/assn2_pref_dpo.jsonl  (also uploads to HF dataset HF_PREF_REPO_ID)

Env vars:
  HF_TOKEN, HF_PREF_REPO_ID
  ASSN2_PREF_MAX_SEED          (default 2000)
  ASSN2_PREF_QWEN_SAMPLES      (default 2)
  ASSN2_PREF_BASE_SAMPLES      (default 5)
  ASSN2_PREF_MIN_LEN           (default 80)
  ASSN2_PREF_MAX_LEN           (default 2400)
  ASSN2_PREF_LEN_DELTA_MAX     (default 400)
  ASSN2_PREF_MAX_PAIRS_PER_PROBLEM (default 2)
  ASSN2_MAX_DPO_PAIRS          (default 1000)
"""

import gc
import json
import os
import re
import statistics
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from vllm import LLM, SamplingParams


# ===== Config =====
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


# ===== Reasoning helpers (mirror openrlhf/assn2/reasoning) =====
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
    """Extract final answer, supporting both '#### N' and '\\boxed{N}' formats."""
    # Prefer #### (matches our prompt template), fallback to \boxed (Qwen-Math style)
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
    """Append '#### N' to text if not present, using extracted answer."""
    if _HASH_RE.search(text):
        return text
    pred = extract_final(text)
    if pred is None:
        return text
    norm = norm_answer(pred)
    return text.rstrip() + f"\n\n#### {norm}"


def cleanup_vllm():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        import vllm.distributed.parallel_state as ps

        ps.destroy_model_parallel()
        if getattr(ps, "_WORLD", None) is not None:
            try:
                ps._WORLD.destroy()
            except Exception:
                pass
            ps._WORLD = None
    except Exception:
        pass


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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GSM8K train, MAX_SEED={MAX_SEED} ...", flush=True)
    ds = load_dataset("talzoomanzoo/gsm8k", split="train")
    ds = ds.shuffle(seed=42).select(range(min(MAX_SEED, len(ds))))
    rows = list(ds)
    questions = [r["Question"] for r in rows]
    golds = [norm_answer(r["answer"]) for r in rows]
    prompts = [build_prompt(q) for q in questions]
    print(f"Seeded {len(rows)} problems", flush=True)

    # ===== Phase 1: Qwen-Math-7B for chosen =====
    print(f"\n[Phase 1] Loading {QWEN_MODEL} for chosen generation ({NUM_QWEN_SAMPLES} samples/problem) ...", flush=True)
    llm = load_vllm(QWEN_MODEL)
    qwen_outs = generate(llm, prompts, NUM_QWEN_SAMPLES)
    del llm
    cleanup_vllm()
    print(f"[Phase 1] done.", flush=True)

    positives_per_problem = []
    qwen_correct_count = 0
    for idx, (gold, out) in enumerate(zip(golds, qwen_outs)):
        comps = [o.text for o in out.outputs]
        pos = []
        for c in comps:
            pred = extract_final(c)
            if pred is not None and norm_answer(pred) == gold:
                pos.append(ensure_hash_format(c, gold))
        if pos:
            qwen_correct_count += 1
        positives_per_problem.append(pos)
    print(f"[Phase 1] Qwen got correct on {qwen_correct_count}/{len(rows)} problems "
          f"({100*qwen_correct_count/len(rows):.1f}%)", flush=True)

    # ===== Phase 2: Llama-1B base for rejected =====
    print(f"\n[Phase 2] Loading {BASE_MODEL} for rejected generation ({NUM_BASE_SAMPLES} samples/problem) ...", flush=True)
    llm = load_vllm(BASE_MODEL)
    base_outs = generate(llm, prompts, NUM_BASE_SAMPLES)
    del llm
    cleanup_vllm()
    print(f"[Phase 2] done.", flush=True)

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
    print(f"[Phase 2] Base got wrong on {base_wrong_count}/{len(rows)} problems "
          f"({100*base_wrong_count/len(rows):.1f}%)", flush=True)

    # ===== Pair them =====
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

        pos.sort(key=len, reverse=True)  # longest correct chosen
        neg.sort(key=len)  # shortest wrong rejected

        # record labels (everything that survived filter)
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
    print(f"  Problems seen:                 {len(rows)}")
    print(f"  Qwen-Math correct (chosen pool): {qwen_correct_count}")
    print(f"  Base wrong (rejected pool):      {base_wrong_count}")
    print(f"  Skipped (no chosen):           {stat_skip_no_pos}")
    print(f"  Skipped (no rejected):         {stat_skip_no_neg}")
    print(f"  Skipped (length window):       {stat_skip_length}")
    print(f"  Pair dropped (length delta):   {stat_skip_length_delta}")
    print(f"  Problems contributing pairs:   {stat_problems_paired}")
    print(f"  DPO pairs kept:                {len(pairs)}")
    print(f"  Label rows kept:               {len(label_rows)}")
    if pairs:
        chos_lens = [len(p["chosen"]) for p in pairs]
        rej_lens = [len(p["rejected"]) for p in pairs]
        deltas = [c - r for c, r in zip(chos_lens, rej_lens)]
        print(f"  chosen   chars  mean={statistics.mean(chos_lens):.0f}  median={statistics.median(chos_lens):.0f}")
        print(f"  rejected chars  mean={statistics.mean(rej_lens):.0f}  median={statistics.median(rej_lens):.0f}")
        print(f"  delta(chosen-rejected) chars  mean={statistics.mean(deltas):+.0f}  median={statistics.median(deltas):+.0f}")
        n_chosen_shorter = sum(1 for d in deltas if d < 0)
        print(f"  pairs where chosen is shorter: {n_chosen_shorter}/{len(deltas)} "
              f"({100*n_chosen_shorter/len(deltas):.1f}%)")

    # ===== Save local =====
    print(f"\nSaving DPO pairs ({len(pairs)}) -> {OUT_JSONL}", flush=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saving label rows ({len(label_rows)}) -> {LABELS_JSONL}", flush=True)
    with LABELS_JSONL.open("w", encoding="utf-8") as f:
        for r in label_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== Upload to HF Hub =====
    pref_repo = os.environ.get("HF_PREF_REPO_ID")
    if pref_repo:
        print(f"\nUploading to HF dataset {pref_repo} ...", flush=True)
        create_repo(pref_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(
            path_or_fileobj=str(OUT_JSONL),
            path_in_repo="assn2_pref.jsonl",
            repo_id=pref_repo,
            repo_type="dataset",
            commit_message="Add weak-strong preference data (Qwen-Math chosen, Llama-1B rejected)",
        )
        print(f"Uploaded -> https://huggingface.co/datasets/{pref_repo}", flush=True)
    else:
        print("HF_PREF_REPO_ID not set — skip upload.", flush=True)


if __name__ == "__main__":
    main()
