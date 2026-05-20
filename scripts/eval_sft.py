"""Standalone SFT 단독 평가 스크립트.

HF Hub에서 SFT 모델 다운로드 → talzoomanzoo/gsm8k test 100문제 평가.
Cell 6와 동일한 prompt + parsing 사용.

환경변수:
  HF_SFT_REPO_ID  필수. 평가할 SFT 모델 HF repo id (e.g., hwanuk16/assn2-sft)
  VESSL_RUN_NAME  결과 JSON 파일명 prefix. 기본값: "eval-sft"
"""

import json
import os
import re

from datasets import load_dataset
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams


def build_prompt(q: str) -> str:
    return (
        "Please answer the following math question.\n"
        "You should provide your final answer in the format\n"
        "\\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{q}\n"
    )


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def norm(x) -> str:
    if x is None:
        return ""
    try:
        return str(int(round(float(x))))
    except (TypeError, ValueError):
        return str(x).strip()


def extract_final(text: str):
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()
    m = list(_HASH_RE.finditer(text))
    return m[-1].group(1).strip() if m else None


def main() -> None:
    sft_repo = os.environ["HF_SFT_REPO_ID"]
    local_dir = "/tmp/sft_model"
    print(f"Downloading {sft_repo} ...", flush=True)
    snapshot_download(repo_id=sft_repo, local_dir=local_dir, repo_type="model")

    ds = load_dataset("talzoomanzoo/gsm8k", split="test").shuffle(seed=42).select(range(100))
    examples = [(ex["Question"], norm(ex["answer"])) for ex in ds]
    print(f"Loaded {len(examples)} eval examples", flush=True)

    llm = LLM(
        model=local_dir,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=768)

    prompts = [build_prompt(q) for q, _ in examples]
    outputs = llm.generate(prompts, sampling)

    correct = 0
    details = []
    for (q, gold), out in zip(examples, outputs):
        text = out.outputs[0].text
        pred_raw = extract_final(text)
        pred_n = norm(pred_raw) if pred_raw is not None else ""
        ok = pred_n == gold
        correct += int(ok)
        details.append({
            "q": q[:300],
            "pred": pred_n,
            "gold": gold,
            "ok": ok,
            "raw": text[:500],
        })
        print(f"Q: {q[:80]}...")
        print(f"  pred={pred_n}  gold={gold}  ok={ok}")
        print(f"  raw[:200]: {text[:200]}")
        print("---")

    acc = correct / len(examples)
    print(f"\n========== SFT_ACC: {acc:.4f} ==========", flush=True)

    out_path = f"/workspace/outputs/{os.environ.get('VESSL_RUN_NAME', 'eval-sft')}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"sft_acc": acc, "n": len(examples), "details": details}, f, indent=2)
    print(f"Saved details to {out_path}")


if __name__ == "__main__":
    main()
