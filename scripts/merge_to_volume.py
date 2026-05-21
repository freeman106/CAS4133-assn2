"""Export DPO/SimPO training output + merged model to vessl-storage.

Strategy: list everything under /workspace/code/checkpoints, copy it to
/workspace/outputs/{run_name}/checkpoints/ via `cp -r`, then ALSO attempt
LoRA merge if an adapter is found. If merge fails or no adapter found,
the raw checkpoint dump is enough for the user to download and inspect.

The previous version (a) only looked for `adapter_config.json` at depth-2 and
missed cases where OpenRLHF saved the merged HF model directly, (b) used
`shutil.copy2` which crashes on directories like `.cache/`. This version
uses `subprocess` + `cp -r` for raw dump and only attempts merge when both
an adapter dir and a base dir are clearly identified.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path


CKPT_ROOT = Path("/workspace/code/checkpoints")
RUN_NAME = os.environ.get("VESSL_RUN_NAME", "merged")
OUT_ROOT = Path(f"/workspace/outputs/{RUN_NAME}")
OUT_CKPT = OUT_ROOT / "checkpoints-raw"
OUT_MERGED = OUT_ROOT / "merged-model"


def run(cmd: list[str]) -> int:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not CKPT_ROOT.is_dir():
        print(f"[merge] no {CKPT_ROOT} — nothing to do", flush=True)
        return

    print(f"[merge] inventory of {CKPT_ROOT}:", flush=True)
    run(["ls", "-laR", str(CKPT_ROOT)])

    print(f"\n[merge] copying raw checkpoint tree -> {OUT_CKPT}", flush=True)
    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    rc = run(["cp", "-r", f"{CKPT_ROOT}/.", str(OUT_CKPT)])
    if rc != 0:
        print(f"[merge] cp -r returned {rc} (continuing)", flush=True)
    run(["du", "-sh", str(OUT_CKPT)])

    # Try to find adapter for explicit merge
    base_dir = CKPT_ROOT / "assn2-sft-merged-hf"
    adapter_dir = None
    for sub in ["assn2-dpo", "assn2-simpo"]:
        root = CKPT_ROOT / sub
        if not root.is_dir():
            continue
        # find any subdir (or itself) containing adapter_config.json
        for p in [root] + sorted(root.rglob("*"), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
            if p.is_dir() and (p / "adapter_config.json").exists():
                adapter_dir = p
                break
        if adapter_dir is not None:
            break

    if adapter_dir is None or not base_dir.is_dir():
        print(f"[merge] no adapter found OR base missing — raw checkpoint dump only.", flush=True)
        print(f"        adapter_dir={adapter_dir}", flush=True)
        print(f"        base_dir={base_dir}", flush=True)
        return

    print(f"\n[merge] attempting merge:", flush=True)
    print(f"  adapter: {adapter_dir}", flush=True)
    print(f"  base:    {base_dir}", flush=True)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"[merge] import failed: {e} — raw dump only.", flush=True)
        return

    try:
        tok_src = str(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_dir)
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

        base = AutoModelForCausalLM.from_pretrained(
            str(base_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

        OUT_MERGED.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(OUT_MERGED, safe_serialization=True)
        tokenizer.save_pretrained(OUT_MERGED)
        print(f"[merge] merged model saved -> {OUT_MERGED}", flush=True)
        run(["du", "-sh", str(OUT_MERGED)])

        del merged, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[merge] merge failed: {type(e).__name__}: {e}", flush=True)
        print(f"        raw checkpoint dump in {OUT_CKPT} is still available.", flush=True)


if __name__ == "__main__":
    main()
