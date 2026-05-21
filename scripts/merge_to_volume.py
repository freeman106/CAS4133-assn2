"""Merge LoRA adapter from DPO/SimPO training into base SFT model and save
to the Vessl-storage-exported outputs directory (no HF Hub upload).

This is a workaround for the HF Hub upload_folder() hangs we observed.
The merged model is exported to /workspace/outputs/${VESSL_RUN_NAME}/merged-model
which Vessl pushes to vessl-storage at Run end. User can then download from
Vessl Volume and manually upload to HF Hub if needed.

Looks for adapter in (in order):
  /workspace/code/checkpoints/assn2-dpo
  /workspace/code/checkpoints/assn2-simpo
Loads base from:
  /workspace/code/checkpoints/assn2-sft-merged-hf
Saves merged to:
  /workspace/outputs/${VESSL_RUN_NAME}/merged-model/
"""

import gc
import json
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_adapter_dir() -> Path | None:
    """Find newest checkpoint directory with adapter_config.json."""
    ckpt_root = Path("/workspace/code/checkpoints")
    if not ckpt_root.is_dir():
        print(f"[merge_to_volume] no {ckpt_root}")
        return None

    candidates = []
    for sub in ["assn2-dpo", "assn2-simpo"]:
        root = ckpt_root / sub
        if not root.is_dir():
            continue
        # walk for adapter_config.json
        for p in [root] + list(root.rglob("*")):
            if not p.is_dir():
                continue
            if (p / "adapter_config.json").exists() and (p / "config.json").exists():
                candidates.append(p)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    base_dir = Path("/workspace/code/checkpoints/assn2-sft-merged-hf")
    if not base_dir.is_dir():
        print(f"[merge_to_volume] base SFT model missing: {base_dir} — skip.")
        return

    adapter_dir = find_adapter_dir()
    if adapter_dir is None:
        print("[merge_to_volume] no adapter found — copying base dir instead.")
        run_name = os.environ.get("VESSL_RUN_NAME", "merged")
        out_dir = Path(f"/workspace/outputs/{run_name}/merged-model")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in base_dir.iterdir():
            shutil.copy2(f, out_dir / f.name)
        print(f"[merge_to_volume] copied base -> {out_dir}")
        return

    print(f"[merge_to_volume] adapter: {adapter_dir}")
    print(f"[merge_to_volume] base:    {base_dir}")

    # Save adapter dir as-is (small, ~50MB) alongside merged version
    run_name = os.environ.get("VESSL_RUN_NAME", "merged")
    out_root = Path(f"/workspace/outputs/{run_name}")
    out_root.mkdir(parents=True, exist_ok=True)

    adapter_copy = out_root / "adapter"
    adapter_copy.mkdir(parents=True, exist_ok=True)
    for f in adapter_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, adapter_copy / f.name)
    print(f"[merge_to_volume] adapter copied -> {adapter_copy}")

    # Load + merge
    print("[merge_to_volume] loading base + adapter for merge ...", flush=True)
    tok_src = str(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    print("[merge_to_volume] merge_and_unload done.", flush=True)

    merged_out = out_root / "merged-model"
    merged_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out, safe_serialization=True)
    tokenizer.save_pretrained(merged_out)
    print(f"[merge_to_volume] merged model saved -> {merged_out}", flush=True)

    # report sizes
    total = 0
    for p in merged_out.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    print(f"[merge_to_volume] merged size: {total/1e9:.2f} GB")

    del merged, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
