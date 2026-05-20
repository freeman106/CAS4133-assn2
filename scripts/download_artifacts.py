"""Download SFT model + preference dataset from HF Hub.

Reads env vars:
  HF_SFT_REPO_ID   e.g. hwanuk16/assn2-sft-v3
  HF_PREF_REPO_ID  e.g. hwanuk16/assn2-pref-v2

Writes:
  /workspace/code/checkpoints/assn2-sft-merged-hf/   (SFT merged model)
  /workspace/code/data/assn2_pref_dpo.jsonl          (preference dataset)
"""
import os
import shutil

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    sft_repo = os.environ["HF_SFT_REPO_ID"]
    sft_dir = "/workspace/code/checkpoints/assn2-sft-merged-hf"
    os.makedirs(sft_dir, exist_ok=True)
    print(f"Downloading SFT model {sft_repo} -> {sft_dir} ...", flush=True)
    snapshot_download(repo_id=sft_repo, local_dir=sft_dir, repo_type="model")
    print(f"Downloaded SFT model.", flush=True)

    pref_repo = os.environ["HF_PREF_REPO_ID"]
    data_dir = "/workspace/code/data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading preference dataset {pref_repo}/assn2_pref.jsonl ...", flush=True)
    src = hf_hub_download(repo_id=pref_repo, filename="assn2_pref.jsonl", repo_type="dataset")
    dst = os.path.join(data_dir, "assn2_pref_dpo.jsonl")
    shutil.copy(src, dst)
    print(f"Downloaded preference dataset -> {dst}", flush=True)


if __name__ == "__main__":
    main()
