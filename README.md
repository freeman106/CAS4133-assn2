# ASSN2: Using vLLM and OpenRLHF in training LLMs for mathematical reasoning

## Setup

### 1) Create and activate a conda env

```bash
conda create -n hw2 python=3.11 -y
conda activate hw2
python -m pip install -U pip
```

### 2) Clone OpenRLHF into this repo (editable workflow)

From the repo root (this directory):

```bash
mkdir -p third_party
git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
```

### 3) Install PyTorch + OpenRLHF **in an order that matches vLLM**

`assn2.ipynb` uses **vLLM** for fast generation and evaluation. OpenRLHF’s optional extra **`[vllm]`** installs **`vllm==0.19.1`**, which declares **exact** PyTorch versions (currently **`torch==2.10.0`**, **`torchvision==0.25.0`**, **`torchaudio==2.10.0`**). If you install an unpinned `torch` first (e.g. 2.11.x), pip will report conflicts when you add vLLM.

**Recommended (vLLM + notebook):** install the PyTorch **CUDA** build that matches your driver, **with those versions**, then install OpenRLHF with the vLLM extra:

```bash
# Pick ONE wheel index that matches your CUDA (examples: cu121, cu124, cu128 — see pytorch.org)
export TORCH_IDX=https://download.pytorch.org/whl/cu124

```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e "third_party/OpenRLHF" --no-build-isolation
```

### 4) Notebook dependencies

```bash
python -m pip install -U jupyter ipykernel transformers datasets accelerate
```

For training (GPU recommended):

```bash
python -m pip install -U deepspeed
```

### 5) Register the Jupyter kernel

```bash
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"
```

---

## Run the assignment notebook

```bash
jupyter lab
```