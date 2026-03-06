---
title: Image Captioning
emoji: 🖼️
colorFrom: indigo
colorTo: pink
sdk: streamlit
python_version: "3.10"
app_file: app.py
pinned: false
---

[**Project Documentation**](docs/index.html) | [**Hugging Face Space**](https://huggingface.co/spaces/pchandragrid/image_captioning)

# Image Captioning with BLIP, ViT‑GPT2 & GIT

End‑to‑end project to **generate natural language descriptions of images**, compare different architectures, run controlled experiments, and deploy a public Streamlit demo backed by Hugging Face Hub.

The goal is:

- Improve **CIDEr** score by **10%+** over baseline models.
- Compare **BLIP vs ViT‑GPT2 vs GIT**.
- Study the effect of **image resolution**, **caption length**, and **decoding parameters**.
- Provide a clean **web UI** anyone can use.

---

## 1. Problem & high‑level idea

Given an input image, we want to produce a short, natural‑sounding caption like:

> “a brown dog running with a tennis ball in the grass”

We treat this as an **image‑to‑text** problem using transformer‑based vision‑language models. The project:

- Trains/fine‑tunes models on **MS COCO captions** (10k–50k subset).
- Evaluates them using **CIDEr** (via `pycocoevalcap`).
- Deploys the best model(s) in a **Streamlit app** on Hugging Face Spaces.

Key one‑liner:

> **Generate natural language descriptions of images, optimize CIDEr, and make it usable via a simple web interface.**

---

## 2. Core stack & libraries

- **PyTorch** (`torch`) – training & tensor operations.
- **Transformers** (`transformers`) – BLIP, ViT‑GPT2, GIT models and tokenizers.
- **Datasets & data tools**
  - COCO captions in JSONL format (`annotations/*.jsonl`).
  - Custom loaders in `src/data/`.
- **Image processing** – `Pillow` (`PIL`), `numpy`.
- **Evaluation** – `pycocoevalcap` (CIDEr metric).
- **Web app** – `streamlit` for UI, `matplotlib` for plots.

> Data: COCO captions via [`whyen-wang/coco_captions`](https://huggingface.co/datasets/whyen-wang/coco_captions)

---

## 3. Training recipe (from basic to advanced)

This is the “blueprint” that guided the experiments and code.

### 3.1 Basic recipe

1. **Get COCO captions**  
   - Use a **10k–50k subset** of COCO captions (JSONL with multiple captions per image).
2. **Fine‑tune BLIP or Vision‑Encoder‑Decoder**
   - Start from `Salesforce/blip-image-captioning-base` or a ViT‑GPT2 model.
3. **Train at 224–384px for ~3 epochs**
   - Begin with 224px to keep memory low; later push to 320/384px.
4. **Use gradient checkpointing**
   - Reduce memory usage, especially on **Mac MPS**.
5. **Optimization goal**
   - Achieve **10%+ improvement in CIDEr** over baseline settings.

### 3.2 Mac acceleration tips (MPS)

BLIP is memory‑heavy, especially at higher resolutions.

- Start with:
  - `batch_size = 4–8` on MPS.
  - Image size **224px** (not 384px) initially.
- Enable:
  - `model.gradient_checkpointing_enable()`
  - Mixed precision on MPS:

    ```python
    with torch.autocast(device_type="mps", dtype=torch.float16):
        ...
    ```

---

## 4. Experiments you can run

The repo is structured so you can reproduce and extend these experiments.

### 4.1 Architecture experiments

- **BLIP vs GIT vs ViT‑GPT2**
  - BLIP: vision+text in one multimodal model.
  - ViT‑GPT2: vision encoder + GPT2 decoder (cross‑attention).
  - GIT: unified transformer for image‑to‑text.
- Try different **cross‑attention patterns** and which layers are unfrozen.

### 4.2 Data preparation experiments

- Caption length:
  - **Short** captions (≤8 words).
  - **Long** captions (>15 words).
  - **Mixed** (all reasonable captions).
- Caption quality:
  - Filter out:
    - Very short captions.
    - Highly repetitive captions.
    - Captions without alphabetic characters.

These are implemented in:

- `src/data/datasets/dataset_advanced.py`
- `src/data/datasets/dataset_git.py`
- `src/data/datasets/dataset_vit_gpt2.py`

### 4.3 Decoding & parameter search

Try combinations like:

- **Beam sizes**: `3, 5, 10`
- **Length penalty**: `0.8, 1.0, 1.2`
- **Max length**: `20, 50`

Plots in `plot/` show:

- Beam size vs CIDEr.
- Caption length vs model performance.

### 4.4 Show it off

The final demo is a **Streamlit uploader**:

- Input: image.
- Output: caption(s) from BLIP / ViT‑GPT2 / GIT.
- Extras:
  - Toxicity filtering (in `app/streamlit_app.py`).
  - Attention heatmap visualization.

---

## 5. Project structure (what each part does)

```text
image-captioning/
├─ apps/
│  └─ app.py                     # Main Streamlit app (BLIP vs ViT-GPT2 vs GIT)
├─ docs/
│  └─ index.html                 # HTML overview with plots
├─ HuggingFaceUploads/
│  └─ uploadtohf.py              # Uploads local models to HF Hub
├─ outputs/
│  ├─ checkpoints/
│  │  └─ latest.pt               # Latest model checkpoint
│  ├─ models/
│  │  ├─ saved_git_model/        # Saved GIT model weights & config
│  │  ├─ saved_model_phase2/     # Saved BLIP Phase 2 model
│  │  └─ saved_vit_gpt2/         # Saved ViT-GPT2 model
│  └─ results/
│     └─ results_summary.csv     # Results and metrics summary
├─ src/
│  ├─ data/
│  │  ├─ datasets/
│  │  │  ├─ dataset_advanced.py        # Advanced COCO dataset (filtered captions)
│  │  │  ├─ dataset_git.py             # Dataset for GIT model
│  │  │  └─ dataset_vit_gpt2.py        # Dataset for ViT-GPT2
│  │  ├─ processed/
│  │  │  ├─ subset_10k.jsonl           # 10k sample subset
│  │  │  └─ subset_20k.jsonl           # 20k sample subset
│  │  └─ raw/
│  │     ├─ captions_train.jsonl       # Raw COCO training captions
│  │     ├─ captions_validation.jsonl  # Raw COCO validation captions
│  │     ├─ train2017/                 # COCO 2017 training images
│  │     └─ val2017/                   # COCO 2017 validation images
│  ├─ evaluation/
│  │  ├─ beam_search_experiments.py    # Beam search parameter experiments
│  │  ├─ evaluate.py                   # Evaluation script (CIDEr & other metrics)
│  │  └─ train_data_experiments.py     # Training data experiments
│  ├─ plot/
│  │  ├─ beam_experiment_plot.py       # Beam size vs CIDEr plot script
│  │  └─ caption_length_analysis.py    # Caption length vs performance plot
│  ├─ training/
│  │  ├─ train_git.py                  # GIT model training script
│  │  ├─ train_phase2.py               # BLIP Phase 2 training (advanced)
│  │  └─ train_vit_gpt2.py             # ViT-GPT2 training script
│  └─ utils/
│     └─ create_subset_20k.py          # Script to create 20k sample subset
├─ requirements.txt                   # Python dependencies
└─ README.md                          # This file
```

---

## 6. How the models are stored (production-friendly)

Fine‑tuned models are **not** committed to this repo. Instead they are pushed to Hugging Face:

- `pchandragrid/blip-caption-model`
- `pchandragrid/vit-gpt2-caption-model`
- `pchandragrid/git-caption-model`

`app.py` loads:

- from **local directories** (e.g. `saved_model_phase2`) if present,
- otherwise from **Hub model IDs** via `from_pretrained(...)`.

This keeps the repo small and makes deployment to **Hugging Face Spaces** and other services feasible.

---

## 7. Running locally (localhost)

### 7.1 Setup

```bash
git clone https://github.com/pchandragrid/ML-Image-Captioning.git
cd ML-Image-Captioning

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Make sure you have access to the model repos (public or via `huggingface-cli login`).

### 7.2 Run the main Streamlit app

```bash
streamlit run app.py
```

Then open the printed URL (usually `http://localhost:8501`) in your browser:

1. Upload an image.
2. In the sidebar, select which models to run:
   - BLIP (default on)
   - ViT‑GPT2 (optional)
   - GIT (optional)
3. Adjust beam size / max length if you want.
4. Click **Generate Captions**.

The app will download models from Hugging Face the first time you use them (cached afterwards).

### 7.3 Run the smaller BLIP demo (optional)

```bash
streamlit run app/streamlit_app.py
```

This version focuses on a single BLIP model with:

- Toxicity filtering.
- Confidence estimation.
- Vision attention heatmap.

---

## 8. Training scripts – how to reproduce the core BLIP model

> Note: these commands assume you have COCO-style JSONL annotations and `train2017/` images prepared.

### 8.1 Phase 1 training

```bash
source .venv/bin/activate
python -m src.training.train_phase1
```

What it does:

- Loads BLIP base model.
- Uses `COCODataset` from `src/data/dataset.py`.
- Trains for a few epochs on a subset.
- Saves to `saved_model_phase2/`.

### 8.2 Phase 2 training (advanced)

```bash
source .venv/bin/activate
python -m src.training.train_phase2
```

What it adds:

- Uses `COCODatasetAdvanced` to filter low‑quality/undesired captions.
- Uses CIDEr evaluation (`src/evaluation/cider_eval.py`) each epoch.
- Early stopping based on CIDEr.
- Saves best model to `saved_model_phase2/`.

### 8.3 Uploading models to Hugging Face

Once local training is done, you can push to your HF account:

```bash
pip install -U transformers huggingface_hub
huggingface-cli login   # once

python uploadtohf.py
```

`uploadtohf.py` is configured to push:

- BLIP → `pchandragrid/blip-caption-model`
- ViT‑GPT2 → `pchandragrid/vit-gpt2-caption-model`
- GIT → `pchandragrid/git-caption-model`

---

## 9. Deployment to Hugging Face Spaces

### 9.1 Create the Space

1. Go to Hugging Face → **Spaces** → **Create new Space**.
2. Choose:
   - Owner: `pchandragrid`
   - SDK: **Streamlit**
   - Visibility: Public
3. Create the Space (e.g. `image_captioning`).

### 9.2 Connect code to the Space

You can deploy by pushing from git:

```bash
git remote add space https://huggingface.co/spaces/pchandragrid/image_captioning
git push space main
```

If the Space already had template commits, you can push from a clean deploy branch (already done in this project).

### 9.3 Configure model IDs (optional)

`app.py` already defaults to your repos:

- `BLIP_MODEL_ID = pchandragrid/blip-caption-model`
- `VITGPT2_MODEL_ID = pchandragrid/vit-gpt2-caption-model`
- `GIT_MODEL_ID = pchandragrid/git-caption-model`

If you prefer setting them explicitly:

1. Open Space → **Settings → Variables and secrets**.
2. Add variables:
   - `BLIP_MODEL_ID`
   - `VITGPT2_MODEL_ID`
   - `GIT_MODEL_ID`
3. If model repos are private, also add:
   - `HF_TOKEN` as a **Secret**.

### 9.4 Rebuild & test

Once you push:

1. Go to the **Logs** tab of the Space.
2. Wait for:
   - Dependencies installation.
   - `streamlit run app.py`.
3. Open the Space URL (for example  
   `https://huggingface.co/spaces/pchandragrid/image_captioning`).

Upload an image and verify captions appear. BLIP is on by default; you can enable ViT‑GPT2 and GIT in the sidebar.

---

## 10. summary

- Built an **image captioning system** that generates natural language descriptions from images.
- Fine‑tuned transformer models (**BLIP**, **ViT‑GPT2**, **GIT**) on COCO‑style data.
- Ran experiments on:
  - **Architecture**: BLIP vs ViT‑GPT2 vs GIT.
  - **Resolution**: 224 → 320 → 384 px.
  - **Caption length**: short vs long vs mixed.
  - **Decoding**: beam size, length penalty, max length.
- Evaluated using **CIDEr** and improved scores by more than **10%** over baseline.
- Optimized training for **Mac MPS** (memory‑efficient training with gradient checkpointing and mixed precision).
- Deployed a **public Streamlit web app** backed by **Hugging Face Hub** so anyone can upload images and get captions in the browser.

