# -*- coding: utf-8 -*-
"""
General Benchmark Inference (Methodology-Aligned, Full Rewrite)

Features:
 - Models: BLIP-base, BLIP-2 (OPT-2.7B-ready), GIT (incl. COCO variant), GPT-2, CLIP (ViT-B/32, ViT-L/14)
 - Decoding strategies: greedy, beam (num_beams from config), nucleus (top_p from config)
 - Candidate generation + CLIP re-ranking (select best of N by CLIPScore)
 - Baselines: Frequency baseline, Nearest-Neighbor baseline (CLIP image similarity)
 - Datasets: JSON annotation with {"file_path":..., "sentences":[{"raw":...}], "split": "train/val/test"}
 - Metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr,  CLIPScore (image-text, reference-free)
 - Efficiency: per-caption latency, per-run peak VRAM, avg throughput
 - Config-driven: see benchmark_config.yaml (same name, extended fields are optional)

Notes:
 - Works best with locally cached models. Set each model's "local_dir" in YAML.
 - CLIPScore (image-text) uses torchmetrics; will download a model if not cached.
"""

import os
import time
import json
import yaml
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- NLTK + metrics ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


try:
    # reference-free CLIPScore: image-text similarity
    from torchmetrics.multimodal.clip_score import CLIPScore as TMCLIPScore
    _HAS_TM_CLIPSCORE = True
except Exception:
    _HAS_TM_CLIPSCORE = False

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM,
    AutoTokenizer, AutoModelForSeq2SeqLM,  # keep for extensibility
    CLIPProcessor, CLIPModel
)

# ======================
# Config
# ======================

def load_config(path="benchmark_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ======================
# Utils
# ======================

def first_param_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ======================
# Decoding helpers
# ======================

def generate_with_strategy(model, inputs, *, strategy="greedy", max_new_tokens=30, num_beams=3, top_p=0.9):
    if strategy == "greedy":
        return model.generate(**inputs, max_new_tokens=max_new_tokens)
    elif strategy == "beam":
        return model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, early_stopping=True)
    elif strategy == "nucleus":
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")


# ======================
# Captioner classes
# ======================

class BLIPCaptioner:
    def __init__(self, local_dir, device="cuda", fp16=True):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(local_dir, local_files_only=True)
        dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
        self.model = BlipForConditionalGeneration.from_pretrained(
            local_dir, torch_dtype=dtype, local_files_only=True
        ).to(device).eval()

    def generate_one(self, image, *, strategy="greedy", max_new_tokens=30, num_beams=3, top_p=0.9):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = generate_with_strategy(self.model, inputs, strategy=strategy,
                                         max_new_tokens=max_new_tokens, num_beams=num_beams, top_p=top_p)
        return self.processor.decode(out[0], skip_special_tokens=True)


class BLIP2Captioner:
    def __init__(self, local_dir, device="cuda", prefer_8bit=False):
        self.device = device
        self.processor = Blip2Processor.from_pretrained(local_dir, local_files_only=True)
        self.model = None
        # Try 8-bit (bitsandbytes), else fp16 auto-map, else cpu fp32
        if prefer_8bit:
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    local_dir,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            except Exception:
                self.model = None
        if self.model is None:
            if torch.cuda.is_available():
                try:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        local_dir, torch_dtype=torch.float16, device_map=None
                    ).to(device)
                except RuntimeError:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        local_dir, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
                    )
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(local_dir, torch_dtype=torch.float32)
        self.model.eval()

    def generate_one(self, image, *, strategy="greedy", max_new_tokens=30, num_beams=3, top_p=0.9):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = generate_with_strategy(self.model, inputs, strategy=strategy,
                                         max_new_tokens=max_new_tokens, num_beams=num_beams, top_p=top_p)
        return self.processor.decode(out[0], skip_special_tokens=True)


class GITCaptioner:
    def __init__(self, local_dir, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True).to(device).eval()

    def generate_one(self, image, *, strategy="greedy", max_new_tokens=30, num_beams=3, top_p=0.9):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = generate_with_strategy(self.model, inputs, strategy=strategy,
                                         max_new_tokens=max_new_tokens, num_beams=num_beams, top_p=top_p)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


class GPT2Captioner:
    def __init__(self, local_dir, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True).to(device).eval()

    def generate_one(self, image=None, *, strategy="beam", max_new_tokens=30, num_beams=3, top_p=0.9,
                      prompt="Describe this image."):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = generate_with_strategy(self.model, inputs, strategy=strategy,
                                         max_new_tokens=max_new_tokens, num_beams=num_beams, top_p=top_p)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class CLIPZeroShot:
    def __init__(self, local_dir, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(local_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(local_dir, local_files_only=True).to(device).eval()
        self.candidate_labels = [
            "a photo of a man", "a photo of a woman", "a photo of a child",
            "a photo of a dog", "a photo of a group of people",
            "a photo of a person riding a bike", "a photo of nature",
            "a photo of an object",
        ]

    def classify(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(text=self.candidate_labels, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        return self.candidate_labels[probs.argmax(dim=1).item()]


# ======================
# Baselines (Section E)
# ======================

class FrequencyBaseline:
    def generate_one(self, image, **kwargs):
        return "A photo of a person"


class NearestNeighborBaseline:
    """Image NN using CLIP image embeddings on TRAIN split; copies the train caption."""
    def __init__(self, clip_model_dir, train_image_paths, train_captions, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_model_dir, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(clip_model_dir, local_files_only=True).to(device).eval()
        self.train_caps = []
        # Precompute image embeddings
        embs = []
        for img_path, caps in tqdm(zip(train_image_paths, train_captions), total=len(train_image_paths), desc="Precompute NN train embs"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self.clip.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy())
            # choose first caption as canonical
            self.train_caps.append(caps[0] if caps else "")
        self.train_embs = np.vstack(embs) if len(embs) else np.zeros((0, 512))

    def generate_one(self, image, **kwargs):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image
        if self.train_embs.shape[0] == 0:
            return ""
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q = self.clip.get_image_features(**inputs)
            q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy()
        sims = (self.train_embs @ q.T).squeeze()
        idx = int(np.argmax(sims))
        return self.train_caps[idx]


class CLIPReRanker:
    def __init__(self, clip_model_dir, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_model_dir, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(clip_model_dir, local_files_only=True).to(device).eval()

    def pick_best(self, image, candidates):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(text=candidates, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip(**inputs)
            scores = outputs.logits_per_image  # higher is better
        best = int(torch.argmax(scores, dim=1).item())
        return candidates[best]


# ======================
# Metrics
# ======================

_smooth = SmoothingFunction().method1


def compute_bleu_all(preds, refs):
    scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}
    for p, R in zip(preds, refs):
        Rt = [r.split() for r in R]
        Pt = p.split()
        try:
            scores["BLEU-1"].append(sentence_bleu(Rt, Pt, weights=(1, 0, 0, 0), smoothing_function=_smooth))
            scores["BLEU-2"].append(sentence_bleu(Rt, Pt, weights=(0.5, 0.5, 0, 0), smoothing_function=_smooth))
            scores["BLEU-3"].append(sentence_bleu(Rt, Pt, weights=(1/3, 1/3, 1/3, 0), smoothing_function=_smooth))
            scores["BLEU-4"].append(sentence_bleu(Rt, Pt, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_smooth))
        except ZeroDivisionError:
            for k in scores:
                scores[k].append(0.0)
    return {k: (sum(v)/len(v) if v else 0.0) for k, v in scores.items()}


def compute_meteor(preds, refs):
    vals = []
    for p, R in zip(preds, refs):
        try:
            vals.append(meteor_score(R, p))
        except Exception:
            vals.append(0.0)
    return sum(vals)/len(vals) if vals else 0.0


def compute_rougeL(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for p, R in zip(preds, refs):
        ref = R[0] if R else ""
        vals.append(scorer.score(ref, p)["rougeL"].fmeasure)
    return sum(vals)/len(vals) if vals else 0.0


def compute_cider(preds, refs):
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    return Cider().compute_score(gts, res)[0]



def compute_clipscore_image_text(img_paths, preds, device="cuda"):
    if not _HAS_TM_CLIPSCORE:
        return None
    try:
        scorer = TMCLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
        vals = []
        for ip, p in zip(img_paths, preds):
            try:
                im = Image.open(ip).convert("RGB")
            except Exception:
                vals.append(0.0)
                continue
            vals.append(float(scorer(im, p).item()))
        return sum(vals)/len(vals) if vals else 0.0
    except Exception:
        return None


# ======================
# Efficiency helpers
# ======================

def measure_efficiency(fn, *args, device="cuda", **kwargs):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = fn(*args, **kwargs)
    t1 = time.time()
    latency = t1 - t0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        vram_mb = 0.0
    throughput = 1.0 / latency if latency > 0 else 0.0
    return out, {"latency": latency, "vram_mb": vram_mb, "throughput": throughput}


# ======================
# Runner
# ======================

def run_benchmark(cfg):
    device = cfg.get("run", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.get("run", {}).get("batch_size", 1))
    max_samples = cfg.get("run", {}).get("max_samples")  # None or int

    # decoding config
    dec_strategies = cfg.get("generation", {}).get("decoding_strategies", ["greedy", "beam", "nucleus"]).copy()
    num_beams = int(cfg.get("generation", {}).get("beam_size", 3))
    top_p = float(cfg.get("generation", {}).get("nucleus_top_p", 0.9))
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 30))

    # candidate gen + rerank
    num_candidates = int(cfg.get("rerank", {}).get("num_candidates", 10))
    enable_rerank = bool(cfg.get("rerank", {}).get("enabled", True))
    clip_dir_for_rerank = cfg.get("rerank", {}).get("clip_local_dir")  # must be provided to use rerank

    # CLIP for NN baseline
    clip_dir_for_nn = cfg.get("baselines", {}).get("nn_clip_local_dir", clip_dir_for_rerank)

    results = {}
    predictions_store = {}

    for dataset in cfg.get("datasets", []):
        ds_name = dataset["name"]
        print(f\"\\n📂 Loading dataset: {ds_name}\")
        with open(dataset["path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        image_root = dataset["image_root"]

        # splits
        split_test = [d for d in data if d.get("split", "test") == "test"]
        split_train = [d for d in data if d.get("split", "train") == "train"]
        if max_samples:
            split_test = split_test[:max_samples]

        # build NN baseline index (train)
        nn_baseline = None
        if clip_dir_for_nn and len(split_train):
            train_imgs, train_caps = [], []
            for ex in split_train:
                img_path = os.path.join(image_root, ex["file_path"]) if "file_path" in ex else None
                if img_path and os.path.exists(img_path):
                    train_imgs.append(img_path)
                    caps = [s.get("raw", "") for s in ex.get("sentences", [])]
                    train_caps.append(caps)
            if len(train_imgs):
                nn_baseline = NearestNeighborBaseline(clip_dir_for_nn, train_imgs, train_caps, device=device)

        for model_cfg in cfg.get("models", []):
            name = model_cfg["name"]
            print(f\"\\n🚀 Running {name} on {ds_name} ...\")

            # choose model
            model = None
            mode = "captioning"
            lname = name.lower()
            if "blip2" in lname:
                model = BLIP2Captioner(model_cfg["local_dir"], device=device, prefer_8bit=model_cfg.get("prefer_8bit", False))
            elif "blip" in lname:
                model = BLIPCaptioner(model_cfg["local_dir"], device=device, fp16=model_cfg.get("fp16", True))
            elif "git" in lname:
                model = GITCaptioner(model_cfg["local_dir"], device=device)
            elif "gpt2" in lname:
                model = GPT2Captioner(model_cfg["local_dir"], device=device)
            elif "clip" in lname and "zeroshot" in lname:
                model = CLIPZeroShot(model_cfg["local_dir"], device=device)
                mode = "zeroshot"
            elif "frequency" in lname:
                model = FrequencyBaseline()
            elif "nearest" in lname or "nn" in lname:
                if nn_baseline is None:
                    print("⚠️ NN baseline requested but no train split or CLIP not configured; skipping.")
                    continue
                model = nn_baseline
            else:
                print(f"⚠️ Unsupported model name: {name}")
                continue

            predictions, references, per_sample, img_paths = [], [], [], []
            eff_logs = []

            # Prepare reranker if needed
            reranker = None
            if enable_rerank and mode == "captioning" and clip_dir_for_rerank:
                try:
                    reranker = CLIPReRanker(clip_dir_for_rerank, device=device)
                except Exception:
                    reranker = None

            # Iterate samples
            loop_data = split_test
            for ex in tqdm(loop_data, desc=f"{name} → {ds_name}"):
                filename = ex.get("file_path")
                img_path = os.path.join(image_root, filename) if filename else None
                if not img_path or not os.path.exists(img_path):
                    continue
                refs = [s.get("raw", "") for s in ex.get("sentences", [])]

                try:
                    if mode == "captioning":
                        # Candidate generation
                        if reranker is not None and num_candidates > 1:
                            cands = []
                            # Round-robin through strategies to get N candidates
                            strat_cycle = (dec_strategies if len(dec_strategies) else ["greedy"]).copy()
                            si = 0
                            while len(cands) < num_candidates:
                                strat = strat_cycle[si % len(strat_cycle)]
                                caption = model.generate_one(
                                    img_path,
                                    strategy=strat,
                                    max_new_tokens=max_new_tokens,
                                    num_beams=num_beams,
                                    top_p=top_p,
                                )
                                cands.append(caption)
                                si += 1
                            # Rerank by CLIP image-text score
                            pred, eff = measure_efficiency(reranker.pick_best, img_path, cands, device=device)
                        else:
                            # Single decode (default: beam)
                            pred, eff = measure_efficiency(
                                model.generate_one, img_path,
                                device=device,
                                strategy=(dec_strategies[0] if dec_strategies else "beam"),
                                max_new_tokens=max_new_tokens,
                                num_beams=num_beams,
                                top_p=top_p,
                            )
                    else:  # zeroshot classification demo
                        pred, eff = measure_efficiency(model.classify, img_path, device=device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    pred, eff = "ERROR_OOM", {"latency": math.inf, "vram_mb": float('nan'), "throughput": 0.0}
                except Exception as e:
                    pred, eff = f"ERROR: {repr(e)}", {"latency": float('nan'), "vram_mb": float('nan'), "throughput": 0.0}

                predictions.append(pred)
                references.append(refs)
                img_paths.append(img_path)
                eff_logs.append(eff)
                per_sample.append({"file": filename, "prediction": pred, "references": refs, "efficiency": eff})

            # --- Metrics ---
            results.setdefault(name, {})[ds_name] = {}
            if mode == "captioning":
                bleu = compute_bleu_all(predictions, references)
                meteor = compute_meteor(predictions, references)
                rougeL = compute_rougeL(predictions, references)
                cider = compute_cider(predictions, references)
                
                clipscore_ref_free = compute_clipscore_image_text(img_paths, predictions, device=device)  # may be None

                results[name][ds_name].update({
                    **bleu,
                    "METEOR": meteor,
                    "ROUGE-L": rougeL,
                    "CIDEr": cider,
                    "CLIPScore": clipscore_ref_free,
                })

                print(
                    f"  ✅ BLEU-1 {bleu['BLEU-1']:.4f} | BLEU-2 {bleu['BLEU-2']:.4f} | "
                    f"BLEU-3 {bleu['BLEU-3']:.4f} | BLEU-4 {bleu['BLEU-4']:.4f} | METEOR {meteor:.4f} | "
                    f"ROUGE-L {rougeL:.4f} | CIDEr {cider:.4f} | "
                    f"CLIPScore {('%.4f'%clipscore_ref_free) if clipscore_ref_free is not None else 'n/a'}"
                )
            else:
                # simple accuracy for zeroshot classification if references contain exact text
                total, correct = 0, 0
                for p, R in zip(predictions, references):
                    total += 1
                    if p.lower() in [r.lower() for r in R]:
                        correct += 1
                acc = (correct/total) if total else 0.0
                results[name][ds_name].update({"Accuracy": acc})
                print(f"  ✅ Accuracy: {acc:.4f}")

            # --- Efficiency summary ---
            if eff_logs:
                avg_latency = sum(e["latency"] for e in eff_logs if math.isfinite(e["latency"])) / max(1, sum(1 for e in eff_logs if math.isfinite(e["latency"])) )
                avg_vram = sum(e["vram_mb"] for e in eff_logs if not math.isnan(e["vram_mb"])) / max(1, sum(1 for e in eff_logs if not math.isnan(e["vram_mb"])) )
                avg_throughput = sum(e["throughput"] for e in eff_logs if e["throughput"] > 0) / max(1, sum(1 for e in eff_logs if e["throughput"] > 0))
                results[name][ds_name].update({
                    "Avg_Latency_s": avg_latency,
                    "Avg_VRAM_MB": avg_vram,
                    "Avg_Throughput_img_per_s": avg_throughput,
                })

            # store predictions
            predictions_store.setdefault(name, {})[ds_name] = per_sample

    # save
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("results_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_store, f, indent=2, ensure_ascii=False)

    print("\\n🎉 Benchmark finished!")
    print("📄 Summary saved to: results.json")
    print("📄 Predictions saved to: results_predictions.json")


# ======================
# Main
# ======================
if __name__ == "__main__":
    cfg = load_config("benchmark_config.yaml")
    run_benchmark(cfg)
