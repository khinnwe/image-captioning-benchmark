# Image Captioning Benchmark

Config-driven benchmarking pipeline for **BLIP**, **BLIP-2**, **GIT**, **GPT-2**, plus baselines and CLIP-based reranking.

## Features
- Decoding: greedy / beam / nucleus
- CLIP reranking (image-text similarity)
- Baselines: frequency, nearest-neighbor (CLIP image embeddings)
- Metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr, CLIPScore (ref-free)
- Efficiency logs: latency, VRAM, throughput
- Config-first design via `benchmark_config.yaml`

## Quickstart

### Option A — pip
```bash
pip install -r requirements.txt
python benchmark.py
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate image-captioning-benchmark
python benchmark.py
```

### Configuration
Edit paths in **benchmark_config.yaml** to match your local datasets and model cache.
- `datasets[].path` → JSON annotation file with items like:
  ```json
  {"file_path": "xxx.jpg", "sentences": [{"raw": "a caption"}], "split": "test"}
  ```
- `datasets[].image_root` → folder where images live
- `models[]` → provide local `local_dir` for each model
- `rerank.clip_local_dir` and `baselines.nn_clip_local_dir` → CLIP directory (e.g., ViT-B/32)

### Outputs
- `results.json` → aggregate metrics and efficiency per model × dataset
- `results_predictions.json` → per-sample predictions + refs + efficiency

> **Note**: By default, large folders like `Data/` and `models/` are ignored by git.

## License
MIT
