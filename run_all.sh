#!/bin/bash
# ==========================================================
# Run all benchmark experiments for:
#   - BLIP, BLIP-2, GIT, GPT-2 (text-only baseline)
#   - Datasets: Flickr8k, COCO, VizWiz
# ==========================================================

# Create logs folder
mkdir -p logs

# -------------------------------
# Flickr8k Experiments
# -------------------------------
echo "Running Flickr8k experiments..."

python evaluate.py --model blip --dataset flickr8k \
    > logs/blip_flickr8k.log 2>&1

python evaluate.py --model blip2 --dataset flickr8k \
    > logs/blip2_flickr8k.log 2>&1

python evaluate.py --model git --dataset flickr8k \
    > logs/git_flickr8k.log 2>&1

python evaluate.py --model gpt2 --dataset flickr8k \
    > logs/gpt2_flickr8k.log 2>&1


# -------------------------------
# COCO Experiments
# -------------------------------
echo "Running COCO experiments..."

python evaluate.py --model blip --dataset coco \
    > logs/blip_coco.log 2>&1

python evaluate.py --model blip2 --dataset coco \
    > logs/blip2_coco.log 2>&1

python evaluate.py --model git --dataset coco \
    > logs/git_coco.log 2>&1

python evaluate.py --model gpt2 --dataset coco \
    > logs/gpt2_coco.log 2>&1


# -------------------------------
# VizWiz Experiments
# -------------------------------
echo "Running VizWiz experiments..."

python evaluate.py --model blip --dataset vizwiz \
    > logs/blip_vizwiz.log 2>&1

python evaluate.py --model blip2 --dataset vizwiz \
    > logs/blip2_vizwiz.log 2>&1

python evaluate.py --model git --dataset vizwiz \
    > logs/git_vizwiz.log 2>&1

python evaluate.py --model gpt2 --dataset vizwiz \
    > logs/gpt2_vizwiz.log 2>&1


# -------------------------------
# Finished
# -------------------------------
echo "All experiments completed. Logs saved in ./logs/"
