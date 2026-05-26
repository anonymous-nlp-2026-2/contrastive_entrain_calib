# The Probe Reward Trap

Code and data for "The Probe Reward Trap: Behavioral Death Before Distributional Collapse in Sycophancy Calibration"

## Abstract

Representation engineering reveals that sycophancy is linearly encoded in LLM residual streams, suggesting that such directions could serve as training signals for alignment. We test this premise systematically and document its failure. A contrastive sycophancy direction d_syc reliably discriminates sycophantic from non-sycophantic activations, yet every formulation that converts it into a Group Relative Policy Optimization (GRPO) reward collapses. We identify the *probe reward trap*: when a periodically retrained linear probe on d_syc is used as a non-differentiable reward signal, behavioral death occurs within 500 steps while generation entropy remains healthy, creating a 1,300-step monitoring blind spot that entropy-based safeguards miss entirely.

## Structure

- `artifacts/training/` — Training scripts (GRPO, SFT probe penalty, ACT, ActAdd)
- `artifacts/eval/` — Evaluation pipeline
- `artifacts/data/` — Data generation and processing
- `docs/paper/` — Paper source (LaTeX)
- `scripts/` — Analysis and visualization scripts
- `training/` — DPO and Llama cross-model training
- `analysis/` — Bootstrap AUROC confidence interval analysis

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- transformers, trl, peft, datasets
- vllm (for evaluation)

## Quick Start

```bash
pip install torch transformers trl peft datasets accelerate deepspeed

# Extract sycophancy direction
python artifacts/data/compute_directions.py

# Run GRPO training (cosine R1)
python artifacts/training/run_exp002v3_dcgrpo_scaled_cosine.py

# Evaluate
python artifacts/eval/eval_pipeline.py
```

## Citation

If you find this work useful, please cite our paper.
