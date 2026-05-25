# Contrastive Entrainment Calibration

Code and data for "When Does Discriminability Fail as Reward Utility?"

## Structure

- `artifacts/training/` — Training scripts (GRPO, SFT probe penalty, ACT, ActAdd)
- `artifacts/eval/` — Evaluation pipeline
- `artifacts/data/` — Data generation and processing
- `docs/paper/` — Paper source (LaTeX)
- `scripts/` — Analysis scripts

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- transformers, trl, peft, datasets
- vllm (for evaluation)

## Citation

If you find this work useful, please cite our paper.
