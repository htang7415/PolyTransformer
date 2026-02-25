# PolyTransformer

Autoregressive Transformer models for polymer prediction and inverse design.

## Scope

This repository currently contains only AR pipelines:

- `AR_Transformer_SMILES`
- `AR_Transformer_SMILES_BPE`
- `AR_Transformer_SELFIES`
- `AR_Transformer_Group_SELFIES`

There are no `Bi_Diffusion_*` or graph-model pipelines in this repo snapshot.

## Shared Assets

- `Data/`: polymer and property datasets plus shared train/val splits.
- `shared/`: common utilities used by AR pipelines (`unlabeled_data.py`, `ood_metrics.py`, `rerank_utils.py`).
- `scripts/`: repo-level submission and aggregation helpers.

## Documentation

Each AR subproject contains:

- `Pipeline.md`: step-by-step run commands.
- `technical_guide.md`: implementation details.
- `results.md`: recorded experiment outputs.
