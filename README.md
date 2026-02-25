# PolyTransformer

Autoregressive Transformer pipelines for inverse polymer design.

## Scope

This workspace contains AR pipelines only:

- `AR_Transformer_SMILES`
- `AR_Transformer_SMILES_BPE`
- `AR_Transformer_SELFIES`
- `AR_Transformer_Group_SELFIES`

There are no `Bi_Diffusion_*` or graph pipelines in this repository snapshot.

## Representation Status (Audited 2026-02-25)

| Method | Tokenizer artifact | Runtime stack used by Steps 1-6 | Transformers migration status |
| --- | --- | --- | --- |
| `AR_Transformer_SMILES` | `results/tokenizer.json` + `results/tokenizer_hf/` | `src.data.hf_tokenizer` + `src.model.hf_ar` | Complete |
| `AR_Transformer_SMILES_BPE` | `results/tokenizer.json` (optional `results/tokenizer_hf/`) | `src.data.hf_tokenizer` + `src.model.hf_ar` | Complete (Steps 1-6) |
| `AR_Transformer_SELFIES` | `results/tokenizer.json` (optional `results/tokenizer_hf/`) | `src.data.hf_tokenizer` + `src.model.hf_ar` | Complete (Steps 1-6) |
| `AR_Transformer_Group_SELFIES` | `results/tokenizer.pkl` (optional `results/tokenizer_hf/`) | `src.data.hf_tokenizer` + `src.model.hf_ar` | Complete (Steps 1-6) |

`hf_ar.py` / `hf_tokenizer.py` modules are now the default runtime path for Steps 1-6 in all methods. Step 0 artifacts remain representation-specific.

## Shared Assets

- `Data/`: polymer/property datasets plus shared train/val splits.
- `shared/`: cross-method utilities and policies.
- `scripts/`: repo-level submission and aggregation helpers.

## Pipeline Summary

Per method:

1. `step0_prepare_data.py`
2. `step1_train_backbone.py`
3. `step2_sample_and_evaluate.py`
4. `step3_train_property_head.py`
5. `step4_inverse_design.py`
6. `step5_class_design.py`
7. `step6_hyperparameter_tuning.py` (optional)

`run_steps1_6.sh` runs Steps 1-6 only. Step 0 must be run separately.

## Documentation

Each subproject keeps:

- `Pipeline.md`: commands and execution flow
- `technical_guide.md`: implementation details and migration state
- `results.md`: artifact layout and file semantics
