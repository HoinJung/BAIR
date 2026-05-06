# BAIR

Clean BAIR repository for the IU-Chest, FACET, and NWPU experiments.

BAIR now exposes one hyperparameter:

- `av`: visual bottleneck strength, default `0.5`

The old `at` and `gs` knobs are fixed internally at `1.0`. Deprecated CLI flags are still accepted in a few scripts for backward compatibility, but they are hidden and forced back to `1.0`.

## What Is Included

- Core BAIR code: `bottleneck_intervention.py`, `bair_efficient.py`
- Functional model helpers: `llm_explainer.py`, `custom_medgemma_model.py`, `rag_core.py`
- Dataset runners:
  - IU-Chest: `unified_medgemma_analysis.py`, `unified_chexagent.py`
  - FACET: `gender_analysis.py`
  - NWPU: `nwpu_analysis.py`
- Preprocessing:
  - IU-Chest: `build_nih_database.py`
  - MIMIC-CXR: `build_nih_mimic.py`
  - NWPU: `build_nwpu_retrieval_dataset.py`
  - Pseudo labels: `pseudo_label.py`
- Evaluation/analysis:
  - Medical: `eval_medical.py`
  - FACET: `evaluate_intervention_gender_fairness.py`, `evaluate_facet_intervention_grid_subset.py`
  - NWPU: `eval_nwpu.py`
  - Supporting analysis helpers: `gtonly_bias_profile_analysis.py`, `gtonly_coincidence_analysis.py`, collection/merge scripts
- Copied metadata in `data/metadata/`
- Generated retrieval/eval metadata in `data/generated/`

Not included: plotting-only scripts, notebooks, visualization assets, bootstrap scripts, and historical output folders.

## Datasets And Models

Supported experiment matrix:

| Dataset | Models |
| --- | --- |
| IU-Chest X-ray | `google/medgemma-4b-it`, `StanfordAIMI/CheXagent-8b` |
| FACET | `Qwen/Qwen2.5-VL-3B-Instruct`, `deepseek-ai/deepseek-vl-7b-chat` |
| NWPU | `ll-13/SkySenseGPT-7B-clip-lora`, `akshaydudhane/EarthDial_4B_RGB` |

## Environment

Use one core environment when possible:

```bash
conda create -n bair python=3.10 -y
conda activate bair
pip install -r requirements.txt
```

This core environment is intended for MedGemma, CheXagent, FACET Qwen, preprocessing, and evaluation.

Some model repos pin incompatible `transformers` versions, so separate environments are recommended:

```bash
conda create -n bair-earthdial python=3.10 -y
conda activate bair-earthdial
pip install -r requirements-earthdial.txt
pip install -e third_party/EarthDial
```

```bash
conda create -n bair-skysense python=3.10 -y
conda activate bair-skysense
pip install -r requirements-skysense.txt
pip install -e third_party/GeoChat
```

DeepSeek-VL requires the official DeepSeek-VL package:

```bash
conda activate bair
pip install -r requirements-deepseek.txt
```

On this machine, compatible existing envs are `lingua` for the core stack and evaluation, `llava-med` for DeepSeek-VL, `nwpu` for SkySense/GeoChat, and `earthdial` for EarthDial.

## Data Preparation

Raw datasets are not copied into git. Put them under `data/raw/` or create symlinks:

```bash
scripts/prepare_raw_data_symlinks.sh /home/jw-server3/a/jung414/data
```

Expected raw layout:

```text
data/raw/iuchest/
  indiana_reports.csv
  indiana_projections.csv
  images_normalized/

data/raw/mimic-cxr/
  metadata.json
  images/

data/raw/facet/
  image/

data/raw/NWPU/
  test/test/<class_name>/*.jpg
```

Already copied into this repo:

```text
data/metadata/
  profession_database.json
  profession_database_fixed.json
  excluding_list.json
  nwpu_database.json
  nwpu_database_fixed.json
  nwpu_exclude.json
  nwpu_keyword_matching.json
  nwpu_label_schema.json
  facet_new_annotations.csv

data/generated/
  iuchest_nih_retrieval_dataset.json
  mimic_nih_retrieval_dataset.json
  mimic_nih_retrieval_dataset_findings_only.json
  nwpu_retrieval_dataset.json
  nwpu_retrieval_dataset_longllmlingua_precomputed.json
  indiana_reports_with_pseudo_labels_dual.csv
  mimic_reports_with_pseudo_labels_dual.csv
```

To rebuild generated retrieval metadata:

```bash
python build_nih_database.py
python build_nih_mimic.py
python build_nwpu_retrieval_dataset.py
```

## External Repositories

Some models need cloned code in `third_party/`:

```bash
scripts/clone_external_repos.sh
```

Then install the needed repo into the matching environment:

```bash
pip install -e third_party/EarthDial
pip install -e third_party/GeoChat
```

`llm_explainer.py` searches both old root-level clone names and `third_party/...` locations.

## Running Smoke Tests

Run all requested 20-sample StandardRAG vs BAIR checks:

```bash
scripts/smoke_all_variants.sh
```

Useful overrides:

```bash
AV=0.1 NUM_SAMPLES=20 DEVICE_ID=0 scripts/smoke_all_variants.sh
AV=0.05 RUN_FACET_DEEPSEEK=0 scripts/smoke_all_variants.sh
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 scripts/smoke_all_variants.sh
DEEPSEEK_ENV=llava-med scripts/smoke_all_variants.sh
```

The smoke script writes to `outputs/smoke_av<av>_n20/` and evaluates each BAIR output against the corresponding StandardRAG baseline. FACET and NWPU eval JSONs can be checked with:

```bash
python scripts/check_smoke_improvement.py outputs/smoke_av0.5_n20/eval/*_eval.json
```

## Efficient Mode

Original BAIR is the default. Efficient mode is enabled with:

```bash
BAIR_EFFICIENT_MODE=1 scripts/smoke_all_variants.sh
```

Efficient mode currently keeps CUDA allocator pools warm and reuses MedGemma pixel-value buffers when shapes match. The BAIR math is unchanged.
