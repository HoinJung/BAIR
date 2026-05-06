# BAIR

This repository applies BAIR to three vision-language evaluation settings, with two models per setting:

| Dataset | Models | Runner | Evaluation |
| --- | --- | --- | --- |
| IU-Chest | `google/medgemma-4b-it`, `StanfordAIMI/CheXagent-8b` | `experiments.iuchest_analysis` | `evaluation.eval_medical` |
| FACET | `Qwen/Qwen2.5-VL-3B-Instruct`, `deepseek-ai/deepseek-vl-7b-chat` | `experiments.gender_analysis` | `evaluation.evaluate_facet` |
| NWPU | `ll-13/SkySenseGPT-7B-clip-lora`, `akshaydudhane/EarthDial_4B_RGB` | `experiments.nwpu_analysis` | `evaluation.eval_nwpu` |

BAIR exposes one experiment hyperparameter:

- `av`: visual bottleneck strength, default `0.5`

Run commands from the repository root so module imports such as `import bair` resolve correctly.

## Repository Layout

| Path | Role |
| --- | --- |
| `bair/` | Importable BAIR library: bottleneck intervention code, model helpers, RAG utilities, optional runtime efficiency helpers. |
| `experiments/` | Dataset runners for IU-Chest, FACET, and NWPU. |
| `evaluation/` | Evaluation scripts for saved result JSON files. |
| `preprocess/` | Builders for included generated retrieval metadata and pseudo labels. |
| `scripts/` | Data symlink setup, external repo cloning, and full experiment orchestration. |
| `data/metadata/` | Included metadata files used by the runners and evaluators. |
| `data/generated/` | Included generated retrieval/evaluation metadata. |
| `data/raw/` | Local raw dataset location; raw data are not committed. |
| `third_party/` | Optional editable installs for EarthDial and GeoChat/SkySense dependencies. |

## Environment Setup

The core environment supports MedGemma, CheXagent, FACET Qwen, preprocessing, and evaluation:

```bash
conda create -n bair python=3.10 -y
conda activate bair
pip install -r requirements.txt
```

DeepSeek-VL needs its official package dependencies. Install them in the core environment or in a separate environment:

```bash
conda activate bair
pip install -r requirements-deepseek.txt
```

EarthDial and SkySense/GeoChat pin dependencies that can conflict with the core environment, so separate environments are recommended:

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

Clone optional external code before installing editable packages:

```bash
scripts/clone_external_repos.sh
```

`scripts/run_full_experiments.sh` uses `conda run -n <env>`. Defaults are:

| Variable | Default | Used for |
| --- | --- | --- |
| `CORE_ENV` | `bair` | MedGemma, CheXagent, Qwen, preprocessing, evaluation |
| `DEEPSEEK_ENV` | `bair` | DeepSeek-VL |
| `SKYSENSE_ENV` | `bair-skysense` | SkySense/GeoChat |
| `EARTHDIAL_ENV` | `bair-earthdial` | EarthDial |

Override these variables only if you choose different environment names.

## Data Setup

Raw datasets are not included. Put them under `data/raw/` or create symlinks:

```bash
scripts/prepare_raw_data_symlinks.sh /path/to/raw/data/root
```

Expected raw layout:

```text
data/raw/iuchest/
  indiana_reports.csv
  indiana_projections.csv
  images_normalized/

data/raw/facet/
  image/

data/raw/NWPU/
  test/test/<class_name>/*.jpg
```

Included metadata:

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
```

Included generated files:

```text
data/generated/
  iuchest_nih_retrieval_dataset.json
  nwpu_retrieval_dataset.json
  nwpu_retrieval_dataset_longllmlingua_precomputed.json
  indiana_reports_with_pseudo_labels_dual.csv
```

To rebuild generated retrieval metadata:

```bash
python -m preprocess.build_nih_iuchest
python -m preprocess.build_nih_nwpu
```

## Full Experiment Script

The orchestrated entry point runs the full IU-Chest, FACET, and NWPU matrix and evaluates completed generations:

```bash
AV=0.5 DEVICE_ID=0 bash scripts/run_full_experiments.sh
```

Default outputs are written under `outputs/full_experiments_av<AV>/`. Useful options:

```bash
OUT=outputs/reproduction_av0.5 AV=0.5 DEVICE_ID=0 SEED=42 bash scripts/run_full_experiments.sh
RUN_EVAL=0 bash scripts/run_full_experiments.sh
RUN_MEDGEMMA=0 RUN_CHEXAGENT=0 bash scripts/run_full_experiments.sh
RUN_FACET_QWEN=0 RUN_FACET_DEEPSEEK=0 bash scripts/run_full_experiments.sh
RUN_NWPU_SKYSENSE=0 RUN_NWPU_EARTHDIAL=0 bash scripts/run_full_experiments.sh
CORE_ENV=bair DEEPSEEK_ENV=bair SKYSENSE_ENV=bair-skysense EARTHDIAL_ENV=bair-earthdial bash scripts/run_full_experiments.sh
```

`RUN_EVAL=0` skips only the evaluation phase after generation. `MAX_MED_TOKENS`, `MAX_FACET_TOKENS`, and `MAX_NWPU_TOKENS` control generation length.

## Manual Runs

Use manual commands for subsets, sweeps, or custom output directories.

### IU-Chest

MedGemma:

```bash
python -m experiments.iuchest_analysis --model medgemma --dataset iuchest \
  --data_dir data/raw/iuchest --device_id 0 --output_dir outputs/iuchest_medgemma \
  --generate_baselines

python -m experiments.iuchest_analysis --model medgemma --dataset iuchest \
  --data_dir data/raw/iuchest --device_id 0 --output_dir outputs/iuchest_medgemma \
  --use_intervention --alpha_v 0.5

python -m evaluation.eval_medical --dataset iu-chest \
  --baseline-json outputs/iuchest_medgemma/iuchest_medgemma_results_baselines.json \
  --intervention-json outputs/iuchest_medgemma/iuchest_medgemma_results_new_bair_av0.5_mid.json \
  --metric f1
```

CheXagent:

```bash
python -m experiments.iuchest_analysis --model chexagent --dataset iuchest \
  --data_dir data/raw/iuchest --device_id 0 --output_dir outputs/iuchest_chexagent \
  --generate_baselines

python -m experiments.iuchest_analysis --model chexagent --dataset iuchest \
  --data_dir data/raw/iuchest --device_id 0 --output_dir outputs/iuchest_chexagent \
  --use_intervention --alpha_v 0.5

python -m evaluation.eval_medical --dataset iu-chest \
  --baseline-json outputs/iuchest_chexagent/iuchest_chexagent_results_baselines.json \
  --intervention-json outputs/iuchest_chexagent/iuchest_chexagent_results_new_bair_av0.5_mid.json \
  --metric f1
```

Pass `--num_samples N` to cap IU-Chest during development. Omit it for the full deterministic split.

### FACET

Qwen:

```bash
python -m experiments.gender_analysis \
  --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
  --find_all --device_id 0 --json_output_dir outputs/facet_qwen --output_dir outputs/facet_qwen \
  --model_names Qwen/Qwen2.5-VL-3B-Instruct --generation_only

python -m experiments.gender_analysis \
  --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
  --find_all --device_id 0 --json_output_dir outputs/facet_qwen --output_dir outputs/facet_qwen \
  --model_names Qwen/Qwen2.5-VL-3B-Instruct --use_intervention --alpha_v 0.5 \
  --from_analysis_results outputs/facet_qwen/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2.json \
  --intervention_output_json outputs/facet_qwen/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2_intervention_av0.5.json

python -m evaluation.evaluate_facet \
  --input-json outputs/facet_qwen/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2_intervention_av0.5.json \
  --baseline-json outputs/facet_qwen/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2.json \
  --intervention-field oracle_with_intervention \
  --save-details-json outputs/facet_qwen/facet_eval.json
```

DeepSeek-VL uses the same commands with:

```text
--model_names deepseek-ai/deepseek-vl-7b-chat
baseline JSON: analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2.json
BAIR JSON: analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2_intervention_av0.5.json
```

### NWPU

SkySense:

```bash
python -m experiments.nwpu_analysis \
  --dataset-json data/generated/nwpu_retrieval_dataset.json \
  --model-name ll-13/SkySenseGPT-7B-clip-lora \
  --device-id 0 --output-dir outputs/nwpu_skysense --generate-baselines

python -m experiments.nwpu_analysis \
  --dataset-json data/generated/nwpu_retrieval_dataset.json \
  --model-name ll-13/SkySenseGPT-7B-clip-lora \
  --device-id 0 --output-dir outputs/nwpu_skysense --use-intervention --alpha-v 0.5

python -m evaluation.eval_nwpu \
  --input-json outputs/nwpu_skysense/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_bair_av0.5.json \
  --baseline-json outputs/nwpu_skysense/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_baselines.json \
  --intervention-field oracle_with_intervention \
  --save-details-json outputs/nwpu_skysense/nwpu_eval.json
```

EarthDial uses the same commands with:

```text
--model-name akshaydudhane/EarthDial_4B_RGB
baseline JSON: nwpu_results_akshaydudhane_EarthDial_4B_RGB_baselines.json
BAIR JSON: nwpu_results_akshaydudhane_EarthDial_4B_RGB_bair_av0.5.json
```

Omit `--num-samples` or pass `--num-samples 0` for the full NWPU retrieval file.

## Efficient Mode

Original BAIR behavior is the default. To reduce runtime overhead in repeated GPU experiments, set:

```bash
BAIR_EFFICIENT_MODE=1 bash scripts/run_full_experiments.sh
```

This skips some aggressive CUDA cache clearing and may reuse compatible MedGemma pixel buffers. BAIR math is unchanged.

## Additional CLI Help

Each runner and evaluator exposes full argument help:

```bash
python -m experiments.iuchest_analysis --help
python -m experiments.gender_analysis --help
python -m experiments.nwpu_analysis --help
python -m evaluation.eval_medical --help
python -m evaluation.evaluate_facet --help
python -m evaluation.eval_nwpu --help
```
