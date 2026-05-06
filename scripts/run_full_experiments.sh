#!/usr/bin/env bash
set -euo pipefail

# Full experiment orchestration: IU-Chest (full deterministic split unless --num_samples),
# FACET (all rows reachable via --find_all), NWPU (entire nwpu_retrieval_dataset.json via default --num-samples 0).
#
# Prerequisites: conda envs, HF weights, data under data/raw/, generated JSONs under data/generated/.
#
# Usage (from repo root):
#   AV=0.5 DEVICE_ID=0 CORE_ENV=bair bash scripts/run_full_experiments.sh
#
# Optional toggles:
#   OUT=outputs/reproduction_av0.5 RUN_MEDGEMMA=1 RUN_EVAL=1 (defaults on)
#   RUN_FACET_DEEPSEEK=0  # skip DeepSeek-VL stage

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AV="${AV:-0.5}"
SEED="${SEED:-42}"
MAX_MED_TOKENS="${MAX_MED_TOKENS:-128}"
MAX_FACET_TOKENS="${MAX_FACET_TOKENS:-64}"
MAX_NWPU_TOKENS="${MAX_NWPU_TOKENS:-64}"

CORE_ENV="${CORE_ENV:-bair}"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-bair}"
SKYSENSE_ENV="${SKYSENSE_ENV:-bair-skysense}"
EARTHDIAL_ENV="${EARTHDIAL_ENV:-bair-earthdial}"

DEVICE_ID="${DEVICE_ID:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_CHEXAGENT="${RUN_CHEXAGENT:-1}"
RUN_FACET_QWEN="${RUN_FACET_QWEN:-1}"
RUN_FACET_DEEPSEEK="${RUN_FACET_DEEPSEEK:-1}"
RUN_NWPU_SKYSENSE="${RUN_NWPU_SKYSENSE:-1}"
RUN_NWPU_EARTHDIAL="${RUN_NWPU_EARTHDIAL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

OUT="${OUT:-outputs/full_experiments_av${AV}}"
mkdir -p "$OUT/eval"

run_py() {
  local env_name="$1"
  shift
  echo
  echo ">>> [$env_name] python $*"
  conda run --no-capture-output -n "$env_name" python "$@"
}

eval_medical_pair() {
  local dataset="$1"
  local baseline_json="$2"
  local bair_json="$3"
  run_py "$CORE_ENV" -m evaluation.eval_medical \
    --dataset "$dataset" \
    --baseline-json "$baseline_json" \
    --intervention-json "$bair_json" \
    --metric f1
}

eval_facet_pair() {
  local baseline_json="$1"
  local bair_json="$2"
  local tag="$3"
  run_py "$CORE_ENV" -m evaluation.evaluate_facet \
    --input-json "$bair_json" \
    --baseline-json "$baseline_json" \
    --intervention-field oracle_with_intervention \
    --save-details-json "$OUT/eval/${tag}_facet_eval.json"
}

eval_nwpu_pair() {
  local baseline_json="$1"
  local bair_json="$2"
  local tag="$3"
  run_py "$CORE_ENV" -m evaluation.eval_nwpu \
    --input-json "$bair_json" \
    --baseline-json "$baseline_json" \
    --intervention-field oracle_with_intervention \
    --save-details-json "$OUT/eval/${tag}_nwpu_eval.json"
}

if [[ "$RUN_MEDGEMMA" == "1" ]]; then
  MED_OUT="$OUT/medgemma"
  run_py "$CORE_ENV" -m experiments.iuchest_analysis --model medgemma \
    --dataset iuchest --data_dir data/raw/iuchest \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$MED_OUT" --generate_baselines
  run_py "$CORE_ENV" -m experiments.iuchest_analysis --model medgemma \
    --dataset iuchest --data_dir data/raw/iuchest \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$MED_OUT" --use_intervention --alpha_v "$AV"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_medical_pair iu-chest \
      "$MED_OUT/iuchest_medgemma_results_baselines.json" \
      "$MED_OUT/iuchest_medgemma_results_new_bair_av${AV}_mid.json"
  fi
fi

if [[ "$RUN_CHEXAGENT" == "1" ]]; then
  CHEX_OUT="$OUT/chexagent"
  run_py "$CORE_ENV" -m experiments.iuchest_analysis --model chexagent \
    --dataset iuchest --data_dir data/raw/iuchest \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$CHEX_OUT" --generate_baselines
  run_py "$CORE_ENV" -m experiments.iuchest_analysis --model chexagent \
    --dataset iuchest --data_dir data/raw/iuchest \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$CHEX_OUT" --use_intervention --alpha_v "$AV"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_medical_pair iu-chest \
      "$CHEX_OUT/iuchest_chexagent_results_baselines.json" \
      "$CHEX_OUT/iuchest_chexagent_results_new_bair_av${AV}_mid.json"
  fi
fi

FACET_INSTRUCTION="You are a helpful assistant. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge."

if [[ "$RUN_FACET_QWEN" == "1" ]]; then
  FQ_OUT="$OUT/facet_qwen"
  run_py "$CORE_ENV" -m experiments.gender_analysis \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --find_all --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FQ_OUT" --output_dir "$FQ_OUT" \
    --model_names Qwen/Qwen2.5-VL-3B-Instruct --generation_only --instruction "$FACET_INSTRUCTION"
  FQ_BASE="$FQ_OUT/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2.json"
  FQ_BAIR="$FQ_OUT/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2_intervention_av${AV}.json"
  run_py "$CORE_ENV" -m experiments.gender_analysis \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --find_all --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FQ_OUT" --output_dir "$FQ_OUT" \
    --model_names Qwen/Qwen2.5-VL-3B-Instruct --use_intervention --alpha_v "$AV" \
    --from_analysis_results "$FQ_BASE" --intervention_output_json "$FQ_BAIR" --instruction "$FACET_INSTRUCTION"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_facet_pair "$FQ_BASE" "$FQ_BAIR" facet_qwen
  fi
fi

if [[ "$RUN_FACET_DEEPSEEK" == "1" ]]; then
  FD_OUT="$OUT/facet_deepseek"
  run_py "$DEEPSEEK_ENV" -m experiments.gender_analysis \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --find_all --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FD_OUT" --output_dir "$FD_OUT" \
    --model_names deepseek-ai/deepseek-vl-7b-chat --generation_only --instruction "$FACET_INSTRUCTION"
  FD_BASE="$FD_OUT/analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2.json"
  FD_BAIR="$FD_OUT/analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2_intervention_av${AV}.json"
  run_py "$DEEPSEEK_ENV" -m experiments.gender_analysis \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --find_all --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FD_OUT" --output_dir "$FD_OUT" \
    --model_names deepseek-ai/deepseek-vl-7b-chat --use_intervention --alpha_v "$AV" \
    --from_analysis_results "$FD_BASE" --intervention_output_json "$FD_BAIR" --instruction "$FACET_INSTRUCTION"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_facet_pair "$FD_BASE" "$FD_BAIR" facet_deepseek
  fi
fi

if [[ "$RUN_NWPU_SKYSENSE" == "1" ]]; then
  NS_OUT="$OUT/nwpu_skysense"
  run_py "$SKYSENSE_ENV" -m experiments.nwpu_analysis \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name ll-13/SkySenseGPT-7B-clip-lora \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NS_OUT" --generate-baselines
  run_py "$SKYSENSE_ENV" -m experiments.nwpu_analysis \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name ll-13/SkySenseGPT-7B-clip-lora \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NS_OUT" --use-intervention --alpha-v "$AV"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_nwpu_pair \
      "$NS_OUT/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_baselines.json" \
      "$NS_OUT/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_bair_av${AV}.json" \
      nwpu_skysense
  fi
fi

if [[ "$RUN_NWPU_EARTHDIAL" == "1" ]]; then
  NE_OUT="$OUT/nwpu_earthdial"
  run_py "$EARTHDIAL_ENV" -m experiments.nwpu_analysis \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name akshaydudhane/EarthDial_4B_RGB \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NE_OUT" --generate-baselines
  run_py "$EARTHDIAL_ENV" -m experiments.nwpu_analysis \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name akshaydudhane/EarthDial_4B_RGB \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NE_OUT" --use-intervention --alpha-v "$AV"
  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_nwpu_pair \
      "$NE_OUT/nwpu_results_akshaydudhane_EarthDial_4B_RGB_baselines.json" \
      "$NE_OUT/nwpu_results_akshaydudhane_EarthDial_4B_RGB_bair_av${AV}.json" \
      nwpu_earthdial
  fi
fi

echo
echo "Full experiment suite finished. Results under: $OUT"
