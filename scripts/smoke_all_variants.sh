#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_SAMPLES="${NUM_SAMPLES:-20}"
AV="${AV:-0.5}"
SEED="${SEED:-42}"
MAX_MED_TOKENS="${MAX_MED_TOKENS:-128}"
MAX_FACET_TOKENS="${MAX_FACET_TOKENS:-64}"
MAX_NWPU_TOKENS="${MAX_NWPU_TOKENS:-64}"

CORE_ENV="${CORE_ENV:-lingua}"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-llava-med}"
SKYSENSE_ENV="${SKYSENSE_ENV:-nwpu}"
EARTHDIAL_ENV="${EARTHDIAL_ENV:-earthdial}"

DEVICE_ID="${DEVICE_ID:-0}"
COMPRESSOR_DEVICE_ID="${COMPRESSOR_DEVICE_ID:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

RUN_MEDGEMMA="${RUN_MEDGEMMA:-1}"
RUN_CHEXAGENT="${RUN_CHEXAGENT:-1}"
RUN_FACET_QWEN="${RUN_FACET_QWEN:-1}"
RUN_FACET_DEEPSEEK="${RUN_FACET_DEEPSEEK:-1}"
RUN_NWPU_SKYSENSE="${RUN_NWPU_SKYSENSE:-1}"
RUN_NWPU_EARTHDIAL="${RUN_NWPU_EARTHDIAL:-1}"

OUT="${OUT:-outputs/smoke_av${AV}_n${NUM_SAMPLES}}"
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
  run_py "$CORE_ENV" eval_medical.py \
    --dataset "$dataset" \
    --baseline-json "$baseline_json" \
    --intervention-json "$bair_json" \
    --metric f1
}

eval_facet_pair() {
  local baseline_json="$1"
  local bair_json="$2"
  local tag="$3"
  run_py "$CORE_ENV" evaluate_intervention_gender_fairness.py \
    --input-json "$bair_json" \
    --baseline-json "$baseline_json" \
    --intervention-field oracle_with_intervention \
    --save-details-json "$OUT/eval/${tag}_facet_eval.json"
}

eval_nwpu_pair() {
  local baseline_json="$1"
  local bair_json="$2"
  local tag="$3"
  run_py "$CORE_ENV" eval_nwpu.py \
    --input-json "$bair_json" \
    --baseline-json "$baseline_json" \
    --intervention-field oracle_with_intervention \
    --save-details-json "$OUT/eval/${tag}_nwpu_eval.json"
}

if [[ "$RUN_MEDGEMMA" == "1" ]]; then
  MED_OUT="$OUT/medgemma"
  run_py "$CORE_ENV" unified_medgemma_analysis.py \
    --dataset iuchest --data_dir data/raw/iuchest --num_samples "$NUM_SAMPLES" \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$MED_OUT" --generate_baselines
  run_py "$CORE_ENV" unified_medgemma_analysis.py \
    --dataset iuchest --data_dir data/raw/iuchest --num_samples "$NUM_SAMPLES" \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$MED_OUT" --use_intervention --alpha_v "$AV" --gt_position mid
  eval_medical_pair iu-chest \
    "$MED_OUT/iuchest_medgemma_results_baselines.json" \
    "$MED_OUT/iuchest_medgemma_results_new_bair_av${AV}_mid.json"
fi

if [[ "$RUN_CHEXAGENT" == "1" ]]; then
  CHEX_OUT="$OUT/chexagent"
  run_py "$CORE_ENV" unified_chexagent.py \
    --dataset iuchest --data_dir data/raw/iuchest --num_samples "$NUM_SAMPLES" \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$CHEX_OUT" --generate_baselines
  run_py "$CORE_ENV" unified_chexagent.py \
    --dataset iuchest --data_dir data/raw/iuchest --num_samples "$NUM_SAMPLES" \
    --device_id "$DEVICE_ID" --seed "$SEED" --max_new_tokens "$MAX_MED_TOKENS" \
    --output_dir "$CHEX_OUT" --use_intervention --alpha_v "$AV" --gt_position mid
  eval_medical_pair iu-chest \
    "$CHEX_OUT/iuchest_chexagent_results_baselines.json" \
    "$CHEX_OUT/iuchest_chexagent_results_new_bair_av${AV}_mid.json"
fi

FACET_INSTRUCTION="You are a helpful assistant. When context is provided, refer to it to accurately describe the image. If no context is provided, describe the image based on your knowledge."

if [[ "$RUN_FACET_QWEN" == "1" ]]; then
  FQ_OUT="$OUT/facet_qwen"
  run_py "$CORE_ENV" gender_analysis.py \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --num_samples "$NUM_SAMPLES" --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FQ_OUT" --output_dir "$FQ_OUT" \
    --model_names Qwen/Qwen2.5-VL-3B-Instruct --generation_only --instruction "$FACET_INSTRUCTION"
  FQ_BASE="$FQ_OUT/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2.json"
  FQ_BAIR="$FQ_OUT/analysis_results_Qwen_Qwen2.5_VL_3B_Instruct_with_instruction_2_intervention_av${AV}.json"
  run_py "$CORE_ENV" gender_analysis.py \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --num_samples "$NUM_SAMPLES" --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FQ_OUT" --output_dir "$FQ_OUT" \
    --model_names Qwen/Qwen2.5-VL-3B-Instruct --use_intervention --alpha_v "$AV" \
    --from_analysis_results "$FQ_BASE" --intervention_output_json "$FQ_BAIR" --instruction "$FACET_INSTRUCTION"
  eval_facet_pair "$FQ_BASE" "$FQ_BAIR" facet_qwen
fi

if [[ "$RUN_FACET_DEEPSEEK" == "1" ]]; then
  FD_OUT="$OUT/facet_deepseek"
  run_py "$DEEPSEEK_ENV" gender_analysis.py \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --num_samples "$NUM_SAMPLES" --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FD_OUT" --output_dir "$FD_OUT" \
    --model_names deepseek-ai/deepseek-vl-7b-chat --generation_only --instruction "$FACET_INSTRUCTION"
  FD_BASE="$FD_OUT/analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2.json"
  FD_BAIR="$FD_OUT/analysis_results_deepseek_ai_deepseek_vl_7b_chat_with_instruction_2_intervention_av${AV}.json"
  run_py "$DEEPSEEK_ENV" gender_analysis.py \
    --facet_csv data/metadata/facet_new_annotations.csv --image_root data/raw/facet/image \
    --num_samples "$NUM_SAMPLES" --device_id "$DEVICE_ID" --seed "$SEED" \
    --max_new_tokens "$MAX_FACET_TOKENS" --json_output_dir "$FD_OUT" --output_dir "$FD_OUT" \
    --model_names deepseek-ai/deepseek-vl-7b-chat --use_intervention --alpha_v "$AV" \
    --from_analysis_results "$FD_BASE" --intervention_output_json "$FD_BAIR" --instruction "$FACET_INSTRUCTION"
  eval_facet_pair "$FD_BASE" "$FD_BAIR" facet_deepseek
fi

if [[ "$RUN_NWPU_SKYSENSE" == "1" ]]; then
  NS_OUT="$OUT/nwpu_skysense"
  run_py "$SKYSENSE_ENV" nwpu_analysis.py \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name ll-13/SkySenseGPT-7B-clip-lora --num-samples "$NUM_SAMPLES" \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NS_OUT" --generate-baselines
  run_py "$SKYSENSE_ENV" nwpu_analysis.py \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name ll-13/SkySenseGPT-7B-clip-lora --num-samples "$NUM_SAMPLES" \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NS_OUT" --use-intervention --alpha-v "$AV"
  eval_nwpu_pair \
    "$NS_OUT/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_baselines.json" \
    "$NS_OUT/nwpu_results_ll_13_SkySenseGPT_7B_clip_lora_bair_av${AV}.json" \
    nwpu_skysense
fi

if [[ "$RUN_NWPU_EARTHDIAL" == "1" ]]; then
  NE_OUT="$OUT/nwpu_earthdial"
  run_py "$EARTHDIAL_ENV" nwpu_analysis.py \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name akshaydudhane/EarthDial_4B_RGB --num-samples "$NUM_SAMPLES" \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NE_OUT" --generate-baselines
  run_py "$EARTHDIAL_ENV" nwpu_analysis.py \
    --dataset-json data/generated/nwpu_retrieval_dataset.json \
    --model-name akshaydudhane/EarthDial_4B_RGB --num-samples "$NUM_SAMPLES" \
    --device-id "$DEVICE_ID" --seed "$SEED" --max-new-tokens "$MAX_NWPU_TOKENS" \
    --output-dir "$NE_OUT" --use-intervention --alpha-v "$AV"
  eval_nwpu_pair \
    "$NE_OUT/nwpu_results_akshaydudhane_EarthDial_4B_RGB_baselines.json" \
    "$NE_OUT/nwpu_results_akshaydudhane_EarthDial_4B_RGB_bair_av${AV}.json" \
    nwpu_earthdial
fi

echo
echo "Smoke suite finished. Outputs are under $OUT"
