#!/bin/bash
# Start timer
start=$(date +%s)

# --- Gemma 3 Evaluation Settings (Applied where applicable below) ---
# Reference: Gemma 3 Technical Report
# Key Settings potentially applied:
# - Model Args: Use bfloat16, trust_remote_code=True, explicit model_max_length
# - Input Formatting: Specific turn structure (<start_of_turn>user/model ... <end_of_turn>) with BOS prepended.
#   -> Implemented via --apply_chat_template and --fewshot_as_multiturn flags in lm-eval for specific benchmarks (HellaSwag, MMLU, WinoGrande, GSM8K).
#   -> Assumes the tokenizer for MODEL_ID implements the correct Gemma 3 template.
# - Default Inference (Sampling): temperature=1.0, top_p=0.95, top_k=64, do_sample=True
#   -> Implemented via --gen_kwargs for sampling tasks (GSM8K).
# - Benchmark Specifics: N-shot counts adjusted for HellaSwag (10), MMLU (5), WinoGrande (5), GSM8K (8).
# ---

# --- Configuration ---
# Set EVAL_LIMIT environment variable to a positive integer to limit tasks to that many instances.
# Example: export EVAL_LIMIT=100

# Detect the number of NVIDIA GPUs
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
echo "Detected $gpu_count NVIDIA GPUs."

# Install dependencies
apt update
apt install -y screen vim git-lfs

# Install common libraries
pip install -q requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4
pip install -U transformers # Ensure latest transformers
pip install -U lm-eval # Install or update lm-eval harness directly

# Check if HUGGINGFACE_TOKEN is set and log in to Hugging Face
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "HUGGINGFACE_TOKEN is defined. Logging in..."
    huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
fi

# Set Verbosity based on DEBUG flag
if [ "$DEBUG" == "True" ]; then
    echo "Launch LLM AutoEval in debug mode"
    VERBOSITY="DEBUG"
else
    VERBOSITY="INFO"
fi

# --- Evaluation Limit Configuration ---
LIMIT_FLAG=""
if [[ -n "$EVAL_LIMIT" && "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    # Check if it's a positive integer
    echo "INFO: EVAL_LIMIT is set to $EVAL_LIMIT. Limiting evaluation instances per task."
    LIMIT_FLAG="--limit $EVAL_LIMIT"
elif [[ -n "$EVAL_LIMIT" ]]; then
    # Check if it was set but invalid
    echo "WARN: Invalid EVAL_LIMIT value '$EVAL_LIMIT'. Must be a positive integer. Running full evaluation."
else
    # Not set, run full evaluation
    echo "INFO: EVAL_LIMIT not set. Running full evaluation for all tasks."
fi

# --- Common Model Arguments (Adjusted for Gemma 3 where applicable) ---
# Using bfloat16 as typically recommended for recent models like Gemma 3
# TRUST_REMOTE_CODE is passed as an environment variable
# **Explicitly set model_max_length to prevent incorrect detection (e.g., 2048) by the harness**
# Using 131072 as specified by user. Reduce if OOM occurs. Gemma 3 base often uses 8192.
MODEL_MAX_LENGTH=131072
echo "INFO: Setting model_max_length to $MODEL_MAX_LENGTH in model args."
BASE_MODEL_ARGS="pretrained=${MODEL_ID},dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE,model_max_length=$MODEL_MAX_LENGTH"

# --- Evaluation Execution ---

# Choose evaluation suite based on BENCHMARK environment variable
if [ "$BENCHMARK" == "nous" ]; then
    # This benchmark suite uses a specific fork and tasks.
    # Applying Gemma 3's bfloat16 dtype, but keeping task/fewshot/prompting structure as defined by this suite.
    # Note: Max length and specific Gemma 3 formatting might not apply correctly here due to different script structure.
    echo "Running Nous Benchmark Suite (Applying bfloat16 dtype)"
    git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness # Using specific fork for nous tasks
    cd lm-evaluation-harness
    pip install -e .

    # Nous model args - applying bfloat16 but *not* max_length override as it might not be supported by this older structure
    NOUS_MODEL_ARGS="pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16"

    benchmark="agieval"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/4] =================="
    python main.py \
        --model hf-causal \
        --model_args $NOUS_MODEL_ARGS \
        --tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
        --device cuda \ # Note: Nous script used explicit device, may need adjustment for multi-GPU
        --batch_size auto \
        --output_path ../${benchmark}.json \ # Save output one level up
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    benchmark="gpt4all" # Note: This group name includes hellaswag/winogrande but uses nous setup
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [2/4] =================="
    python main.py \
        --model hf-causal \
        --model_args $NOUS_MODEL_ARGS \
        --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
        --device cuda \
        --batch_size auto \
        --output_path ../${benchmark}.json \ # Save output one level up
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    benchmark="truthfulqa"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [3/4] =================="
    python main.py \
        --model hf-causal \
        --model_args $NOUS_MODEL_ARGS \
        --tasks truthfulqa_mc \
        --device cuda \
        --batch_size auto \
        --output_path ../${benchmark}.json \ # Save output one level up
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    benchmark="bigbench"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [4/4] =================="
    python main.py \
        --model hf-causal \
        --model_args $NOUS_MODEL_ARGS \
        --tasks bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects \
        --device cuda \
        --batch_size auto \
        --output_path ../${benchmark}.json \ # Save output one level up
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    cd .. # Return to base directory

elif [ "$BENCHMARK" == "openllm" ] || [ "$BENCHMARK" == "gemma3" ]; then
    # This block runs the standard OpenLLM leaderboard tasks, applying Gemma 3 specific settings where appropriate.
    if [ "$BENCHMARK" == "openllm" ]; then
        echo "Running OpenLLM Benchmark Suite (Applying Gemma 3 settings where applicable)"
    else
         echo "Running dedicated Gemma 3 Benchmark Suite (Subset of OpenLLM tasks with Gemma 3 settings)"
    fi

    # Use the official lm-evaluation-harness installed via pip
    # Common model args defined above: $BASE_MODEL_ARGS (includes max_length override)

    # --- Gemma 3 Specific Formatting/Generation Arguments ---
    # Apply chat template and format few-shot as multi-turn for specific tasks
    GEMMA_CHAT_ARGS="--apply_chat_template --fewshot_as_multiturn"
    # Sampling parameters for GSM8K - IMPORTANT: Added do_sample=True
    GEMMA_GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=64,do_sample=True"

    # --- Benchmark Runs ---

    # ARC Challenge (Original OpenLLM setup: 25-shot)
    # Using BASE_MODEL_ARGS (incl max_length), keeping original shot count. Chat template not applied.
    benchmark="arc"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/6] (Original 25-shot) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size auto \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    # HellaSwag (Gemma 3 Spec: 10-shot, Chat Format)
    benchmark="hellaswag"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [2/6] (Gemma 3 Spec: 10-shot, Chat Format) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks hellaswag \
        --num_fewshot 10 `# Gemma 3 Spec` \
        --batch_size auto \
        $GEMMA_CHAT_ARGS `# Gemma 3 Spec` \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    # MMLU (Gemma 3 Spec: 5-shot, Chat Format)
    # Using BASE_MODEL_ARGS (incl max_length override)
    benchmark="mmlu"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [3/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks mmlu \
        --num_fewshot 5 `# Gemma 3 Spec` \
        --batch_size auto \
        $GEMMA_CHAT_ARGS `# Gemma 3 Spec` \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    # TruthfulQA (Original OpenLLM setup: 0-shot)
    # Using BASE_MODEL_ARGS (incl max_length), keeping original shot count. Chat template not applied (0-shot).
    benchmark="truthfulqa"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [4/6] (Original 0-shot) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks truthfulqa \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    # WinoGrande (Gemma 3 Spec: 5-shot, Chat Format)
    benchmark="winogrande"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [5/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks winogrande \
        --num_fewshot 5 `# Gemma 3 Spec` \
        --batch_size auto \
        $GEMMA_CHAT_ARGS `# Gemma 3 Spec` \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    # GSM8K (Gemma 3 Spec: 8-shot, Chat Format, Sampling Params)
    # Using BASE_MODEL_ARGS (incl max_length), applying specific gen_kwargs (incl do_sample=True)
    benchmark="gsm8k"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [6/6] (Gemma 3 Spec: 8-shot, Chat Format, Sampling) =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks gsm8k \
        --num_fewshot 8 `# Gemma 3 Spec` \
        --batch_size auto \
        $GEMMA_CHAT_ARGS `# Gemma 3 Spec` \
        --gen_kwargs "$GEMMA_GEN_KWARGS" `# Gemma 3 Spec + do_sample=True` \
        --output_path ./${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

elif [ "$BENCHMARK" == "lighteval" ]; then
    # This benchmark uses the lighteval framework.
    # Applying Gemma 3's bfloat16 dtype. Using lighteval's own --use_chat_template.
    # Adding model_max_length override here too for consistency.
    echo "Running Lighteval Benchmark Suite (Applying bfloat16 dtype, max_length, using lighteval's chat template flag)"
    git clone https://github.com/huggingface/lighteval.git
    cd lighteval
    pip install '.[accelerate,quantization,adapters]' # Install lighteval with extras

    # Lighteval model args - Add model_max_length
    LIGHTEVAL_MODEL_ARGS="pretrained=${MODEL_ID},trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,model_max_length=$MODEL_MAX_LENGTH"

    echo "Running lighteval with accelerate..."
    accelerate launch run_evals_accelerate.py \
        --model_args "$LIGHTEVAL_MODEL_ARGS" \
        --use_chat_template `# Lighteval's flag, assumes it uses the correct template from tokenizer` \
        --tasks "${LIGHT_EVAL_TASK}" \
        --output_dir="./evals/" \
        $LIMIT_FLAG # Apply limit if set (assuming lighteval uses --limit)
        # Add other lighteval specific args as needed, e.g., --override_batch_size

    cd .. # Return to base directory

elif [ "$BENCHMARK" == "eq-bench" ]; then
    # This benchmark uses lm-eval harness for the eq-bench task.
    # Applying Gemma 3's bfloat16 dtype and max_length. Keeping 0-shot setup. Chat template not applied (0-shot).
    echo "Running EQ-Bench (Applying bfloat16 dtype, max_length, original 0-shot)"
    # Assuming lm-eval is already installed

    benchmark="eq-bench"
    echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/1] =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args $BASE_MODEL_ARGS \
        --tasks eq_bench \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./evals/${benchmark}_eval.json \
        --verbosity $VERBOSITY \
        $LIMIT_FLAG # Apply limit if set

    mkdir -p ./evals # Ensure output directory exists

else
    echo "Error: Invalid BENCHMARK value. Please set BENCHMARK to 'nous', 'openllm', 'gemma3', 'lighteval', or 'eq-bench'."
    exit 1
fi

# --- Post-processing and Cleanup ---
end=$(date +%s)
elapsed_time=$(($end-$start))
echo "Evaluation Complete. Elapsed Time: $elapsed_time seconds"

# Consolidate results (assuming llm-autoeval/main.py exists in the current dir or one level up)
# Adjust the path passed to main.py based on where the output files are saved.
output_dir="." # Default for nous, openllm/gemma3 benchmarks saved in current dir relative to script exec
if [ "$BENCHMARK" == "lighteval" ]; then
    # Lighteval results are inside ./lighteval/evals/
    output_dir="./lighteval/evals/"
elif [ "$BENCHMARK" == "eq-bench" ]; then
    output_dir="./evals" # Specific path for eq-bench results
elif [ "$BENCHMARK" == "nous" ]; then
    output_dir="." # nous results were saved relative to where python main.py ran (../ -> .)
fi


# Find the consolidation script robustly relative to this script's location
script_dir=$(dirname "$0")
consolidation_script_path=""
# Look relative to script dir
if [ -f "${script_dir}/main.py" ]; then
     consolidation_script_path="${script_dir}/main.py"
# Look in parent directory (if script is inside llm-autoeval dir)
elif [ -f "${script_dir}/../main.py" ]; then
     consolidation_script_path="${script_dir}/../main.py"
fi


if [ -n "$consolidation_script_path" ]; then
    echo "Running consolidation script from: $consolidation_script_path"
    # Ensure output_dir exists before passing it
    mkdir -p "$output_dir"
    # Check if results files/dirs actually exist before calling consolidation
    # Heuristic: Check for any *.json item (file or dir) in the expected output directory
    if ls "${output_dir}" | grep -q '\.json'; then
       python "$consolidation_script_path" "$output_dir" $elapsed_time
    else
       echo "Warning: No JSON result files/dirs found in '$output_dir'. Skipping result consolidation."
    fi
else
    echo "Warning: Consolidation script 'main.py' (from llm-autoeval) not found. Skipping result consolidation."
fi

# Optional: Cleanup or keep pod running for debugging
if [ "$DEBUG" == "False" ]; then
    echo "Evaluation finished. Removing pod."
    # Check if runpodctl exists and RUNPOD_POD_ID is set before attempting removal
    if command -v runpodctl &> /dev/null && [ -n "$RUNPOD_POD_ID" ]; then
        runpodctl remove pod $RUNPOD_POD_ID
    else
        echo "runpodctl not found or RUNPOD_POD_ID not set. Cannot remove pod automatically."
    fi
else
    echo "DEBUG mode is True. Pod will remain running. Connect via SSH or RunPod UI."
    sleep infinity # Keep the pod running indefinitely
fi

exit 0
