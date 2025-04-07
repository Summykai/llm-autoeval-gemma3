#!/bin/bash
start=$(date +%s)

# --- Dependencies ---
apt update > /dev/null
apt install -y -qq --no-install-recommends screen vim git-lfs > /dev/null
# Ensure bitsandbytes is installed for optional 4-bit quantization
pip install -q requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4 bitsandbytes
pip install -U -q transformers lm-eval

# --- Environment Setup ---
# Read Env Vars including new QUANTIZE flag
# MODEL_ID, BENCHMARK, QUANTIZE, EVAL_LIMIT, DEBUG, HUGGINGFACE_TOKEN,
# GITHUB_API_TOKEN, TRUST_REMOTE_CODE, PRIVATE_GIST, LIGHT_EVAL_TASK, RUNPOD_POD_ID

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "INFO: Logging in to Hugging Face Hub..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
fi

VERBOSITY="INFO"
if [ "$DEBUG" == "True" ]; then
    echo "INFO: Debug mode enabled."
    VERBOSITY="DEBUG"
fi

LIMIT_FLAG=""
if [[ -n "$EVAL_LIMIT" && "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "INFO: EVAL_LIMIT is set to $EVAL_LIMIT. Limiting evaluation instances per task."
    LIMIT_FLAG="--limit $EVAL_LIMIT"
elif [[ -n "$EVAL_LIMIT" ]]; then
    echo "WARN: Invalid EVAL_LIMIT value '$EVAL_LIMIT'. Using full evaluation."
fi

# --- Model & Evaluation Arguments ---

# Define common base arguments (no quantization yet)
# Using harness 'max_length' argument to suggest context size (e.g., 8192).
MODEL_CONTEXT_LENGTH=8192
BASE_ARGS_COMMON="pretrained=${MODEL_ID},dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE,max_length=$MODEL_CONTEXT_LENGTH"

# Define the 4-bit quantization arguments
QUANT_ARGS_4BIT=",load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True"

# Initialize BASE_MODEL_ARGS with common args
BASE_MODEL_ARGS="$BASE_ARGS_COMMON"
QUANTIZATION_ENABLED="false" # Flag to track status

# Conditionally add quantization arguments based on QUANTIZE env var (case-insensitive check)
if [[ "${QUANTIZE,,}" == "true" ]]; then
    echo "INFO: 4-bit quantization ENABLED."
    BASE_MODEL_ARGS+="$QUANT_ARGS_4BIT"
    QUANTIZATION_ENABLED="true"
else
    echo "INFO: 4-bit quantization DISABLED. Ensure sufficient VRAM for full precision."
fi
echo "INFO: Final BASE_MODEL_ARGS: $BASE_MODEL_ARGS"


# Gemma 3 specific args for applicable benchmarks
GEMMA_CHAT_ARGS="--apply_chat_template --fewshot_as_multiturn"
# GSM8K generation args (includes sampling enabled)
GEMMA_GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=64,do_sample=True"

# --- Evaluation Execution ---

if [ "$BENCHMARK" == "nous" ]; then
    # Nous benchmark uses a specific fork/script setup. Quantization args likely NOT compatible.
    echo "INFO: Running Nous Benchmark Suite (Applying bfloat16 dtype - Quantization may not apply)"
    git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness nous-lm-eval-harness > /dev/null
    cd nous-lm-eval-harness
    pip install -q -e .
    NOUS_MODEL_ARGS="pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16"
    NOUS_TASKS=( "agieval:..." "gpt4all:..." "truthfulqa:..." "bigbench:..." ) # Truncated for brevity
    # (Loop for nous tasks remains the same)
     for item in "${NOUS_TASKS[@]}"; do IFS=":" read -r bn tl <<< "$item"; echo "=== NOUS - $(echo "$bn" | tr '[:lower:]' '[:upper:]') ==="; python main.py --model hf-causal --model_args "$NOUS_MODEL_ARGS" --tasks "$tl" --device cuda --batch_size auto --output_path "../${bn}.json" --verbosity "$VERBOSITY" $LIMIT_FLAG; done
    cd ..

elif [ "$BENCHMARK" == "openllm" ] || [ "$BENCHMARK" == "gemma3" ]; then
    if [ "$BENCHMARK" == "openllm" ]; then echo "INFO: Running OpenLLM Benchmark Suite (Applying Gemma 3 settings where applicable)"; else echo "INFO: Running dedicated Gemma 3 Benchmark Suite"; fi
    echo "INFO: Using BASE_MODEL_ARGS: $BASE_MODEL_ARGS" # Log the final args

    # ARC
    echo "================== ARC [1/6] (Original 25-shot) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks arc_challenge --num_fewshot 25 --batch_size auto --output_path ./arc_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG
    # HellaSwag
    echo "================== HELLASWAG [2/6] (Gemma 3 Spec: 10-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks hellaswag --num_fewshot 10 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./hellaswag_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG
    # MMLU
    echo "================== MMLU [3/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks mmlu --num_fewshot 5 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./mmlu_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG
    # TruthfulQA
    echo "================== TRUTHFULQA [4/6] (Original 0-shot) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks truthfulqa --num_fewshot 0 --batch_size auto --output_path ./truthfulqa_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG
    # WinoGrande
    echo "================== WINOGRANDE [5/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks winogrande --num_fewshot 5 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./winogrande_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG
    # GSM8K
    echo "================== GSM8K [6/6] (Gemma 3 Spec: 8-shot, Chat Format, Sampling) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks gsm8k --num_fewshot 8 --batch_size auto $GEMMA_CHAT_ARGS --gen_kwargs "$GEMMA_GEN_KWARGS" --output_path ./gsm8k_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

elif [ "$BENCHMARK" == "lighteval" ]; then
    echo "INFO: Running Lighteval Benchmark Suite..."
    git clone https://github.com/huggingface/lighteval.git > /dev/null
    cd lighteval
    pip install -q '.[accelerate,quantization,adapters]'

    # Adapt model args for lighteval format - Conditionally add quantization
    LIGHTEVAL_MODEL_ARGS="$BASE_ARGS_COMMON" # Start with common args (incl max_length)
    if [[ "$QUANTIZATION_ENABLED" == "true" ]]; then
        echo "INFO: Enabling 4-bit quantization for Lighteval."
        # Assuming lighteval uses same/similar args, check its docs if needed
        LIGHTEVAL_MODEL_ARGS+="$QUANT_ARGS_4BIT"
    else
         echo "INFO: Running Lighteval without 4-bit quantization."
    fi
    echo "INFO: Lighteval model_args: $LIGHTEVAL_MODEL_ARGS"

    echo "INFO: Running lighteval with accelerate..."
    accelerate launch run_evals_accelerate.py \
        --model_args "$LIGHTEVAL_MODEL_ARGS" \
        --use_chat_template \
        --tasks "${LIGHT_EVAL_TASK}" \
        --output_dir="./evals/" \
        $LIMIT_FLAG
    cd ..

elif [ "$BENCHMARK" == "eq-bench" ]; then
    echo "INFO: Running EQ-Bench (Using quantization setting: $QUANTIZATION_ENABLED)"
    benchmark="eq-bench"
    echo "================== $(echo "$benchmark" | tr '[:lower:]' '[:upper:]') [1/1] =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args "$BASE_MODEL_ARGS" \ # Uses the conditionally quantized args
        --tasks eq_bench \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path ./evals/${benchmark}_eval.json \
        --verbosity "$VERBOSITY" \
        $LIMIT_FLAG

    mkdir -p ./evals

else
    echo "ERROR: Invalid BENCHMARK value. Please set BENCHMARK to 'nous', 'openllm', 'gemma3', 'lighteval', or 'eq-bench'."
    exit 1
fi

# --- Post-processing and Cleanup ---
end=$(date +%s)
elapsed_time=$(($end-$start))
echo "INFO: Evaluation Complete. Elapsed Time: $elapsed_time seconds"

output_dir="."
if [ "$BENCHMARK" == "lighteval" ]; then output_dir="./lighteval/evals/"; fi
if [ "$BENCHMARK" == "eq-bench" ]; then output_dir="./evals"; fi
if [ "$BENCHMARK" == "nous" ]; then output_dir="."; fi

script_dir=$(dirname "$0")
consolidation_script_path=""
if [ -f "${script_dir}/main.py" ]; then consolidation_script_path="${script_dir}/main.py";
elif [ -f "${script_dir}/../main.py" ]; then consolidation_script_path="${script_dir}/../main.py"; fi

if [ -n "$consolidation_script_path" ]; then
    echo "INFO: Running consolidation script from: $consolidation_script_path"
    mkdir -p "$output_dir"
    if ls "$output_dir" | grep -q '\.json'; then
       echo "INFO: Attempting to run consolidation. Ensure main.py handles nested directories if applicable."
       python "$consolidation_script_path" "$output_dir" "$elapsed_time"
    else
       echo "WARN: No JSON result items found in '$output_dir'. Skipping result consolidation."
    fi
else
    echo "WARN: Consolidation script 'main.py' not found relative to runpod.sh. Skipping result consolidation."
fi

if [ "$DEBUG" == "False" ]; then
    echo "INFO: Evaluation finished. Removing pod."
    if command -v runpodctl &> /dev/null && [ -n "$RUNPOD_POD_ID" ]; then
        runpodctl remove pod "$RUNPOD_POD_ID"
    else
        echo "WARN: runpodctl not found or RUNPOD_POD_ID not set. Cannot remove pod automatically."
    fi
else
    echo "INFO: DEBUG mode is True. Pod will remain running."
    sleep infinity
fi

exit 0
