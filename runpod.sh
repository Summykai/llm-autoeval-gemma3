#!/bin/bash
start=$(date +%s)

# --- Dependencies ---
apt update > /dev/null
apt install -y -qq --no-install-recommends screen vim git-lfs > /dev/null
pip install -q requests accelerate sentencepiece pytablewriter einops protobuf huggingface_hub==0.21.4 bitsandbytes
pip install -U -q transformers lm-eval

# --- Environment Setup ---
# Read Env Vars: MODEL_ID, BENCHMARK, EVAL_LIMIT, DEBUG,
# HUGGINGFACE_TOKEN, GITHUB_API_TOKEN, TRUST_REMOTE_CODE,
# PRIVATE_GIST, LIGHT_EVAL_TASK, RUNPOD_POD_ID

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

# Base args for HF models: 4-bit NF4 quantization, bfloat16 compute, trust remote code
# Relying on model's config.json for max sequence length.
BASE_MODEL_ARGS="pretrained=${MODEL_ID},dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE,load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True"
echo "INFO: Using BASE_MODEL_ARGS with 4-bit NF4 quantization: $BASE_MODEL_ARGS"

# Gemma 3 specific args for applicable benchmarks
GEMMA_CHAT_ARGS="--apply_chat_template --fewshot_as_multiturn"
# GSM8K generation args (includes sampling enabled)
GEMMA_GEN_KWARGS="temperature=1.0,top_p=0.95,top_k=64,do_sample=True"

# --- Evaluation Execution ---

if [ "$BENCHMARK" == "nous" ]; then
    # Nous benchmark uses a specific fork/script setup. Quantization args may not apply.
    echo "INFO: Running Nous Benchmark Suite (Applying bfloat16 dtype)"
    git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness nous-lm-eval-harness > /dev/null
    cd nous-lm-eval-harness
    pip install -q -e .

    NOUS_MODEL_ARGS="pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16"
    NOUS_TASKS=(
        "agieval:agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math"
        "gpt4all:hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa"
        "truthfulqa:truthfulqa_mc"
        "bigbench:bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects"
    )
    for item in "${NOUS_TASKS[@]}"; do
        IFS=":" read -r benchmark_name task_list <<< "$item"
        echo "================== NOUS - $(echo "$benchmark_name" | tr '[:lower:]' '[:upper:]') =================="
        python main.py --model hf-causal --model_args "$NOUS_MODEL_ARGS" --tasks "$task_list" --device cuda --batch_size auto --output_path "../${benchmark_name}.json" --verbosity "$VERBOSITY" $LIMIT_FLAG
    done
    cd ..

elif [ "$BENCHMARK" == "openllm" ] || [ "$BENCHMARK" == "gemma3" ]; then
    if [ "$BENCHMARK" == "openllm" ]; then
        echo "INFO: Running OpenLLM Benchmark Suite (Applying Gemma 3 settings where applicable)"
    else
         echo "INFO: Running dedicated Gemma 3 Benchmark Suite"
    fi
    echo "INFO: Using BASE_MODEL_ARGS: $BASE_MODEL_ARGS"

    # ARC (Original 25-shot)
    echo "================== ARC [1/6] (Original 25-shot) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks arc_challenge --num_fewshot 25 --batch_size auto --output_path ./arc_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

    # HellaSwag (Gemma 3 Spec)
    echo "================== HELLASWAG [2/6] (Gemma 3 Spec: 10-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks hellaswag --num_fewshot 10 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./hellaswag_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

    # MMLU (Gemma 3 Spec)
    echo "================== MMLU [3/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks mmlu --num_fewshot 5 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./mmlu_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

    # TruthfulQA (Original 0-shot)
    echo "================== TRUTHFULQA [4/6] (Original 0-shot) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks truthfulqa --num_fewshot 0 --batch_size auto --output_path ./truthfulqa_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

    # WinoGrande (Gemma 3 Spec)
    echo "================== WINOGRANDE [5/6] (Gemma 3 Spec: 5-shot, Chat Format) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks winogrande --num_fewshot 5 --batch_size auto $GEMMA_CHAT_ARGS --output_path ./winogrande_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

    # GSM8K (Gemma 3 Spec)
    echo "================== GSM8K [6/6] (Gemma 3 Spec: 8-shot, Chat Format, Sampling) =================="
    accelerate launch -m lm_eval --model hf --model_args "$BASE_MODEL_ARGS" --tasks gsm8k --num_fewshot 8 --batch_size auto $GEMMA_CHAT_ARGS --gen_kwargs "$GEMMA_GEN_KWARGS" --output_path ./gsm8k_eval.json --verbosity "$VERBOSITY" $LIMIT_FLAG

elif [ "$BENCHMARK" == "lighteval" ]; then
    echo "INFO: Running Lighteval Benchmark Suite (Applying 4-bit NF4, using lighteval's chat template flag)"
    git clone https://github.com/huggingface/lighteval.git > /dev/null
    cd lighteval
    pip install -q '.[accelerate,quantization,adapters]'

    # Adapt model args for lighteval format
    LIGHTEVAL_MODEL_ARGS="pretrained=${MODEL_ID},trust_remote_code=${TRUST_REMOTE_CODE},dtype=bfloat16,load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True"
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
    echo "INFO: Running EQ-Bench (Applying 4-bit NF4, original 0-shot)"
    benchmark="eq-bench"
    echo "================== $(echo "$benchmark" | tr '[:lower:]' '[:upper:]') [1/1] =================="
    accelerate launch -m lm_eval \
        --model hf \
        --model_args "$BASE_MODEL_ARGS" \
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

# Define output directory based on benchmark for consolidation
output_dir="." # Default for openllm/gemma3/nous
if [ "$BENCHMARK" == "lighteval" ]; then
    output_dir="./lighteval/evals/"
elif [ "$BENCHMARK" == "eq-bench" ]; then
    output_dir="./evals"
fi

# Find consolidation script relative to this script
script_dir=$(dirname "$0")
consolidation_script_path=""
if [ -f "${script_dir}/main.py" ]; then
     consolidation_script_path="${script_dir}/main.py"
elif [ -f "${script_dir}/../main.py" ]; then # If runpod.sh is in a subdir
     consolidation_script_path="${script_dir}/../main.py"
fi

# Run consolidation if script found and results exist
if [ -n "$consolidation_script_path" ]; then
    echo "INFO: Running consolidation script from: $consolidation_script_path"
    mkdir -p "$output_dir"
    # Check if any .json file/dir exists in the output dir
    if ls "$output_dir" | grep -q '\.json'; then
       echo "INFO: Attempting to run consolidation. Ensure main.py handles nested directories if applicable."
       python "$consolidation_script_path" "$output_dir" "$elapsed_time"
    else
       echo "WARN: No JSON result items found in '$output_dir'. Skipping result consolidation."
    fi
else
    echo "WARN: Consolidation script 'main.py' not found relative to runpod.sh. Skipping result consolidation."
fi

# Cleanup or keep pod running
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
