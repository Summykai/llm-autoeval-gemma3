import math
import os
import logging # Added for better debugging

from pytablewriter import MarkdownTableWriter

logger = logging.getLogger(__name__) # Added logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


BENCHMARK = os.getenv("BENCHMARK", "Unknown_Benchmark") # Added default

# --- Helper functions to extract metrics ---
# These seem generic enough based on lm-harness output structure

def get_acc_norm(data):
    """Calculates average acc_norm or acc across tasks in results."""
    try:
        accs = []
        for k in data.get("results", {}):
            result = data["results"][k]
            if "acc_norm" in result:
                accs.append(result["acc_norm"])
            elif "acc" in result:
                accs.append(result["acc"])
            else:
                logger.warning(f"Task {k} has no 'acc_norm' or 'acc' metric.")
        
        if not accs:
             logger.error("No accuracy values found to calculate average acc_norm.")
             return 0.0 # Or raise error

        acc = sum(accs) / len(accs) * 100
        return acc
    except Exception as e:
        logger.error(f"Error in get_acc_norm: {e}")
        return 0.0

def get_mcg(data):
    """Calculates average multiple_choice_grade across tasks in results."""
    try:
        # Ensure 'results' key exists and is a dictionary
        results_data = data.get("results", {})
        if not isinstance(results_data, dict):
             logger.error("'results' key not found or is not a dictionary in get_mcg data.")
             return 0.0

        accs = []
        for k in results_data:
            if "multiple_choice_grade" in results_data[k]:
                accs.append(results_data[k]["multiple_choice_grade"])
            else:
                 logger.warning(f"Task {k} has no 'multiple_choice_grade' metric.")

        if not accs:
            logger.error("No multiple_choice_grade values found.")
            return 0.0

        acc = sum(accs) / len(accs) * 100
        return acc
    except Exception as e:
        logger.error(f"Error in get_mcg: {e}")
        return 0.0

# --- Main function to calculate the single average score for a benchmark ---

def calculate_average(data, task):
    """
    Calculates the primary average score for a specific task *suite*
    based on the benchmark type. Note: 'task' here often refers to the
    benchmark suite name (e.g., "ARC", "MMLU") passed from main.py,
    not the individual subtask keys inside the JSON (e.g., "arc_challenge").
    """
    task_suite_name = task.lower() # Use the overall task name passed from main.py
    logger.info(f"Calculating average for task suite '{task_suite_name}' within benchmark '{BENCHMARK}'")
    # logger.debug(f"Full data received: {data}") # Optional: Log full data for deep debug

    results = data.get("results", {})
    if not results:
        logger.error("No 'results' key found in the provided data.")
        return "Error: No results data"

    try:
        # --- OpenLLM / Gemma3 Benchmarks ---
        if BENCHMARK == "openllm" or BENCHMARK == "gemma3": # Added gemma3 recognition
            if task_suite_name == "arc":
                # ARC uses arc_challenge task results
                value = results.get("arc_challenge", {}).get("acc_norm,none", None)
                metric_key = "acc_norm,none"
                task_key = "arc_challenge"
            elif task_suite_name == "hellaswag":
                value = results.get("hellaswag", {}).get("acc_norm,none", None)
                metric_key = "acc_norm,none"
                task_key = "hellaswag"
            elif task_suite_name == "mmlu":
                 # MMLU overall score might be calculated differently or use a specific key if available
                 # Often, people average the subtask accs. Let's check for a dedicated key first.
                 value = results.get("mmlu", {}).get("acc,none", None) # Check if harness provides aggregate
                 metric_key = "acc,none"
                 task_key = "mmlu"
                 # If aggregate "mmlu" key doesn't exist, maybe calculate from subtasks? (More complex)
                 # For now, rely on the harness providing the aggregate if this key is used.
            elif task_suite_name == "truthfulqa":
                 # OpenLLM runs truthfulqa_mc2 usually
                 value = results.get("truthfulqa_mc2", {}).get("acc,none", None)
                 metric_key = "acc,none"
                 task_key = "truthfulqa_mc2"
            elif task_suite_name == "winogrande":
                 value = results.get("winogrande", {}).get("acc,none", None)
                 metric_key = "acc,none"
                 task_key = "winogrande"
            elif task_suite_name == "gsm8k":
                 value = results.get("gsm8k", {}).get("exact_match,strict-match", None)
                 metric_key = "exact_match,strict-match"
                 task_key = "gsm8k"
            else:
                 logger.error(f"Unknown task suite '{task_suite_name}' for benchmark '{BENCHMARK}'")
                 return f"Error: Unknown task suite {task_suite_name}"

            if value is None:
                 logger.error(f"Metric '{metric_key}' not found for task key '{task_key}' in results.")
                 return f"Error: Metric not found ({task_key}/{metric_key})"

            # Handle potential NaN values, convert to float, multiply by 100
            return 0.0 if math.isnan(float(value)) else float(value) * 100

        # --- Nous Benchmarks ---
        elif BENCHMARK == "nous":
            # Nous tasks often group results differently (e.g., under "AGIEval")
            # The helper functions might be more appropriate here if the JSON structure reflects that
            if task_suite_name == "agieval":
                return get_acc_norm(data) # Assumes AGIEval results are structured for this helper
            elif task_suite_name == "gpt4all":
                return get_acc_norm(data) # Assumes GPT4All results are structured for this helper
            elif task_suite_name == "bigbench":
                return get_mcg(data) # Assumes BigBench results are structured for this helper
            elif task_suite_name == "truthfulqa":
                # Nous runs truthfulqa_mc (which has mc1/mc2 inside)
                value = results.get("truthfulqa_mc", {}).get("mc2", None) # Use mc2 score
                metric_key = "mc2"
                task_key = "truthfulqa_mc"

                if value is None:
                     logger.error(f"Metric '{metric_key}' not found for task key '{task_key}' in results.")
                     return f"Error: Metric not found ({task_key}/{metric_key})"
                return 0.0 if math.isnan(float(value)) else float(value) * 100
            else:
                 logger.error(f"Unknown task suite '{task_suite_name}' for benchmark '{BENCHMARK}'")
                 return f"Error: Unknown task suite {task_suite_name}"

        # --- EQ-Bench Benchmark ---
        elif BENCHMARK == "eq-bench":
            if task_suite_name == "eq-bench":
                # Assuming eq_bench task key and eqbench,none metric key
                value = results.get("eq_bench", {}).get("eqbench,none", None)
                metric_key = "eqbench,none"
                task_key = "eq_bench"

                if value is None:
                    logger.error(f"Metric '{metric_key}' not found for task key '{task_key}' in results.")
                    return f"Error: Metric not found ({task_key}/{metric_key})"
                # EQ-Bench score might not need multiplying by 100, check its scale
                # Assuming it's a direct score here.
                return float(value)
            else:
                 logger.error(f"Unknown task suite '{task_suite_name}' for benchmark '{BENCHMARK}'")
                 return f"Error: Unknown task suite {task_suite_name}"

        else:
             logger.error(f"Benchmark '{BENCHMARK}' not implemented in calculate_average.")
             return f"Error: Benchmark {BENCHMARK} not recognized"

    except KeyError as e:
        logger.error(f"KeyError accessing results data for task '{task_suite_name}': {e}. Check JSON structure.")
        return f"Error: Missing key {e}"
    except Exception as e:
        logger.error(f"Unexpected error calculating average for task '{task_suite_name}': {e}")
        return "Error: Calculation failed"

# --- Table Formatting Functions ---

def make_table(result_dict, task):
    """Generate detailed markdown table of results for a task suite."""
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    values = []

    results_data = result_dict.get("results", {})
    versions_data = result_dict.get("versions", {})

    if not results_data:
         logger.warning(f"No 'results' key found in data for task suite '{task}'. Returning empty table.")
         return "", "Error: No results data" # Return empty table and error status

    for k, dic in sorted(results_data.items()):
        # Use N/A version as default
        version = versions_data.get(k, "N/A")
        
        # Simple check for percentage metrics (adjust if needed)
        # This check is heuristic and might need refinement based on actual metrics
        is_percent_metric = "acc" in k or "mc" in k or k == "squad2" or "bleu" in k # Example heuristic

        for m, v in dic.items():
            # Skip stderr entries, they are handled below
            if m.endswith("_stderr"):
                continue

            stderr_val = dic.get(m + "_stderr", None)
            stderr_str = ""
            stderr_disp = ""

            # Format value and stderr
            try:
                # Attempt to convert v to float for formatting
                v_float = float(v)
                # Default formatting: assume needs * 100 unless specifically known otherwise
                if m == "ppl" or not is_percent_metric : # Lower ppl is better, other non-% metrics
                    value_formatted = f"{v_float:.4f}" # More precision for non-%
                else: # Assumed percentage metric
                    value_formatted = f"{v_float * 100:.2f}"

                if stderr_val is not None:
                    se_float = float(stderr_val)
                    stderr_str = "±"
                    # Format stderr consistent with value
                    if m == "ppl" or not is_percent_metric:
                       stderr_disp = f"{se_float:.4f}"
                    else:
                       stderr_disp = f"{se_float * 100:.2f}"

            except (ValueError, TypeError):
                # If conversion fails, use the original string value
                value_formatted = str(v) # Keep original string if not float
                stderr_disp = str(stderr_val) if stderr_val is not None else ""
                if stderr_val is not None:
                    stderr_str = "±"


            values.append([k, version, m, value_formatted, stderr_str, stderr_disp])
            # Avoid repeating task and version in subsequent rows for the same task
            k = ""
            version = ""

    md_writer.value_matrix = values
    table_dump = md_writer.dumps() if values else "*No results data found for this task suite.*"

    # Calculate the average score for the *entire task suite* using the dedicated function
    average = calculate_average(result_dict, task) # `task` here is the suite name like "ARC", "MMLU"

    # Round average if it's numeric, otherwise keep the error string
    if isinstance(average, (int, float)):
        average = round(average, 2)

    return table_dump, average


def make_final_table(result_dict, model_name):
    """Generate summary markdown table of results with model name."""
    md_writer = MarkdownTableWriter()

    # Sanitize model name for URL (basic)
    model_path = model_name.strip()
    model_display_name = model_path.split('/')[-1] # Get last part for display
    model_link = f"[{model_display_name}](https://huggingface.co/{model_path})"

    # Prepare headers and values
    headers = ["Model"]
    values_row = [model_link]

    for key, value in result_dict.items():
         headers.append(key)
         # Format numeric values, keep strings (like errors or N/A) as is
         if isinstance(value, (int, float)):
              values_row.append(f"{value:.2f}") # Ensure 2 decimal places for numbers
         else:
              values_row.append(str(value)) # Keep error strings etc.

    md_writer.headers = headers
    md_writer.value_matrix = [values_row]

    return md_writer.dumps()
