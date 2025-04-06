import argparse
import json
import logging
import os
import time
import glob # Import glob for finding files with patterns

# Assuming these modules exist in the llm_autoeval package/directory
# Make sure these imports work relative to where main.py is run
try:
    from llm_autoeval.table import make_final_table, make_table
    from llm_autoeval.upload import upload_to_github_gist
except ImportError:
    # Fallback if running script directly and modules are in the same dir
    from table import make_final_table, make_table
    from upload import upload_to_github_gist


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "Unknown_Model") # Add default
BENCHMARK = os.getenv("BENCHMARK", "Unknown_Benchmark") # Add default
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")


def find_actual_json(base_path: str, benchmark_type: str) -> str | None:
    """
    Finds the actual results_*.json file.
    For nous, assumes base_path is the file.
    For others, assumes base_path is a directory containing a model subdir.
    """
    if benchmark_type == "nous":
        # Nous benchmark expects flat files
        if os.path.isfile(base_path):
            logger.info(f"Found potential flat file for 'nous' benchmark: {base_path}")
            return base_path
        else:
            logger.warning(f"'nous' benchmark specified, but path is not a file: {base_path}")
            return None
    else:
        # Other benchmarks (openllm, gemma3, eq-bench) expect nested dirs
        if not os.path.isdir(base_path):
            logger.warning(f"Expected results directory not found or is not a directory: {base_path}")
            return None

        try:
            # Find the model subdirectory (take the first one found)
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not subdirs:
                logger.warning(f"No model subdirectories found inside {base_path}")
                return None

            model_subdir_name = subdirs[0]
            model_subdir_path = os.path.join(base_path, model_subdir_name)
            logger.info(f"Searching within model subdirectory: {model_subdir_path}")

            # Find the results_*.json file using glob for pattern matching
            results_files = glob.glob(os.path.join(model_subdir_path, "results_*.json"))
            if not results_files:
                logger.warning(f"No 'results_*.json' file found inside {model_subdir_path}")
                return None

            actual_json_path = results_files[0] # Use the first match
            logger.info(f"Found results JSON file: {actual_json_path}")
            return actual_json_path

        except OSError as e:
            logger.error(f"Error accessing directory {base_path} or its subdirectories: {e}")
            return None
        except Exception as e: # Catch other potential errors
            logger.error(f"Unexpected error traversing directory {base_path}: {e}")
            return None


def _make_autoeval_summary(directory: str, elapsed_time: float) -> str:
    """Generates summary table for lm-evaluation-harness results."""
    tables = []
    averages = []
    file_suffix = "" # Default suffix

    # Define tasks and filename suffix based on BENCHMARK
    # This determines the base path name (file for nous, dir for others)
    if BENCHMARK == "openllm" or BENCHMARK == "gemma3":
        tasks = ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]
        file_suffix = "_eval" # Expected base name suffix for dir/file
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks}. Expecting dirs ending '{file_suffix}.json'")
    elif BENCHMARK == "nous":
        tasks = ["AGIEval", "GPT4All", "TruthfulQA", "Bigbench"]
        file_suffix = "" # Nous expects plain filenames
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks}. Expecting files ending '.json'")
    elif BENCHMARK == "eq-bench":
        tasks = ["EQ-Bench"]
        file_suffix = "_eval" # Expected base name suffix for dir/file
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks}. Expecting dirs ending '{file_suffix}.json'")
    else:
        logger.error(f"Benchmark '{BENCHMARK}' is not recognized for summary generation.")
        return f"Error: Benchmark '{BENCHMARK}' not recognized."

    # --- Loop through tasks to find and process results ---
    for task in tasks:
        task_lower = task.lower()
        # Construct the expected base path name (could be file or dir)
        filename_base = task_lower
        if BENCHMARK == "eq-bench" and task == "EQ-Bench":
             filename_base = "eq-bench"

        # This is the expected name of the file (for nous) or directory (for others)
        expected_base_path = os.path.join(directory, f"{filename_base}{file_suffix}.json")

        # Find the actual path to the results JSON, handling nesting
        actual_json_path = find_actual_json(expected_base_path, BENCHMARK)

        # Process the found JSON file (if any)
        if actual_json_path:
            try:
                with open(actual_json_path, "r") as f:
                    json_data = f.read()
                    data = json.loads(json_data, strict=False)
                # Pass the loaded data to table/average functions
                table, average = make_table(data, task) # Assumes make_table takes data dict and task name
                logger.info(f"Successfully processed '{task}' from '{actual_json_path}'. Average: {average}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {actual_json_path}: {e}")
                table = f"Error: Invalid JSON"
                average = f"Error: Invalid JSON"
            except Exception as e:
                logger.error(f"Error processing file {actual_json_path} for task '{task}': {e}")
                table = f"Error: Processing failed"
                average = f"Error: Processing failed"
        else:
            # Handle case where JSON path wasn't found
            logger.warning(f"Results JSON file could not be located for task '{task}' (expected path: {expected_base_path})")
            table = f"Error: Results file not found for {task}"
            average = f"Error: Results file not found"

        tables.append(table)
        averages.append(average)
        # Ensure data is not accidentally carried over from previous loop iteration
        data = None

    # --- Generate summary sections ---
    summary = ""
    result_dict = {}
    for index, task in enumerate(tasks):
        # Only add % sign if average is a float/number
        avg_display = f"{averages[index]}%" if isinstance(averages[index], (int, float)) else averages[index]
        # Make table heading bold
        summary += f"### **{task}**\n{tables[index]}\nAverage: {avg_display}\n\n"
        result_dict[task] = averages[index] # Store original value for final average calculation

    # --- Calculate the final average, excluding strings/errors ---
    valid_averages = [avg for avg in averages if isinstance(avg, (int, float))]
    if valid_averages:
        final_average = round(sum(valid_averages) / len(valid_averages), 2)
        summary += f"Average score (across valid tasks): **{final_average}%**" # Make average bold
        result_dict["Average"] = final_average
    else:
        summary += "Average score: **Not available** (no valid task averages)" # Make bold
        result_dict["Average"] = "N/A"


    # --- Generate final summary table ---
    final_table = make_final_table(result_dict, MODEL_ID) # Assumes make_final_table exists
    summary = final_table + "\n\n---\n\n" + summary # Add separator
    return summary


def _get_result_dict(directory: str) -> dict:
    """Walk down directories to get the first JSON file found (for Lighteval)."""
    logger.info(f"Searching for JSON results (Lighteval) in directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Lighteval might not follow the 'results_' prefix convention consistently
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                logger.info(f"Found potential Lighteval results JSON file: {json_path}")
                try:
                    with open(json_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                     logger.error(f"Error reading/parsing JSON file {json_path}: {e}")
                     # Continue searching if one file fails
    # If loop finishes without returning/raising
    logger.error(f"No JSON file found in {directory} or its subdirectories.")
    raise FileNotFoundError(f"No JSON file found for Lighteval in {directory}")


def _make_lighteval_summary(directory: str, elapsed_time: float) -> str:
    """Generates summary table for lighteval results."""
    try:
        # lighteval might be an optional dependency
        from lighteval.evaluator import make_results_table
    except ImportError:
        logger.error("lighteval library not found. Cannot generate lighteval summary.")
        return "Error: lighteval library not installed."

    try:
        result_dict = _get_result_dict(directory) # Find the JSON first
        # Assuming make_results_table works on the loaded dictionary
        # LightEval's structure might vary; adjust parsing if needed based on its output
        final_table = make_results_table(result_dict)

        model_name = MODEL_ID.split('/')[-1] if MODEL_ID else "Unknown_Model"
        benchmark_name = BENCHMARK.capitalize() if BENCHMARK else "LightEval"
        summary = f"## {model_name} - {benchmark_name}\n\n"
        summary += final_table
        return summary
    except FileNotFoundError as e:
        logger.error(f"Could not find results file for lighteval: {e}")
        return f"Error: Could not find lighteval results JSON in {directory}"
    except Exception as e:
        logger.error(f"Error generating lighteval summary: {e}")
        return f"Error: Failed to generate lighteval summary from {directory}"


def main(directory: str, elapsed_time: float) -> None:
    """Main function to generate and upload summary."""
    logger.info(f"Starting summary generation for benchmark '{BENCHMARK}' in directory '{directory}'")

    # Determine which summary function to use
    if BENCHMARK in ["openllm", "nous", "eq-bench", "gemma3"]: # Added gemma3
        summary = _make_autoeval_summary(directory, elapsed_time)
    elif BENCHMARK == "lighteval":
        summary = _make_lighteval_summary(directory, elapsed_time)
    else:
        logger.error(f"Unsupported BENCHMARK value: {BENCHMARK}")
        summary = f"Error: Unsupported BENCHMARK value '{BENCHMARK}'. Cannot generate summary."

    # Add elapsed time to summary
    convert = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    summary += f"\n\nElapsed time: {convert}"
    logger.info(f"Summary generated. Elapsed time: {convert}")

    # Define Gist filename
    model_name = MODEL_ID.split('/')[-1] if MODEL_ID else "Unknown_Model"
    benchmark_name = BENCHMARK.capitalize() if BENCHMARK else "Unknown_Benchmark"
    gist_filename = f"{model_name}-{benchmark_name}-EvalSummary.md"

    # Upload to GitHub Gist if API token is provided
    if GITHUB_API_TOKEN:
        logger.info(f"Attempting to upload summary to GitHub Gist as '{gist_filename}'...")
        try:
            upload_to_github_gist(summary, gist_filename, GITHUB_API_TOKEN)
            logger.info("Successfully uploaded summary to GitHub Gist.")
        except Exception as e:
            logger.error(f"Failed to upload summary to GitHub Gist: {e}")
    else:
        logger.warning("GITHUB_API_TOKEN not set. Skipping GitHub Gist upload.")
        print("\n--- EVALUATION SUMMARY ---\n")
        print(summary) # Print summary to console if not uploading
        print("\n--- END SUMMARY ---\n")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Summarize evaluation results and upload them to GitHub Gist.")
    parser.add_argument(
        "directory", type=str, help="The path to the directory containing the evaluation results (JSON files or dirs)"
    )
    parser.add_argument(
        "elapsed_time",
        type=float,
        help="Total elapsed time for the evaluation in seconds",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.directory):
        logger.error(f"The specified base directory does not exist: {args.directory}")
        print(f"Error: The specified base directory does not exist: {args.directory}")
        exit(1)

    if args.elapsed_time < 0:
         logger.warning(f"Elapsed time is negative ({args.elapsed_time}). Using 0.")
         args.elapsed_time = 0

    # Call the main function with the directory argument
    main(args.directory, args.elapsed_time)
