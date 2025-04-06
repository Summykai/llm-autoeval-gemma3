import argparse
import json
import logging
import os
import time

# Assuming these modules exist in the llm_autoeval package/directory
from llm_autoeval.table import make_final_table, make_table
from llm_autoeval.upload import upload_to_github_gist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "Unknown_Model") # Add default
BENCHMARK = os.getenv("BENCHMARK", "Unknown_Benchmark") # Add default
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")


def _make_autoeval_summary(directory: str, elapsed_time: float) -> str:
    """Generates summary table for lm-evaluation-harness results."""
    tables = []
    averages = []
    file_suffix = "" # Default suffix

    # Define tasks and filename suffix based on BENCHMARK
    if BENCHMARK == "openllm" or BENCHMARK == "gemma3":
        tasks = ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]
        file_suffix = "_eval" # Filename suffix used in runpod.py
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks} and filename suffix: '{file_suffix}.json'")
    elif BENCHMARK == "nous":
        tasks = ["AGIEval", "GPT4All", "TruthfulQA", "Bigbench"]
        file_suffix = "" # Nous uses plain filenames
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks} and filename suffix: '.json'")
    elif BENCHMARK == "eq-bench":
        tasks = ["EQ-Bench"]
        file_suffix = "_eval" # EQ-Bench also uses _eval suffix in runpod.py
        logger.info(f"Processing benchmark '{BENCHMARK}' with tasks: {tasks} and filename suffix: '{file_suffix}.json'")
    else:
        # This case should ideally be caught in main(), but added defensively
        logger.error(f"Benchmark '{BENCHMARK}' is not recognized for summary generation.")
        return f"Error: Benchmark '{BENCHMARK}' not recognized."

    # Load results for each task
    for task in tasks:
        task_lower = task.lower()
        # Construct the expected filename
        # Special case for eq-bench task name vs filename base
        if BENCHMARK == "eq-bench" and task == "EQ-Bench":
             filename_base = "eq-bench"
        else:
             filename_base = task_lower

        file_path = os.path.join(directory, f"{filename_base}{file_suffix}.json")
        logger.info(f"Looking for results file: {file_path}")

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    json_data = f.read()
                    # Handle potential trailing commas or other minor JSON issues if necessary
                    data = json.loads(json_data, strict=False)
                table, average = make_table(data, task) # Assumes make_table handles parsing
                logger.info(f"Successfully processed {task}. Average: {average}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path}: {e}")
                table = f"Error: Invalid JSON in {file_path}"
                average = f"Error: Invalid JSON"
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                table = f"Error: Processing failed for {file_path}"
                average = f"Error: Processing failed"
        else:
            logger.warning(f"Results file not found: {file_path}")
            table = f"Error: File does not exist ({os.path.basename(file_path)})"
            average = f"Error: File does not exist"

        tables.append(table)
        averages.append(average)

    # Generate summary sections
    summary = ""
    result_dict = {}
    for index, task in enumerate(tasks):
        # Only add % sign if average is a float/number
        avg_display = f"{averages[index]}%" if isinstance(averages[index], (int, float)) else averages[index]
        summary += f"### {task}\n{tables[index]}\nAverage: {avg_display}\n\n"
        result_dict[task] = averages[index] # Store original value for final average calculation

    # Calculate the final average, excluding strings/errors
    valid_averages = [avg for avg in averages if isinstance(avg, (int, float))]
    if valid_averages:
        final_average = round(sum(valid_averages) / len(valid_averages), 2)
        summary += f"Average score (across valid tasks): {final_average}%"
        result_dict["Average"] = final_average
    else:
        summary += "Average score: Not available (no valid task averages)"
        result_dict["Average"] = "N/A"


    # Generate final summary table
    final_table = make_final_table(result_dict, MODEL_ID) # Assumes make_final_table exists
    summary = final_table + "\n\n---\n\n" + summary # Add separator
    return summary


def _get_result_dict(directory: str) -> dict:
    """Walk down directories to get the first JSON file found."""
    logger.info(f"Searching for JSON results in directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                logger.info(f"Found results JSON file: {json_path}")
                try:
                    with open(json_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                     logger.error(f"Error reading/parsing JSON file {json_path}: {e}")
                     raise FileNotFoundError(f"Error reading JSON in {directory}") # Re-raise or handle
    logger.error(f"No JSON file found in {directory} or its subdirectories.")
    raise FileNotFoundError(f"No JSON file found in {directory}")


def _make_lighteval_summary(directory: str, elapsed_time: float) -> str:
    """Generates summary table for lighteval results."""
    try:
        # lighteval might be an optional dependency
        from lighteval.evaluator import make_results_table
    except ImportError:
        logger.error("lighteval library not found. Cannot generate lighteval summary.")
        return "Error: lighteval library not installed."

    try:
        result_dict = _get_result_dict(directory)
        # Ensure the results are in the expected format for make_results_table
        # This might need adjustment depending on lighteval's output structure
        if "results" in result_dict:
             final_table = make_results_table(result_dict["results"])
        else:
             # Fallback if structure is different, maybe root is the results dict?
             logger.warning("Lighteval results JSON might not have top-level 'results' key. Attempting to use root.")
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
        # Avoid raising error, allow script to finish gracefully if needed
        summary = f"Error: Unsupported BENCHMARK value '{BENCHMARK}'. Cannot generate summary."
        # Or re-raise if script should halt:
        # raise NotImplementedError(
        #     f"BENCHMARK should be 'openllm', 'nous', 'lighteval', 'eq-bench', or 'gemma3' (current value = {BENCHMARK})"
        # )

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
        "directory", type=str, help="The path to the directory containing the JSON result files"
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
        logger.error(f"The specified directory does not exist: {args.directory}")
        # Instead of raising ValueError, exit gracefully
        print(f"Error: The specified directory does not exist: {args.directory}")
        exit(1) # Exit with a non-zero code to indicate failure
        # raise ValueError(f"The directory {args.directory} does not exist.")

    if args.elapsed_time < 0:
         logger.warning(f"Elapsed time is negative ({args.elapsed_time}). Using 0.")
         args.elapsed_time = 0

    # Call the main function with the directory argument
    main(args.directory, args.elapsed_time)
