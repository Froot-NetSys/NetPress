import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json
from solid_step_helper import get_node_value_ranges, getGraphData, \
solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes
from dy_query_generation import QueryGenerator
from malt_env import BenchmarkEvaluator
import argparse
from scipy import stats
import math

# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default='AzureGPT4Agent', help='Choose the LLM agent', choices=['AzureGPT4Agent', 
                                                                                                                      'GoogleGeminiAgent',     
                                                                                                                      'Qwen2.5-72B-Instruct', 
                                                                                                                      'QwenModel_finetuned', 
                                                                                                                      'ReAct_Agent'])
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model (for local models/finetunes).')
    parser.add_argument('--prompt_type', type=str, default='base', help='Choose the prompt type', choices=['base', 'cot', 'few_shot_basic', 'few_shot_semantic', 'few_shot_knn'])
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', nargs='+', default=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--output_dir', type=str, default='logs/llm_agents', help='Directory to save output JSONL file')
    parser.add_argument('--output_file', type=str, default='gpt4o.jsonl', help='Name of the output JSONL file')
    parser.add_argument('--dynamic_benchmark_path', type=str, default='data/benchmark_malt.jsonl', help='Path to save dynamic dataset')
    parser.add_argument('--regenerate_query', action='store_true', help='Whether to regenerate benchmark queries or load existing ones')
    parser.add_argument('--start_index', type=int, default=0, help='Start index of the queries to run (zero indexed).')
    parser.add_argument('--end_index', type=int, default=None, help='End index of the queries to run (zero indexed).')
    return parser.parse_args()

# An example of how to use main.py with input args
# Example usage:
# python main.py --llm_agent_type AzureGPT4Agent --num_queries 2 --complexity_level level1 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl

def main(args):

    benchmark_config = {
        'llm_agent_type': args.llm_agent_type,
        'model_path': args.model_path,
        'prompt_type': args.prompt_type,
        'num_queries': args.num_queries,
        'complexity_level': args.complexity_level,
        'output_dir': args.output_dir,
        'output_file': args.output_file,
        'dynamic_benchmark_path': args.dynamic_benchmark_path,
        'regenerate_query': args.regenerate_query,
        'start_index': args.start_index,
        'end_index': args.end_index
    }

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create the output file if it does not exist
    output_path = os.path.join(args.output_dir, args.output_file)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            pass

    # dynamically generate or load existing queries
    query_generator = QueryGenerator()
    benchmark_path = benchmark_config['dynamic_benchmark_path']
    
    if benchmark_config['regenerate_query']:
        print("Generating new queries due to regenerate_query=True")
        query_generator.generate_queries(num_each_type=benchmark_config['num_queries'], complexity_level=benchmark_config['complexity_level'])
        query_generator.save_queries_to_file(benchmark_path)
    else:
        if not os.path.exists(benchmark_path):
            print(f"Benchmark file {benchmark_path} does not exist. Generating new queries...")
            query_generator.generate_queries(num_each_type=benchmark_config['num_queries'], complexity_level=benchmark_config['complexity_level'])
            query_generator.save_queries_to_file(benchmark_path)
        else:
            print(f"Loading existing benchmark from {benchmark_path}")
            query_generator.load_queries_from_file(benchmark_path)

    # Load the evaluator
    evaluator = BenchmarkEvaluator(graph_data=query_generator.malt_real_graph, llm_agent_type=benchmark_config['llm_agent_type'], 
                                   prompt_type=benchmark_config['prompt_type'], model_path=benchmark_config['model_path'])

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = []
    with jsonlines.open(benchmark_path) as reader:
        for obj in reader:
            benchmark_data.append(obj['messages'])
    
    # Skip to start_index if specified
    start_idx = max(benchmark_config['start_index'], 0)
    end_idx = len(benchmark_data) if not isinstance(benchmark_config['end_index'], int) else min(benchmark_config['end_index'], len(benchmark_data))
    if 0 < start_idx or end_idx < len(benchmark_data):
        print(f"Starting from query index {start_idx} (skipping {start_idx} queries) and ending at {end_idx} (processing {end_idx - start_idx} queries).")
        if start_idx >= end_idx:
            print(f"Warning: start_index {start_idx} is greater than or equal to end index ({len(benchmark_data)})")
            return
        benchmark_data = benchmark_data[start_idx:end_idx]
    
    # for each object in the benchmark list, get the question and answer
    for i, obj in enumerate(benchmark_data):
        # Calculate actual index (for logging)
        actual_idx = i + start_idx
        print(f"Processing query {actual_idx} of {len(benchmark_data) + start_idx - 1}")
        
        # obj is a list of dictionaries, load question, answer, task_label from it
        for item in obj:
            if 'question' in item:
                current_query = item['question']
            elif 'answer' in item:
                golden_answer = item['answer']
            elif 'task_label' in item:
                task_label = item['task_label']
            
        ret, ground_truth_ret, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, ret_graph_copy = evaluator.userQuery(current_query, golden_answer)
        evaluator.ground_truth_check(current_query, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error, gt_verifier_results, gt_verifier_error, query_run_latency, output_path)

        # have to sleep for Gemini API quota
        if benchmark_config['llm_agent_type'] == 'GoogleGeminiAgent':
            time.sleep(10)

    # Analyze the results
    # load the data from output_path
    results = []
    with jsonlines.open(output_path) as reader:
        for obj in reader:
            results.append(obj)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # group the results by task label
    grouped_results = {}
    for result in results:
        task_label = result["Label"]
        if task_label not in grouped_results:
            grouped_results[task_label] = []
        grouped_results[task_label].append(result)

    task_labels = list(grouped_results.keys())
    avg_latencies = []
    std_latencies = []
    for task_label in task_labels:
        latencies = [result["Result-Latency"] for result in grouped_results[task_label]]
        avg_latencies.append(np.mean(latencies))
        std_latencies.append(np.std(latencies))

    # create figs directory if it doesn't exist
    figs_dir = os.path.join(args.output_dir, 'figs')
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    # plot the average query run latency for each task label with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(task_labels, avg_latencies, color='skyblue', yerr=std_latencies, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Average Query Run Latency (seconds)')
    plt.title(f'Average Query Run Latency ({args.llm_agent_type}, Avg={np.mean(avg_latencies):.2f}s ±{np.mean(std_latencies):.2f}s)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_latencies[i],
                f'±{std_latencies[i]:.2f}',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'average_latency_{args.llm_agent_type}_{timestamp}.png'), dpi=300)
    # plt.show()

    # plot the pass rate of correctness for each task label
    correctness_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Correctness"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]
    
    # Calculate Standard Error of Mean (SEM) for each task label's pass rate
    sem_values = []
    sample_sizes = []   
    
    for task_label in task_labels:
        # Get sample size for this task
        n = len(grouped_results[task_label])
        sample_sizes.append(n)
        
        # Calculate SEM using scipy
        binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in grouped_results[task_label]]
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    error_margins = [1.96 * sem for sem in sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(task_labels, correctness_pass_rates, color='green', yerr=error_margins, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Correctness Pass Rate (%)')
    # print the average pass rate and error margin on the title
    avg_pass_rate = np.mean(correctness_pass_rates)
    error_margin = np.mean(error_margins)
    plt.title(f'Correctness Pass Rate ({args.llm_agent_type}, N={sample_sizes}, Avg={avg_pass_rate:.2f}% ±{error_margin:.2f}%)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error_margins[i],
                f'±{error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'correctness_pass_rate_{args.llm_agent_type}_{timestamp}.png'), dpi=300)
    # plt.show()
    
    # plot the pass rate of safety for each task label
    safety_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Safety"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]
    
    # Calculate SEM for safety pass rates
    safety_sem_values = []
    for task_label in task_labels:
        binary_outcomes = [1 if result["Result-Safety"] == "Pass" else 0 for result in grouped_results[task_label]]
        scipy_sem = stats.sem(binary_outcomes, ddof=0) * 100
        safety_sem_values.append(scipy_sem)
    
    # Calculate 95% confidence interval (1.96 * SEM)
    safety_error_margins = [1.96 * sem for sem in safety_sem_values]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(task_labels, safety_pass_rates, color='orange', yerr=safety_error_margins, capsize=5)
    plt.xlabel('Task Label')
    plt.ylabel('Safety Pass Rate (%)')
    # print the average pass rate and error margin on the title
    avg_pass_rate = np.mean(safety_pass_rates)
    error_margin = np.mean(safety_error_margins)
    plt.title(f'Safety Pass Rate ({args.llm_agent_type}, N={sample_sizes}, Avg={avg_pass_rate:.2f}% ±{error_margin:.2f}%)')
    # Add error values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + safety_error_margins[i],
                f'±{safety_error_margins[i]:.2f}%',
                ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'safety_pass_rate_{args.llm_agent_type}_{timestamp}.png'), dpi=300)
    # plt.show()


# run the main function
if __name__ == "__main__":
    args = parse_args()
    main(args)