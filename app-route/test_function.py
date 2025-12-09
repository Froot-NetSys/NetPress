from llm_model import LLMModel  
from mininet.log import lg
from file_utils import prepare_file, initialize_json_file, static_summarize_results, static_plot_metrics
from topology import initialize_network
from safety_check import safety_check
import argparse
import json
from datetime import datetime
import os
import subprocess
import time
from multiprocessing import Process
from file_utils import process_results, plot_results
from advanced_error_function import generate_config, process_single_error
import shutil
from parallel_ping import parallelPing

def static_benchmark_run_modify(args):
    """
    Run a separate Mininet instance for each benchmark test.
    Assign a unique root directory for each instance.
    """
    start_time_2 = datetime.now()
    # Get the unique process ID to distinguish between different instances
    unique_id = os.getpid()
    if args.parallel == 1:
        args.root_dir = os.path.join(args.root_dir)
        if args.llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            result_path = os.path.join(args.root_dir, args.prompt_type+"_Qwen")
        elif args.llm_agent_type == "ReAct_Agent":
            result_path = os.path.join(args.root_dir, args.prompt_type+"_React")
        else:      
            result_path = os.path.join(args.root_dir, args.prompt_type+"_GPT")
    else:
        args.root_dir = os.path.join(args.root_dir, args.llm_agent_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
        result_path = args.root_dir
    os.makedirs(args.root_dir, exist_ok=True)

    # Generate or load the error configuration file
    # If benchmark_path is relative, make it relative to root_dir
    if not os.path.isabs(args.benchmark_path):
        file_path = os.path.join(args.root_dir, args.benchmark_path)
    else:
        file_path = args.benchmark_path
    
    # Generate config if requested OR if the file doesn't exist
    if args.static_benchmark_generation == 1 or (args.parallel == 0 and not os.path.exists(file_path)):
        generate_config(file_path, num_errors_per_type=args.num_queries, 
                       num_switches=args.num_switches, num_hosts_per_subnet=args.num_hosts_per_subnet)
        print(f"Process {unique_id}: Generated error configuration file: {file_path}")
    print(f"Process {unique_id}: Running benchmark with prompt type {args.prompt_type}")
    print(file_path)
    # Load the error configuration
    with open(file_path, 'r') as f:
        config = json.load(f)
    queries = config.get("queries", [])

    print(f"Number of queries: {len(queries)}")

    # Initialize the LLM model
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm, prompt_type=args.prompt_type, num_gpus=args.num_gpus)
    print("agenttype", args.llm_agent_type)
    if args.llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
        result_path = os.path.join(args.root_dir, args.prompt_type+"_Qwen")
    elif args.llm_agent_type == "ReAct_Agent":
        result_path = os.path.join(args.root_dir, args.prompt_type+"_React")
    else:      
        result_path = os.path.join(args.root_dir, args.prompt_type+"_GPT")
    for i, query in enumerate(queries):
        start_time_1 = datetime.now()
        print(f'Process {unique_id}: Injecting errors for query {i}')

        # Extract parameters from the query
        num_hosts_per_subnet = query.get("num_hosts_per_subnet", 1)
        num_switches = query.get("num_switches")
        errortype = query.get("errortype")
        errordetail = query.get("errordetail")
        errornumber = query.get("errornumber")

        print(f"Process {unique_id}: Initializing Mininet instance")
        print(f"  -> Topology: {num_switches} switches, {num_hosts_per_subnet} hosts/subnet")
        print(f"  -> Error type: {errortype}")
        import sys; sys.stdout.flush()  # Force output
        start_time = datetime.now()

        # Initialize the network
        subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches, unique_id)

        end_time = datetime.now()
        print(f"Process {unique_id}: Network initialization took {end_time - start_time}")
        print(f"Process {unique_id}: Subnets created: {[s[2] for s in subnets]}")
        sys.stdout.flush()

        # Inject errors into the network
        print(f"Process {unique_id}: Injecting errors into network...")
        sys.stdout.flush()
        if errornumber == 1:
            print(f"Process {unique_id}: Injecting single error: {errortype}")
            process_single_error(router, subnets, errortype, errordetail, unique_id)
            print(f"Process {unique_id}: Error injected successfully")
        else:
            if isinstance(errortype, list) and isinstance(errordetail, list) and len(errortype) == errornumber and len(errordetail) == errornumber:
                for et, ed in zip(errortype, errordetail):
                    print(f"Process {unique_id}: Injecting error: {et}")
                    process_single_error(router, subnets, et, ed, unique_id)
                print(f"Process {unique_id}: All errors injected successfully")
            else:
                print(f"Process {unique_id}: Error: For multiple error injection, errortype and errordetail must be lists of length equal to errornumber")
                continue
        sys.stdout.flush()
        # CLI(net)   
        if isinstance(errortype, list):
            errortype = '+'.join(errortype)  
        # Create result directory and files
        result_dir = os.path.join(result_path, errortype)
        os.makedirs(result_dir, exist_ok=True)

        result_file_path = os.path.join(result_dir, f'result_{i+1}.txt')
        json_path = os.path.join(result_dir, f'result_{i+1}.json')

        prepare_file(result_file_path)
        initialize_json_file(json_path)

        # LLM interacts with Mininet
        print(f"Process {unique_id}: Starting LLM interaction loop (max {args.max_iteration} iterations)")
        sys.stdout.flush()
        iter = 0
        while iter < args.max_iteration:
            # Execute LLM command
            if iter != 0:

                lg.output(f"Machine: {machine}\n")
                lg.output(f'Iteration: {iter}\n')
                lg.output(f"Command: {commands}\n")

                if safety_check(commands):
                    try:
                        # Try executing the command
                        command_output = net[machine].cmd(commands)
                        print("LLM command executed successfully")

                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}\n")
                    except Exception as e:
                        # Handle exceptions, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}\n")

            # Ping all hosts in the network
            start_time = datetime.now()
            print(f"Process {unique_id}: Iteration {iter} - Running pingAll test...")
            sys.stdout.flush()
            try:
                pingall, loss_percent = parallelPing(net, timeout=0.1)
                print(f"Process {unique_id}: PingAll completed - Loss: {loss_percent}%")
            except Exception as e:
                print(f"Process {unique_id}: Error during pingAll: {e}")
                if e == "Command execution timed out":
                    break
            end_time = datetime.now()
            print(f"Time taken for pingAll: {end_time - start_time}")
            
            # Read log file content
            if iter != 0:
                log_content = f"Machine: {machine}\n" + f"Command: {commands}\n" + f"Command Output: \n{command_output}\n" + f"Pingall result:\n{pingall}\n"
            else:
                log_content = f"Pingall result:\n{pingall}\n"
            print(f"\n**LOG CONTENT**\n{log_content}")

            # Get LLM response
            attempt = 0
            while True:
                attempt += 1
                print(f"Attempt {attempt}: Calling LLM...")
                try:
                    machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
                    print(f"Generated LLM command ([machine] [command]): {machine} {commands}")
                    break
                except Exception as e:
                    print(f"Error while generating LLM command: {e}")
                    time.sleep(3)

            # Check log content, exit loop if successful
            if loss_percent == 0:
                print(f"Query {i}: Success in {iter} iterations")
                break
            end_time = datetime.now()
            print(f"Time taken for LLM response: {end_time - start_time}")
            iter += 1

        # Stop the Mininet instance
        print(f"Process {unique_id}: Stopping Mininet instance")
        net.stop()

        end_time_1 = datetime.now()
        print(f"Process {unique_id}: Time taken for query {i}: {end_time_1 - start_time_1}")

    print(f"Process {unique_id}: Benchmark finished for {args.prompt_type}")



    for subdir in os.listdir(result_path):
        subdir_path = os.path.join(result_path, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    static_plot_metrics(result_path)
    end_time_2 = datetime.now()
    print(f"Process {unique_id}: Total time taken for all queries: {end_time_2 - start_time_2}")


def run_benchmark_parallel(args):
    """
    Run static benchmark tests in parallel using multiple processes.

    Args:
        args (argparse.Namespace): The parsed arguments containing configuration.
    """
    # Clean up any existing Mininet resources
    subprocess.run(["sudo", "mn", "-c"], check=True)

    # Create a directory to save results
    save_result_path = os.path.join(args.root_dir, 'result', args.llm_agent_type, "agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_result_path, exist_ok=True)

    # Update the root directory in args
    args.root_dir = save_result_path
    args.llm_agent_type = "GPT-Agent"
    # Generate the error configuration file and update benchmark_path
    args.benchmark_path = os.path.join(save_result_path, "error_config.json")
    generate_config(args.benchmark_path, num_errors_per_type=args.num_queries,
                   num_switches=args.num_switches, num_hosts_per_subnet=args.num_hosts_per_subnet)

    # Define a wrapper function to run static benchmarks
    def run_static_benchmark(prompt_type, static_benchmark_generation,llm_agent_type):
        """
        Wrapper function to create an independent args instance per process.
        This ensures no conflicts between parallel processes.
        """
        args_copy = argparse.Namespace(**vars(args))  # Deep copy args to avoid conflicts
        args_copy.prompt_type = prompt_type
        args_copy.llm_agent_type = llm_agent_type
        args_copy.static_benchmark_generation = static_benchmark_generation
        static_benchmark_run_modify(args_copy)

    # Get the list of prompt types from args (comma-separated)

    prompt_types = ["cot", "few_shot_basic"]

    # Create and start processes for each prompt type
    processes = []
    for prompt_type in prompt_types:
        process = Process(target=run_static_benchmark, args=(prompt_type, args.static_benchmark_generation, args.llm_agent_type))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    logs_path = os.path.join(save_result_path, "logs")
    if os.path.exists(logs_path):
        print(f"Deleting logs folder: {logs_path}")
        shutil.rmtree(logs_path)

    # Process the results and generate plots
    process_results(save_result_path)
    plot_results(save_result_path, args.num_queries)

    print(f"✅ Benchmark completed. Results saved to: {save_result_path}")