#!/usr/bin/env python3
"""
Route Benchmark Runner
======================
Simple entry point for running route evaluation.

Usage:
    python run.py                    # Uses config.toml
    python run.py --config my.toml   # Uses custom config file
"""

import argparse
import os
import sys
from datetime import datetime

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from test_function import static_benchmark_run_modify, run_benchmark_parallel


def load_config(config_path: str) -> dict:
    """Load and validate TOML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        template_path = os.path.join(os.path.dirname(config_path) or ".", "config.template.toml")
        if os.path.exists(template_path):
            print(f"\nTo get started, copy the template and add your credentials:")
            print(f"  cp {template_path} {config_path}")
        else:
            print("Please create a config.toml file or specify a valid config path.")
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config


def setup_environment(config: dict) -> None:
    """Set environment variables from config if not already set."""
    # Azure OpenAI settings
    azure = config.get("model", {}).get("azure", {})
    if azure.get("endpoint") and "AZURE_OPENAI_ENDPOINT" not in os.environ:
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure["endpoint"]
    if azure.get("deployment_name") and "AZURE_OPENAI_DEPLOYMENT_NAME" not in os.environ:
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = azure["deployment_name"]
    if azure.get("api_version") and "AZURE_OPENAI_API_VERSION" not in os.environ:
        os.environ["AZURE_OPENAI_API_VERSION"] = azure["api_version"]
    if azure.get("api_key") and "AZURE_OPENAI_API_KEY" not in os.environ:
        os.environ["AZURE_OPENAI_API_KEY"] = azure["api_key"]
    
    # Hugging Face token
    hf = config.get("model", {}).get("huggingface", {})
    if hf.get("token") and "HUGGINGFACE_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_TOKEN"] = hf["token"]


class ConfigArgs:
    """Convert TOML config to argparse-like namespace for compatibility with main.py."""
    
    def __init__(self, config: dict):
        model = config.get("model", {})
        benchmark = config.get("benchmark", {})
        topology = config.get("topology", {})
        output = config.get("output", {})
        
        # Model settings
        self.llm_agent_type = model.get("agent_type", "GPT-Agent")
        self.prompt_type = model.get("prompt_type", "base")
        self.vllm = model.get("vllm", 1)
        self.num_gpus = model.get("num_gpus", 1)
        
        # Benchmark settings
        self.num_queries = benchmark.get("num_queries", 10)
        self.max_iteration = benchmark.get("max_iteration", 10)
        self.static_benchmark_generation = 1 if benchmark.get("regenerate", False) else 0
        self.benchmark_path = benchmark.get("benchmark_path", "error_config.json")
        self.parallel = 1 if benchmark.get("parallel", False) else 0
        
        # Topology settings
        self.num_switches = topology.get("num_switches", 2)
        self.num_hosts_per_subnet = topology.get("num_hosts_per_subnet", 1)
        
        # Output settings
        self.root_dir = output.get("output_dir", "results")


def print_config_summary(config: dict) -> None:
    """Print a summary of the configuration being used."""
    model = config.get("model", {})
    benchmark = config.get("benchmark", {})
    topology = config.get("topology", {})
    output = config.get("output", {})
    
    print("\n" + "=" * 60)
    print("Route Benchmark Configuration")
    print("=" * 60)
    print(f"  Agent:        {model.get('agent_type', 'GPT-Agent')}")
    print(f"  Prompt Type:  {model.get('prompt_type', 'base')}")
    print(f"  Queries:      {benchmark.get('num_queries', 10)}")
    print(f"  Max Iter:     {benchmark.get('max_iteration', 10)}")
    print(f"  Topology:     {topology.get('num_switches', 2)} switches, {topology.get('num_hosts_per_subnet', 1)} hosts/subnet")
    print(f"  Output:       {output.get('output_dir', 'results')}/")
    print(f"  Parallel:     {'Yes' if benchmark.get('parallel', False) else 'No'}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run route benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Use default config.toml
    python run.py -c custom.toml     # Use custom config file
    python run.py --show-config      # Show current configuration
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="config.toml",
        help="Path to TOML configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit without running"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print summary
    print_config_summary(config)
    
    if args.show_config:
        print("Configuration loaded successfully. Use without --show-config to run.")
        return
    
    # Setup environment variables
    setup_environment(config)
    
    # Convert to args namespace for main.py compatibility
    run_args = ConfigArgs(config)
    
    # Run the benchmark
    print("Starting route benchmark evaluation...\n")
    start_time = datetime.now()
    
    if run_args.parallel == 1:
        run_benchmark_parallel(run_args)
    else:
        static_benchmark_run_modify(run_args)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nBenchmark completed in {duration}")


if __name__ == "__main__":
    main()

