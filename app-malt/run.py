#!/usr/bin/env python3
"""
MALT Benchmark Runner
=====================
Simple entry point for running MALT evaluation.

Usage:
    python run.py                    # Uses config.toml
    python run.py --config my.toml   # Uses custom config file
"""

import argparse
import os
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from main import main as run_benchmark


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
    
    # Google API key
    google = config.get("model", {}).get("google", {})
    if google.get("api_key") and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = google["api_key"]
    
    # Hugging Face token
    hf = config.get("model", {}).get("huggingface", {})
    if hf.get("token") and "HUGGINGFACE_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_TOKEN"] = hf["token"]


class ConfigArgs:
    """Convert TOML config to argparse-like namespace for compatibility with main.py."""
    
    def __init__(self, config: dict):
        model = config.get("model", {})
        benchmark = config.get("benchmark", {})
        output = config.get("output", {})
        
        # Model settings
        self.llm_agent_type = model.get("agent_type", "AzureGPT4Agent")
        self.prompt_type = model.get("prompt_type", "base")
        self.model_path = model.get("model_path") or None
        
        # Benchmark settings
        self.num_queries = benchmark.get("num_queries", 10)
        self.complexity_level = benchmark.get("complexity_levels", ["level1", "level2"])
        self.dynamic_benchmark_path = benchmark.get("benchmark_path", "data/benchmark_malt.jsonl")
        self.regenerate_query = benchmark.get("regenerate_queries", False)
        self.start_index = benchmark.get("start_index", 0)
        end_idx = benchmark.get("end_index", 0)
        self.end_index = None if end_idx == 0 else end_idx
        
        # Output settings
        self.output_dir = output.get("output_dir", "logs/llm_agents")
        self.output_file = output.get("output_file", "results.jsonl")


def print_config_summary(config: dict) -> None:
    """Print a summary of the configuration being used."""
    model = config.get("model", {})
    benchmark = config.get("benchmark", {})
    output = config.get("output", {})
    
    print("\n" + "=" * 60)
    print("MALT Benchmark Configuration")
    print("=" * 60)
    print(f"  Agent:        {model.get('agent_type', 'AzureGPT4Agent')}")
    print(f"  Prompt Type:  {model.get('prompt_type', 'base')}")
    print(f"  Complexity:   {', '.join(benchmark.get('complexity_levels', ['level1', 'level2']))}")
    print(f"  Queries/Type: {benchmark.get('num_queries', 10)}")
    print(f"  Output:       {output.get('output_dir', 'logs')}/{output.get('output_file', 'results.jsonl')}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run MALT benchmark evaluation",
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
    print("Starting MALT benchmark evaluation...\n")
    run_benchmark(run_args)
    
    print("\n" + "=" * 60)
    print("Benchmark evaluation complete!")
    print("=" * 60)
    print(f"  Results:  {run_args.output_dir}/{run_args.output_file}")
    print(f"  Figures:  {run_args.output_dir}/figs/")
    print("=" * 60)


if __name__ == "__main__":
    main()

