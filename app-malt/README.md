# MALT: Datacenter Capacity Planning Benchmark

Evaluate LLM agent performance on datacenter planning tasks. Agents operate in a mock datacenter environment, making topology modifications while respecting operational constraints.

## Quick Start

```bash
# 1. Copy template and add your credentials
cp config.template.toml config.toml
nano config.toml

# 2. Run evaluation
python run.py
```

That's it! All configuration is in `config.toml`.

## Setup

### Docker (Recommended)
```bash
# Mount local directory to use your latest changes
docker run -it --rm -v $(pwd):/NetPress netpress:latest /bin/bash
conda activate ai_gym_env
cd app-malt
python run.py
```

> **Note**: The `-v $(pwd):/NetPress` flag mounts your local directory into the container, so any config changes you make locally are immediately available inside Docker.

### Local Installation
```bash
conda env create -f environment_ai_gym.yml
conda activate ai_gym
```

## Configuration

All settings are in **`config.toml`**:

```toml
[model]
agent_type = "AzureGPT4Agent"    # Which LLM to use
prompt_type = "base"             # Prompt strategy

[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "gpt-4.1"
api_version = "2024-12-01-preview"
api_key = ""                     # Leave empty to use Azure credential

[benchmark]
num_queries = 10                 # Queries per error type
complexity_levels = ["level1", "level2"]
benchmark_path = "data/benchmark_malt.jsonl"
regenerate_queries = false

[output]
output_dir = "logs/llm_agents"
output_file = "results.jsonl"
```

### Supported Models

| Agent Type | Description |
|------------|-------------|
| `AzureGPT4Agent` | Azure OpenAI (GPT-4, GPT-4.1) |
| `GoogleGeminiAgent` | Google Gemini 1.5 Pro |
| `Qwen2.5-72B-Instruct` | Local Qwen model via vLLM |
| `QwenModel_finetuned` | Fine-tuned Qwen model |
| `ReAct_Agent` | ReAct reasoning agent |

### Prompt Types

| Type | Description |
|------|-------------|
| `base` | Basic zero-shot prompt |
| `cot` | Chain-of-thought reasoning |
| `few_shot_basic` | Few-shot with fixed examples |
| `few_shot_semantic` | Few-shot with semantic similarity |

### Complexity Levels

- **level1**: Basic queries (4 types)
- **level2**: Intermediate queries
- **level3**: Advanced queries

## Authentication

### Azure OpenAI
Set credentials in `config.toml` under `[model.azure]`:
```toml
[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "your-deployment"
api_key = "your-api-key"  # Or leave empty for Azure credential
```

Or use environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
```

### Google Gemini
```bash
export GOOGLE_API_KEY="your-api-key"
```

### Hugging Face (Local Models)
```bash
export HUGGINGFACE_TOKEN="your-token"
```

## Usage Examples

### Basic Run
```bash
python run.py
```

### Custom Config File
```bash
python run.py --config my_experiment.toml
```

### Preview Configuration
```bash
python run.py --show-config
```

### Legacy CLI Mode
The original CLI is still available via `main.py`:
```bash
python main.py \
    --llm_agent_type AzureGPT4Agent \
    --num_queries 10 \
    --complexity_level level1 level2 \
    --output_dir logs/llm_agents \
    --output_file results.jsonl
```

## Output

Results are saved to `{output_dir}/{output_file}` as JSONL with:
- Query and response
- Correctness/safety evaluation
- Latency metrics

Visualization plots are saved to `{output_dir}/figs/`:
- Latency by task type
- Correctness pass rates
- Safety pass rates

## Configuration Reference

### `[model]` Section
| Key | Type | Description |
|-----|------|-------------|
| `agent_type` | string | LLM agent to use |
| `prompt_type` | string | Prompting strategy |
| `model_path` | string | Path for local models |

### `[model.azure]` Section
| Key | Type | Description |
|-----|------|-------------|
| `endpoint` | string | Azure OpenAI endpoint URL |
| `deployment_name` | string | Deployment name |
| `api_version` | string | API version |
| `api_key` | string | API key (empty = use Azure credential) |

### `[benchmark]` Section
| Key | Type | Description |
|-----|------|-------------|
| `num_queries` | int | Queries per error type |
| `complexity_levels` | array | Levels to evaluate |
| `benchmark_path` | string | Path for benchmark data |
| `regenerate_queries` | bool | Generate new queries |
| `start_index` | int | Start query index |
| `end_index` | int | End query index (0 = all) |

### `[output]` Section
| Key | Type | Description |
|-----|------|-------------|
| `output_dir` | string | Output directory |
| `output_file` | string | Results filename |
