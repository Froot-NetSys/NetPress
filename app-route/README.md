# Route: Network Routing Benchmark

Evaluate LLM agent performance on dynamically generated routing configuration tasks in a simulated network environment.

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
sudo docker run -it --rm \
  --privileged \
  --net=host \
  -v /lib/modules:/lib/modules \
  -v $(pwd):/NetPress \
  netpress:latest \
  /bin/bash
```

Inside the Docker shell:
```bash
# Start Open vSwitch
service openvswitch-switch start 2>/dev/null || \
  /etc/init.d/openvswitch-switch start 2>/dev/null || true

# Test Mininet
mn --test pingall

# Run benchmark
conda activate mininet
cd app-route
python run.py
```

## Configuration

All settings are in **`config.toml`**:

```toml
[model]
agent_type = "GPT-Agent"         # Which LLM to use
prompt_type = "base"             # Prompt strategy

[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "gpt-4.1"
api_key = ""                     # Leave empty to use Azure credential

[benchmark]
num_queries = 10                 # Queries per error type
max_iteration = 10               # Max iterations per query
regenerate = false               # Generate new benchmark
parallel = false                 # Run in parallel

[topology]
num_switches = 2
num_hosts_per_subnet = 1

[output]
output_dir = "results"
```

### Supported Models

| Agent Type | Description |
|------------|-------------|
| `GPT-Agent` | Azure OpenAI (GPT-4, GPT-4.1) |
| `Qwen/Qwen2.5-72B-Instruct` | Local Qwen model via vLLM |
| `ReAct_Agent` | ReAct reasoning agent |

### Prompt Types

| Type | Description |
|------|-------------|
| `base` | Basic zero-shot prompt |
| `cot` | Chain-of-thought reasoning |
| `few_shot_basic` | Few-shot with fixed examples |

## Authentication

### Azure OpenAI
Set credentials in `config.toml` under `[model.azure]`:
```toml
[model.azure]
endpoint = "https://your-resource.openai.azure.com/"
deployment_name = "your-deployment"
api_key = "your-api-key"
```

Or use environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
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
    --llm_agent_type GPT-Agent \
    --num_queries 10 \
    --root_dir results \
    --max_iteration 10 \
    --prompt_type base \
    --parallel 0
```

## Configuration Reference

### `[model]` Section
| Key | Type | Description |
|-----|------|-------------|
| `agent_type` | string | LLM agent to use |
| `prompt_type` | string | Prompting strategy |
| `vllm` | int | Enable vLLM for local models (0/1) |
| `num_gpus` | int | GPUs for tensor parallelism |

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
| `max_iteration` | int | Max iterations per query |
| `regenerate` | bool | Generate new benchmark |
| `benchmark_path` | string | Path for benchmark config |
| `parallel` | bool | Run prompt types in parallel |

### `[topology]` Section
| Key | Type | Description |
|-----|------|-------------|
| `num_switches` | int | Number of switches |
| `num_hosts_per_subnet` | int | Hosts per subnet |

### `[output]` Section
| Key | Type | Description |
|-----|------|-------------|
| `output_dir` | string | Root directory for outputs |

## Testing Your Own Model

To integrate your own model:

1. **Model Initialization** - Update `_create_model` and `_initialize_YourModel` in `llm_model.py`
2. **Model Loading** - Update `_load_model` in your model class
3. **Prediction** - Implement the `predict` method

Look for `# ====== TODO:` comments in the code for guidance.
