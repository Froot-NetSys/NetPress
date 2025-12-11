# Adapting NetPress to AgentX Competition

This document provides a step-by-step guide for adapting the NetPress benchmark suite to work with the AgentX (Agentbeats) competition platform.

---

## Table of Contents

1. [Background](#background)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [TODO List](#todo-list)
5. [Detailed Implementation Guide](#detailed-implementation-guide)
6. [Expected Outputs](#expected-outputs)
7. [Testing & Verification](#testing--verification)
8. [Reference Materials](#reference-materials)

---

## Background

### What is NetPress?

NetPress is a benchmark suite for evaluating LLM agents on networking tasks. It contains three applications:

| App | Description | Environment |
|-----|-------------|-------------|
| **app-malt** | Graph-based network topology queries | Python only |
| **app-route** | Network routing diagnosis/fix with Mininet | Requires privileged Docker + Open vSwitch |
| **app-k8s** | Kubernetes network policy debugging | Requires Docker-in-Docker + KIND cluster |

Each app currently runs via CLI (`python main.py` or `python run.py`) and outputs results to JSONL/JSON files.

### What is AgentX/Agentbeats?

[Agentbeats](https://agentbeats.org) is an open platform for standardized agent evaluations. It uses the **A2A (Agent-to-Agent) protocol** for communication.

Key concepts:
- **Green Agent**: The benchmark harness/judge that orchestrates evaluations
- **Purple Agent**: The agent being evaluated (optional - can be internal to green agent)
- **Assessment**: A single evaluation session
- **A2A Protocol**: HTTP-based protocol for agent communication

### Goal of This Task

Convert NetPress benchmarks into **A2A green agents** that can:
1. Receive evaluation requests via HTTP
2. Run the benchmark internally
3. Report progress and results via the A2A protocol
4. Be registered on agentbeats.org for public evaluation

---

## Architecture Overview

### Current Architecture (CLI-based)

```
┌─────────────────┐
│  User           │
│  (runs CLI)     │
└────────┬────────┘
         │ python main.py --llm_agent_type GPT-4o ...
         ▼
┌─────────────────────────────────────────┐
│  NetPress Benchmark                      │
│  - Generates queries                     │
│  - Calls LLM API                         │
│  - Evaluates responses                   │
│  - Writes results to JSONL               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  results/*.jsonl│
└─────────────────┘
```

### Target Architecture (A2A HTTP Server)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Docker Container (--privileged for route/k8s)                       │
│                                                                      │
│  ┌────────────────────────┐     ┌──────────────────────────────────┐│
│  │  A2A HTTP Server       │     │  Existing NetPress Code          ││
│  │  (Green Agent)         │────►│  - QueryGenerator                ││
│  │                        │     │  - BenchmarkEvaluator            ││
│  │  Listens on port 9009  │     │  - Mininet / K8s (if needed)     ││
│  │                        │     │                                  ││
│  │  Receives:             │     │  Returns:                        ││
│  │  - EvalRequest (JSON)  │     │  - Progress updates              ││
│  │                        │     │  - Final results (artifact)      ││
│  └───────────┬────────────┘     └──────────────────────────────────┘│
│              │ port 9009                                             │
└──────────────┼───────────────────────────────────────────────────────┘
               │
               ▼
       ┌───────────────────┐
       │  Cloudflare       │
       │  Tunnel           │──► https://abc-123.trycloudflare.com
       └───────────────────┘
               │
               ▼
       ┌───────────────────┐
       │  AgentX Platform  │
       │  (agentbeats.org) │
       └───────────────────┘
```

### Key Mapping: CLI Args → A2A Config

| Current (CLI) | Target (A2A EvalRequest.config) |
|---------------|--------------------------------|
| `--llm_agent_type GPT-4o` | `{"llm_agent_type": "GPT-4o"}` |
| `--num_queries 10` | `{"num_queries": 10}` |
| `--prompt_type base` | `{"prompt_type": "base"}` |
| Output: `results/*.jsonl` | Output: A2A artifact |

---

## Prerequisites

Before starting, ensure you understand:

1. **Python async/await** - A2A servers use asyncio
2. **HTTP servers** - Basic understanding of uvicorn/FastAPI
3. **Docker** - For running Mininet/K8s environments
4. **The existing NetPress codebase** - Read the READMEs in `app-malt/`, `app-route/`, `app-k8s/`

### Required Reading

1. `tutorial/README.md` - Agentbeats tutorial overview
2. `tutorial/scenarios/debate/debate_judge.py` - Example green agent implementation
3. `tutorial/src/agentbeats/green_executor.py` - Base class for green agents
4. `app-malt/main.py`, `app-route/main.py`, `app-k8s/run.py` - Current entry points

---

## TODO List

### Phase 1: Setup (Estimated: 2 hours)

- [ ] **1.1** Read and understand the Agentbeats tutorial (`tutorial/README.md`)
- [ ] **1.2** Run the debate example locally to understand the flow:
  ```bash
  cd tutorial
  uv sync
  cp sample.env .env  # Add your Google API key
  uv run agentbeats-run scenarios/debate/scenario.toml
  ```
- [ ] **1.3** Study `tutorial/scenarios/debate/debate_judge.py` - this is the pattern to follow
- [ ] **1.4** Read existing NetPress code:
  - `app-malt/main.py` and `app-malt/malt_env.py`
  - `app-route/main.py` and `app-route/test_function.py`
  - `app-k8s/run.py` and `app-k8s/run_workflow.py`

### Phase 2: Dependencies (Estimated: 30 minutes)

- [ ] **2.1** Create `requirements-agentx.txt` in repo root:
  ```
  a2a-sdk>=0.3.5
  uvicorn>=0.35.0
  pydantic>=2.11.9
  python-dotenv>=1.1.1
  ```

- [ ] **2.2** Update `environment_ai_gym.yml` - add these packages to the pip section:
  ```yaml
  - pip:
    - a2a-sdk>=0.3.5
    - uvicorn>=0.35.0
    # ... existing packages ...
  ```

- [ ] **2.3** Copy the agentbeats library from tutorial:
  ```bash
  mkdir -p src/agentbeats
  cp tutorial/src/agentbeats/*.py src/agentbeats/
  ```

### Phase 3: Create Directory Structure (Estimated: 15 minutes)

- [ ] **3.1** Create the scenarios directory:
  ```bash
  mkdir -p scenarios/netpress
  ```

- [ ] **3.2** Expected final structure:
  ```
  scenarios/
  └── netpress/
      ├── malt_judge.py          # Green agent for app-malt
      ├── route_judge.py         # Green agent for app-route
      ├── k8s_judge.py           # Green agent for app-k8s
      ├── netpress_common.py     # Shared utilities and models
      ├── scenario_malt.toml     # Config for malt benchmark
      ├── scenario_route.toml    # Config for route benchmark
      └── scenario_k8s.toml      # Config for k8s benchmark
  ```

### Phase 4: Implement app-malt Judge (Estimated: 3-4 hours)

**Start with app-malt because it has no special environment requirements.**

- [ ] **4.1** Create `scenarios/netpress/netpress_common.py`:
  - Define shared Pydantic models for results
  - Define agent card factory function
  - See [Implementation Guide: netpress_common.py](#netpress_commonpy)

- [ ] **4.2** Create `scenarios/netpress/malt_judge.py`:
  - Inherit from `GreenAgent` base class
  - Implement `validate_request()` to check required config keys
  - Implement `run_eval()` to:
    1. Generate benchmark queries (using existing `QueryGenerator`)
    2. Run evaluation (using existing `BenchmarkEvaluator`)
    3. Report progress via `updater.update_status()`
    4. Output results via `updater.add_artifact()`
  - See [Implementation Guide: malt_judge.py](#malt_judgepy)

- [ ] **4.3** Create `scenarios/netpress/scenario_malt.toml`:
  ```toml
  [green_agent]
  endpoint = "http://127.0.0.1:9009"
  cmd = "python scenarios/netpress/malt_judge.py --host 127.0.0.1 --port 9009"

  [config]
  llm_agent_type = "AzureGPT4Agent"
  prompt_type = "base"
  num_queries = 10
  complexity_level = ["level1", "level2"]
  ```

- [ ] **4.4** Test locally:
  ```bash
  conda activate ai_gym_env
  cd /path/to/NetPress
  
  # Option 1: Run scenario (starts server + runs assessment)
  python -m agentbeats.run_scenario scenarios/netpress/scenario_malt.toml --show-logs
  
  # Option 2: Start server only (for debugging)
  python scenarios/netpress/malt_judge.py --host 127.0.0.1 --port 9009
  # Then in another terminal, send a test request
  ```

### Phase 5: Implement app-route Judge (Estimated: 3-4 hours)

**Requires privileged Docker container with Mininet.**

- [ ] **5.1** Create `scenarios/netpress/route_judge.py`:
  - Similar structure to malt_judge.py
  - Import from `app-route/test_function.py`
  - Handle the fact that Mininet operations are blocking (use `asyncio.to_thread()`)
  - See [Implementation Guide: route_judge.py](#route_judgepy)

- [ ] **5.2** Create `scenarios/netpress/scenario_route.toml`:
  ```toml
  [green_agent]
  endpoint = "http://127.0.0.1:9010"
  cmd = "python scenarios/netpress/route_judge.py --host 127.0.0.1 --port 9010"

  [config]
  llm_agent_type = "GPT-Agent"
  prompt_type = "base"
  num_queries = 5
  max_iteration = 10
  num_switches = 2
  num_hosts_per_subnet = 1
  ```

- [ ] **5.3** Test in Docker container:
  ```bash
  # Start privileged container
  sudo docker run -it --privileged --net=host \
    -v $(pwd):/NetPress netpress:latest /bin/bash
  
  # Inside container
  service openvswitch-switch start
  conda activate mininet
  cd /NetPress
  python scenarios/netpress/route_judge.py --host 0.0.0.0 --port 9010
  ```

### Phase 6: Implement app-k8s Judge (Estimated: 3-4 hours)

**Requires Docker-in-Docker with KIND cluster.**

- [ ] **6.1** Create `scenarios/netpress/k8s_judge.py`:
  - Similar structure to others
  - Import from `app-k8s/run_workflow.py`
  - Handle async nature of K8s operations
  - See [Implementation Guide: k8s_judge.py](#k8s_judgepy)

- [ ] **6.2** Create `scenarios/netpress/scenario_k8s.toml`:
  ```toml
  [green_agent]
  endpoint = "http://127.0.0.1:9011"
  cmd = "python scenarios/netpress/k8s_judge.py --host 127.0.0.1 --port 9011"

  [config]
  llm_agent_type = "GPT-4o"
  prompt_type = "base"
  num_queries = 10
  max_iteration = 10
  ```

- [ ] **6.3** Test in Docker container:
  ```bash
  # Start container with Docker socket
  sudo docker run -it --privileged --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/NetPress netpress:latest /bin/bash
  
  # Inside container - setup K8s (one-time, ~20 min)
  kind create cluster
  cd /microservices-demo && skaffold run
  
  # Run the judge
  cd /NetPress
  conda activate ai_gym_env
  python scenarios/netpress/k8s_judge.py --host 0.0.0.0 --port 9011
  ```

### Phase 7: Documentation & Polish (Estimated: 2 hours)

- [ ] **7.1** Create `scenarios/netpress/README.md` with:
  - Overview of each judge
  - How to run locally
  - How to expose via Cloudflare tunnel
  - Configuration options

- [ ] **7.2** Update main `README.md` with AgentX section

- [ ] **7.3** Add error handling for common failures:
  - LLM API rate limits
  - Mininet/K8s not available
  - Missing config keys

- [ ] **7.4** Test all three judges end-to-end

### Phase 8: Public Deployment (Optional)

- [ ] **8.1** Set up Cloudflare tunnel:
  ```bash
  # Install cloudflared
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
  
  # Start tunnel
  cloudflared tunnel --url http://127.0.0.1:9009
  # Note the URL: https://abc-123.trycloudflare.com
  ```

- [ ] **8.2** Start judge with public URL:
  ```bash
  python scenarios/netpress/malt_judge.py \
    --host 0.0.0.0 --port 9009 \
    --card-url https://abc-123.trycloudflare.com
  ```

- [ ] **8.3** Register on agentbeats.org

---

## Detailed Implementation Guide

### netpress_common.py

```python
"""Shared utilities and models for NetPress green agents."""

from pydantic import BaseModel
from typing import Any, Literal
from a2a.types import AgentCard, AgentSkill, AgentCapabilities


class BenchmarkResult(BaseModel):
    """Standard result format for all NetPress benchmarks."""
    total_queries: int
    correct_count: int
    safe_count: int
    correctness_rate: float  # percentage
    safety_rate: float  # percentage
    avg_latency_sec: float
    per_task_results: dict[str, Any] = {}


def create_agent_card(
    name: str,
    description: str,
    url: str,
    skill_id: str,
    skill_name: str,
    skill_description: str,
    tags: list[str],
) -> AgentCard:
    """Factory function to create agent cards for NetPress judges."""
    skill = AgentSkill(
        id=skill_id,
        name=skill_name,
        description=skill_description,
        tags=tags,
        examples=['''
{
  "participants": {},
  "config": {
    "llm_agent_type": "AzureGPT4Agent",
    "num_queries": 10,
    "prompt_type": "base"
  }
}
''']
    )
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
```

### malt_judge.py

```python
"""Green agent for the MALT network graph benchmark."""

import argparse
import asyncio
import os
import sys

# Add app-malt to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../app-malt"))

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message

# Import agentbeats base classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult

# Import existing NetPress code (NO CHANGES NEEDED TO THESE)
from dy_query_generation import QueryGenerator
from malt_env import BenchmarkEvaluator

# Import shared utilities
from netpress_common import create_agent_card, BenchmarkResult


class MaltJudge(GreenAgent):
    """Green agent that evaluates LLMs on the MALT benchmark."""
    
    def __init__(self):
        self._required_config_keys = ["llm_agent_type"]
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate that required config keys are present."""
        missing = set(self._required_config_keys) - set(request.config.keys())
        if missing:
            return False, f"Missing required config keys: {missing}"
        return True, "ok"
    
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Run the MALT benchmark evaluation."""
        config = req.config
        
        # Extract config with defaults
        llm_agent_type = config.get("llm_agent_type", "AzureGPT4Agent")
        num_queries = config.get("num_queries", 10)
        complexity_level = config.get("complexity_level", ["level1", "level2"])
        prompt_type = config.get("prompt_type", "base")
        model_path = config.get("model_path", None)
        
        # Step 1: Generate benchmark queries
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Generating {num_queries} queries per type...")
        )
        
        query_generator = QueryGenerator()
        query_generator.generate_queries(
            num_each_type=num_queries,
            complexity_level=complexity_level
        )
        
        # Step 2: Initialize evaluator
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Initializing {llm_agent_type} evaluator...")
        )
        
        evaluator = BenchmarkEvaluator(
            graph_data=query_generator.malt_real_graph,
            llm_agent_type=llm_agent_type,
            prompt_type=prompt_type,
            model_path=model_path
        )
        
        # Step 3: Run evaluation on each query
        benchmark_data = query_generator.get_all_queries()  # Adapt to actual API
        results = {
            "total": 0,
            "correct": 0,
            "safe": 0,
            "latencies": [],
            "per_task": {}
        }
        
        for i, query_obj in enumerate(benchmark_data):
            # Progress update
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Processing query {i+1}/{len(benchmark_data)}")
            )
            
            # Extract question, answer, task_label from query_obj
            question = query_obj.get("question", "")
            answer = query_obj.get("answer", "")
            task_label = query_obj.get("task_label", "unknown")
            
            # Run evaluation (this calls the LLM)
            ret, ground_truth_ret, verifier_results, verifier_error, \
                gt_verifier_results, gt_verifier_error, latency, ret_graph_copy = \
                await asyncio.to_thread(
                    evaluator.userQuery, question, answer
                )
            
            # Aggregate results
            results["total"] += 1
            results["latencies"].append(latency)
            
            # Check correctness (simplified - adapt based on actual logic)
            is_correct = (ret.get("type") != "error" and 
                         ret.get("data") == ground_truth_ret.get("data"))
            if is_correct:
                results["correct"] += 1
            
            if verifier_results:
                results["safe"] += 1
            
            # Track per-task results
            if task_label not in results["per_task"]:
                results["per_task"][task_label] = {"correct": 0, "total": 0, "safe": 0}
            results["per_task"][task_label]["total"] += 1
            if is_correct:
                results["per_task"][task_label]["correct"] += 1
            if verifier_results:
                results["per_task"][task_label]["safe"] += 1
        
        # Step 4: Calculate final metrics
        correctness_rate = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
        safety_rate = (results["safe"] / results["total"] * 100) if results["total"] > 0 else 0
        avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
        
        benchmark_result = BenchmarkResult(
            total_queries=results["total"],
            correct_count=results["correct"],
            safe_count=results["safe"],
            correctness_rate=correctness_rate,
            safety_rate=safety_rate,
            avg_latency_sec=avg_latency,
            per_task_results=results["per_task"]
        )
        
        # Step 5: Output artifact
        eval_result = EvalResult(
            winner=llm_agent_type,
            detail=benchmark_result.model_dump()
        )
        
        summary = f"""
MALT Benchmark Results
======================
Agent: {llm_agent_type}
Total Queries: {results['total']}
Correctness: {correctness_rate:.1f}% ({results['correct']}/{results['total']})
Safety: {safety_rate:.1f}% ({results['safe']}/{results['total']})
Avg Latency: {avg_latency:.2f}s
"""
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=TextPart(text=eval_result.model_dump_json(indent=2))),
            ],
            name="MALT Benchmark Results",
        )


async def main():
    parser = argparse.ArgumentParser(description="MALT Benchmark Green Agent")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, help="Public URL for agent card")
    args = parser.parse_args()
    
    agent = MaltJudge()
    executor = GreenExecutor(agent)
    
    agent_card = create_agent_card(
        name="MaltBenchmarkJudge",
        description="Evaluates LLM agents on the MALT network graph benchmark",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        skill_id="malt_eval",
        skill_name="MALT Graph Evaluation",
        skill_description="Tests LLM ability to query and modify network graph structures",
        tags=["networking", "graph", "malt"],
    )
    
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    
    print(f"Starting MALT Judge at http://{args.host}:{args.port}")
    config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(main())
```

### route_judge.py

Similar structure to `malt_judge.py`, but:
- Imports from `app-route/test_function.py`
- Uses `asyncio.to_thread()` for blocking Mininet operations
- Config includes topology parameters (`num_switches`, `num_hosts_per_subnet`)

### k8s_judge.py

Similar structure to `malt_judge.py`, but:
- Imports from `app-k8s/run_workflow.py`
- May need to handle async K8s operations
- Config includes K8s-specific parameters

---

## Expected Outputs

### After Phase 4 (app-malt complete)

You should be able to run:
```bash
python -m agentbeats.run_scenario scenarios/netpress/scenario_malt.toml --show-logs
```

And see output like:
```
Starting green agent at 127.0.0.1:9009
Waiting for 1 agent(s) to be ready...
  1/1 agents ready, waiting...
Agents started. Press Ctrl+C to stop.

--- Assessment Started ---
Starting assessment.
{"participants": {}, "config": {"llm_agent_type": "AzureGPT4Agent", ...}}
Generating 10 queries per type...
Initializing AzureGPT4Agent evaluator...
Processing query 1/60
Processing query 2/60
...
Processing query 60/60

--- Artifact: MALT Benchmark Results ---
MALT Benchmark Results
======================
Agent: AzureGPT4Agent
Total Queries: 60
Correctness: 72.5% (43/60)
Safety: 95.0% (57/60)
Avg Latency: 2.34s

{"winner": "AzureGPT4Agent", "detail": {...}}

--- Assessment Complete ---
```

### After Phase 6 (all three complete)

You should have:
1. Three working green agents that can be started independently
2. Three scenario.toml files for running assessments
3. All existing benchmark functionality accessible via HTTP

### After Phase 8 (public deployment)

Your agents should be:
1. Accessible via public Cloudflare URLs
2. Registered on agentbeats.org
3. Available for anyone to run evaluations against

---

## Testing & Verification

### Unit Tests for Each Judge

```bash
# Test malt judge
python scenarios/netpress/malt_judge.py --host 127.0.0.1 --port 9009 &
sleep 5
curl -X POST http://127.0.0.1:9009/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tasks/send", "params": {"message": {"role": "user", "parts": [{"text": "{\"participants\": {}, \"config\": {\"llm_agent_type\": \"AzureGPT4Agent\", \"num_queries\": 1}}"}]}}}'
```

### Full Scenario Test

```bash
# Run with visible logs
python -m agentbeats.run_scenario scenarios/netpress/scenario_malt.toml --show-logs

# Serve only (for debugging)
python -m agentbeats.run_scenario scenarios/netpress/scenario_malt.toml --serve-only
```

### Checklist for Each Judge

- [ ] Server starts without errors
- [ ] Agent card is accessible at `GET /`
- [ ] Assessment request is validated correctly
- [ ] Progress updates are emitted during evaluation
- [ ] Final artifact contains valid results
- [ ] Server handles errors gracefully (bad config, API failures)

---

## Reference Materials

### Key Files to Study

| File | Purpose |
|------|---------|
| `tutorial/src/agentbeats/green_executor.py` | Base class for green agents |
| `tutorial/src/agentbeats/models.py` | `EvalRequest` and `EvalResult` models |
| `tutorial/scenarios/debate/debate_judge.py` | Complete example green agent |
| `tutorial/scenarios/debate/scenario.toml` | Example scenario config |
| `app-malt/main.py` | Current MALT entry point |
| `app-malt/malt_env.py` | MALT evaluation logic |
| `app-route/test_function.py` | Route benchmark logic |
| `app-k8s/run_workflow.py` | K8s benchmark logic |

### A2A Protocol Resources

- [A2A Protocol Spec](https://a2a-protocol.org/latest/)
- [A2A SDK Docs](https://a2a-protocol.org/latest/sdk/)
- [Agentbeats Platform](https://agentbeats.org)

### Important Classes/Functions

```python
# From agentbeats
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult

# From a2a-sdk
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState, Part, TextPart, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
```

---

## Questions?

If you have questions about this task, check:
1. The tutorial README and example code
2. The existing NetPress app READMEs
3. The A2A protocol documentation

---

## Estimated Total Time

| Phase | Estimated Time |
|-------|---------------|
| Phase 1: Setup | 2 hours |
| Phase 2: Dependencies | 30 minutes |
| Phase 3: Directory Structure | 15 minutes |
| Phase 4: app-malt Judge | 3-4 hours |
| Phase 5: app-route Judge | 3-4 hours |
| Phase 6: app-k8s Judge | 3-4 hours |
| Phase 7: Documentation | 2 hours |
| Phase 8: Public Deployment | 1-2 hours |
| **Total** | **~16-20 hours** |

Start with app-malt since it has no special environment requirements, then move to route and k8s.
