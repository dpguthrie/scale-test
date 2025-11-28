# AI Observability Scale Test Framework

Scale testing framework for AI observability platforms using realistic OpenTelemetry traces.

## Features

- **Framework-free**: Pure Python + OTEL SDK, no agent frameworks
- **Multi-platform**: Braintrust, LangSmith, OTLP-compatible backends
- **Realistic traces**: Inspired by real travel agent workloads (100-250KB, 50-150 spans, 5-8 levels deep)
- **Configurable scale**: 10-100 req/sec with async execution
- **Comprehensive metrics**: Latency percentiles, throughput, data volume, per-scenario breakdowns

## Architecture

Three-layer design:
1. **Trace Generation**: Declarative scenario definitions with workflow steps
2. **OTEL Instrumentation**: Automatic span creation with GenAI semantic conventions
3. **Workload Execution**: Async executor with rate limiting and metrics collection

## Setup

```bash
# Install dependencies with uv
uv sync

# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

## Configuration

All configuration is via environment variables. See `.env.example` for full options.

### Braintrust Example

```bash
OTEL_PLATFORM=braintrust
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-..., x-bt-parent=project_name:scale-test"
SCALE_TEST_CONCURRENCY=50
SCALE_TEST_DURATION=300
SCALE_TEST_RATE_LIMIT=50
```

### LangSmith Example

```bash
OTEL_PLATFORM=langsmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
OTEL_EXPORTER_OTLP_HEADERS="x-api-key=lsv2_...,Langsmith-Project=scale-test"
SCALE_TEST_CONCURRENCY=50
SCALE_TEST_DURATION=300
```

## Usage

```bash
# Run scale test
uv run python scripts/run_scale_test.py
```

## Built-in Scenarios

1. **simple_query** (40% default): Simple questions (~5K tokens, 5 spans, ~35KB)
2. **single_service_search** (30% default): Service searches (~18K tokens, 12 spans, ~270KB)
3. **delegated_booking** (20% default): Specialist delegation (~40K tokens, 25 spans, ~470KB)
4. **multi_service_complex** (10% default): Multi-service workflows (~185K tokens, 80 spans, ~2.3MB)

## Example Output

```
📊 Scale Test Results:
   Duration: 300.0s
   Total requests: 15,000
   Success rate: 99.8%
   Throughput: 50.0 req/s

⏱️  Latency:
   P50: 45ms
   P95: 120ms
   P99: 250ms

📦 Data Volume:
   Total: 11.2 GB
   Avg per request: 747 KB

📈 Query Mix Breakdown:
   simple_query: 6000 traces, P50=30ms, 35KB avg (~5K tokens, 5 spans)
   single_service_search: 4500 traces, P50=250ms, 270KB avg (~18K tokens, 12 spans)
   delegated_booking: 3000 traces, P50=1200ms, 470KB avg (~40K tokens, 25 spans)
   multi_service_complex: 1500 traces, P50=4500ms, 2.3MB avg (~185K tokens, 80 spans)
```

## Code Structure

The framework is organized in three layers:

### Core Framework (`src/`)

- **`workflow.py`**: Workflow step definitions
  - Abstract `WorkflowStep` base class
  - Concrete steps: `LLMStep`, `ToolStep`, `DelegationStep`, `RoutingStep`, `ParallelStep`
  - Each step has an async `execute()` method that creates OTEL spans

- **`scenarios.py`**: Pre-defined trace scenarios
  - `TraceScenario` dataclass with workflow steps, expected span counts, and sizes
  - 4 built-in scenarios: `simple_query`, `single_service_search`, `delegated_booking`, `multi_service_complex`
  - Scenario registry and `get_scenario()` factory function

- **`instrumentation.py`**: OpenTelemetry span creation
  - `create_llm_span()`: LLM calls with GenAI semantic conventions
  - `create_tool_span()`: Tool invocations with inputs/outputs
  - `create_agent_span()`: Agent delegation spans
  - `set_platform_attributes()`: Platform-specific attributes (Braintrust, LangSmith)

- **`platforms.py`**: Platform configurations
  - Abstract `Platform` base class
  - Platform implementations: `BraintrustPlatform`, `LangSmithPlatform`, `OTLPPlatform`, `ConsolePlatform`
  - Each platform configures endpoint URLs, headers, and span attributes

- **`payloads.py`**: Realistic payload generation
  - `generate_flight_search_results()`: Flight search data
  - `generate_hotel_search_results()`: Hotel search data
  - `generate_llm_messages()`: LLM conversation messages
  - `generate_tool_result()`: Generic tool outputs

- **`metrics.py`**: Metrics collection and reporting
  - `MetricsCollector`: Tracks latencies, throughput, data volumes
  - `ScenarioMetrics`: Per-scenario breakdown
  - Calculates percentiles (P50, P95, P99) and formats reports

- **`executor.py`**: Async workload execution
  - `ScaleTestExecutor`: Main test runner with worker pool
  - `TokenBucketRateLimiter`: Smooth request rate limiting
  - Runs scenarios for specified duration with configured concurrency

### Entry Point (`scripts/`)

- **`run_scale_test.py`**: CLI entry point
  - Loads configuration from environment variables
  - Parses `OTEL_PLATFORM`, `SCALE_TEST_*` variables
  - Creates and runs `ScaleTestExecutor`
  - Prints formatted metrics report

### Tests (`tests/`)

- **`test_workflow.py`**: Workflow step creation tests
- **`test_scenarios.py`**: Scenario definition tests
- **`test_instrumentation.py`**: OTEL span creation tests
- **`test_platforms.py`**: Platform configuration tests
- **`test_payloads.py`**: Payload generation tests
- **`test_metrics.py`**: Metrics collection tests
- **`test_executor.py`**: Executor and rate limiter tests
- **`test_workflow_execution.py`**: End-to-end workflow execution tests
- **`test_integration.py`**: Full integration tests

## Development

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_scenarios.py -v

# Run with output
uv run pytest -v -s
```

## Design

See `docs/plans/2025-11-27-ai-observability-scale-test-design.md` for complete design documentation.

## License

MIT
