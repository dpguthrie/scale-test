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
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-..., x-bt-parent=project_id:..."
SCALE_TEST_CONCURRENCY=50
SCALE_TEST_DURATION=300
SCALE_TEST_RATE_LIMIT=50
```

### LangSmith Example

```bash
OTEL_PLATFORM=langsmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
OTEL_EXPORTER_OTLP_HEADERS="x-api-key=lsv2_..., Langsmith-Project=scale-test"
SCALE_TEST_CONCURRENCY=50
SCALE_TEST_DURATION=300
```

## Usage

```bash
# Run scale test
uv run python scripts/run_scale_test.py
```

## Built-in Scenarios

1. **simple_query** (40% default): Simple questions (5-10 spans, ~20KB)
2. **single_service_search** (30% default): Service searches (10-20 spans, ~50KB)
3. **delegated_booking** (20% default): Specialist delegation (30-50 spans, ~150KB)
4. **multi_service_complex** (10% default): Multi-service workflows (80-150 spans, ~250KB)

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
   Total: 2.25 GB
   Avg per request: 150 KB

📈 Query Mix Breakdown:
   simple_query: 6000 traces, P50=20ms, 20KB avg
   single_service_search: 4500 traces, P50=35ms, 50KB avg
   delegated_booking: 3000 traces, P50=80ms, 150KB avg
   multi_service_complex: 1500 traces, P50=180ms, 250KB avg
```

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
