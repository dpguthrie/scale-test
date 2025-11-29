# AI Observability Scale Test Framework

Scale testing framework for AI observability platforms using realistic OpenTelemetry traces.

## Features

- **Framework-free**: Pure Python + OTEL SDK, no agent frameworks
- **Multi-platform**: Braintrust, LangSmith, OTLP-compatible backends
- **Configurable scale**: 10-100 req/sec with async execution
- **Production-ready**: OpenTelemetry Collector support for high-throughput testing

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Copy configuration file
cp .env.example .env

# Edit .env with your API key
nano .env
```

### Configuration

**All configuration is in a single `.env` file**. This file configures both your test script AND the OTel Collector (if using).

```bash
# Edit .env file:
# 1. Add your API key to BRAINTRUST_API_KEY or LANGSMITH_API_KEY
# 2. Choose deployment mode (see below)
# 3. Adjust test parameters (concurrency, duration, etc.)
```

See `.env.example` for full configuration options with comments.

## Deployment Modes

### Option 1: Direct Export (Simple, Low Scale)

For quick tests with < 50 traces. Your script connects directly to Braintrust/LangSmith.

**When to use:** Quick validation, low concurrency (< 10 workers), short duration (< 5 minutes)

**Setup (.env):**
```bash
OTEL_PLATFORM=braintrust
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-YOUR_KEY, x-bt-parent=project_name:YOUR_PROJECT"
```

**Run:**
```bash
uv run python scripts/run_scale_test.py
```

**Limitations:**
- ❌ Script hangs 5-10 minutes on shutdown waiting for spans to export
- ❌ Timeouts on large tests (500+ traces)

---

### Option 2: Collector Export (Production, High Scale)

For serious scale testing with 100+ traces. Your script sends spans to a local OpenTelemetry Collector that handles export in the background.

**When to use:** Scale testing (100+ traces), high concurrency (10+ workers), production workloads

**Architecture:**
```
Your Script → localhost:4318 (fast) → Collector → Braintrust/LangSmith
   ↓
Exits in ~5 seconds
(Collector continues exporting in background)
```

**Setup:**

1. **Configure .env file:**
```bash
# Deployment mode (send to local collector)
OTEL_PLATFORM=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
# No OTEL_EXPORTER_OTLP_HEADERS needed!

# API keys (for collector to authenticate with platform)
BRAINTRUST_API_KEY=sk-YOUR_ACTUAL_KEY
BRAINTRUST_PROJECT=scale-test

# Enable fast shutdown
OTEL_SKIP_SHUTDOWN=true
```

2. **Edit collector pipeline** (`otel-collector-config.yaml`):

**IMPORTANT:** Enable ONLY ONE platform pipeline. Having both active sends duplicate spans and causes errors.

```yaml
service:
  pipelines:
    # For Braintrust (enabled by default)
    traces/braintrust:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlphttp/braintrust]

    # For LangSmith (comment out Braintrust above first!)
    # traces/langsmith:
    #   receivers: [otlp]
    #   processors: [memory_limiter, batch]
    #   exporters: [otlphttp/langsmith]
```

3. **Start collector:**
```bash
docker-compose up -d
```

4. **Run test:**
```bash
uv run python scripts/run_scale_test.py
# Script exits in ~5 seconds
# Collector continues exporting in background
```

5. **Monitor collector:**
```bash
docker-compose logs -f otel-collector
```

**Benefits:**
- ✅ Script exits immediately (< 5 seconds)
- ✅ Automatic retries with exponential backoff
- ✅ Built-in rate limiting and backpressure
- ✅ Can handle 100+ req/s sustained
- ✅ Spans continue exporting after script exits
- ✅ 10x faster iteration

---

## Configuration Details

### Single .env File Approach

**One file configures everything:**
- Your test script reads: `OTEL_PLATFORM`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `SCALE_TEST_*`
- Docker Compose reads: `BRAINTRUST_API_KEY`, `LANGSMITH_API_KEY` (via `env_file: .env`)
- Collector reads: API keys passed from Docker Compose as environment variables

### Key Configuration Variables

#### Deployment Mode
```bash
# Direct export to platform
OTEL_PLATFORM=braintrust  # or langsmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-..., x-bt-parent=project_name:..."

# OR collector export
OTEL_PLATFORM=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
# No headers needed!
```

#### API Keys (for Collector)
```bash
# Braintrust
BRAINTRUST_API_KEY=sk-...
BRAINTRUST_PROJECT=scale-test

# LangSmith
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=scale-test
```

#### Test Parameters
```bash
SCALE_TEST_CONCURRENCY=50     # Number of concurrent workers
SCALE_TEST_DURATION=300       # Test duration in seconds
SCALE_TEST_RATE_LIMIT=50      # Max requests per second

# Query mix (must sum to 1.0)
SCALE_TEST_MIX_SIMPLE=0.40    # Simple queries
SCALE_TEST_MIX_SEARCH=0.30    # Service searches
SCALE_TEST_MIX_BOOKING=0.20   # Delegated bookings
SCALE_TEST_MIX_COMPLEX=0.10   # Complex workflows
```

#### OTEL BatchSpanProcessor Tuning
```bash
OTEL_BSP_SCHEDULE_DELAY=1000         # Export every 1s (for collector)
OTEL_BSP_MAX_EXPORT_BATCH_SIZE=100   # Large batches for local network
OTEL_BSP_MAX_QUEUE_SIZE=8192         # Large queue for high concurrency
```

#### Shutdown Behavior
```bash
OTEL_SKIP_SHUTDOWN=false  # Wait for spans to export (direct mode)
OTEL_SKIP_SHUTDOWN=true   # Exit immediately (recommended for collector mode)
```

### Understanding the Queue

When you see `"Queue is full, likely spans will be dropped"`:

- **What queue?** The BatchSpanProcessor queue in YOUR app (not the platform)
- **Where?** In-memory on your machine
- **Size:** Configured by `OTEL_BSP_MAX_QUEUE_SIZE` (default: 8192)
- **Memory usage:** ~4KB per span × queue size (8192 spans ≈ 32MB RAM)

**Solution:** Increase `OTEL_BSP_MAX_QUEUE_SIZE` or use collector mode

### Tuning Guidelines

| Concurrency | Mode      | Schedule Delay | Queue Size | Notes |
|-------------|-----------|----------------|------------|-------|
| 1-10        | Direct    | 5000ms         | 4096       | Works for small tests |
| 10-50       | Direct    | 3000ms         | 8192       | Will timeout on shutdown |
| 50-100      | Collector | 1000ms         | 8192       | Recommended |
| 100+        | Collector | 1000ms         | 16384      | High throughput |

---

## Usage

### Running Tests

```bash
# Basic run
uv run python scripts/run_scale_test.py

# Quick test
SCALE_TEST_DURATION=10 SCALE_TEST_CONCURRENCY=5 uv run python scripts/run_scale_test.py

# With debug logging
LOG_LEVEL=DEBUG uv run python scripts/run_scale_test.py
```

### Logging Levels

```bash
LOG_LEVEL=ERROR    # Only errors
LOG_LEVEL=WARNING  # Warnings and errors
LOG_LEVEL=INFO     # High-level progress (default)
LOG_LEVEL=DEBUG    # Detailed execution (every scenario, iteration)
```

**Troubleshooting tip:** If the script seems stuck, run with `LOG_LEVEL=DEBUG` to see where it's hanging.

### Monitoring Collector

#### Real-time Logs
```bash
# View real-time logs
docker-compose logs -f otel-collector

# Check container status
docker-compose ps

# Restart collector (after editing .env or config)
docker-compose restart

# Stop collector
docker-compose down
```

#### Drop Analysis

After running tests, analyze dropped spans and platform rejections:

```bash
./scripts/analyze_collector_drops.sh
```

**Example Output:**
```
🔍 OpenTelemetry Collector Drop Analysis
==========================================

📉 Dropped Spans Summary:
------------------------
Total spans dropped: 1812

🚫 Platform Rejection Breakdown:
--------------------------------
HTTP 413 (Payload Too Large):
  Occurrences: 1
  Spans dropped: 112

HTTP 429 (Rate Limited):
  Occurrences: 172
  Spans dropped: 0

HTTP 409 (Conflict):
  Occurrences: 17
  Spans dropped: 1700

💡 Recommendations:
⚠️  HTTP 413: Reduce OTEL_BSP_MAX_EXPORT_BATCH_SIZE in collector config
```

**What the metrics mean:**

- **HTTP 413 (Payload Too Large)**: Batch size exceeds platform's limit (10MB for Braintrust)
  - **Fix**: Reduce `max_export_batch_size` in `otel-collector-config.yaml` batch processor

- **HTTP 429 (Rate Limited)**: Platform is rate limiting your requests
  - **Fix**: Reduce test concurrency, increase retry intervals, or contact platform for higher limits
  - **Note**: Retries usually succeed, so drops are rare unless sustained

- **HTTP 503 (Service Unavailable)**: Platform temporarily unavailable
  - **Fix**: Usually transient, retries handle it. If persistent, check platform status

- **HTTP 409 (Conflict)**: Duplicate trace IDs or pipeline conflicts
  - **Fix**: Ensure only ONE platform pipeline is active in collector config

**Comparing Platform Ingestion:**

To compare ingestion capabilities between Braintrust and LangSmith:

1. Run identical test with Braintrust (enable only `traces/braintrust` pipeline)
2. Run drop analysis and save results
3. Switch to LangSmith (disable Braintrust, enable `traces/langsmith` pipeline)
4. Restart collector: `docker-compose restart`
5. Run identical test with LangSmith
6. Run drop analysis and compare:
   - Total spans dropped
   - HTTP error distribution (413 vs 429 vs 503)
   - Memory warnings
   - Retry success rate

#### Prometheus Metrics (Optional)

The collector exposes Prometheus metrics on port 8888. To enable detailed telemetry, add this to `otel-collector-config.yaml`:

```yaml
service:
  telemetry:
    metrics:
      level: detailed
      address: :8888
```

Then query metrics:

```bash
# View all metrics
curl http://localhost:8888/metrics

# Filter for exporter metrics
curl http://localhost:8888/metrics | grep exporter

# Filter for span counts
curl http://localhost:8888/metrics | grep spans
```

**Key metrics:**
- `otelcol_exporter_sent_spans`: Successfully exported spans
- `otelcol_exporter_send_failed_spans`: Failed span exports
- `otelcol_processor_batch_batch_send_size`: Batch sizes being sent
- `otelcol_processor_batch_timeout_trigger`: How often batches timeout

**Note:** The drop analysis script (`analyze_collector_drops.sh`) provides more actionable insights than raw Prometheus metrics.

---

## Built-in Scenarios

1. **simple_query** (40% default): Simple questions (~5K tokens, 5 spans, ~35KB)
2. **single_service_search** (30% default): Service searches (~18K tokens, 12 spans, ~270KB)
3. **delegated_booking** (20% default): Specialist delegation (~40K tokens, 25 spans, ~470KB)
4. **multi_service_complex** (10% default): Multi-service workflows (~185K tokens, 80 spans, ~500KB)

---

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
   simple_query: 6000 traces, P50=30ms, 35KB avg
   single_service_search: 4500 traces, P50=250ms, 270KB avg
   delegated_booking: 3000 traces, P50=1200ms, 470KB avg
   multi_service_complex: 1500 traces, P50=4500ms, 500KB avg
```

---

## Architecture

Three-layer design:
1. **Trace Generation**: Declarative scenario definitions with workflow steps
2. **OTEL Instrumentation**: Automatic span creation with GenAI semantic conventions
3. **Workload Execution**: Async executor with rate limiting and metrics collection

### Code Structure

```
src/
├── workflow.py          # Workflow step definitions (LLM, Tool, Delegation, etc.)
├── scenarios.py         # Pre-defined trace scenarios
├── instrumentation.py   # OTEL span creation with GenAI conventions
├── platforms.py         # Platform configurations (Braintrust, LangSmith, etc.)
├── payloads.py          # Realistic payload generation
├── metrics.py           # Metrics collection and reporting
└── executor.py          # Async workload execution with rate limiting

scripts/
├── run_scale_test.py             # CLI entry point
└── analyze_collector_drops.sh    # Collector drop analysis tool

tests/
└── test_*.py            # Comprehensive test suite
```

---

## Troubleshooting

### Script hangs at shutdown
**Problem:** "Shutting down tracer provider..." hangs for 5+ minutes

**Solution:**
1. Use collector mode (recommended)
2. Or set `OTEL_SKIP_SHUTDOWN=true` (may lose some spans)
3. Or reduce test duration/concurrency

### "Queue is full" warnings
**Problem:** `"Queue is full, likely spans will be dropped"`

**Solution:** Increase queue size:
```bash
OTEL_BSP_MAX_QUEUE_SIZE=16384
```

### Collector "connection refused"
**Problem:** Cannot connect to localhost:4318

**Solution:**
```bash
# Check collector is running
docker-compose ps

# Start if not running
docker-compose up -d
```

### Spans not appearing in platform
**Problem:** Spans aren't showing up in Braintrust/LangSmith

**Solution:**
1. Check API key is set in `.env`
2. Restart collector: `docker-compose restart`
3. Check collector logs: `docker-compose logs otel-collector`
4. Look for "401 Unauthorized" or "429 Rate Limit" errors
5. Run drop analysis: `./scripts/analyze_collector_drops.sh` to see if spans are being rejected

### Collector dropping spans
**Problem:** Collector logs show "Exporting failed. Dropping data" messages

**Solution:**
1. Run drop analysis to identify the root cause:
   ```bash
   ./scripts/analyze_collector_drops.sh
   ```

2. Address based on HTTP status code:
   - **HTTP 413**: Batch size too large
     - Reduce `max_export_batch_size` in `otel-collector-config.yaml` (try 50 or 10)
   - **HTTP 429**: Rate limiting
     - Reduce test concurrency (`SCALE_TEST_CONCURRENCY`)
     - Increase batch schedule delay in collector config
   - **HTTP 409**: Multiple pipelines active
     - Ensure only ONE platform pipeline is uncommented in collector config
     - Restart collector after changing config

3. For persistent issues, check platform-specific limits:
   - Braintrust: 10MB per batch, rate limits vary by plan
   - LangSmith: Contact support for specific limits

### Collector errors on startup
**Problem:** Collector crashes with config errors

**Solution:**
1. Check `.env` has API keys set
2. Verify pipeline is uncommented in `otel-collector-config.yaml`
3. Check logs: `docker-compose logs otel-collector`

---

## Development

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_scenarios.py -v

# Run with output
uv run pytest -v -s
```

---

## Performance Comparison

### Direct Export (549 traces, concurrency=50)
- Test duration: 100s
- Shutdown wait: 300s (timed out!)
- **Total time: 400s+**
- Spans lost: ~10-20%

### Collector Export (same test)
- Test duration: 100s
- Shutdown wait: < 5s
- **Total time: 105s**
- Spans lost: 0%

**10x faster iteration, 100% data delivery!**

---

## License

MIT
