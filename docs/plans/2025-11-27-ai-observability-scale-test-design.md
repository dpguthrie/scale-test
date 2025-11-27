# AI Observability Scale Test Framework - Design Document

**Date**: 2025-11-27
**Status**: Design Complete, Ready for Implementation
**Target Scale**: 10-100 requests/second (medium production load)

## Overview

A framework for scale testing AI observability platforms (Braintrust, LangSmith, and OTLP-compatible backends) using realistic agent workloads. The system generates production-like traces inspired by a travel customer support agent, with configurable concurrency, trace characteristics, and query mix distributions.

## Goals

1. **Realistic Workloads**: Generate traces that mimic real AI agent behavior with proper nesting, LLM calls, tool invocations, and routing decisions
2. **Multi-Platform Support**: Export to Braintrust, LangSmith, and any OTLP-compatible backend using OpenTelemetry
3. **Scale Testing**: Support 10-100 req/sec with configurable concurrency, rate limiting, and ramp-up periods
4. **Configurable Characteristics**: Test large payloads (100-250KB), deep nesting (5-10 levels), and high span counts (50-200+ spans)
5. **Framework-Free**: No agent frameworks (LangGraph, LangChain, etc.) - pure Python + OTEL SDK

## Architecture

### Three-Layer Design

#### 1. Trace Generation Layer
Models realistic agent workflows inspired by the expedia travel customer support agent:

**TraceScenario**: Declarative workflow definitions
```python
class TraceScenario:
    name: str                           # e.g., "hotel_booking_complex"
    workflow_steps: List[WorkflowStep]  # Ordered execution steps
    expected_span_count: int            # 50-200 for high count tests
    total_payload_size: str             # e.g., "150-250KB"
```

**WorkflowStep Types**:
- `LLMStep`: Simulates LLM call with input/output tokens, model name, latency (time.sleep)
- `ToolStep`: Simulates tool invocation (search_flights, book_hotel) with realistic payloads
- `RoutingStep`: Decision points that branch to different sub-workflows
- `DelegationStep`: Nested sub-agent execution (primary → specialist assistant)
- `ParallelStep`: Multiple tools executing concurrently

**Built-in Scenarios** (based on expedia patterns):
1. **Simple query** (5-10 spans, ~20KB): "What's my flight status?"
2. **Single-service search** (10-20 spans, ~50KB): "Find flights to NYC"
3. **Delegated booking** (30-50 spans, ~150KB): "Book a hotel in Zurich"
4. **Multi-service complex** (80-150 spans, ~250KB): "Plan trip: flight + hotel + car"
5. **Deep nested workflow** (50-100 spans, 8-10 levels deep): Multiple delegation layers

#### 2. OTEL Instrumentation Layer
Converts workflows to OpenTelemetry spans:

**Automatic Span Creation**:
- Proper parent-child relationships based on workflow nesting
- Context propagation for nested and parallel operations
- BatchSpanProcessor for production-scale throughput

**Span Attributes** (GenAI semantic conventions):

*LLM Spans*:
- `gen_ai.request.model` / `gen_ai.response.model`
- `gen_ai.prompt_json` / `gen_ai.completion_json`
- `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens`
- `gen_ai.request.temperature`, `gen_ai.request.max_tokens`

*Tool Spans*:
- `gen_ai.tool.name`
- `gen_ai.operation.name` = "execute_tool"
- Tool-specific parameters and results

*Agent Spans*:
- `gen_ai.agent.tools` (JSON array of available tools)
- Custom attributes for delegation depth, routing decisions

**Platform-Specific Attributes**:

*Braintrust*:
- `braintrust.input_json` / `braintrust.output_json`
- `braintrust.metadata` (JSON dict of model params)
- `braintrust.metrics` (performance metrics)
- Root span requirement enforced

*LangSmith*:
- `langsmith.span.kind` (llm/chain/tool/retriever)
- `langsmith.metadata.{key}` for custom metadata
- Indexed message format: `gen_ai.prompt.{n}.content`, `gen_ai.prompt.{n}.role`

**Payload Generation**:
- Template-based realistic data (flight results, hotel listings from expedia DB schema)
- Variable injection for uniqueness (IDs, timestamps, names)
- Configurable padding to hit target KB sizes

#### 3. Workload Execution Layer
Manages concurrent execution and metrics collection:

**Async Execution**:
- `asyncio` worker pool with configurable concurrency
- Token bucket rate limiter (configurable req/sec limit)
- Weighted random scenario selection based on query mix
- Gradual ramp-up period to avoid cold start issues

**Metrics Collection** (inspired by expedia's scale_test.py):
- Throughput: actual req/sec, total requests
- Latency: P50, P95, P99, min, max
- Data volume: total bytes, avg per trace
- Success rate: completed vs failed
- Per-scenario breakdown
- Platform-specific: export failures, retries

## Configuration

### Environment Variables

```bash
# Platform Configuration
OTEL_PLATFORM=braintrust|langsmith|otlp|console
OTEL_EXPORTER_OTLP_ENDPOINT=<platform endpoint>
OTEL_EXPORTER_OTLP_HEADERS=<auth headers>
OTEL_PROJECT_NAME=scale-test

# Scale Test Parameters
SCALE_TEST_CONCURRENCY=50              # Concurrent workers
SCALE_TEST_DURATION=300                # Test duration in seconds
SCALE_TEST_RATE_LIMIT=50               # Max requests/second (optional)
SCALE_TEST_RAMP_UP=30                  # Gradual ramp-up period

# Query Mix (weights, must sum to 1.0)
SCALE_TEST_MIX_SIMPLE=0.40             # Simple queries (40%)
SCALE_TEST_MIX_SEARCH=0.30             # Single-service searches (30%)
SCALE_TEST_MIX_BOOKING=0.20            # Delegated bookings (20%)
SCALE_TEST_MIX_COMPLEX=0.10            # Multi-service complex (10%)

# Trace Characteristics
SCALE_TEST_PAYLOAD_SIZE=realistic|small|large  # realistic=100-250KB
SCALE_TEST_SPAN_COUNT=realistic|low|high       # realistic=50-150 spans
SCALE_TEST_NESTING_DEPTH=realistic|shallow|deep # realistic=5-8 levels
```

### Platform-Specific Configuration

**Braintrust**:
```bash
OTEL_PLATFORM=braintrust
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <API_KEY>, x-bt-parent=project_id:<PROJECT_ID>"
```

**LangSmith**:
```bash
OTEL_PLATFORM=langsmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
OTEL_EXPORTER_OTLP_HEADERS="x-api-key=<API_KEY>, Langsmith-Project=<PROJECT_NAME>"
```

**Generic OTLP**:
```bash
OTEL_PLATFORM=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=""
```

## Project Structure

```
scale-test/
├── src/
│   ├── __init__.py
│   ├── scenarios.py          # TraceScenario definitions
│   ├── workflow.py            # WorkflowStep types (LLMStep, ToolStep, etc.)
│   ├── instrumentation.py     # OTEL span creation logic
│   ├── platforms.py           # Platform-specific configs (Braintrust/LangSmith)
│   ├── executor.py            # Async workload execution engine
│   ├── metrics.py             # Metrics collection and reporting
│   └── payloads.py            # Realistic data generation templates
├── tests/
│   ├── test_scenarios.py
│   ├── test_instrumentation.py
│   └── test_platforms.py
├── scripts/
│   └── run_scale_test.py      # Main CLI entry point
├── docs/
│   └── plans/
│       └── 2025-11-27-ai-observability-scale-test-design.md
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

## Component Details

### scenarios.py

Defines the built-in trace scenarios:

```python
SIMPLE_QUERY = TraceScenario(
    name="simple_query",
    workflow_steps=[
        LLMStep(name="primary_assistant", tokens_in=50, tokens_out=100, latency_ms=500),
        ToolStep(name="fetch_user_info", payload_kb=5, latency_ms=100),
        LLMStep(name="primary_assistant_response", tokens_in=150, tokens_out=50, latency_ms=300),
    ],
    expected_span_count=5,
    total_payload_size="20KB"
)

DELEGATED_BOOKING = TraceScenario(
    name="hotel_booking",
    workflow_steps=[
        LLMStep(name="primary_assistant", tokens_in=100, tokens_out=50),
        DelegationStep(
            name="hotel_assistant",
            sub_workflow=[
                LLMStep(name="hotel_assistant_plan", tokens_in=200, tokens_out=100),
                ToolStep(name="search_hotels", payload_kb=30, latency_ms=200),
                LLMStep(name="hotel_assistant_select", tokens_in=500, tokens_out=100),
                ToolStep(name="book_hotel", payload_kb=10, latency_ms=150),
            ]
        ),
        LLMStep(name="primary_assistant_confirm", tokens_in=300, tokens_out=100),
    ],
    expected_span_count=30,
    total_payload_size="150KB"
)
```

### workflow.py

Base classes for workflow steps:

```python
class WorkflowStep(ABC):
    name: str

    @abstractmethod
    async def execute(self, tracer, parent_span_context):
        """Execute this workflow step and create appropriate OTEL spans"""
        pass

class LLMStep(WorkflowStep):
    tokens_in: int
    tokens_out: int
    model: str = "claude-sonnet-4"
    latency_ms: int = 500

class ToolStep(WorkflowStep):
    tool_name: str
    payload_kb: int
    latency_ms: int = 100

class DelegationStep(WorkflowStep):
    sub_workflow: List[WorkflowStep]
    assistant_type: str
```

### instrumentation.py

OTEL span creation with proper attributes:

```python
def create_llm_span(tracer, name, tokens_in, tokens_out, model, parent_ctx):
    """Create an LLM span with GenAI semantic conventions"""
    with tracer.start_as_current_span(name, context=parent_ctx) as span:
        # GenAI attributes
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.usage.prompt_tokens", tokens_in)
        span.set_attribute("gen_ai.usage.completion_tokens", tokens_out)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Platform-specific
        if platform == "braintrust":
            span.set_attribute("braintrust.input_json", generate_prompt())
            span.set_attribute("braintrust.output_json", generate_completion())
        elif platform == "langsmith":
            span.set_attribute("langsmith.span.kind", "llm")
            for i, msg in enumerate(messages):
                span.set_attribute(f"gen_ai.prompt.{i}.content", msg.content)
                span.set_attribute(f"gen_ai.prompt.{i}.role", msg.role)
```

### platforms.py

Platform-specific configuration:

```python
class BraintrustPlatform:
    """Braintrust OTEL configuration"""
    endpoint = "https://api.braintrust.dev/otel/v1/traces"

    def get_headers(self, api_key, project_id):
        return {
            "Authorization": f"Bearer {api_key}",
            "x-bt-parent": f"project_id:{project_id}"
        }

    def ensure_root_span(self, trace):
        """Ensure trace has root span (Braintrust requirement)"""
        # Implementation ensures every trace has a root span
        pass

class LangSmithPlatform:
    """LangSmith OTEL configuration"""
    endpoint = "https://api.smith.langchain.com/otel"

    def get_headers(self, api_key, project_name):
        return {
            "x-api-key": api_key,
            "Langsmith-Project": project_name
        }
```

### executor.py

Async workload execution:

```python
class ScaleTestExecutor:
    def __init__(self, config):
        self.concurrency = config.concurrency
        self.rate_limiter = TokenBucketRateLimiter(config.rate_limit)
        self.scenarios = self._build_scenario_pool(config.query_mix)
        self.metrics = MetricsCollector()

    async def run(self, duration_seconds):
        """Run scale test for specified duration"""
        tasks = []
        for _ in range(self.concurrency):
            tasks.append(self._worker())

        await asyncio.gather(*tasks)
        return self.metrics.report()

    async def _worker(self):
        """Worker coroutine that executes scenarios"""
        while not self.should_stop:
            await self.rate_limiter.acquire()
            scenario = random.choices(self.scenarios, weights=self.weights)[0]
            await self._execute_scenario(scenario)
```

### metrics.py

Metrics collection and reporting:

```python
class MetricsCollector:
    def __init__(self):
        self.latencies = []
        self.data_volumes = []
        self.success_count = 0
        self.failure_count = 0
        self.per_scenario_metrics = defaultdict(list)

    def report(self):
        """Generate comprehensive metrics report"""
        return {
            "throughput": self.success_count / self.total_duration,
            "latency_p50": np.percentile(self.latencies, 50),
            "latency_p95": np.percentile(self.latencies, 95),
            "latency_p99": np.percentile(self.latencies, 99),
            "total_data_gb": sum(self.data_volumes) / (1024**3),
            "success_rate": self.success_count / (self.success_count + self.failure_count),
            "per_scenario": self._scenario_breakdown()
        }
```

### payloads.py

Realistic data generation:

```python
def generate_flight_search_results(count=20, size_kb=30):
    """Generate realistic flight search results matching expedia schema"""
    flights = []
    for i in range(count):
        flights.append({
            "flight_id": random.randint(1000, 9999),
            "flight_no": f"AA{random.randint(100, 999)}",
            "departure_airport": random.choice(["JFK", "LAX", "ORD", "SFO"]),
            "arrival_airport": random.choice(["LHR", "CDG", "FRA", "AMS"]),
            "scheduled_departure": generate_timestamp(),
            "scheduled_arrival": generate_timestamp(),
            # Pad to reach target size
            "details": generate_padding(size_kb // count)
        })
    return flights
```

## Usage Example

```bash
# Setup
export OTEL_PLATFORM=braintrust
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-..., x-bt-parent=project_id:..."

# Run scale test
export SCALE_TEST_CONCURRENCY=50
export SCALE_TEST_DURATION=300
export SCALE_TEST_RATE_LIMIT=50
export SCALE_TEST_MIX_SIMPLE=0.4
export SCALE_TEST_MIX_SEARCH=0.3
export SCALE_TEST_MIX_BOOKING=0.2
export SCALE_TEST_MIX_COMPLEX=0.1

python scripts/run_scale_test.py
```

**Expected Output**:
```
📊 Scale Test Results:
   Duration: 300.0s
   Total traces: 15,000
   Success rate: 99.8%
   Throughput: 50.0 req/s

⏱️  Latency:
   P50: 45ms | P95: 120ms | P99: 250ms

📦 Data Volume:
   Total: 2.25 GB
   Avg per trace: 150 KB

📈 Query Mix Breakdown:
   Simple (40%): 6000 traces, P50=20ms, 20KB avg
   Search (30%): 4500 traces, P50=35ms, 50KB avg
   Booking (20%): 3000 traces, P50=80ms, 150KB avg
   Complex (10%): 1500 traces, P50=180ms, 250KB avg
```

## Design Decisions

### 1. Framework-Free Approach
**Decision**: Build without LangGraph, LangChain, or other agent frameworks
**Rationale**:
- Maximum control over trace structure and span attributes
- No framework overhead or version conflicts
- Easier to understand and maintain
- Can generate exact trace patterns we want to test

### 2. Time.sleep() for LLM Simulation
**Decision**: Use time.sleep() to simulate LLM latency instead of real API calls
**Rationale**:
- Zero cost for generating millions of traces
- Predictable, controllable behavior
- Focus on testing observability platform scale, not LLM APIs
- Can still add real LLM mode later if needed

### 3. Environment Variables for Configuration
**Decision**: Use environment variables instead of CLI args or config files
**Rationale**:
- Better for containerized environments
- Clean separation of secrets (API keys) from test parameters
- Easy to override in CI/CD pipelines
- Follows 12-factor app principles

### 4. GenAI Semantic Conventions
**Decision**: Use OpenTelemetry GenAI semantic conventions as primary attribute schema
**Rationale**:
- Compatible with both Braintrust and LangSmith
- Industry standard for AI/ML observability
- Future-proof as more platforms adopt OTEL
- Add platform-specific attributes only when necessary

### 5. Inspired by Expedia, Not Copied
**Decision**: Take inspiration from expedia's patterns but rebuild from scratch
**Rationale**:
- Expedia uses LangGraph (framework dependency we want to avoid)
- We only need the trace *patterns*, not the actual agent logic
- Simpler implementation focused on trace generation
- No database or external service dependencies

## Platform Integration Details

### Braintrust Critical Requirements

1. **Root Span Requirement**: Every trace MUST have a root span (traces with only child spans won't appear in UI)
   - Implementation: Ensure first span in every scenario has no parent context

2. **Recommended Attributes**:
   - Use `braintrust.*` namespace for direct field mapping
   - `braintrust.input_json` / `braintrust.output_json` for messages
   - `braintrust.metadata` for model params (JSON dict)
   - `braintrust.metrics` for custom metrics

3. **BatchSpanProcessor**: Required for production scale
   - Configure with appropriate batch size and timeout
   - Monitor export failures

### LangSmith Critical Requirements

1. **Span Kind Attribute**: Must set `langsmith.span.kind` (llm/chain/tool/retriever)
   - Maps OTEL spans to LangSmith's run type system

2. **Message Format Options**:
   - Indexed: `gen_ai.prompt.{n}.content`, `gen_ai.prompt.{n}.role`
   - Array: `gen_ai.input.messages`, `gen_ai.output.messages`
   - Implementation supports both

3. **Token Conventions**: Supports multiple naming conventions
   - `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
   - `gen_ai.usage.prompt_tokens` / `gen_ai.usage.completion_tokens`
   - Use GenAI standard for maximum compatibility

## Testing Strategy

### Unit Tests
- `test_scenarios.py`: Validate scenario definitions and execution
- `test_instrumentation.py`: Verify correct OTEL span attributes
- `test_platforms.py`: Test platform-specific configuration

### Integration Tests
- Console exporter: Verify spans generated correctly
- Local OTEL collector: Test OTLP export pipeline
- Small-scale runs: Validate metrics collection

### Scale Tests
- Ramp up to target concurrency gradually
- Monitor memory usage and CPU utilization
- Verify export success rates at high throughput
- Compare results across platforms

## Future Enhancements

1. **Additional Platforms**: Jaeger, Honeycomb, Datadog, New Relic
2. **Real LLM Mode**: Optional flag to make actual API calls for integration testing
3. **Custom Scenarios**: YAML/JSON format for user-defined workflows
4. **Streaming Simulation**: Generate span events for streaming LLM responses
5. **Error Injection**: Configurable failure rates to test error handling
6. **Multi-Region Testing**: Distributed load generation across regions

## Success Criteria

1. ✅ Generate 50 req/sec sustained for 5+ minutes
2. ✅ Traces appear correctly in both Braintrust and LangSmith
3. ✅ Realistic trace characteristics (100-250KB, 50-150 spans, 5-8 levels deep)
4. ✅ <1% export failure rate at target scale
5. ✅ Comprehensive metrics reporting
6. ✅ Zero external dependencies (no frameworks, no databases)

## References

- Expedia travel agent: ~/repos/expedia
- Braintrust OTEL integration: https://www.braintrust.dev/docs/integrations/sdk-integrations/opentelemetry
- LangSmith OTEL integration: https://docs.langchain.com/langsmith/trace-with-opentelemetry
- OpenTelemetry GenAI conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
