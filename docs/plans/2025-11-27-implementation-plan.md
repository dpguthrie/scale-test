# AI Observability Scale Test Framework - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a framework-free OpenTelemetry-based scale testing system that generates realistic AI agent traces for testing Braintrust, LangSmith, and other observability platforms at 10-100 req/sec.

**Architecture:** Three-layer design with declarative scenario definitions, OTEL instrumentation middleware, and async workload executor. Scenarios model expedia-style agent workflows (LLM calls, tool invocations, delegation, routing) that generate proper OTEL spans with GenAI semantic conventions and platform-specific attributes.

**Tech Stack:** Python 3.11+, OpenTelemetry SDK, asyncio, no agent frameworks

---

## Task 1: Project Foundation

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/__init__.py`

**Step 1: Create requirements.txt with dependencies**

Create `requirements.txt`:
```txt
# OpenTelemetry core
opentelemetry-api==1.25.0
opentelemetry-sdk==1.25.0
opentelemetry-exporter-otlp-proto-http==1.25.0

# Async and utilities
aiohttp==3.9.5
python-dotenv==1.0.1
numpy==1.26.4

# Testing
pytest==8.2.0
pytest-asyncio==0.23.7
```

**Step 2: Create .env.example**

Create `.env.example`:
```bash
# Platform Configuration
OTEL_PLATFORM=braintrust
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.braintrust.dev/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer YOUR_API_KEY, x-bt-parent=project_id:YOUR_PROJECT_ID"
OTEL_PROJECT_NAME=scale-test

# Scale Test Parameters
SCALE_TEST_CONCURRENCY=50
SCALE_TEST_DURATION=300
SCALE_TEST_RATE_LIMIT=50
SCALE_TEST_RAMP_UP=30

# Query Mix (must sum to 1.0)
SCALE_TEST_MIX_SIMPLE=0.40
SCALE_TEST_MIX_SEARCH=0.30
SCALE_TEST_MIX_BOOKING=0.20
SCALE_TEST_MIX_COMPLEX=0.10

# Trace Characteristics
SCALE_TEST_PAYLOAD_SIZE=realistic
SCALE_TEST_SPAN_COUNT=realistic
SCALE_TEST_NESTING_DEPTH=realistic
```

**Step 3: Create .gitignore**

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# OS
.DS_Store
Thumbs.db
```

**Step 4: Create basic README**

Create `README.md`:
```markdown
# AI Observability Scale Test Framework

Scale testing framework for AI observability platforms using realistic OpenTelemetry traces.

## Features

- Framework-free trace generation inspired by real agent workloads
- Multi-platform support: Braintrust, LangSmith, OTLP-compatible backends
- Configurable scale: 10-100 req/sec with async execution
- Realistic characteristics: 100-250KB traces, 50-150 spans, 5-8 nesting levels

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

```bash
python scripts/run_scale_test.py
```

See `docs/plans/2025-11-27-ai-observability-scale-test-design.md` for full design.
```

**Step 5: Create src package**

Create `src/__init__.py`:
```python
"""AI Observability Scale Test Framework"""

__version__ = "0.1.0"
```

**Step 6: Commit project foundation**

```bash
git add requirements.txt .env.example .gitignore README.md src/__init__.py docs/
git commit -m "feat: add project foundation with dependencies and documentation"
```

---

## Task 2: Workflow Step Base Classes

**Files:**
- Create: `src/workflow.py`
- Create: `tests/test_workflow.py`

**Step 1: Write failing test for WorkflowStep ABC**

Create `tests/test_workflow.py`:
```python
import pytest
from src.workflow import WorkflowStep, LLMStep, ToolStep, DelegationStep


def test_workflow_step_is_abstract():
    """WorkflowStep cannot be instantiated directly"""
    with pytest.raises(TypeError):
        WorkflowStep(name="test")


def test_llm_step_creation():
    """LLMStep can be created with required parameters"""
    step = LLMStep(
        name="test_llm",
        tokens_in=100,
        tokens_out=50,
        model="claude-sonnet-4"
    )
    assert step.name == "test_llm"
    assert step.tokens_in == 100
    assert step.tokens_out == 50
    assert step.model == "claude-sonnet-4"
    assert step.latency_ms == 500  # default


def test_tool_step_creation():
    """ToolStep can be created with required parameters"""
    step = ToolStep(
        name="search_flights",
        payload_kb=30,
        latency_ms=200
    )
    assert step.name == "search_flights"
    assert step.payload_kb == 30
    assert step.latency_ms == 200


def test_delegation_step_creation():
    """DelegationStep can contain sub-workflow"""
    sub_steps = [
        LLMStep(name="sub_llm", tokens_in=50, tokens_out=25),
        ToolStep(name="sub_tool", payload_kb=10)
    ]
    step = DelegationStep(
        name="hotel_assistant",
        assistant_type="hotel",
        sub_workflow=sub_steps
    )
    assert step.name == "hotel_assistant"
    assert step.assistant_type == "hotel"
    assert len(step.sub_workflow) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL with "No module named 'src.workflow'"

**Step 3: Implement workflow base classes**

Create `src/workflow.py`:
```python
"""Workflow step definitions for trace generation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorkflowStep(ABC):
    """Base class for workflow steps that generate OTEL spans"""
    name: str

    @abstractmethod
    async def execute(self, tracer, parent_span_context):
        """Execute this workflow step and create appropriate OTEL spans

        Args:
            tracer: OpenTelemetry tracer instance
            parent_span_context: Parent span context for nesting

        Returns:
            Span context of created span(s)
        """
        pass


@dataclass
class LLMStep(WorkflowStep):
    """Simulates an LLM call with token usage and latency"""
    tokens_in: int
    tokens_out: int
    model: str = "claude-sonnet-4"
    latency_ms: int = 500
    temperature: float = 1.0
    max_tokens: int = 4096


@dataclass
class ToolStep(WorkflowStep):
    """Simulates a tool invocation with payload size and latency"""
    payload_kb: int
    latency_ms: int = 100
    tool_parameters: dict = field(default_factory=dict)


@dataclass
class DelegationStep(WorkflowStep):
    """Simulates delegation to a sub-agent with nested workflow"""
    assistant_type: str
    sub_workflow: List[WorkflowStep]


@dataclass
class RoutingStep(WorkflowStep):
    """Simulates a routing decision point that branches workflow"""
    decision: str
    branches: dict = field(default_factory=dict)


@dataclass
class ParallelStep(WorkflowStep):
    """Simulates parallel execution of multiple tools"""
    parallel_steps: List[WorkflowStep]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit workflow base classes**

```bash
git add src/workflow.py tests/test_workflow.py
git commit -m "feat: add workflow step base classes with dataclass definitions"
```

---

## Task 3: Realistic Payload Generation

**Files:**
- Create: `src/payloads.py`
- Create: `tests/test_payloads.py`

**Step 1: Write failing test for payload generators**

Create `tests/test_payloads.py`:
```python
import json
from src.payloads import (
    generate_flight_search_results,
    generate_hotel_search_results,
    generate_llm_messages,
    generate_tool_result,
    estimate_json_size_kb
)


def test_flight_search_results_structure():
    """Flight search results have correct schema"""
    results = generate_flight_search_results(count=5, target_size_kb=30)
    assert len(results) == 5
    assert "flight_id" in results[0]
    assert "flight_no" in results[0]
    assert "departure_airport" in results[0]
    assert "arrival_airport" in results[0]


def test_flight_search_results_size():
    """Flight search results approximately match target size"""
    results = generate_flight_search_results(count=20, target_size_kb=50)
    size_kb = estimate_json_size_kb(results)
    assert 45 <= size_kb <= 55  # Within 10% tolerance


def test_hotel_search_results_structure():
    """Hotel search results have correct schema"""
    results = generate_hotel_search_results(count=10, target_size_kb=40)
    assert len(results) == 10
    assert "id" in results[0]
    assert "name" in results[0]
    assert "location" in results[0]
    assert "price_tier" in results[0]


def test_llm_messages_format():
    """LLM messages have proper chat format"""
    messages = generate_llm_messages(
        user_prompt="Book a hotel in Zurich",
        assistant_response="I'll help you search for hotels",
        target_size_kb=10
    )
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "assistant"


def test_tool_result_format():
    """Tool results have proper structure"""
    result = generate_tool_result(
        tool_name="search_flights",
        success=True,
        data={"flights": [{"id": 1}]},
        target_size_kb=5
    )
    assert result["tool"] == "search_flights"
    assert result["success"] is True
    assert "data" in result
    assert "timestamp" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_payloads.py -v`
Expected: FAIL with "No module named 'src.payloads'"

**Step 3: Implement payload generators**

Create `src/payloads.py`:
```python
"""Realistic payload generation for trace data"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


AIRPORTS = ["JFK", "LAX", "ORD", "SFO", "BOS", "DFW", "ATL", "MIA",
            "LHR", "CDG", "FRA", "AMS", "ZRH", "BCN", "FCO"]
AIRLINES = ["AA", "UA", "DL", "BA", "LH", "AF", "KL", "LX"]
HOTEL_NAMES = ["Grand Hotel", "City Inn", "Plaza Hotel", "Resort & Spa",
               "Boutique Hotel", "Business Hotel", "Airport Hotel"]
LOCATIONS = ["New York", "Los Angeles", "Chicago", "San Francisco",
             "London", "Paris", "Frankfurt", "Zurich", "Barcelona"]
PRICE_TIERS = ["Midscale", "Upper Midscale", "Upscale", "Luxury"]


def generate_flight_search_results(count: int = 20, target_size_kb: int = 30) -> List[Dict]:
    """Generate realistic flight search results matching expedia schema

    Args:
        count: Number of flight results to generate
        target_size_kb: Target size in KB for the result set

    Returns:
        List of flight result dictionaries
    """
    flights = []
    base_time = datetime.now()

    for i in range(count):
        departure_time = base_time + timedelta(days=random.randint(1, 30))
        arrival_time = departure_time + timedelta(hours=random.randint(2, 12))

        flight = {
            "flight_id": random.randint(1000, 9999),
            "flight_no": f"{random.choice(AIRLINES)}{random.randint(100, 999)}",
            "departure_airport": random.choice(AIRPORTS),
            "arrival_airport": random.choice(AIRPORTS),
            "scheduled_departure": departure_time.isoformat(),
            "scheduled_arrival": arrival_time.isoformat(),
            "fare_conditions": random.choice(["Economy", "Business", "First"]),
            "price": random.randint(200, 2000),
            "available_seats": random.randint(0, 50),
        }
        flights.append(flight)

    # Pad to reach target size
    current_size = estimate_json_size_kb(flights)
    if current_size < target_size_kb:
        padding_per_flight = (target_size_kb - current_size) * 1024 // count
        for flight in flights:
            flight["details"] = "x" * padding_per_flight

    return flights


def generate_hotel_search_results(count: int = 10, target_size_kb: int = 40) -> List[Dict]:
    """Generate realistic hotel search results

    Args:
        count: Number of hotel results
        target_size_kb: Target size in KB

    Returns:
        List of hotel result dictionaries
    """
    hotels = []

    for i in range(count):
        hotel = {
            "id": random.randint(1000, 9999),
            "name": f"{random.choice(HOTEL_NAMES)} {random.choice(LOCATIONS)}",
            "location": random.choice(LOCATIONS),
            "price_tier": random.choice(PRICE_TIERS),
            "price_per_night": random.randint(100, 800),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "amenities": random.sample(
                ["WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Parking"],
                k=random.randint(3, 7)
            ),
            "available_rooms": random.randint(0, 20),
        }
        hotels.append(hotel)

    # Pad to target size
    current_size = estimate_json_size_kb(hotels)
    if current_size < target_size_kb:
        padding_per_hotel = (target_size_kb - current_size) * 1024 // count
        for hotel in hotels:
            hotel["description"] = "x" * padding_per_hotel

    return hotels


def generate_llm_messages(
    user_prompt: str,
    assistant_response: str,
    target_size_kb: int = 10
) -> List[Dict[str, str]]:
    """Generate LLM message array in chat format

    Args:
        user_prompt: User's input message
        assistant_response: Assistant's response
        target_size_kb: Target total size in KB

    Returns:
        List of message dictionaries with role and content
    """
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

    # Pad assistant response to reach target size
    current_size = estimate_json_size_kb(messages)
    if current_size < target_size_kb:
        padding = "x" * ((target_size_kb - current_size) * 1024)
        messages[-1]["content"] += f"\n\nAdditional context: {padding}"

    return messages


def generate_tool_result(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    target_size_kb: int = 5
) -> Dict[str, Any]:
    """Generate a tool execution result

    Args:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        data: Tool result data
        target_size_kb: Target size in KB

    Returns:
        Tool result dictionary
    """
    result = {
        "tool": tool_name,
        "success": success,
        "data": data or {},
        "timestamp": datetime.now().isoformat(),
    }

    # Pad to target size
    current_size = estimate_json_size_kb(result)
    if current_size < target_size_kb:
        padding = "x" * ((target_size_kb - current_size) * 1024)
        result["metadata"] = {"padding": padding}

    return result


def estimate_json_size_kb(obj: Any) -> float:
    """Estimate size of object when serialized to JSON

    Args:
        obj: Object to estimate size of

    Returns:
        Estimated size in KB
    """
    json_str = json.dumps(obj)
    return len(json_str.encode('utf-8')) / 1024
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_payloads.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit payload generation**

```bash
git add src/payloads.py tests/test_payloads.py
git commit -m "feat: add realistic payload generators for flights, hotels, and LLM messages"
```

---

## Task 4: Platform Configuration

**Files:**
- Create: `src/platforms.py`
- Create: `tests/test_platforms.py`

**Step 1: Write failing test for platform configs**

Create `tests/test_platforms.py`:
```python
import pytest
from src.platforms import (
    BraintrustPlatform,
    LangSmithPlatform,
    OTLPPlatform,
    ConsolePlatform,
    get_platform
)


def test_braintrust_platform_configuration():
    """Braintrust platform has correct endpoint and headers"""
    platform = BraintrustPlatform(
        api_key="test-key",
        project_id="test-project"
    )
    assert platform.endpoint == "https://api.braintrust.dev/otel/v1/traces"
    headers = platform.get_headers()
    assert "Authorization" in headers
    assert "Bearer test-key" in headers["Authorization"]
    assert "x-bt-parent" in headers


def test_langsmith_platform_configuration():
    """LangSmith platform has correct endpoint and headers"""
    platform = LangSmithPlatform(
        api_key="test-key",
        project_name="test-project"
    )
    assert platform.endpoint == "https://api.smith.langchain.com/otel"
    headers = platform.get_headers()
    assert headers["x-api-key"] == "test-key"
    assert headers["Langsmith-Project"] == "test-project"


def test_otlp_platform_configuration():
    """Generic OTLP platform accepts custom endpoint"""
    platform = OTLPPlatform(
        endpoint="http://localhost:4318/v1/traces"
    )
    assert platform.endpoint == "http://localhost:4318/v1/traces"
    assert platform.get_headers() == {}


def test_console_platform_no_export():
    """Console platform is for debugging only"""
    platform = ConsolePlatform()
    assert platform.endpoint is None


def test_get_platform_by_name():
    """get_platform factory creates correct platform type"""
    config = {
        "platform": "braintrust",
        "api_key": "test",
        "project_id": "proj"
    }
    platform = get_platform(config)
    assert isinstance(platform, BraintrustPlatform)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_platforms.py -v`
Expected: FAIL with "No module named 'src.platforms'"

**Step 3: Implement platform configurations**

Create `src/platforms.py`:
```python
"""Platform-specific OTEL configuration for observability backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


class Platform(ABC):
    """Base class for observability platform configuration"""

    @property
    @abstractmethod
    def endpoint(self) -> Optional[str]:
        """OTLP endpoint URL"""
        pass

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """HTTP headers for OTLP exporter"""
        pass

    @abstractmethod
    def get_span_attributes(self) -> Dict[str, str]:
        """Platform-specific span attributes to add"""
        pass


@dataclass
class BraintrustPlatform(Platform):
    """Braintrust OTEL configuration

    Docs: https://www.braintrust.dev/docs/integrations/sdk-integrations/opentelemetry

    Critical requirements:
    - Every trace MUST have a root span
    - Use GenAI semantic conventions OR braintrust.* namespace
    - BatchSpanProcessor recommended for scale
    """
    api_key: str
    project_id: str
    endpoint_url: str = "https://api.braintrust.dev/otel/v1/traces"

    @property
    def endpoint(self) -> str:
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-bt-parent": f"project_id:{self.project_id}"
        }

    def get_span_attributes(self) -> Dict[str, str]:
        """Braintrust-specific attributes for direct field mapping"""
        return {
            "braintrust.project_id": self.project_id,
        }


@dataclass
class LangSmithPlatform(Platform):
    """LangSmith OTEL configuration

    Docs: https://docs.langchain.com/langsmith/trace-with-opentelemetry

    Critical requirements:
    - Must set langsmith.span.kind attribute (llm/chain/tool/retriever)
    - Supports indexed messages: gen_ai.prompt.{n}.content
    - Or array format: gen_ai.input.messages
    """
    api_key: str
    project_name: str
    endpoint_url: str = "https://api.smith.langchain.com/otel"
    region: str = "us"  # or "eu"

    @property
    def endpoint(self) -> str:
        if self.region == "eu":
            return "https://eu.api.smith.langchain.com/otel"
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Langsmith-Project": self.project_name
        }

    def get_span_attributes(self) -> Dict[str, str]:
        """LangSmith-specific attributes"""
        return {
            "langsmith.project": self.project_name,
        }


@dataclass
class OTLPPlatform(Platform):
    """Generic OTLP exporter for any compatible backend"""
    endpoint_url: str
    headers: Dict[str, str] = None

    @property
    def endpoint(self) -> str:
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return self.headers or {}

    def get_span_attributes(self) -> Dict[str, str]:
        return {}


@dataclass
class ConsolePlatform(Platform):
    """Console exporter for debugging (no network export)"""

    @property
    def endpoint(self) -> None:
        return None

    def get_headers(self) -> Dict[str, str]:
        return {}

    def get_span_attributes(self) -> Dict[str, str]:
        return {}


def get_platform(config: Dict) -> Platform:
    """Factory function to create platform instance from config

    Args:
        config: Configuration dictionary with platform details

    Returns:
        Platform instance

    Raises:
        ValueError: If platform type is unsupported
    """
    platform_type = config.get("platform", "").lower()

    if platform_type == "braintrust":
        return BraintrustPlatform(
            api_key=config["api_key"],
            project_id=config.get("project_id", config.get("project_name"))
        )
    elif platform_type == "langsmith":
        return LangSmithPlatform(
            api_key=config["api_key"],
            project_name=config["project_name"]
        )
    elif platform_type == "otlp":
        return OTLPPlatform(
            endpoint_url=config["endpoint"],
            headers=config.get("headers", {})
        )
    elif platform_type == "console":
        return ConsolePlatform()
    else:
        raise ValueError(f"Unsupported platform: {platform_type}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_platforms.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit platform configuration**

```bash
git add src/platforms.py tests/test_platforms.py
git commit -m "feat: add platform-specific configurations for Braintrust, LangSmith, and OTLP"
```

---

## Task 5: OTEL Instrumentation Layer

**Files:**
- Create: `src/instrumentation.py`
- Create: `tests/test_instrumentation.py`

**Step 1: Write failing test for span creation**

Create `tests/test_instrumentation.py`:
```python
import pytest
from unittest.mock import Mock, MagicMock
from src.instrumentation import (
    create_llm_span,
    create_tool_span,
    create_agent_span,
    set_genai_attributes,
    set_platform_attributes
)
from src.platforms import BraintrustPlatform, LangSmithPlatform


def test_create_llm_span_genai_attributes():
    """LLM spans have GenAI semantic convention attributes"""
    tracer = Mock()
    span = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = Mock()

    create_llm_span(
        tracer=tracer,
        name="test_llm",
        tokens_in=100,
        tokens_out=50,
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "test"}],
        platform=None
    )

    # Verify GenAI attributes were set
    calls = span.set_attribute.call_args_list
    attr_dict = {call[0][0]: call[0][1] for call in calls}

    assert attr_dict["gen_ai.request.model"] == "claude-sonnet-4"
    assert attr_dict["gen_ai.usage.prompt_tokens"] == 100
    assert attr_dict["gen_ai.usage.completion_tokens"] == 50
    assert attr_dict["gen_ai.operation.name"] == "chat"


def test_create_tool_span_attributes():
    """Tool spans have correct tool attributes"""
    tracer = Mock()
    span = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = Mock()

    create_tool_span(
        tracer=tracer,
        name="search_flights",
        tool_result={"flights": []},
        platform=None
    )

    calls = span.set_attribute.call_args_list
    attr_dict = {call[0][0]: call[0][1] for call in calls}

    assert attr_dict["gen_ai.tool.name"] == "search_flights"
    assert attr_dict["gen_ai.operation.name"] == "execute_tool"


def test_platform_specific_attributes_braintrust():
    """Braintrust platform adds braintrust.* attributes"""
    span = MagicMock()
    platform = BraintrustPlatform(api_key="test", project_id="proj")

    set_platform_attributes(span, platform, span_type="llm", data={})

    calls = span.set_attribute.call_args_list
    attr_dict = {call[0][0]: call[0][1] for call in calls}

    assert "braintrust.project_id" in attr_dict


def test_platform_specific_attributes_langsmith():
    """LangSmith platform adds langsmith.span.kind attribute"""
    span = MagicMock()
    platform = LangSmithPlatform(api_key="test", project_name="proj")

    set_platform_attributes(span, platform, span_type="llm", data={})

    calls = span.set_attribute.call_args_list
    attr_dict = {call[0][0]: call[0][1] for call in calls}

    assert attr_dict["langsmith.span.kind"] == "llm"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_instrumentation.py -v`
Expected: FAIL with "No module named 'src.instrumentation'"

**Step 3: Implement OTEL instrumentation functions**

Create `src/instrumentation.py`:
```python
"""OpenTelemetry instrumentation for creating realistic agent traces"""

import json
import time
from typing import Any, Dict, List, Optional
from opentelemetry import trace
from opentelemetry.trace import Span

from src.platforms import Platform, BraintrustPlatform, LangSmithPlatform


def create_llm_span(
    tracer: trace.Tracer,
    name: str,
    tokens_in: int,
    tokens_out: int,
    model: str,
    messages: List[Dict[str, str]],
    platform: Optional[Platform] = None,
    latency_ms: int = 500,
    temperature: float = 1.0,
    max_tokens: int = 4096
) -> Span:
    """Create an LLM span with GenAI semantic conventions

    Args:
        tracer: OpenTelemetry tracer
        name: Span name
        tokens_in: Input token count
        tokens_out: Output token count
        model: Model name
        messages: Chat messages
        platform: Platform for platform-specific attributes
        latency_ms: Simulated latency
        temperature: Model temperature
        max_tokens: Max tokens setting

    Returns:
        Created span
    """
    with tracer.start_as_current_span(name) as span:
        # Simulate LLM latency
        time.sleep(latency_ms / 1000.0)

        # GenAI semantic conventions
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.response.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Token usage
        span.set_attribute("gen_ai.usage.prompt_tokens", tokens_in)
        span.set_attribute("gen_ai.usage.completion_tokens", tokens_out)
        span.set_attribute("gen_ai.usage.total_tokens", tokens_in + tokens_out)

        # Request parameters
        span.set_attribute("gen_ai.request.temperature", temperature)
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)

        # Messages - use indexed format for compatibility
        for i, msg in enumerate(messages):
            span.set_attribute(f"gen_ai.prompt.{i}.role", msg["role"])
            span.set_attribute(f"gen_ai.prompt.{i}.content", msg["content"])

        # Platform-specific attributes
        if platform:
            set_platform_attributes(
                span,
                platform,
                span_type="llm",
                data={"messages": messages, "model": model}
            )

        return span


def create_tool_span(
    tracer: trace.Tracer,
    name: str,
    tool_result: Dict[str, Any],
    platform: Optional[Platform] = None,
    latency_ms: int = 100
) -> Span:
    """Create a tool execution span

    Args:
        tracer: OpenTelemetry tracer
        name: Tool name
        tool_result: Tool result data
        platform: Platform for platform-specific attributes
        latency_ms: Simulated latency

    Returns:
        Created span
    """
    with tracer.start_as_current_span(name) as span:
        # Simulate tool latency
        time.sleep(latency_ms / 1000.0)

        # GenAI tool attributes
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("gen_ai.operation.name", "execute_tool")

        # Tool result
        result_json = json.dumps(tool_result)
        span.set_attribute("tool.result", result_json)
        span.set_attribute("tool.result.size", len(result_json))

        # Platform-specific attributes
        if platform:
            set_platform_attributes(
                span,
                platform,
                span_type="tool",
                data={"tool": name, "result": tool_result}
            )

        return span


def create_agent_span(
    tracer: trace.Tracer,
    name: str,
    agent_type: str,
    available_tools: List[str],
    platform: Optional[Platform] = None
) -> Span:
    """Create an agent span

    Args:
        tracer: OpenTelemetry tracer
        name: Agent name
        agent_type: Type of agent (primary, hotel, flight, etc.)
        available_tools: List of tools available to agent
        platform: Platform for platform-specific attributes

    Returns:
        Created span
    """
    with tracer.start_as_current_span(name) as span:
        # Agent attributes
        span.set_attribute("gen_ai.agent.tools", json.dumps(available_tools))
        span.set_attribute("agent.type", agent_type)

        # Platform-specific attributes
        if platform:
            set_platform_attributes(
                span,
                platform,
                span_type="agent",
                data={"agent_type": agent_type, "tools": available_tools}
            )

        return span


def set_genai_attributes(span: Span, attributes: Dict[str, Any]):
    """Set GenAI semantic convention attributes on a span

    Args:
        span: OpenTelemetry span
        attributes: Dictionary of attributes to set
    """
    for key, value in attributes.items():
        if isinstance(value, (dict, list)):
            span.set_attribute(key, json.dumps(value))
        else:
            span.set_attribute(key, value)


def set_platform_attributes(
    span: Span,
    platform: Platform,
    span_type: str,
    data: Dict[str, Any]
):
    """Set platform-specific attributes

    Args:
        span: OpenTelemetry span
        platform: Platform instance
        span_type: Type of span (llm, tool, agent)
        data: Span data for platform-specific formatting
    """
    # Add platform base attributes
    for key, value in platform.get_span_attributes().items():
        span.set_attribute(key, value)

    # Platform-specific handling
    if isinstance(platform, BraintrustPlatform):
        _set_braintrust_attributes(span, span_type, data)
    elif isinstance(platform, LangSmithPlatform):
        _set_langsmith_attributes(span, span_type, data)


def _set_braintrust_attributes(span: Span, span_type: str, data: Dict[str, Any]):
    """Set Braintrust-specific attributes using braintrust.* namespace

    Args:
        span: OpenTelemetry span
        span_type: Type of span
        data: Span data
    """
    if span_type == "llm" and "messages" in data:
        # Use braintrust namespace for direct field mapping
        span.set_attribute("braintrust.input_json", json.dumps(data["messages"]))
        span.set_attribute("braintrust.output_json", json.dumps(data["messages"][-1:]))
        span.set_attribute("braintrust.metadata", json.dumps({
            "model": data.get("model", "unknown")
        }))


def _set_langsmith_attributes(span: Span, span_type: str, data: Dict[str, Any]):
    """Set LangSmith-specific attributes

    Args:
        span: OpenTelemetry span
        span_type: Type of span
        data: Span data
    """
    # Map span type to LangSmith span kind
    span_kind_map = {
        "llm": "llm",
        "tool": "tool",
        "agent": "chain"
    }
    span.set_attribute("langsmith.span.kind", span_kind_map.get(span_type, "chain"))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_instrumentation.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit instrumentation layer**

```bash
git add src/instrumentation.py tests/test_instrumentation.py
git commit -m "feat: add OTEL instrumentation with GenAI conventions and platform-specific attributes"
```

---

## Task 6: Trace Scenarios

**Files:**
- Create: `src/scenarios.py`
- Create: `tests/test_scenarios.py`

**Step 1: Write failing test for scenario definitions**

Create `tests/test_scenarios.py`:
```python
from src.scenarios import (
    TraceScenario,
    get_scenario,
    SIMPLE_QUERY,
    SINGLE_SERVICE_SEARCH,
    DELEGATED_BOOKING,
    MULTI_SERVICE_COMPLEX
)


def test_simple_query_scenario():
    """Simple query scenario has expected structure"""
    assert SIMPLE_QUERY.name == "simple_query"
    assert len(SIMPLE_QUERY.workflow_steps) >= 3
    assert SIMPLE_QUERY.expected_span_count >= 5
    assert "20KB" in SIMPLE_QUERY.total_payload_size or "20" in str(SIMPLE_QUERY.expected_size_kb)


def test_delegated_booking_scenario():
    """Delegated booking has delegation step"""
    assert DELEGATED_BOOKING.name == "delegated_booking"
    # Should have delegation step
    has_delegation = any(
        step.__class__.__name__ == "DelegationStep"
        for step in DELEGATED_BOOKING.workflow_steps
    )
    assert has_delegation
    assert DELEGATED_BOOKING.expected_span_count >= 30


def test_get_scenario_by_name():
    """get_scenario retrieves correct scenario"""
    scenario = get_scenario("simple_query")
    assert scenario.name == "simple_query"

    scenario = get_scenario("delegated_booking")
    assert scenario.name == "delegated_booking"


def test_all_scenarios_registered():
    """All built-in scenarios are available"""
    scenarios = ["simple_query", "single_service_search", "delegated_booking", "multi_service_complex"]
    for name in scenarios:
        scenario = get_scenario(name)
        assert scenario is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenarios.py -v`
Expected: FAIL with "No module named 'src.scenarios'"

**Step 3: Implement trace scenarios**

Create `src/scenarios.py`:
```python
"""Pre-defined trace scenarios inspired by expedia travel agent patterns"""

from dataclasses import dataclass
from typing import List

from src.workflow import (
    WorkflowStep,
    LLMStep,
    ToolStep,
    DelegationStep,
    RoutingStep
)


@dataclass
class TraceScenario:
    """A complete trace scenario with workflow steps"""
    name: str
    description: str
    workflow_steps: List[WorkflowStep]
    expected_span_count: int
    expected_size_kb: int


# Scenario 1: Simple Query
# Pattern: User asks simple question, agent responds from context
SIMPLE_QUERY = TraceScenario(
    name="simple_query",
    description="Simple question answered from context (e.g., 'What's my flight status?')",
    workflow_steps=[
        LLMStep(
            name="primary_assistant_initial",
            tokens_in=50,
            tokens_out=100,
            latency_ms=500
        ),
        ToolStep(
            name="fetch_user_info",
            payload_kb=5,
            latency_ms=100
        ),
        LLMStep(
            name="primary_assistant_response",
            tokens_in=150,
            tokens_out=50,
            latency_ms=300
        ),
    ],
    expected_span_count=5,
    expected_size_kb=20
)


# Scenario 2: Single Service Search
# Pattern: User searches for one service (flights, hotels, cars)
SINGLE_SERVICE_SEARCH = TraceScenario(
    name="single_service_search",
    description="Search a single service (e.g., 'Find flights to NYC')",
    workflow_steps=[
        LLMStep(
            name="primary_assistant_parse",
            tokens_in=100,
            tokens_out=50,
            latency_ms=400
        ),
        ToolStep(
            name="search_flights",
            payload_kb=30,
            latency_ms=200
        ),
        LLMStep(
            name="primary_assistant_present",
            tokens_in=500,
            tokens_out=150,
            latency_ms=600
        ),
    ],
    expected_span_count=10,
    expected_size_kb=50
)


# Scenario 3: Delegated Booking
# Pattern: Primary delegates to specialist assistant for booking
DELEGATED_BOOKING = TraceScenario(
    name="delegated_booking",
    description="Delegated booking through specialist (e.g., 'Book a hotel in Zurich')",
    workflow_steps=[
        LLMStep(
            name="primary_assistant_delegate",
            tokens_in=100,
            tokens_out=50,
            latency_ms=400
        ),
        DelegationStep(
            name="hotel_assistant",
            assistant_type="hotel",
            sub_workflow=[
                LLMStep(
                    name="hotel_assistant_plan",
                    tokens_in=200,
                    tokens_out=100,
                    latency_ms=500
                ),
                ToolStep(
                    name="search_hotels",
                    payload_kb=40,
                    latency_ms=250
                ),
                LLMStep(
                    name="hotel_assistant_select",
                    tokens_in=600,
                    tokens_out=100,
                    latency_ms=600
                ),
                ToolStep(
                    name="book_hotel",
                    payload_kb=10,
                    latency_ms=150
                ),
            ]
        ),
        LLMStep(
            name="primary_assistant_confirm",
            tokens_in=300,
            tokens_out=100,
            latency_ms=400
        ),
    ],
    expected_span_count=40,
    expected_size_kb=150
)


# Scenario 4: Multi-Service Complex
# Pattern: Complex multi-step journey across multiple services
MULTI_SERVICE_COMPLEX = TraceScenario(
    name="multi_service_complex",
    description="Complex multi-service request (e.g., 'Plan trip: flight + hotel + car')",
    workflow_steps=[
        LLMStep(
            name="primary_assistant_plan",
            tokens_in=200,
            tokens_out=100,
            latency_ms=600
        ),
        # Flight booking
        DelegationStep(
            name="flight_assistant",
            assistant_type="flight",
            sub_workflow=[
                LLMStep(name="flight_assistant_search", tokens_in=150, tokens_out=50),
                ToolStep(name="search_flights", payload_kb=30, latency_ms=200),
                LLMStep(name="flight_assistant_book", tokens_in=400, tokens_out=50),
                ToolStep(name="book_flight", payload_kb=15, latency_ms=150),
            ]
        ),
        # Hotel booking
        DelegationStep(
            name="hotel_assistant",
            assistant_type="hotel",
            sub_workflow=[
                LLMStep(name="hotel_assistant_search", tokens_in=150, tokens_out=50),
                ToolStep(name="search_hotels", payload_kb=40, latency_ms=250),
                LLMStep(name="hotel_assistant_book", tokens_in=500, tokens_out=50),
                ToolStep(name="book_hotel", payload_kb=10, latency_ms=150),
            ]
        ),
        # Car rental
        DelegationStep(
            name="car_rental_assistant",
            assistant_type="car_rental",
            sub_workflow=[
                LLMStep(name="car_assistant_search", tokens_in=150, tokens_out=50),
                ToolStep(name="search_cars", payload_kb=25, latency_ms=200),
                LLMStep(name="car_assistant_book", tokens_in=400, tokens_out=50),
                ToolStep(name="book_car", payload_kb=8, latency_ms=150),
            ]
        ),
        # Final summary
        LLMStep(
            name="primary_assistant_summary",
            tokens_in=800,
            tokens_out=200,
            latency_ms=800
        ),
    ],
    expected_span_count=100,
    expected_size_kb=250
)


# Scenario registry
_SCENARIOS = {
    "simple_query": SIMPLE_QUERY,
    "single_service_search": SINGLE_SERVICE_SEARCH,
    "delegated_booking": DELEGATED_BOOKING,
    "multi_service_complex": MULTI_SERVICE_COMPLEX,
}


def get_scenario(name: str) -> TraceScenario:
    """Get a scenario by name

    Args:
        name: Scenario name

    Returns:
        TraceScenario instance

    Raises:
        KeyError: If scenario not found
    """
    return _SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List all available scenario names"""
    return list(_SCENARIOS.keys())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scenarios.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit scenario definitions**

```bash
git add src/scenarios.py tests/test_scenarios.py
git commit -m "feat: add built-in trace scenarios inspired by expedia patterns"
```

---

## Task 7: Workflow Execution with OTEL

**Files:**
- Modify: `src/workflow.py`
- Create: `tests/test_workflow_execution.py`

**Step 1: Write failing test for workflow execution**

Create `tests/test_workflow_execution.py`:
```python
import pytest
from unittest.mock import Mock, MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from src.workflow import LLMStep, ToolStep, DelegationStep
from src.platforms import ConsolePlatform


@pytest.mark.asyncio
async def test_llm_step_execution():
    """LLMStep creates proper OTEL span"""
    # Setup tracer
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer(__name__)

    platform = ConsolePlatform()

    step = LLMStep(
        name="test_llm",
        tokens_in=100,
        tokens_out=50,
        model="claude-sonnet-4"
    )

    # Execute should create span
    await step.execute(tracer, None, platform)
    # No exception = success


@pytest.mark.asyncio
async def test_tool_step_execution():
    """ToolStep creates proper OTEL span"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer(__name__)

    platform = ConsolePlatform()

    step = ToolStep(
        name="search_flights",
        payload_kb=30,
        latency_ms=100
    )

    await step.execute(tracer, None, platform)
    # No exception = success


@pytest.mark.asyncio
async def test_delegation_step_nesting():
    """DelegationStep creates nested spans"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer(__name__)

    platform = ConsolePlatform()

    step = DelegationStep(
        name="hotel_assistant",
        assistant_type="hotel",
        sub_workflow=[
            LLMStep(name="sub_llm", tokens_in=50, tokens_out=25),
            ToolStep(name="sub_tool", payload_kb=10)
        ]
    )

    await step.execute(tracer, None, platform)
    # No exception = success
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_execution.py -v`
Expected: FAIL with "execute() missing implementation"

**Step 3: Implement execute methods for workflow steps**

Modify `src/workflow.py`, add these implementations:
```python
# Add these imports at top
import asyncio
from typing import Optional
from opentelemetry import trace, context
from opentelemetry.trace import Span

from src.instrumentation import create_llm_span, create_tool_span, create_agent_span
from src.payloads import (
    generate_llm_messages,
    generate_flight_search_results,
    generate_hotel_search_results,
    generate_tool_result
)
from src.platforms import Platform


# Add to LLMStep class:
    async def execute(self, tracer: trace.Tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute LLM step and create span"""
        # Generate realistic messages
        messages = generate_llm_messages(
            user_prompt="User request placeholder",
            assistant_response="Assistant response placeholder",
            target_size_kb=self.tokens_out // 100  # Rough estimate
        )

        # Create span with platform-specific attributes
        create_llm_span(
            tracer=tracer,
            name=self.name,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model=self.model,
            messages=messages,
            platform=platform,
            latency_ms=self.latency_ms,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


# Add to ToolStep class:
    async def execute(self, tracer: trace.Tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute tool step and create span"""
        # Generate realistic tool result based on tool name
        if "flight" in self.name.lower():
            data = generate_flight_search_results(count=10, target_size_kb=self.payload_kb)
        elif "hotel" in self.name.lower():
            data = generate_hotel_search_results(count=5, target_size_kb=self.payload_kb)
        else:
            data = {"result": "x" * (self.payload_kb * 1024)}

        tool_result = generate_tool_result(
            tool_name=self.name,
            success=True,
            data=data,
            target_size_kb=self.payload_kb
        )

        # Create tool span
        create_tool_span(
            tracer=tracer,
            name=self.name,
            tool_result=tool_result,
            platform=platform,
            latency_ms=self.latency_ms
        )


# Add to DelegationStep class:
    async def execute(self, tracer: trace.Tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute delegation step with nested sub-workflow"""
        # Create agent span for this assistant
        with tracer.start_as_current_span(self.name) as span:
            # Set agent attributes
            tools = [step.name for step in self.sub_workflow if isinstance(step, ToolStep)]
            span.set_attribute("gen_ai.agent.tools", str(tools))
            span.set_attribute("agent.type", self.assistant_type)

            if platform:
                from src.instrumentation import set_platform_attributes
                set_platform_attributes(
                    span,
                    platform,
                    span_type="agent",
                    data={"agent_type": self.assistant_type, "tools": tools}
                )

            # Execute sub-workflow steps
            for step in self.sub_workflow:
                await step.execute(tracer, context.get_current(), platform)


# Add to RoutingStep class (stub for now):
    async def execute(self, tracer: trace.Tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute routing decision"""
        with tracer.start_as_current_span(f"{self.name}_routing") as span:
            span.set_attribute("routing.decision", self.decision)
            # Could execute one branch based on decision
            pass


# Add to ParallelStep class (stub for now):
    async def execute(self, tracer: trace.Tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute parallel steps concurrently"""
        with tracer.start_as_current_span(f"{self.name}_parallel") as span:
            # Execute all parallel steps concurrently
            tasks = [
                step.execute(tracer, context.get_current(), platform)
                for step in self.parallel_steps
            ]
            await asyncio.gather(*tasks)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_execution.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit workflow execution**

```bash
git add src/workflow.py tests/test_workflow_execution.py
git commit -m "feat: implement workflow step execution with OTEL span creation"
```

---

## Task 8: Metrics Collection

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write failing test for metrics**

Create `tests/test_metrics.py`:
```python
import pytest
import time
from src.metrics import MetricsCollector


def test_metrics_collector_initialization():
    """MetricsCollector initializes with empty state"""
    collector = MetricsCollector()
    assert collector.success_count == 0
    assert collector.failure_count == 0
    assert len(collector.latencies) == 0


def test_record_success():
    """Recording success increments count and tracks latency"""
    collector = MetricsCollector()
    collector.record_success(
        scenario_name="test",
        latency_ms=100.0,
        data_size_bytes=1024
    )
    assert collector.success_count == 1
    assert len(collector.latencies) == 1
    assert collector.latencies[0] == 100.0


def test_record_failure():
    """Recording failure increments failure count"""
    collector = MetricsCollector()
    collector.record_failure(scenario_name="test", error="Test error")
    assert collector.failure_count == 1


def test_metrics_report():
    """Report generates comprehensive statistics"""
    collector = MetricsCollector()
    collector.start_time = time.time() - 10  # 10 seconds ago

    # Record some metrics
    for i in range(100):
        collector.record_success("test", latency_ms=float(i), data_size_bytes=1024 * 150)

    report = collector.report()

    assert "throughput" in report
    assert "latency_p50" in report
    assert "latency_p95" in report
    assert "latency_p99" in report
    assert "total_data_gb" in report
    assert "success_rate" in report
    assert report["success_rate"] == 1.0


def test_per_scenario_breakdown():
    """Report includes per-scenario breakdown"""
    collector = MetricsCollector()
    collector.start_time = time.time() - 10

    collector.record_success("scenario_a", latency_ms=50, data_size_bytes=1024 * 20)
    collector.record_success("scenario_a", latency_ms=60, data_size_bytes=1024 * 25)
    collector.record_success("scenario_b", latency_ms=100, data_size_bytes=1024 * 100)

    report = collector.report()

    assert "per_scenario" in report
    assert "scenario_a" in report["per_scenario"]
    assert "scenario_b" in report["per_scenario"]
    assert report["per_scenario"]["scenario_a"]["count"] == 2
    assert report["per_scenario"]["scenario_b"]["count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with "No module named 'src.metrics'"

**Step 3: Implement metrics collection**

Create `src/metrics.py`:
```python
"""Metrics collection and reporting for scale tests"""

import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScenarioMetrics:
    """Metrics for a specific scenario"""
    count: int = 0
    latencies: List[float] = field(default_factory=list)
    data_sizes: List[int] = field(default_factory=list)
    failures: int = 0

    def add_success(self, latency_ms: float, data_size_bytes: int):
        """Record successful execution"""
        self.count += 1
        self.latencies.append(latency_ms)
        self.data_sizes.append(data_size_bytes)

    def add_failure(self):
        """Record failed execution"""
        self.failures += 1

    def summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.latencies:
            return {
                "count": self.count,
                "failures": self.failures,
                "avg_latency_ms": 0,
                "avg_data_kb": 0
            }

        return {
            "count": self.count,
            "failures": self.failures,
            "avg_latency_ms": np.mean(self.latencies),
            "p50_latency_ms": np.percentile(self.latencies, 50),
            "p95_latency_ms": np.percentile(self.latencies, 95),
            "avg_data_kb": np.mean(self.data_sizes) / 1024
        }


class MetricsCollector:
    """Collects and reports metrics during scale testing"""

    def __init__(self):
        self.start_time = time.time()
        self.success_count = 0
        self.failure_count = 0
        self.latencies: List[float] = []
        self.data_sizes: List[int] = []
        self.per_scenario: Dict[str, ScenarioMetrics] = defaultdict(ScenarioMetrics)
        self.errors: List[str] = []

    def record_success(
        self,
        scenario_name: str,
        latency_ms: float,
        data_size_bytes: int
    ):
        """Record a successful trace execution

        Args:
            scenario_name: Name of the scenario
            latency_ms: Execution latency in milliseconds
            data_size_bytes: Total data size in bytes
        """
        self.success_count += 1
        self.latencies.append(latency_ms)
        self.data_sizes.append(data_size_bytes)
        self.per_scenario[scenario_name].add_success(latency_ms, data_size_bytes)

    def record_failure(self, scenario_name: str, error: str):
        """Record a failed trace execution

        Args:
            scenario_name: Name of the scenario
            error: Error message
        """
        self.failure_count += 1
        self.per_scenario[scenario_name].add_failure()
        self.errors.append(f"{scenario_name}: {error}")

    def report(self) -> Dict:
        """Generate comprehensive metrics report

        Returns:
            Dictionary with all collected metrics
        """
        elapsed_time = time.time() - self.start_time
        total_requests = self.success_count + self.failure_count

        report = {
            "duration_seconds": elapsed_time,
            "total_requests": total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "throughput": self.success_count / elapsed_time if elapsed_time > 0 else 0,
        }

        # Latency statistics
        if self.latencies:
            report.update({
                "latency_p50": np.percentile(self.latencies, 50),
                "latency_p95": np.percentile(self.latencies, 95),
                "latency_p99": np.percentile(self.latencies, 99),
                "latency_min": min(self.latencies),
                "latency_max": max(self.latencies),
            })

        # Data volume statistics
        if self.data_sizes:
            total_bytes = sum(self.data_sizes)
            report.update({
                "total_data_gb": total_bytes / (1024 ** 3),
                "avg_data_per_request_kb": (total_bytes / len(self.data_sizes)) / 1024,
            })

        # Per-scenario breakdown
        report["per_scenario"] = {
            name: metrics.summary()
            for name, metrics in self.per_scenario.items()
        }

        return report

    def format_report(self, report: Optional[Dict] = None) -> str:
        """Format report as human-readable string

        Args:
            report: Report dict (or generate new one if None)

        Returns:
            Formatted report string
        """
        if report is None:
            report = self.report()

        lines = [
            "📊 Scale Test Results:",
            f"   Duration: {report['duration_seconds']:.1f}s",
            f"   Total requests: {report['total_requests']}",
            f"   Success rate: {report['success_rate']*100:.1f}%",
            f"   Throughput: {report['throughput']:.1f} req/s",
            "",
        ]

        if "latency_p50" in report:
            lines.extend([
                "⏱️  Latency:",
                f"   P50: {report['latency_p50']:.1f}ms",
                f"   P95: {report['latency_p95']:.1f}ms",
                f"   P99: {report['latency_p99']:.1f}ms",
                "",
            ])

        if "total_data_gb" in report:
            lines.extend([
                "📦 Data Volume:",
                f"   Total: {report['total_data_gb']:.2f} GB",
                f"   Avg per request: {report['avg_data_per_request_kb']:.1f} KB",
                "",
            ])

        if report.get("per_scenario"):
            lines.append("📈 Query Mix Breakdown:")
            for name, metrics in report["per_scenario"].items():
                if metrics["count"] > 0:
                    lines.append(
                        f"   {name}: {metrics['count']} traces, "
                        f"P50={metrics.get('p50_latency_ms', 0):.1f}ms, "
                        f"{metrics['avg_data_kb']:.1f}KB avg"
                    )

        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit metrics collection**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add comprehensive metrics collection and reporting"
```

---

## Task 9: Async Workload Executor

**Files:**
- Create: `src/executor.py`
- Create: `tests/test_executor.py`

**Step 1: Write failing test for executor**

Create `tests/test_executor.py`:
```python
import pytest
import asyncio
from src.executor import ScaleTestExecutor, TokenBucketRateLimiter
from src.scenarios import SIMPLE_QUERY


@pytest.mark.asyncio
async def test_rate_limiter():
    """TokenBucketRateLimiter enforces rate limit"""
    limiter = TokenBucketRateLimiter(rate=10)  # 10 per second

    # Should allow immediate requests
    start = asyncio.get_event_loop().time()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.3  # Should be nearly instant


@pytest.mark.asyncio
async def test_executor_initialization():
    """ScaleTestExecutor initializes with config"""
    config = {
        "concurrency": 5,
        "rate_limit": 10,
        "query_mix": {
            "simple_query": 1.0
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)
    assert executor.concurrency == 5
    assert executor.rate_limiter is not None


@pytest.mark.asyncio
async def test_executor_runs_for_duration():
    """Executor runs for specified duration"""
    config = {
        "concurrency": 2,
        "rate_limit": 100,
        "query_mix": {
            "simple_query": 1.0
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)

    # Run for 2 seconds
    start = asyncio.get_event_loop().time()
    metrics = await executor.run(duration_seconds=2)
    elapsed = asyncio.get_event_loop().time() - start

    # Should complete in approximately 2 seconds
    assert 1.5 <= elapsed <= 3.0
    assert metrics["total_requests"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL with "No module named 'src.executor'"

**Step 3: Implement async executor**

Create `src/executor.py`:
```python
"""Async workload executor for scale testing"""

import asyncio
import random
import time
from typing import Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from src.scenarios import TraceScenario, get_scenario
from src.platforms import get_platform, Platform
from src.metrics import MetricsCollector


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rate"""

    def __init__(self, rate: float, burst: Optional[float] = None):
        """Initialize rate limiter

        Args:
            rate: Requests per second
            burst: Burst capacity (defaults to rate)
        """
        self.rate = rate
        self.burst = burst or rate
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # Wait for next token
            wait_time = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0.0


class ScaleTestExecutor:
    """Executes scale test workload with async workers"""

    def __init__(self, config: Dict):
        """Initialize executor with configuration

        Args:
            config: Configuration dictionary with:
                - concurrency: Number of concurrent workers
                - rate_limit: Max requests per second (optional)
                - query_mix: Dict mapping scenario names to weights
                - platform_config: Platform configuration
                - ramp_up_seconds: Gradual ramp-up period (optional)
        """
        self.concurrency = config["concurrency"]
        self.rate_limit = config.get("rate_limit")
        self.query_mix = config["query_mix"]
        self.ramp_up_seconds = config.get("ramp_up_seconds", 0)

        # Build scenario pool with weights
        self.scenarios: List[TraceScenario] = []
        self.weights: List[float] = []
        for scenario_name, weight in self.query_mix.items():
            self.scenarios.append(get_scenario(scenario_name))
            self.weights.append(weight)

        # Setup rate limiter
        self.rate_limiter = None
        if self.rate_limit:
            self.rate_limiter = TokenBucketRateLimiter(rate=self.rate_limit)

        # Setup platform and OTEL
        self.platform = get_platform(config["platform_config"])
        self.tracer_provider = self._setup_tracer()
        self.tracer = self.tracer_provider.get_tracer(__name__)

        # Metrics
        self.metrics = MetricsCollector()

        # Control flags
        self.should_stop = False

    def _setup_tracer(self) -> TracerProvider:
        """Setup OpenTelemetry tracer with platform exporter"""
        provider = TracerProvider()

        # Setup exporter based on platform
        if self.platform.endpoint:
            exporter = OTLPSpanExporter(
                endpoint=self.platform.endpoint,
                headers=self.platform.get_headers()
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        return provider

    async def run(self, duration_seconds: float) -> Dict:
        """Run scale test for specified duration

        Args:
            duration_seconds: How long to run test

        Returns:
            Metrics report dictionary
        """
        print(f"Starting scale test: {self.concurrency} workers, {duration_seconds}s duration")

        # Schedule stop
        asyncio.create_task(self._stop_after(duration_seconds))

        # Start workers
        workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.concurrency)
        ]

        # Wait for all workers to complete
        await asyncio.gather(*workers)

        # Shutdown tracer provider to flush remaining spans
        self.tracer_provider.shutdown()

        return self.metrics.report()

    async def _stop_after(self, seconds: float):
        """Stop execution after specified time"""
        await asyncio.sleep(seconds)
        self.should_stop = True

    async def _worker(self, worker_id: int):
        """Worker coroutine that executes scenarios

        Args:
            worker_id: Unique worker identifier
        """
        # Gradual ramp-up: stagger worker starts
        if self.ramp_up_seconds > 0:
            delay = (worker_id / self.concurrency) * self.ramp_up_seconds
            await asyncio.sleep(delay)

        while not self.should_stop:
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Select scenario based on weights
            scenario = random.choices(self.scenarios, weights=self.weights)[0]

            # Execute scenario
            await self._execute_scenario(scenario)

    async def _execute_scenario(self, scenario: TraceScenario):
        """Execute a single scenario and collect metrics

        Args:
            scenario: Scenario to execute
        """
        start_time = time.time()
        data_size = 0

        try:
            # Create root span for trace (required for Braintrust)
            with self.tracer.start_as_current_span(f"trace_{scenario.name}") as root_span:
                root_span.set_attribute("scenario.name", scenario.name)
                root_span.set_attribute("scenario.expected_spans", scenario.expected_span_count)

                # Execute workflow steps
                for step in scenario.workflow_steps:
                    await step.execute(self.tracer, None, self.platform)

                # Estimate data size (rough approximation)
                data_size = scenario.expected_size_kb * 1024

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_success(
                scenario_name=scenario.name,
                latency_ms=latency_ms,
                data_size_bytes=data_size
            )

        except Exception as e:
            # Record failure
            self.metrics.record_failure(
                scenario_name=scenario.name,
                error=str(e)
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_executor.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit executor**

```bash
git add src/executor.py tests/test_executor.py
git commit -m "feat: add async workload executor with rate limiting and OTEL integration"
```

---

## Task 10: Main CLI Script

**Files:**
- Create: `scripts/run_scale_test.py`
- Create: `tests/__init__.py`

**Step 1: Create test init file**

Create `tests/__init__.py`:
```python
"""Test package"""
```

**Step 2: Create main CLI script**

Create `scripts/run_scale_test.py`:
```python
#!/usr/bin/env python3
"""Main CLI script for running scale tests"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.executor import ScaleTestExecutor


def load_config_from_env() -> dict:
    """Load configuration from environment variables

    Returns:
        Configuration dictionary
    """
    # Platform configuration
    platform = os.getenv("OTEL_PLATFORM", "console").lower()
    platform_config = {
        "platform": platform,
    }

    if platform == "braintrust":
        # Parse headers for API key and project
        headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        api_key = ""
        project_id = ""

        for part in headers.split(","):
            part = part.strip()
            if "Bearer" in part:
                api_key = part.split("Bearer")[1].strip()
            elif "x-bt-parent" in part:
                project_id = part.split("project_id:")[1].strip()

        platform_config.update({
            "api_key": api_key,
            "project_id": project_id
        })

    elif platform == "langsmith":
        headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        api_key = ""
        project_name = ""

        for part in headers.split(","):
            part = part.strip()
            if "x-api-key" in part:
                api_key = part.split("=")[1].strip()
            elif "Langsmith-Project" in part:
                project_name = part.split("=")[1].strip()

        platform_config.update({
            "api_key": api_key,
            "project_name": project_name
        })

    elif platform == "otlp":
        platform_config.update({
            "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "headers": {}
        })

    # Scale test parameters
    concurrency = int(os.getenv("SCALE_TEST_CONCURRENCY", "50"))
    duration = int(os.getenv("SCALE_TEST_DURATION", "300"))
    rate_limit = os.getenv("SCALE_TEST_RATE_LIMIT")
    ramp_up = int(os.getenv("SCALE_TEST_RAMP_UP", "0"))

    # Query mix
    query_mix = {
        "simple_query": float(os.getenv("SCALE_TEST_MIX_SIMPLE", "0.40")),
        "single_service_search": float(os.getenv("SCALE_TEST_MIX_SEARCH", "0.30")),
        "delegated_booking": float(os.getenv("SCALE_TEST_MIX_BOOKING", "0.20")),
        "multi_service_complex": float(os.getenv("SCALE_TEST_MIX_COMPLEX", "0.10")),
    }

    # Validate query mix sums to 1.0
    total = sum(query_mix.values())
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Query mix sums to {total}, normalizing to 1.0")
        query_mix = {k: v/total for k, v in query_mix.items()}

    config = {
        "concurrency": concurrency,
        "duration": duration,
        "rate_limit": float(rate_limit) if rate_limit else None,
        "ramp_up_seconds": ramp_up,
        "query_mix": query_mix,
        "platform_config": platform_config
    }

    return config


async def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config_from_env()

    # Print configuration
    print("Configuration:")
    print(f"  Platform: {config['platform_config']['platform']}")
    print(f"  Concurrency: {config['concurrency']} workers")
    print(f"  Duration: {config['duration']}s")
    if config['rate_limit']:
        print(f"  Rate limit: {config['rate_limit']} req/s")
    if config['ramp_up_seconds']:
        print(f"  Ramp-up: {config['ramp_up_seconds']}s")
    print(f"  Query mix: {config['query_mix']}")
    print()

    # Create and run executor
    executor = ScaleTestExecutor(config)
    report = await executor.run(duration_seconds=config["duration"])

    # Print results
    print("\n" + "="*60)
    print(executor.metrics.format_report(report))
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Make script executable**

Run: `chmod +x scripts/run_scale_test.py`

**Step 4: Test script with console output**

Run:
```bash
export OTEL_PLATFORM=console
export SCALE_TEST_CONCURRENCY=2
export SCALE_TEST_DURATION=3
python scripts/run_scale_test.py
```

Expected: Script runs for 3 seconds, prints configuration and results

**Step 5: Commit CLI script**

```bash
git add scripts/run_scale_test.py tests/__init__.py
git commit -m "feat: add main CLI script with environment variable configuration"
```

---

## Task 11: Documentation and Final Testing

**Files:**
- Modify: `README.md`
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:
```python
"""Integration tests for complete workflow"""

import pytest
import asyncio
from src.executor import ScaleTestExecutor
from src.scenarios import list_scenarios


@pytest.mark.asyncio
async def test_end_to_end_console_platform():
    """End-to-end test with console platform"""
    config = {
        "concurrency": 2,
        "rate_limit": 100,
        "query_mix": {
            "simple_query": 0.5,
            "single_service_search": 0.5
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)
    metrics = await executor.run(duration_seconds=2)

    assert metrics["total_requests"] > 0
    assert metrics["success_rate"] > 0.9  # Most should succeed
    assert "per_scenario" in metrics


@pytest.mark.asyncio
async def test_all_scenarios_executable():
    """Verify all scenarios can be executed"""
    for scenario_name in list_scenarios():
        config = {
            "concurrency": 1,
            "query_mix": {scenario_name: 1.0},
            "platform_config": {"platform": "console"}
        }

        executor = ScaleTestExecutor(config)
        metrics = await executor.run(duration_seconds=1)

        assert metrics["total_requests"] > 0
        assert scenario_name in metrics["per_scenario"]
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v -s`
Expected: PASS (both tests, may see console span output)

**Step 3: Update README with complete usage**

Modify `README.md`:
```markdown
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
# Install dependencies
pip install -r requirements.txt

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
python scripts/run_scale_test.py
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
pytest

# Run specific test file
pytest tests/test_scenarios.py -v

# Run with output
pytest -v -s
```

## Design

See `docs/plans/2025-11-27-ai-observability-scale-test-design.md` for complete design documentation.

## License

MIT
```

**Step 4: Run all tests**

Run: `pytest -v`
Expected: All tests pass

**Step 5: Commit final documentation**

```bash
git add README.md tests/test_integration.py
git commit -m "docs: update README with complete usage and add integration tests"
```

---

## Completion

All tasks completed! The framework is ready for use.

**Next steps:**
1. Test with real Braintrust/LangSmith credentials
2. Run longer duration tests (5-10 minutes)
3. Experiment with different query mixes
4. Monitor platform dashboards for trace accuracy

**To execute this plan:**
- Use `superpowers:executing-plans` in a new session
- Or use `superpowers:subagent-driven-development` in current session
