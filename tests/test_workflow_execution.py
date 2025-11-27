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
