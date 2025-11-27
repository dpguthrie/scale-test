"""OpenTelemetry instrumentation for creating realistic agent traces"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from opentelemetry import trace
from opentelemetry.trace import Span

from src.platforms import Platform, BraintrustPlatform, LangSmithPlatform


async def create_llm_span(
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
        await asyncio.sleep(latency_ms / 1000.0)

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


async def create_tool_span(
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
        await asyncio.sleep(latency_ms / 1000.0)

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
