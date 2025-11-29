"""OpenTelemetry instrumentation for creating realistic agent traces"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from opentelemetry import trace
from opentelemetry.trace import Span

from src.platforms import Platform, BraintrustPlatform, LangSmithPlatform
from src.metadata import (
    generate_llm_metadata,
    generate_tool_metadata,
    generate_agent_metadata
)
from src.span_attributes import set_metadata_attributes


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
            # Content can be string or array (Anthropic format)
            content = msg["content"]
            if isinstance(content, list):
                # Anthropic multi-part content - serialize as JSON
                span.set_attribute(f"gen_ai.prompt.{i}.content", json.dumps(content))
            else:
                span.set_attribute(f"gen_ai.prompt.{i}.content", content)

        # Also set completion attributes for output
        # Find assistant messages for output
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if assistant_messages:
            # Set single completion string (most recent assistant message)
            last_content = assistant_messages[-1]["content"]
            if isinstance(last_content, list):
                span.set_attribute("gen_ai.completion", json.dumps(last_content))
            else:
                span.set_attribute("gen_ai.completion", last_content)

            # Set indexed completion format for all assistant messages
            for i, msg in enumerate(assistant_messages):
                span.set_attribute(f"gen_ai.completion.{i}.role", "assistant")
                content = msg["content"]
                if isinstance(content, list):
                    span.set_attribute(f"gen_ai.completion.{i}.content", json.dumps(content))
                else:
                    span.set_attribute(f"gen_ai.completion.{i}.content", content)

        # Add realistic metadata for filtering/analytics
        llm_metadata = generate_llm_metadata(model)
        set_metadata_attributes(span, llm_metadata)

        # Platform-specific attributes
        if platform:
            from src.platforms import BraintrustPlatform, OTLPPlatform

            # Set Braintrust attributes directly for Braintrust or OTLP mode
            # (OTLP may forward to Braintrust)
            if isinstance(platform, (BraintrustPlatform, OTLPPlatform)):
                # Set span type for filtering
                span.set_attribute("braintrust.span_attributes.type", "llm")

                # Parse messages into input (user) and output (assistant)
                input_msgs = [msg for msg in messages if msg.get("role") in ("user", "system")]
                output_msgs = [msg for msg in messages if msg.get("role") == "assistant"]

                # Set as JSON that Braintrust will parse
                span.set_attribute("braintrust.input_json", json.dumps(input_msgs if input_msgs else messages, separators=(',', ':')))
                span.set_attribute("braintrust.output_json", json.dumps(output_msgs if output_msgs else messages[-1:], separators=(',', ':')))

            # Always set other platform attributes too (for LangSmith, etc.)
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

        # Set input/output for UI display
        # Input is the tool invocation
        tool_input = f"Execute tool: {name}"
        span.set_attribute("gen_ai.prompt", tool_input)

        # Output is the result
        span.set_attribute("gen_ai.completion", result_json)

        # Add realistic metadata for filtering/analytics
        tool_metadata = generate_tool_metadata(name)
        set_metadata_attributes(span, tool_metadata)

        # Platform-specific attributes
        if platform:
            from src.platforms import BraintrustPlatform, OTLPPlatform

            # Set Braintrust attributes directly for Braintrust or OTLP mode
            # (OTLP may forward to Braintrust)
            if isinstance(platform, (BraintrustPlatform, OTLPPlatform)):
                # Set span type for filtering
                span.set_attribute("braintrust.span_attributes.type", "tool")

                # Set input as plain text, output as JSON object
                span.set_attribute("braintrust.input", tool_input)
                span.set_attribute("braintrust.output_json", json.dumps(tool_result, separators=(',', ':')))

            # Always set other platform attributes too (for LangSmith, etc.)
            set_platform_attributes(
                span,
                platform,
                span_type="tool",
                data={"tool": name, "result": tool_result, "input": tool_input, "output": result_json}
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

        # Add realistic metadata for filtering/analytics
        agent_metadata = generate_agent_metadata(agent_type)
        set_metadata_attributes(span, agent_metadata)

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
    from src.platforms import BraintrustPlatform, LangSmithPlatform, OTLPPlatform

    # Add platform base attributes
    for key, value in platform.get_span_attributes().items():
        span.set_attribute(key, value)

    # Platform-specific handling
    if isinstance(platform, BraintrustPlatform):
        _set_braintrust_attributes(span, span_type, data)
    elif isinstance(platform, OTLPPlatform):
        # OTLP may forward to Braintrust, so set those attributes
        _set_braintrust_attributes(span, span_type, data)
        # Also set LangSmith attributes for multi-platform support
        _set_langsmith_attributes(span, span_type, data)
    elif isinstance(platform, LangSmithPlatform):
        _set_langsmith_attributes(span, span_type, data)


def _set_braintrust_attributes(span: Span, span_type: str, data: Dict[str, Any]):
    """Set Braintrust-specific attributes using braintrust.* namespace

    Args:
        span: OpenTelemetry span
        span_type: Type of span
        data: Span data
    """
    # CRITICAL: Set span type for thread view filtering
    span.set_attribute("braintrust.span_attributes.type", span_type)

    if span_type == "llm" and "messages" in data:
        # Use braintrust namespace for direct field mapping
        # Input: all user/system messages
        input_msgs = [msg for msg in data["messages"] if msg.get("role") in ("user", "system")]
        # Output: assistant messages
        output_msgs = [msg for msg in data["messages"] if msg.get("role") == "assistant"]

        # Braintrust parses attributes with _json suffix into actual objects/arrays
        # Use compact JSON without extra whitespace
        span.set_attribute("braintrust.input_json", json.dumps(input_msgs if input_msgs else data["messages"], separators=(',', ':')))
        span.set_attribute("braintrust.output_json", json.dumps(output_msgs if output_msgs else data["messages"][-1:], separators=(',', ':')))
        span.set_attribute("braintrust.metadata", json.dumps({
            "model": data.get("model", "unknown")
        }, separators=(',', ':')))

    elif span_type == "tool":
        # For tools, use input/output from data
        # Tool results should be JSON objects, use _json suffix for parsing
        if "input" in data:
            span.set_attribute("braintrust.input", data["input"])
        if "output" in data:
            # Output is the full tool result dict - serialize it for Braintrust to parse
            span.set_attribute("braintrust.output_json", json.dumps(data["output"], separators=(',', ':')))


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

    # Set input/output for display in LangSmith UI
    if span_type == "llm" and "messages" in data:
        # For LLM spans, use messages for input/output
        input_msgs = [msg for msg in data["messages"] if msg.get("role") in ("user", "system")]
        output_msgs = [msg for msg in data["messages"] if msg.get("role") == "assistant"]

        if input_msgs:
            span.set_attribute("input.value", json.dumps(input_msgs))
        if output_msgs:
            span.set_attribute("output.value", json.dumps(output_msgs))

    elif span_type == "tool":
        # For tool spans, use input/output from data
        if "input" in data:
            span.set_attribute("input.value", data["input"])
        if "output" in data:
            span.set_attribute("output.value", data["output"])
