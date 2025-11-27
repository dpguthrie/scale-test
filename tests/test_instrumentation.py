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
