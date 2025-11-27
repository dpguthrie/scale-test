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
