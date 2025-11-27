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

    async def execute(self, tracer, parent_span_context):
        """Execute LLM step and create span"""
        pass


@dataclass
class ToolStep(WorkflowStep):
    """Simulates a tool invocation with payload size and latency"""
    payload_kb: int
    latency_ms: int = 100
    tool_parameters: dict = field(default_factory=dict)

    async def execute(self, tracer, parent_span_context):
        """Execute tool step and create span"""
        pass


@dataclass
class DelegationStep(WorkflowStep):
    """Simulates delegation to a sub-agent with nested workflow"""
    assistant_type: str
    sub_workflow: List[WorkflowStep]

    async def execute(self, tracer, parent_span_context):
        """Execute delegation step with nested sub-workflow"""
        pass


@dataclass
class RoutingStep(WorkflowStep):
    """Simulates a routing decision point that branches workflow"""
    decision: str
    branches: dict = field(default_factory=dict)

    async def execute(self, tracer, parent_span_context):
        """Execute routing decision"""
        pass


@dataclass
class ParallelStep(WorkflowStep):
    """Simulates parallel execution of multiple tools"""
    parallel_steps: List[WorkflowStep]

    async def execute(self, tracer, parent_span_context):
        """Execute parallel steps concurrently"""
        pass
