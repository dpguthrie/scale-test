"""Workflow step definitions for trace generation"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from opentelemetry import trace, context
from opentelemetry.trace import Span

from src.instrumentation import create_llm_span, create_tool_span, create_agent_span, set_platform_attributes
from src.payloads import (
    generate_llm_messages,
    generate_flight_search_results,
    generate_hotel_search_results,
    generate_tool_result
)
from src.platforms import Platform


@dataclass
class WorkflowStep(ABC):
    """Base class for workflow steps that generate OTEL spans"""
    name: str

    @abstractmethod
    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute this workflow step and create appropriate OTEL spans

        Args:
            tracer: OpenTelemetry tracer instance
            parent_span_context: Parent span context for nesting
            platform: Platform for platform-specific attributes

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

    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute LLM step and create span"""
        # Generate realistic messages
        messages = generate_llm_messages(
            user_prompt="User request placeholder",
            assistant_response="Assistant response placeholder",
            target_size_kb=self.tokens_out // 100  # Rough estimate
        )

        # Create span with platform-specific attributes
        await create_llm_span(
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


@dataclass
class ToolStep(WorkflowStep):
    """Simulates a tool invocation with payload size and latency"""
    payload_kb: int
    latency_ms: int = 100
    tool_parameters: dict = field(default_factory=dict)

    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
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
        await create_tool_span(
            tracer=tracer,
            name=self.name,
            tool_result=tool_result,
            platform=platform,
            latency_ms=self.latency_ms
        )


@dataclass
class DelegationStep(WorkflowStep):
    """Simulates delegation to a sub-agent with nested workflow"""
    assistant_type: str
    sub_workflow: List[WorkflowStep]

    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute delegation step with nested sub-workflow"""
        # Create agent span for this assistant
        with tracer.start_as_current_span(self.name) as span:
            # Set agent attributes
            tools = [step.name for step in self.sub_workflow if isinstance(step, ToolStep)]
            span.set_attribute("gen_ai.agent.tools", str(tools))
            span.set_attribute("agent.type", self.assistant_type)

            if platform:
                set_platform_attributes(
                    span,
                    platform,
                    span_type="agent",
                    data={"agent_type": self.assistant_type, "tools": tools}
                )

            # Execute sub-workflow steps
            for step in self.sub_workflow:
                await step.execute(tracer, context.get_current(), platform)


@dataclass
class RoutingStep(WorkflowStep):
    """Simulates a routing decision point that branches workflow"""
    decision: str
    branches: dict = field(default_factory=dict)

    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute routing decision"""
        with tracer.start_as_current_span(f"{self.name}_routing") as span:
            span.set_attribute("routing.decision", self.decision)
            # Could execute one branch based on decision
            pass


@dataclass
class ParallelStep(WorkflowStep):
    """Simulates parallel execution of multiple tools"""
    parallel_steps: List[WorkflowStep]

    async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
        """Execute parallel steps concurrently"""
        with tracer.start_as_current_span(f"{self.name}_parallel") as span:
            # Execute all parallel steps concurrently
            tasks = [
                step.execute(tracer, context.get_current(), platform)
                for step in self.parallel_steps
            ]
            await asyncio.gather(*tasks)
