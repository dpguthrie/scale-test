"""Workflow step definitions for trace generation"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from opentelemetry import trace, context
from opentelemetry.trace import Span

from src.instrumentation import create_llm_span, create_tool_span, create_agent_span, set_platform_attributes
from src.payloads import (
    generate_anthropic_messages,
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
        # Get scenario name from parent context (if available)
        scenario_name = getattr(self, '_scenario_name', 'unknown')

        # Generate realistic messages with Anthropic format
        # This returns only assistant messages
        assistant_messages = generate_anthropic_messages(
            scenario_name=scenario_name,
            span_name=self.name,
            target_tokens_out=self.tokens_out
        )

        # Build full conversation with user message first
        from src.payloads import generate_user_query
        user_query = generate_user_query(scenario_name)

        # Full conversation: user message + assistant response
        messages = [
            {"role": "user", "content": user_query}
        ] + assistant_messages

        # Collect messages in accumulator if available
        messages_accumulator = getattr(self, '_messages_accumulator', None)
        if messages_accumulator is not None:
            messages_accumulator.extend(assistant_messages)

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
        # Import at function level to avoid circular dependency
        from src.payloads import generate_agent_plan, generate_agent_reflection
        import json
        import random

        # Create agent span for this assistant
        with tracer.start_as_current_span(self.name) as span:
            # Set agent attributes
            tools = [step.name for step in self.sub_workflow if isinstance(step, ToolStep)]
            span.set_attribute("gen_ai.agent.tools", json.dumps(tools))
            span.set_attribute("agent.type", self.assistant_type)

            # Add planning structure at start
            plan = generate_agent_plan(self.assistant_type, tools)
            span.set_attribute("agent.plan", json.dumps(plan))

            if platform:
                set_platform_attributes(
                    span,
                    platform,
                    span_type="agent",
                    data={"agent_type": self.assistant_type, "tools": tools}
                )

            # Execute sub-workflow steps
            # Propagate messages_accumulator to sub-workflow steps
            messages_accumulator = getattr(self, '_messages_accumulator', None)
            for step in self.sub_workflow:
                if messages_accumulator is not None:
                    step._messages_accumulator = messages_accumulator
                await step.execute(tracer, context.get_current(), platform)

            # Add reflection structure at end
            results_count = len([s for s in self.sub_workflow if isinstance(s, ToolStep)])
            filters_applied = [f"filter_{i}" for i in range(random.randint(2, 5))]
            reflection = generate_agent_reflection(
                self.assistant_type,
                success=True,
                results_count=results_count,
                filters_applied=filters_applied
            )
            span.set_attribute("agent.reflection", json.dumps(reflection))


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
