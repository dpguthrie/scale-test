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
