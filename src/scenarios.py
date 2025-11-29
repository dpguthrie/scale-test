"""Pre-defined trace scenarios inspired by travel agent patterns"""

from dataclasses import dataclass
from typing import List

from src.workflow import DelegationStep, LLMStep, ToolStep, WorkflowStep


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
# Realistic: conversation history + system prompt + reasoning = 2-5K tokens
SIMPLE_QUERY = TraceScenario(
    name="simple_query",
    description="Simple question answered from context (e.g., 'What's my flight status?')",
    workflow_steps=[
        LLMStep(
            name="primary_assistant_initial",
            tokens_in=1500,  # System prompt + conversation history + user query
            tokens_out=500,  # Reasoning + response
            latency_ms=10,
        ),
        ToolStep(
            name="fetch_user_info",
            payload_kb=15,  # User profile + booking history
            latency_ms=10,
        ),
        LLMStep(
            name="primary_assistant_response",
            tokens_in=2500,  # Previous context + tool results + prompt
            tokens_out=300,  # Final response
            latency_ms=10,
        ),
    ],
    expected_span_count=5,
    expected_size_kb=35,  # ~4800 tokens × 4 bytes/token = 19KB + 15KB tools = 34KB
)


# Scenario 2: Single Service Search
# Pattern: User searches for one service (flights, hotels, cars)
# Realistic: large search results + formatting = 8-15K tokens
SINGLE_SERVICE_SEARCH = TraceScenario(
    name="single_service_search",
    description="Search a single service (e.g., 'Find flights to NYC')",
    workflow_steps=[
        # Understanding phase
        LLMStep(
            name="primary_assistant_parse",
            tokens_in=2000,  # System prompt + conversation history
            tokens_out=400,  # Extract search parameters + reasoning
            latency_ms=70,
        ),
        ToolStep(name="fetch_user_preferences", payload_kb=10, latency_ms=10),
        LLMStep(
            name="primary_assistant_refine_search",
            tokens_in=2500,
            tokens_out=300,
            latency_ms=60,
        ),
        # Search phase
        ToolStep(
            name="search_flights",
            payload_kb=120,  # 50-100 flight options with details
            latency_ms=40,
        ),
        ToolStep(name="get_price_alerts", payload_kb=30, latency_ms=20),
        # Analysis phase
        LLMStep(
            name="primary_assistant_analyze",
            tokens_in=6000,
            tokens_out=700,
            latency_ms=120,
        ),
        ToolStep(name="check_alternative_airports", payload_kb=60, latency_ms=30),
        LLMStep(
            name="primary_assistant_compare",
            tokens_in=7000,
            tokens_out=800,
            latency_ms=140,
        ),
        # Presentation phase
        LLMStep(
            name="primary_assistant_present",
            tokens_in=8000,  # All search results + formatting instructions
            tokens_out=1200,  # Formatted presentation with recommendations
            latency_ms=150,
        ),
    ],
    expected_span_count=12,  # ~10 workflow steps + root = 11-12 spans
    expected_size_kb=270,  # ~17800 tokens × 4 bytes/token = 71KB + 220KB tools = 291KB (adjusted)
)


# Scenario 3: Delegated Booking
# Pattern: Primary delegates to specialist assistant for booking
# Realistic: full context passing + specialist reasoning = 15-30K tokens
DELEGATED_BOOKING = TraceScenario(
    name="delegated_booking",
    description="Delegated booking through specialist (e.g., 'Book a hotel in Zurich')",
    workflow_steps=[
        # Initial understanding
        LLMStep(
            name="primary_assistant_understand",
            tokens_in=2000,
            tokens_out=400,
            latency_ms=80,
        ),
        ToolStep(name="fetch_user_context", payload_kb=15, latency_ms=10),
        # Delegation
        LLMStep(
            name="primary_assistant_delegate",
            tokens_in=2500,  # Full conversation context
            tokens_out=600,  # Delegation instructions + context summary
            latency_ms=90,
        ),
        DelegationStep(
            name="hotel_assistant",
            assistant_type="hotel",
            sub_workflow=[
                # Planning phase
                LLMStep(
                    name="hotel_assistant_plan",
                    tokens_in=3500,  # Specialist system prompt + delegated context
                    tokens_out=800,  # Search strategy + reasoning
                    latency_ms=100,
                ),
                ToolStep(name="get_location_info", payload_kb=30, latency_ms=20),
                LLMStep(
                    name="hotel_assistant_refine_search",
                    tokens_in=4000,
                    tokens_out=500,
                    latency_ms=90,
                ),
                # Search phase
                ToolStep(
                    name="search_hotels",
                    payload_kb=150,  # 30-50 hotels with full details
                    latency_ms=50,
                ),
                ToolStep(name="get_hotel_reviews", payload_kb=80, latency_ms=40),
                LLMStep(
                    name="hotel_assistant_analyze_reviews",
                    tokens_in=7000,
                    tokens_out=700,
                    latency_ms=140,
                ),
                # Selection phase
                ToolStep(name="check_availability", payload_kb=40, latency_ms=30),
                LLMStep(
                    name="hotel_assistant_select",
                    tokens_in=9000,  # All hotel options + selection criteria
                    tokens_out=1000,  # Analysis + selection + booking parameters
                    latency_ms=180,
                ),
                # Price validation
                ToolStep(name="get_price_comparison", payload_kb=50, latency_ms=35),
                LLMStep(
                    name="hotel_assistant_validate_price",
                    tokens_in=5000,
                    tokens_out=600,
                    latency_ms=110,
                ),
                # Booking phase
                LLMStep(
                    name="hotel_assistant_prepare_booking",
                    tokens_in=4000,
                    tokens_out=500,
                    latency_ms=90,
                ),
                ToolStep(
                    name="book_hotel",
                    payload_kb=25,  # Booking confirmation details
                    latency_ms=30,
                ),
                ToolStep(name="get_confirmation", payload_kb=20, latency_ms=20),
            ],
        ),
        # Confirmation
        LLMStep(
            name="primary_assistant_confirm",
            tokens_in=5000,  # Full conversation + booking results
            tokens_out=800,  # Confirmation message + summary
            latency_ms=100,
        ),
        ToolStep(name="send_confirmation_email", payload_kb=15, latency_ms=15),
    ],
    expected_span_count=25,  # ~20 workflow steps + wrappers = 22-25 spans
    expected_size_kb=470,  # ~40K tokens × 4 bytes/token = 160KB + 425KB tools = 585KB (adjusted lower for realistic estimate)
)


# Scenario 4: Multi-Service Complex
# Pattern: Complex multi-step journey across multiple services
# Realistic: multiple specialists + huge context = 50-100K tokens, 80-120 spans
MULTI_SERVICE_COMPLEX = TraceScenario(
    name="multi_service_complex",
    description="Complex multi-service request (e.g., 'Plan trip: flight + hotel + car')",
    workflow_steps=[
        # Initial understanding and context gathering
        LLMStep(
            name="primary_assistant_understand",
            tokens_in=2000,
            tokens_out=400,
            latency_ms=80,
        ),
        ToolStep(name="fetch_user_preferences", payload_kb=5, latency_ms=10),
        ToolStep(name="fetch_travel_history", payload_kb=5, latency_ms=15),
        LLMStep(
            name="primary_assistant_analyze_context",
            tokens_in=3500,
            tokens_out=600,
            latency_ms=100,
        ),
        # Trip planning
        LLMStep(
            name="primary_assistant_plan",
            tokens_in=4000,
            tokens_out=1200,
            latency_ms=150,
        ),
        ToolStep(name="check_travel_restrictions", payload_kb=5, latency_ms=20),
        LLMStep(
            name="primary_assistant_adjust_plan",
            tokens_in=5000,
            tokens_out=800,
            latency_ms=120,
        ),
        # Flight booking with multiple sub-agents
        DelegationStep(
            name="flight_assistant",
            assistant_type="flight",
            sub_workflow=[
                # Flight search phase
                LLMStep(
                    name="flight_assistant_parse_requirements",
                    tokens_in=3500,
                    tokens_out=500,
                    latency_ms=90,
                ),
                ToolStep(name="search_flights_outbound", payload_kb=40, latency_ms=60),
                ToolStep(name="search_flights_return", payload_kb=40, latency_ms=60),
                LLMStep(
                    name="flight_assistant_compare_options",
                    tokens_in=9000,
                    tokens_out=1000,
                    latency_ms=180,
                ),
                # Price optimization
                ToolStep(
                    name="check_flight_prices_cache", payload_kb=10, latency_ms=20
                ),
                LLMStep(
                    name="flight_assistant_price_analysis",
                    tokens_in=6000,
                    tokens_out=700,
                    latency_ms=130,
                ),
                ToolStep(name="get_alternative_airports", payload_kb=15, latency_ms=40),
                LLMStep(
                    name="flight_assistant_alternative_analysis",
                    tokens_in=7000,
                    tokens_out=800,
                    latency_ms=150,
                ),
                # Booking phase
                LLMStep(
                    name="flight_assistant_finalize_choice",
                    tokens_in=8000,
                    tokens_out=600,
                    latency_ms=140,
                ),
                ToolStep(name="check_flight_availability", payload_kb=5, latency_ms=30),
                LLMStep(
                    name="flight_assistant_prepare_booking",
                    tokens_in=4000,
                    tokens_out=500,
                    latency_ms=100,
                ),
                ToolStep(name="book_flight", payload_kb=8, latency_ms=40),
                ToolStep(name="get_booking_confirmation", payload_kb=5, latency_ms=25),
            ],
        ),
        # Intermediate coordination
        LLMStep(
            name="primary_assistant_update_context",
            tokens_in=6000,
            tokens_out=700,
            latency_ms=130,
        ),
        # Hotel booking with validation
        DelegationStep(
            name="hotel_assistant",
            assistant_type="hotel",
            sub_workflow=[
                # Hotel search phase
                LLMStep(
                    name="hotel_assistant_understand_needs",
                    tokens_in=4000,
                    tokens_out=600,
                    latency_ms=110,
                ),
                ToolStep(name="search_hotels_location", payload_kb=45, latency_ms=70),
                ToolStep(name="get_hotel_reviews", payload_kb=20, latency_ms=50),
                LLMStep(
                    name="hotel_assistant_filter_options",
                    tokens_in=10000,
                    tokens_out=900,
                    latency_ms=180,
                ),
                # Amenity checking
                ToolStep(name="check_hotel_amenities", payload_kb=12, latency_ms=35),
                LLMStep(
                    name="hotel_assistant_match_preferences",
                    tokens_in=8000,
                    tokens_out=700,
                    latency_ms=150,
                ),
                ToolStep(name="check_hotel_availability", payload_kb=8, latency_ms=30),
                # Price comparison
                ToolStep(name="get_competitor_prices", payload_kb=15, latency_ms=45),
                LLMStep(
                    name="hotel_assistant_price_analysis",
                    tokens_in=7000,
                    tokens_out=800,
                    latency_ms=140,
                ),
                # Booking phase
                LLMStep(
                    name="hotel_assistant_finalize_selection",
                    tokens_in=9000,
                    tokens_out=900,
                    latency_ms=170,
                ),
                ToolStep(name="validate_booking_details", payload_kb=5, latency_ms=20),
                ToolStep(name="book_hotel", payload_kb=6, latency_ms=35),
                ToolStep(name="get_hotel_confirmation", payload_kb=6, latency_ms=25),
            ],
        ),
        # Another coordination step
        LLMStep(
            name="primary_assistant_check_progress",
            tokens_in=8000,
            tokens_out=600,
            latency_ms=140,
        ),
        # Car rental with insurance options
        DelegationStep(
            name="car_rental_assistant",
            assistant_type="car_rental",
            sub_workflow=[
                # Car search phase
                LLMStep(
                    name="car_assistant_analyze_needs",
                    tokens_in=4000,
                    tokens_out=600,
                    latency_ms=110,
                ),
                ToolStep(name="search_cars_airport", payload_kb=25, latency_ms=50),
                ToolStep(name="search_cars_downtown", payload_kb=25, latency_ms=50),
                LLMStep(
                    name="car_assistant_compare_locations",
                    tokens_in=8000,
                    tokens_out=800,
                    latency_ms=160,
                ),
                # Insurance and extras
                ToolStep(name="get_insurance_options", payload_kb=8, latency_ms=30),
                LLMStep(
                    name="car_assistant_insurance_recommendation",
                    tokens_in=6000,
                    tokens_out=700,
                    latency_ms=130,
                ),
                ToolStep(name="check_car_availability", payload_kb=6, latency_ms=25),
                # Booking phase
                LLMStep(
                    name="car_assistant_prepare_booking",
                    tokens_in=7000,
                    tokens_out=800,
                    latency_ms=150,
                ),
                ToolStep(name="book_car", payload_kb=5, latency_ms=30),
                ToolStep(name="get_car_confirmation", payload_kb=5, latency_ms=20),
            ],
        ),
        # Final validation and summary
        LLMStep(
            name="primary_assistant_validate_itinerary",
            tokens_in=12000,
            tokens_out=1000,
            latency_ms=200,
        ),
        ToolStep(name="calculate_total_cost", payload_kb=3, latency_ms=10),
        LLMStep(
            name="primary_assistant_optimize_itinerary",
            tokens_in=10000,
            tokens_out=1200,
            latency_ms=200,
        ),
        ToolStep(name="check_schedule_conflicts", payload_kb=4, latency_ms=15),
        LLMStep(
            name="primary_assistant_final_summary",
            tokens_in=15000,
            tokens_out=2000,
            latency_ms=250,
        ),
        ToolStep(name="send_confirmation_email", payload_kb=6, latency_ms=20),
    ],
    expected_span_count=80,  # ~54 workflow steps + wrappers + root = 70-80 spans
    expected_size_kb=500,  # ~185K tokens × 4 bytes/token = 740KB + ~340KB tools = ~1.08MB scaled down to 500KB
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
