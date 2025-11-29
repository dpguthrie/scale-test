"""Realistic metadata generation for spans to enable filtering and analytics"""

import random
from typing import Dict, Any


# Realistic metadata pools for different attributes
ENVIRONMENTS = ["production", "staging", "development"]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]
USER_TIERS = ["free", "basic", "premium", "enterprise"]
FEATURE_FLAGS = {
    "streaming_enabled": [True, False],
    "cache_enabled": [True, False],
    "experimental_model": [True, False],
    "parallel_tools": [True, False],
    "advanced_routing": [True, False]
}
CUSTOMER_SEGMENTS = ["retail", "enterprise", "smb", "startup", "agency"]
API_VERSIONS = ["v1", "v2", "v3-beta"]
REQUEST_PRIORITIES = ["low", "normal", "high", "critical"]
DEPLOYMENT_VERSIONS = ["1.2.3", "1.2.4", "1.3.0", "2.0.0-rc1"]
AGENT_MODES = ["autonomous", "guided", "interactive", "batch"]
CACHE_STATUS = ["hit", "miss", "partial"]
ERROR_HANDLING_STRATEGIES = ["retry", "fallback", "fail-fast", "circuit-breaker"]


def generate_trace_metadata() -> Dict[str, Any]:
    """Generate metadata for root trace spans

    Returns realistic metadata that would be useful for filtering entire traces:
    - Environment (prod/staging/dev)
    - Region (AWS regions)
    - User tier (free/premium/enterprise)
    - Feature flags
    - Customer segment
    - API version
    """
    metadata = {
        "environment": random.choice(ENVIRONMENTS),
        "region": random.choice(REGIONS),
        "user_tier": random.choice(USER_TIERS),
        "customer_segment": random.choice(CUSTOMER_SEGMENTS),
        "api_version": random.choice(API_VERSIONS),
        "deployment_version": random.choice(DEPLOYMENT_VERSIONS),
        "request_priority": random.choice(REQUEST_PRIORITIES),
        "agent_mode": random.choice(AGENT_MODES),
    }

    # Add some random feature flags (not all enabled)
    num_flags = random.randint(2, 4)
    selected_flags = random.sample(list(FEATURE_FLAGS.keys()), num_flags)
    for flag in selected_flags:
        metadata[f"feature.{flag}"] = random.choice(FEATURE_FLAGS[flag])

    # Add request metadata
    metadata["request_id"] = f"req_{random.randint(100000, 999999)}"
    metadata["session_id"] = f"sess_{random.randint(10000, 99999)}"

    # Weighted: premium users get higher priority
    if metadata["user_tier"] == "enterprise":
        metadata["request_priority"] = random.choice(["high", "high", "critical", "normal"])
    elif metadata["user_tier"] == "free":
        metadata["request_priority"] = random.choice(["low", "low", "normal"])

    return metadata


def generate_llm_metadata(model: str) -> Dict[str, Any]:
    """Generate metadata for LLM spans

    Returns metadata specific to LLM calls:
    - Model provider details
    - Cache status
    - Token limits
    - Cost tracking
    """
    # Extract provider from model name (e.g., "claude-3" -> "anthropic")
    if "claude" in model.lower():
        provider = "anthropic"
    elif "gpt" in model.lower():
        provider = "openai"
    else:
        provider = "custom"

    metadata = {
        "llm.provider": provider,
        "llm.cache_status": random.choice(CACHE_STATUS),
        "llm.stream_enabled": random.choice([True, False]),
    }

    # Add cost tracking (realistic prices per 1K tokens)
    if "claude-3-opus" in model:
        metadata["llm.cost_per_1k_input"] = 0.015
        metadata["llm.cost_per_1k_output"] = 0.075
    elif "claude-3-sonnet" in model:
        metadata["llm.cost_per_1k_input"] = 0.003
        metadata["llm.cost_per_1k_output"] = 0.015
    elif "gpt-4" in model:
        metadata["llm.cost_per_1k_input"] = 0.03
        metadata["llm.cost_per_1k_output"] = 0.06
    else:
        metadata["llm.cost_per_1k_input"] = 0.001
        metadata["llm.cost_per_1k_output"] = 0.002

    # Retry information (some calls retry)
    if random.random() < 0.15:  # 15% of calls have retries
        metadata["llm.retry_count"] = random.randint(1, 3)
        metadata["llm.retry_reason"] = random.choice(["rate_limit", "timeout", "service_unavailable"])

    return metadata


def generate_tool_metadata(tool_name: str) -> Dict[str, Any]:
    """Generate metadata for tool execution spans

    Returns metadata specific to tool calls:
    - Tool version
    - Error handling strategy
    - Cache status
    - External API details
    """
    metadata = {
        "tool.version": random.choice(["1.0.0", "1.1.0", "2.0.0", "2.1.0-beta"]),
        "tool.execution_mode": random.choice(["sync", "async", "parallel"]),
        "tool.error_handling": random.choice(ERROR_HANDLING_STRATEGIES),
    }

    # Tool-specific metadata based on tool type
    if "flight" in tool_name.lower():
        # Flight-specific metadata
        metadata["flight.origin_airport"] = random.choice(["SFO", "JFK", "LAX", "ORD", "ATL", "DFW", "DEN"])
        metadata["flight.destination_airport"] = random.choice(["LHR", "CDG", "NRT", "SYD", "DXB", "FRA", "HKG"])
        metadata["flight.cabin_class"] = random.choice(["economy", "premium_economy", "business", "first"])
        metadata["flight.airline_alliance"] = random.choice(["star_alliance", "oneworld", "skyteam", "none"])
        metadata["flight.direct_only"] = random.choice([True, False])
        metadata["flight.search_results_count"] = random.randint(10, 100)
        metadata["flight.price_range_min"] = random.randint(200, 500)
        metadata["flight.price_range_max"] = random.randint(800, 3000)
        metadata["flight.amadeus_api_version"] = random.choice(["v1", "v2"])

    elif "hotel" in tool_name.lower():
        # Hotel-specific metadata
        metadata["hotel.city"] = random.choice(["New York", "London", "Paris", "Tokyo", "Dubai", "Singapore"])
        metadata["hotel.star_rating_filter"] = random.choice([3, 4, 5])
        metadata["hotel.room_type"] = random.choice(["standard", "deluxe", "suite", "executive"])
        metadata["hotel.amenities_required"] = random.choice([
            ["wifi", "parking"],
            ["pool", "gym"],
            ["breakfast", "spa"],
            ["business_center", "airport_shuttle"]
        ])
        metadata["hotel.booking_provider"] = random.choice(["expedia", "booking_com", "hotels_com", "direct"])
        metadata["hotel.loyalty_program"] = random.choice(["marriott_bonvoy", "hilton_honors", "ihg_rewards", "none"])
        metadata["hotel.guest_rating_min"] = random.choice([7.0, 8.0, 8.5, 9.0])
        metadata["hotel.search_results_count"] = random.randint(15, 75)

    elif "car" in tool_name.lower() or "rental" in tool_name.lower():
        # Car rental metadata
        metadata["car.pickup_location"] = random.choice(["airport", "downtown", "hotel"])
        metadata["car.vehicle_class"] = random.choice(["economy", "compact", "midsize", "suv", "luxury", "van"])
        metadata["car.transmission"] = random.choice(["automatic", "manual"])
        metadata["car.rental_company"] = random.choice(["hertz", "enterprise", "avis", "budget", "sixt"])
        metadata["car.insurance_included"] = random.choice([True, False])
        metadata["car.unlimited_mileage"] = random.choice([True, False])
        metadata["car.driver_age_check"] = random.choice([True, False])

    elif "user" in tool_name.lower() or "profile" in tool_name.lower():
        # User data fetching metadata
        metadata["user.data_source"] = random.choice(["primary_db", "cache", "crm_api", "auth_service"])
        metadata["user.pii_redacted"] = random.choice([True, False])
        metadata["user.preferences_loaded"] = random.choice([True, False])
        metadata["user.loyalty_tier"] = random.choice(["bronze", "silver", "gold", "platinum"])
        metadata["user.account_age_days"] = random.randint(30, 3650)
        metadata["user.previous_bookings_count"] = random.randint(0, 50)
        metadata["user.preferred_airlines"] = random.choice(["united,lufthansa", "delta,air_france", "american,ba", "none"])
        metadata["user.communication_preferences"] = random.choice(["email", "sms", "push", "all"])

    elif "book" in tool_name.lower() or "payment" in tool_name.lower():
        # Booking/payment metadata
        metadata["booking.payment_provider"] = random.choice(["stripe", "paypal", "braintree", "adyen"])
        metadata["booking.transaction_type"] = random.choice(["immediate", "hold", "split_payment", "installment"])
        metadata["booking.fraud_check_enabled"] = random.choice([True, False])
        metadata["booking.currency"] = random.choice(["USD", "EUR", "GBP", "JPY", "AUD"])
        metadata["booking.total_amount_cents"] = random.randint(50000, 500000)
        metadata["booking.confirmation_sent"] = random.choice([True, False])
        metadata["booking.cancellation_policy"] = random.choice(["flexible", "moderate", "strict", "non_refundable"])

    elif "search" in tool_name.lower():
        # General search metadata
        metadata["tool.search_provider"] = random.choice(["elasticsearch", "algolia", "custom"])
        metadata["tool.search_index"] = random.choice(["flights_v2", "hotels_v3", "products_v1"])
        metadata["tool.result_count"] = random.randint(5, 50)
        metadata["tool.cache_enabled"] = random.choice([True, False])
        if metadata["tool.cache_enabled"]:
            metadata["tool.cache_status"] = random.choice(CACHE_STATUS)

    elif "database" in tool_name.lower() or "db" in tool_name.lower():
        metadata["tool.database_type"] = random.choice(["postgres", "mysql", "mongodb", "redis"])
        metadata["tool.query_optimizer"] = random.choice([True, False])
        metadata["tool.connection_pool_size"] = random.choice([10, 25, 50, 100])

    # External API call metadata (if tool calls external service)
    if random.random() < 0.7:  # 70% of tools call external APIs
        metadata["tool.external_api_call"] = True
        metadata["tool.external_api_latency_ms"] = random.randint(50, 500)
        metadata["tool.external_api_status_code"] = random.choice([200, 200, 200, 200, 201, 429, 503])

        if metadata["tool.external_api_status_code"] >= 400:
            metadata["tool.retry_attempted"] = True

    return metadata


def generate_agent_metadata(agent_type: str) -> Dict[str, Any]:
    """Generate metadata for agent/delegation spans

    Returns metadata specific to agent orchestration:
    - Agent capabilities
    - Delegation strategy
    - Performance metrics
    """
    metadata = {
        "agent.type": agent_type,
        "agent.delegation_strategy": random.choice(["sequential", "parallel", "adaptive", "priority_based"]),
        "agent.max_iterations": random.choice([5, 10, 15, 20]),
        "agent.tool_selection_mode": random.choice(["manual", "automatic", "hybrid"]),
    }

    # Agent-specific capabilities
    if agent_type == "primary":
        metadata["agent.can_delegate"] = True
        metadata["agent.specialist_pool_size"] = random.randint(3, 8)
    elif agent_type in ["flight", "hotel", "booking"]:
        metadata["agent.specialist_domain"] = agent_type
        metadata["agent.confidence_threshold"] = random.choice([0.7, 0.8, 0.85, 0.9])

    # Performance tracking
    metadata["agent.memory_usage_mb"] = random.randint(50, 500)
    metadata["agent.context_window_utilized"] = random.uniform(0.3, 0.95)

    return metadata


def add_metadata_to_span(span, metadata: Dict[str, Any]):
    """Add metadata dictionary to a span as attributes

    Args:
        span: OpenTelemetry span
        metadata: Dictionary of metadata key-value pairs
    """
    for key, value in metadata.items():
        # Handle boolean, numeric, and string types
        if isinstance(value, bool):
            span.set_attribute(key, value)
        elif isinstance(value, (int, float)):
            span.set_attribute(key, value)
        elif isinstance(value, str):
            span.set_attribute(key, value)
        else:
            # Fallback to string representation
            span.set_attribute(key, str(value))
