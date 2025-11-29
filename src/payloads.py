"""Realistic payload generation for trace data"""

import json
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

# Initialize Faker for generating realistic text
fake = Faker()


def generate_user_query(scenario_name: str) -> str:
    """Generate a realistic user query based on scenario type

    Args:
        scenario_name: Name of the scenario

    Returns:
        Realistic user query string
    """
    if "simple" in scenario_name.lower():
        queries = [
            "What's the weather like in San Francisco?",
            "Can you help me find a good restaurant nearby?",
            "What time is it in Tokyo right now?",
            "Tell me about the latest tech news",
            "How do I convert USD to EUR?"
        ]
    elif "search" in scenario_name.lower():
        origin = random.choice(AIRPORTS)
        destination = random.choice([a for a in AIRPORTS if a != origin])
        queries = [
            f"I need flights from {origin} to {destination} next week",
            f"Show me hotels in {fake.city()} for 3 nights",
            f"Find me a business class ticket from {origin} to {destination}",
            f"I'm looking for luxury hotels in {fake.city()}",
            f"Search for weekend flights from {origin} to {destination}"
        ]
    elif "booking" in scenario_name.lower() or "delegated" in scenario_name.lower():
        origin = random.choice(AIRPORTS)
        destination = random.choice([a for a in AIRPORTS if a != origin])
        city = fake.city()
        queries = [
            f"Book a trip from {origin} to {destination} with hotel for 5 days",
            f"I need a complete package: flight to {city} and accommodation",
            f"Plan my business trip to {destination} including flight and hotel",
            f"Book round-trip from {origin} to {destination} with 3-star hotel",
            f"Arrange travel to {city} with mid-range hotel for a week"
        ]
    elif "complex" in scenario_name.lower() or "multi" in scenario_name.lower():
        origin = random.choice(AIRPORTS)
        destination = random.choice([a for a in AIRPORTS if a != origin])
        city = fake.city()
        queries = [
            f"Plan a 2-week vacation from {origin} to {destination} with multiple hotels and car rental",
            f"I need flights from {origin} to {destination}, hotels in 2 cities, and local transportation",
            f"Organize a complex trip: {origin} to {destination}, stopover in {city}, hotels, and activities",
            f"Book multi-city itinerary with flights, accommodations, and ground transport",
            f"Create a detailed travel plan for {origin} to {destination} with all services"
        ]
    else:
        queries = [
            f"Help me with {scenario_name}",
            f"I need assistance with {scenario_name}",
            f"Can you handle {scenario_name}?"
        ]

    return random.choice(queries)

AIRPORTS = [
    "JFK",
    "LAX",
    "ORD",
    "SFO",
    "BOS",
    "DFW",
    "ATL",
    "MIA",
    "LHR",
    "CDG",
    "FRA",
    "AMS",
    "ZRH",
    "BCN",
    "FCO",
]
AIRLINES = ["AA", "UA", "DL", "BA", "LH", "AF", "KL", "LX"]
HOTEL_NAMES = [
    "Grand Hotel",
    "City Inn",
    "Plaza Hotel",
    "Resort & Spa",
    "Boutique Hotel",
    "Business Hotel",
    "Airport Hotel",
]
LOCATIONS = [
    "New York",
    "Los Angeles",
    "Chicago",
    "San Francisco",
    "London",
    "Paris",
    "Frankfurt",
    "Zurich",
    "Barcelona",
]
PRICE_TIERS = ["Midscale", "Upper Midscale", "Upscale", "Luxury"]


def generate_flight_search_results(
    count: int = 20, target_size_kb: int = 30
) -> List[Dict]:
    """Generate realistic flight search results

    Args:
        count: Number of flight results to generate
        target_size_kb: Target size in KB for the result set

    Returns:
        List of flight result dictionaries
    """
    flights = []
    base_time = datetime.now()

    for i in range(count):
        departure_time = base_time + timedelta(days=random.randint(1, 30))
        arrival_time = departure_time + timedelta(hours=random.randint(2, 12))

        flight = {
            "flight_id": random.randint(1000, 9999),
            "flight_no": f"{random.choice(AIRLINES)}{random.randint(100, 999)}",
            "departure_airport": random.choice(AIRPORTS),
            "arrival_airport": random.choice(AIRPORTS),
            "scheduled_departure": departure_time.isoformat(),
            "scheduled_arrival": arrival_time.isoformat(),
            "fare_conditions": random.choice(["Economy", "Business", "First"]),
            "price": random.randint(200, 2000),
            "available_seats": random.randint(0, 50),
        }
        flights.append(flight)

    # Pad to reach target size
    current_size = estimate_json_size_kb(flights)
    if current_size < target_size_kb:
        padding_per_flight = int((target_size_kb - current_size) * 1024 // count)
        for flight in flights:
            flight["details"] = "x" * padding_per_flight

    return flights


def generate_hotel_search_results(
    count: int = 10, target_size_kb: int = 40
) -> List[Dict]:
    """Generate realistic hotel search results

    Args:
        count: Number of hotel results
        target_size_kb: Target size in KB

    Returns:
        List of hotel result dictionaries
    """
    hotels = []

    for i in range(count):
        hotel = {
            "id": random.randint(1000, 9999),
            "name": f"{random.choice(HOTEL_NAMES)} {random.choice(LOCATIONS)}",
            "location": random.choice(LOCATIONS),
            "price_tier": random.choice(PRICE_TIERS),
            "price_per_night": random.randint(100, 800),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "amenities": random.sample(
                ["WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Parking"],
                k=random.randint(3, 7),
            ),
            "available_rooms": random.randint(0, 20),
        }
        hotels.append(hotel)

    # Pad to target size
    current_size = estimate_json_size_kb(hotels)
    if current_size < target_size_kb:
        padding_per_hotel = int((target_size_kb - current_size) * 1024 // count)
        for hotel in hotels:
            hotel["description"] = "x" * padding_per_hotel

    return hotels


def generate_anthropic_messages(
    scenario_name: str,
    span_name: str,
    target_tokens_out: int
) -> List[Dict[str, Any]]:
    """Generate messages with Anthropic multi-part content format

    Args:
        scenario_name: Scenario name (simple_query, single_service_search, etc.)
        span_name: Span name to determine type (parse, refine, analyze, etc.)
        target_tokens_out: Total target tokens for assistant message

    Returns:
        List of message dictionaries with content arrays
    """
    # User message is always simple text
    user_content = f"Help me with {scenario_name.replace('_', ' ')}"

    # Determine span type from name
    span_type = "default"
    if "parse" in span_name.lower() or "understand" in span_name.lower():
        span_type = "parse"
    elif "refine" in span_name.lower():
        span_type = "refine"
    elif "search" in span_name.lower():
        span_type = "search"
    elif "analyze" in span_name.lower():
        span_type = "analyze"
    elif "compare" in span_name.lower():
        span_type = "compare"
    elif "present" in span_name.lower():
        span_type = "present"
    elif "plan" in span_name.lower():
        span_type = "plan"
    elif "delegate" in span_name.lower():
        span_type = "delegate"

    # Determine content structure based on scenario and span type
    content_blocks = []

    # Simple queries: just text
    if "simple" in scenario_name:
        content_blocks.append(generate_text_block("summary", target_tokens_out))

    # Service searches: thinking + tool_use + text
    elif "search" in scenario_name or "service" in scenario_name:
        if span_type in ("parse", "refine"):
            # Thinking + tool_use + text
            thinking_tokens = int(target_tokens_out * 0.3)
            tool_tokens = int(target_tokens_out * 0.1)
            text_tokens = target_tokens_out - thinking_tokens - tool_tokens

            content_blocks.append(generate_thinking_block(span_type, thinking_tokens))
            content_blocks.append(generate_tool_use_block(span_name, scenario_name))
            content_blocks.append(generate_text_block("results", text_tokens))
        else:
            # Thinking + text (analysis/comparison)
            thinking_tokens = int(target_tokens_out * 0.4)
            text_tokens = target_tokens_out - thinking_tokens

            content_blocks.append(generate_thinking_block(span_type, thinking_tokens))
            content_blocks.append(generate_text_block("analysis", text_tokens))

    # Booking scenarios: thinking + multiple tool_uses + text
    elif "booking" in scenario_name:
        thinking_tokens = int(target_tokens_out * 0.25)
        tool_tokens = int(target_tokens_out * 0.15)
        text_tokens = target_tokens_out - thinking_tokens - tool_tokens

        content_blocks.append(generate_thinking_block(span_type, thinking_tokens))

        # Add 1-2 tool calls for booking steps
        if "specialist" in span_name or "assistant" in span_name:
            content_blocks.append(generate_tool_use_block(span_name, scenario_name))

        content_blocks.append(generate_text_block("recommendation", text_tokens))

    # Complex multi-service: extensive thinking + multiple tools + text
    elif "complex" in scenario_name or "multi" in scenario_name:
        thinking_tokens = int(target_tokens_out * 0.35)
        tool_tokens = int(target_tokens_out * 0.15)
        text_tokens = target_tokens_out - thinking_tokens - tool_tokens

        content_blocks.append(generate_thinking_block(span_type, thinking_tokens))

        # Add 2-3 tool calls for complex coordination
        if "check" in span_name or "search" in span_name or "get" in span_name:
            content_blocks.append(generate_tool_use_block(span_name, scenario_name))

        if "assistant" in span_name:
            content_blocks.append(generate_tool_use_block(span_name.replace("assistant", "coordinator"), scenario_name))

        content_blocks.append(generate_text_block("results", text_tokens))

    else:
        # Default: thinking + text
        thinking_tokens = int(target_tokens_out * 0.3)
        text_tokens = target_tokens_out - thinking_tokens

        content_blocks.append(generate_thinking_block("default", thinking_tokens))
        content_blocks.append(generate_text_block("summary", text_tokens))

    # Only return assistant message - user message is added at root level
    return [
        {"role": "assistant", "content": content_blocks}
    ]


def generate_llm_messages(
    user_prompt: str, assistant_response: str, target_size_kb: int = 10
) -> List[Dict[str, str]]:
    """DEPRECATED: Use generate_anthropic_messages() instead

    Generate LLM message array in chat format

    Args:
        user_prompt: User's input message
        assistant_response: Assistant's response
        target_size_kb: Target total size in KB

    Returns:
        List of message dictionaries with role and content
    """
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    # Pad assistant response to reach target size
    current_size = estimate_json_size_kb(messages)
    if current_size < target_size_kb:
        padding = "x" * int((target_size_kb - current_size) * 1024)
        messages[-1]["content"] += f"\n\nAdditional context: {padding}"

    return messages


def generate_search_metadata(tool_name: str, result_count: int) -> Dict[str, Any]:
    """Generate search metadata for tool results

    Args:
        tool_name: Name of the tool
        result_count: Number of results returned

    Returns:
        Search metadata dictionary
    """
    total_results = result_count + random.randint(10, 50)

    return {
        "query_time_ms": random.randint(50, 300),
        "cache_hit": random.choice([True, False]),
        "total_results": total_results,
        "filtered_to": result_count,
        "data_sources": random.sample(
            ["gds_amadeus", "gds_sabre", "expedia_api", "booking_com", "internal_db"],
            k=random.randint(1, 3)
        ),
        "pricing_currency": "USD"
    }


def generate_execution_context(tool_name: str) -> Dict[str, Any]:
    """Generate execution context for tool calls

    Args:
        tool_name: Name of the tool

    Returns:
        Execution context dictionary
    """
    return {
        "request_id": f"req_{generate_random_id(16)}",
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": random.randint(50, 500),
        "retry_count": random.choice([0, 0, 0, 1])  # Mostly no retries
    }


def generate_debug_info(tool_name: str, result_data: Any) -> Dict[str, Any]:
    """Generate full diagnostic context for tool execution

    Args:
        tool_name: Name of the tool
        result_data: The result data to generate debug info for

    Returns:
        Debug info dictionary with SQL, cache, upstream calls, timing
    """
    # SQL query based on tool type
    sql_queries = {
        "search_flights": "SELECT * FROM flights WHERE origin = $1 AND destination = $2 AND departure_date >= $3",
        "search_hotels": "SELECT * FROM hotels WHERE city = $1 AND check_in >= $2 AND check_out <= $3",
        "fetch_user": "SELECT * FROM users WHERE user_id = $1",
        "get_price": "SELECT * FROM price_history WHERE service_type = $1 AND service_id = $2"
    }

    base_query = sql_queries.get(tool_name, "SELECT * FROM data WHERE id = $1")

    # Generate realistic execution plan
    execution_plan = {
        "scan_type": random.choice(["index_scan", "bitmap_scan", "seq_scan"]),
        "index_used": f"idx_{tool_name.split('_')[0]}_{random.choice(['primary', 'composite', 'covering'])}",
        "rows_examined": random.randint(100, 5000),
        "rows_returned": random.randint(10, 100)
    }

    # Cache information
    cache_keys = [
        f"{tool_name}:{generate_random_id(8)}",
        f"result:{generate_random_id(8)}"
    ]
    cache_hit = random.choice([True, False])

    # Upstream API calls
    services = {
        "search_flights": [("gds_amadeus", "/v1/shopping/flight-offers"), ("gds_sabre", "/v2/search/flights")],
        "search_hotels": [("booking_api", "/v1/hotels/search"), ("expedia_api", "/v2/properties")],
        "fetch_user": [("user_service", "/v1/users/:id"), ("preference_service", "/v1/preferences")],
    }

    upstream_calls = []
    for service, endpoint in services.get(tool_name, [("data_service", "/v1/query")]):
        upstream_calls.append({
            "service": service,
            "endpoint": endpoint,
            "duration_ms": random.randint(30, 200),
            "status_code": 200,
            "retry_count": 0
        })

    # Performance breakdown
    total_time = random.randint(100, 500)
    timing_breakdown = {
        "parse_request_ms": random.randint(1, 5),
        "validate_params_ms": random.randint(1, 3),
        "query_database_ms": int(total_time * 0.6),
        "fetch_pricing_ms": int(total_time * 0.25),
        "format_response_ms": int(total_time * 0.1)
    }

    # Data lineage
    data_lineage = {
        "source_systems": random.sample(
            ["amadeus_gds", "sabre_gds", "pricing_engine", "user_db", "cache_layer"],
            k=random.randint(2, 4)
        ),
        "last_updated": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
        "data_freshness_seconds": random.randint(60, 1800)
    }

    return {
        "sql_query": base_query,
        "execution_plan": execution_plan,
        "cache_keys": cache_keys,
        "cache_hit": cache_hit,
        "cache_write_time_ms": random.randint(5, 20) if not cache_hit else 0,
        "upstream_calls": upstream_calls,
        "timing_breakdown": timing_breakdown,
        "data_lineage": data_lineage
    }


def generate_agent_plan(agent_type: str, available_tools: List[str]) -> Dict[str, Any]:
    """Generate agent planning structure

    Args:
        agent_type: Type of agent (hotel, flight, primary, etc.)
        available_tools: List of tools available to agent

    Returns:
        Agent plan dictionary
    """
    plans = {
        "hotel": {
            "goal": "Find hotels matching user preferences and budget constraints",
            "steps": [
                "Parse user preferences from context",
                "Apply budget and amenity filters",
                "Search hotels with geographic constraints",
                "Rank results by preference match score",
                "Format top recommendations with details"
            ]
        },
        "flight": {
            "goal": "Search and recommend flights based on travel requirements",
            "steps": [
                "Parse travel dates and destinations",
                "Apply cabin class and airline preferences",
                "Search available flights",
                "Compare fares and connection times",
                "Recommend best value options"
            ]
        },
        "primary": {
            "goal": "Coordinate multi-service booking workflow",
            "steps": [
                "Understand user request and requirements",
                "Identify required services (flights, hotels, etc.)",
                "Delegate to specialist agents",
                "Coordinate timing and availability",
                "Synthesize recommendations"
            ]
        }
    }

    plan_template = plans.get(agent_type, {
        "goal": f"Execute {agent_type} workflow",
        "steps": ["Parse request", "Execute actions", "Return results"]
    })

    return {
        "goal": plan_template["goal"],
        "steps": plan_template["steps"],
        "estimated_steps": len(plan_template["steps"]),
        "parallel_capable": agent_type != "primary",
        "available_tools": available_tools
    }


def generate_agent_reflection(
    agent_type: str,
    success: bool,
    results_count: int,
    filters_applied: List[str]
) -> Dict[str, Any]:
    """Generate agent reflection after execution

    Args:
        agent_type: Type of agent
        success: Whether execution succeeded
        results_count: Number of results returned
        filters_applied: List of filters that were applied

    Returns:
        Agent reflection dictionary
    """
    quality_score = random.uniform(0.75, 0.95) if success else random.uniform(0.3, 0.6)

    suggestions = [
        "Consider expanding search radius for more options",
        "User prefers boutique properties based on history",
        "Price sensitivity detected - prioritize value options",
        "Loyalty program membership could provide benefits"
    ]

    return {
        "success": success,
        "goal_achieved": success and results_count > 0,
        "quality_score": round(quality_score, 2),
        "results_count": results_count,
        "filters_applied": filters_applied,
        "execution_notes": f"Successfully processed {agent_type} request with {len(filters_applied)} filters",
        "suggestions": random.sample(suggestions, k=random.randint(1, 2))
    }


def generate_tool_result(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    target_size_kb: int = 5,
    tool_call_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a tool execution result with MCP format

    Args:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        data: Tool result data
        target_size_kb: Target size in KB
        tool_call_id: Optional tool call ID to link to tool_use block

    Returns:
        Tool result dictionary in MCP format
    """
    # Handle different data formats and normalize to dict
    # Check type of data directly before any defaulting
    if data is None:
        result_data = {}
        result_count = 0
        normalized_data = {}
    elif isinstance(data, list):
        # Data is a list of results (flights or hotels) - wrap in appropriate key
        result_data = data
        result_count = len(data)
        if "flight" in tool_name.lower():
            normalized_data = {"flights": data}
        elif "hotel" in tool_name.lower():
            normalized_data = {"hotels": data}
        else:
            normalized_data = {"results": data}
    elif isinstance(data, dict):
        # Data is already a dict
        result_data = data
        result_count = len(data.get("flights", data.get("hotels", [])))
        normalized_data = data
    else:
        # Unexpected type, default to empty
        result_data = {}
        result_count = 0
        normalized_data = {}

    # Build MCP-style result
    result = {
        "tool_call_id": tool_call_id or f"call_{generate_random_id(16)}",
        "function": {
            "name": tool_name,
            "arguments": generate_tool_parameters(tool_name, "default")
        },
        "result": {
            **normalized_data,
            "search_metadata": generate_search_metadata(tool_name, result_count) if result_count > 0 else None
        },
        "execution_context": generate_execution_context(tool_name),
        "debug_info": generate_debug_info(tool_name, result_data)
    }

    # Remove None values
    if result["result"]["search_metadata"] is None:
        del result["result"]["search_metadata"]

    # Pad to target size if needed
    current_size = estimate_json_size_kb(result)
    if current_size < target_size_kb:
        padding_needed = int((target_size_kb - current_size) * 1024)
        # Add realistic verbose field instead of "xxx"
        result["debug_info"]["trace_context"] = {
            "span_id": generate_random_id(16),
            "trace_id": generate_random_id(32),
            "additional_context": "x" * padding_needed
        }

    return result


def estimate_json_size_kb(obj: Any) -> float:
    """Estimate size of object when serialized to JSON

    Args:
        obj: Object to estimate size of

    Returns:
        Estimated size in KB
    """
    json_str = json.dumps(obj)
    return len(json_str.encode("utf-8")) / 1024


def generate_random_id(length: int = 22) -> str:
    """Generate a random alphanumeric ID

    Args:
        length: Length of ID to generate

    Returns:
        Random alphanumeric string
    """
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_thinking_block(scenario_context: str, target_tokens: int) -> Dict[str, str]:
    """Generate a thinking content block with natural language reasoning

    Uses Faker to generate varied, realistic reasoning text.

    Args:
        scenario_context: What the agent is thinking about
        target_tokens: Target token count for thinking text (1 token ≈ 4 chars)

    Returns:
        Thinking block dictionary
    """
    # Template strings for different contexts
    thinking_templates = {
        "parse": "Let me analyze the user's request to understand what they need. ",
        "refine": "I need to refine the search parameters based on the user's preferences. ",
        "search": "I should search for options that match the user's criteria. ",
        "analyze": "Let me analyze these results to find the best options. ",
        "compare": "I need to compare these options based on price, convenience, and quality. ",
        "present": "I should present these results in a clear and organized way. ",
        "plan": "Let me plan out the steps needed to complete this booking. ",
        "delegate": "I need to delegate this to a specialist who can handle the details. ",
        "coordinate": "I should coordinate between the different services to ensure everything aligns. ",
    }

    # Start with appropriate template
    base_thinking = thinking_templates.get(scenario_context, "Let me think about how to approach this. ")

    # Use Faker to generate natural reasoning
    target_chars = target_tokens * 4
    thinking_text = base_thinking

    while len(thinking_text) < target_chars:
        thinking_text += fake.paragraph(nb_sentences=3) + " "

    return {
        "type": "thinking",
        "thinking": thinking_text[:target_chars].strip()
    }


def generate_tool_use_block(tool_name: str, scenario_type: str) -> Dict[str, Any]:
    """Generate a tool_use content block with Anthropic format

    Args:
        tool_name: Name of tool (e.g., "search_flights")
        scenario_type: Scenario context for realistic parameters

    Returns:
        Tool use block dictionary
    """
    return {
        "type": "tool_use",
        "id": f"toolu_{generate_random_id(22)}",
        "name": tool_name,
        "input": generate_tool_parameters(tool_name, scenario_type)
    }


def generate_text_block(response_context: str, target_tokens: int) -> Dict[str, str]:
    """Generate a text content block with natural language response

    Uses Faker to generate varied, realistic response text.

    Args:
        response_context: What the response is about
        target_tokens: Target token count for response text (1 token ≈ 4 chars)

    Returns:
        Text block dictionary
    """
    # Template strings for different response contexts
    response_templates = {
        "results": "Based on the search results, I found several great options that match your criteria. ",
        "analysis": "After analyzing the available options, here's what I recommend. ",
        "summary": "Here's a summary of what I found. ",
        "recommendation": "I recommend the following based on your preferences. ",
        "confirmation": "I've completed the requested action. ",
        "clarification": "Let me clarify the details. ",
        "explanation": "Here's a detailed explanation. ",
    }

    base_response = response_templates.get(response_context, "Here's what I found. ")

    # Use Faker to generate natural explanatory text
    target_chars = target_tokens * 4
    response_text = base_response

    while len(response_text) < target_chars:
        response_text += fake.paragraph(nb_sentences=4) + " "

    return {
        "type": "text",
        "text": response_text[:target_chars].strip()
    }


def generate_tool_parameters(tool_name: str, scenario_type: str) -> Dict[str, Any]:
    """Generate realistic tool parameters based on tool and scenario

    Args:
        tool_name: Name of the tool
        scenario_type: Type of scenario for context

    Returns:
        Dictionary of tool parameters
    """
    # Generate realistic parameters based on tool name
    if "flight" in tool_name.lower():
        return {
            "origin": random.choice(AIRPORTS),
            "destination": random.choice(AIRPORTS),
            "departure_date": (datetime.now() + timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d"),
            "return_date": (datetime.now() + timedelta(days=random.randint(14, 90))).strftime("%Y-%m-%d"),
            "cabin_class": random.choice(["economy", "premium_economy", "business", "first"]),
            "passengers": random.randint(1, 4),
            "filters": {
                "max_stops": random.choice([0, 1, 2]),
                "preferred_airlines": random.sample(AIRLINES, k=random.randint(1, 3)),
                "max_price": random.choice([1000, 2000, 5000, 10000])
            }
        }

    elif "hotel" in tool_name.lower():
        return {
            "location": random.choice(LOCATIONS),
            "check_in": (datetime.now() + timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d"),
            "check_out": (datetime.now() + timedelta(days=random.randint(10, 65))).strftime("%Y-%m-%d"),
            "guests": random.randint(1, 4),
            "rooms": random.randint(1, 2),
            "filters": {
                "min_rating": random.choice([3.0, 3.5, 4.0, 4.5]),
                "price_range": {
                    "min": random.choice([50, 100, 200]),
                    "max": random.choice([300, 500, 1000])
                },
                "amenities": random.sample(["wifi", "pool", "gym", "spa", "restaurant", "parking"], k=random.randint(2, 4))
            }
        }

    elif "user" in tool_name.lower():
        return {
            "user_id": f"user_{random.randint(10000, 99999)}",
            "include_preferences": True,
            "include_history": True,
            "include_loyalty_status": True
        }

    elif "price" in tool_name.lower():
        return {
            "service_type": random.choice(["flight", "hotel", "car"]),
            "track_changes": True,
            "alert_threshold": random.choice([50, 100, 200])
        }

    elif "location" in tool_name.lower():
        return {
            "city": random.choice(LOCATIONS),
            "include_attractions": True,
            "include_transportation": True,
            "radius_miles": random.choice([5, 10, 25])
        }

    else:
        # Generic tool parameters
        return {
            "query": "search query",
            "limit": random.choice([10, 20, 50]),
            "offset": 0
        }
