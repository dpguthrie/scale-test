"""Realistic payload generation for trace data"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


AIRPORTS = ["JFK", "LAX", "ORD", "SFO", "BOS", "DFW", "ATL", "MIA",
            "LHR", "CDG", "FRA", "AMS", "ZRH", "BCN", "FCO"]
AIRLINES = ["AA", "UA", "DL", "BA", "LH", "AF", "KL", "LX"]
HOTEL_NAMES = ["Grand Hotel", "City Inn", "Plaza Hotel", "Resort & Spa",
               "Boutique Hotel", "Business Hotel", "Airport Hotel"]
LOCATIONS = ["New York", "Los Angeles", "Chicago", "San Francisco",
             "London", "Paris", "Frankfurt", "Zurich", "Barcelona"]
PRICE_TIERS = ["Midscale", "Upper Midscale", "Upscale", "Luxury"]


def generate_flight_search_results(count: int = 20, target_size_kb: int = 30) -> List[Dict]:
    """Generate realistic flight search results matching expedia schema

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


def generate_hotel_search_results(count: int = 10, target_size_kb: int = 40) -> List[Dict]:
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
                k=random.randint(3, 7)
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


def generate_llm_messages(
    user_prompt: str,
    assistant_response: str,
    target_size_kb: int = 10
) -> List[Dict[str, str]]:
    """Generate LLM message array in chat format

    Args:
        user_prompt: User's input message
        assistant_response: Assistant's response
        target_size_kb: Target total size in KB

    Returns:
        List of message dictionaries with role and content
    """
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

    # Pad assistant response to reach target size
    current_size = estimate_json_size_kb(messages)
    if current_size < target_size_kb:
        padding = "x" * int((target_size_kb - current_size) * 1024)
        messages[-1]["content"] += f"\n\nAdditional context: {padding}"

    return messages


def generate_tool_result(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    target_size_kb: int = 5
) -> Dict[str, Any]:
    """Generate a tool execution result

    Args:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        data: Tool result data
        target_size_kb: Target size in KB

    Returns:
        Tool result dictionary
    """
    result = {
        "tool": tool_name,
        "success": success,
        "data": data or {},
        "timestamp": datetime.now().isoformat(),
    }

    # Pad to target size
    current_size = estimate_json_size_kb(result)
    if current_size < target_size_kb:
        padding = "x" * int((target_size_kb - current_size) * 1024)
        result["metadata"] = {"padding": padding}

    return result


def estimate_json_size_kb(obj: Any) -> float:
    """Estimate size of object when serialized to JSON

    Args:
        obj: Object to estimate size of

    Returns:
        Estimated size in KB
    """
    json_str = json.dumps(obj)
    return len(json_str.encode('utf-8')) / 1024
