import json
from src.payloads import (
    generate_flight_search_results,
    generate_hotel_search_results,
    generate_llm_messages,
    generate_tool_result,
    estimate_json_size_kb
)


def test_flight_search_results_structure():
    """Flight search results have correct schema"""
    results = generate_flight_search_results(count=5, target_size_kb=30)
    assert len(results) == 5
    assert "flight_id" in results[0]
    assert "flight_no" in results[0]
    assert "departure_airport" in results[0]
    assert "arrival_airport" in results[0]


def test_flight_search_results_size():
    """Flight search results approximately match target size"""
    results = generate_flight_search_results(count=20, target_size_kb=50)
    size_kb = estimate_json_size_kb(results)
    assert 45 <= size_kb <= 55  # Within 10% tolerance


def test_hotel_search_results_structure():
    """Hotel search results have correct schema"""
    results = generate_hotel_search_results(count=10, target_size_kb=40)
    assert len(results) == 10
    assert "id" in results[0]
    assert "name" in results[0]
    assert "location" in results[0]
    assert "price_tier" in results[0]


def test_llm_messages_format():
    """LLM messages have proper chat format"""
    messages = generate_llm_messages(
        user_prompt="Book a hotel in Zurich",
        assistant_response="I'll help you search for hotels",
        target_size_kb=10
    )
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "assistant"


def test_tool_result_format():
    """Tool results have proper structure"""
    result = generate_tool_result(
        tool_name="search_flights",
        success=True,
        data={"flights": [{"id": 1}]},
        target_size_kb=5
    )
    assert result["tool"] == "search_flights"
    assert result["success"] is True
    assert "data" in result
    assert "timestamp" in result
