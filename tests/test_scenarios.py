from src.scenarios import (
    TraceScenario,
    get_scenario,
    SIMPLE_QUERY,
    SINGLE_SERVICE_SEARCH,
    DELEGATED_BOOKING,
    MULTI_SERVICE_COMPLEX
)


def test_simple_query_scenario():
    """Simple query scenario has expected structure"""
    assert SIMPLE_QUERY.name == "simple_query"
    assert len(SIMPLE_QUERY.workflow_steps) >= 3
    assert SIMPLE_QUERY.expected_span_count >= 5
    assert SIMPLE_QUERY.expected_size_kb == 35


def test_delegated_booking_scenario():
    """Delegated booking has delegation step"""
    assert DELEGATED_BOOKING.name == "delegated_booking"
    # Should have delegation step
    has_delegation = any(
        step.__class__.__name__ == "DelegationStep"
        for step in DELEGATED_BOOKING.workflow_steps
    )
    assert has_delegation
    assert DELEGATED_BOOKING.expected_span_count >= 20


def test_get_scenario_by_name():
    """get_scenario retrieves correct scenario"""
    scenario = get_scenario("simple_query")
    assert scenario.name == "simple_query"

    scenario = get_scenario("delegated_booking")
    assert scenario.name == "delegated_booking"


def test_all_scenarios_registered():
    """All built-in scenarios are available"""
    scenarios = ["simple_query", "single_service_search", "delegated_booking", "multi_service_complex"]
    for name in scenarios:
        scenario = get_scenario(name)
        assert scenario is not None
