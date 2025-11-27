"""Integration tests for complete workflow"""

import pytest
import asyncio
from src.executor import ScaleTestExecutor
from src.scenarios import list_scenarios


@pytest.mark.asyncio
async def test_end_to_end_console_platform():
    """End-to-end test with console platform"""
    config = {
        "concurrency": 2,
        "rate_limit": 100,
        "query_mix": {
            "simple_query": 0.5,
            "single_service_search": 0.5
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)
    metrics = await executor.run(duration_seconds=2)

    assert metrics["total_requests"] > 0
    assert metrics["success_rate"] > 0.9  # Most should succeed
    assert "per_scenario" in metrics


@pytest.mark.asyncio
async def test_all_scenarios_executable():
    """Verify all scenarios can be executed"""
    for scenario_name in list_scenarios():
        config = {
            "concurrency": 1,
            "query_mix": {scenario_name: 1.0},
            "platform_config": {"platform": "console"}
        }

        executor = ScaleTestExecutor(config)
        metrics = await executor.run(duration_seconds=1)

        assert metrics["total_requests"] > 0
        assert scenario_name in metrics["per_scenario"]
