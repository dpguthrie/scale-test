import pytest
import asyncio
from src.executor import ScaleTestExecutor, TokenBucketRateLimiter
from src.scenarios import SIMPLE_QUERY


@pytest.mark.asyncio
async def test_rate_limiter():
    """TokenBucketRateLimiter enforces rate limit"""
    limiter = TokenBucketRateLimiter(rate=10)  # 10 per second

    # Should allow immediate requests
    start = asyncio.get_event_loop().time()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.3  # Should be nearly instant


@pytest.mark.asyncio
async def test_executor_initialization():
    """ScaleTestExecutor initializes with config"""
    config = {
        "concurrency": 5,
        "rate_limit": 10,
        "query_mix": {
            "simple_query": 1.0
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)
    assert executor.concurrency == 5
    assert executor.rate_limiter is not None


@pytest.mark.asyncio
async def test_executor_runs_for_duration():
    """Executor runs for specified duration"""
    config = {
        "concurrency": 2,
        "rate_limit": 100,
        "query_mix": {
            "simple_query": 1.0
        },
        "platform_config": {
            "platform": "console"
        }
    }

    executor = ScaleTestExecutor(config)

    # Run for 2 seconds
    start = asyncio.get_event_loop().time()
    metrics = await executor.run(duration_seconds=2)
    elapsed = asyncio.get_event_loop().time() - start

    # Should complete in approximately 2 seconds
    assert 1.5 <= elapsed <= 3.0
    assert metrics["total_requests"] > 0
