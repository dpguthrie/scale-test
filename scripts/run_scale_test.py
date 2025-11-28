#!/usr/bin/env python3
"""Main CLI script for running scale tests"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.executor import ScaleTestExecutor


def load_config_from_env() -> dict:
    """Load configuration from environment variables

    Returns:
        Configuration dictionary
    """
    # Platform configuration
    platform = os.getenv("OTEL_PLATFORM", "console").lower()
    platform_config = {
        "platform": platform,
    }

    if platform == "braintrust":
        # Parse headers for API key and project
        headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        api_key = ""
        project_name = ""

        for part in headers.split(","):
            part = part.strip()
            if "Bearer" in part:
                api_key = part.split("Bearer")[1].strip()
            elif "x-bt-parent" in part:
                # Extract value after colon
                if ":" in part:
                    project_name = part.split(":", 1)[1].strip()

        platform_config.update({
            "api_key": api_key,
            "project_name": project_name
        })

    elif platform == "langsmith":
        # LangSmith uses LANGSMITH_* environment variables
        api_key = os.getenv("LANGSMITH_API_KEY", "")
        # LangSmith project is optional, defaults to "default" if not specified
        project_name = os.getenv("LANGSMITH_PROJECT", "default")

        platform_config.update({
            "api_key": api_key,
            "project_name": project_name
        })

    elif platform == "otlp":
        platform_config.update({
            "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "headers": {}
        })

    # Scale test parameters
    concurrency = int(os.getenv("SCALE_TEST_CONCURRENCY", "50"))
    duration = int(os.getenv("SCALE_TEST_DURATION", "300"))
    rate_limit = os.getenv("SCALE_TEST_RATE_LIMIT")
    ramp_up = int(os.getenv("SCALE_TEST_RAMP_UP", "0"))

    # Query mix
    query_mix = {
        "simple_query": float(os.getenv("SCALE_TEST_MIX_SIMPLE", "0.40")),
        "single_service_search": float(os.getenv("SCALE_TEST_MIX_SEARCH", "0.30")),
        "delegated_booking": float(os.getenv("SCALE_TEST_MIX_BOOKING", "0.20")),
        "multi_service_complex": float(os.getenv("SCALE_TEST_MIX_COMPLEX", "0.10")),
    }

    # Validate query mix sums to 1.0
    total = sum(query_mix.values())
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Query mix sums to {total}, normalizing to 1.0")
        query_mix = {k: v/total for k, v in query_mix.items()}

    config = {
        "concurrency": concurrency,
        "duration": duration,
        "rate_limit": float(rate_limit) if rate_limit else None,
        "ramp_up_seconds": ramp_up,
        "query_mix": query_mix,
        "platform_config": platform_config
    }

    return config


async def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config_from_env()

    # Print configuration
    print("Configuration:")
    print(f"  Platform: {config['platform_config']['platform']}")
    print(f"  Concurrency: {config['concurrency']} workers")
    print(f"  Duration: {config['duration']}s")
    if config['rate_limit']:
        print(f"  Rate limit: {config['rate_limit']} req/s")
    if config['ramp_up_seconds']:
        print(f"  Ramp-up: {config['ramp_up_seconds']}s")
    print(f"  Query mix: {config['query_mix']}")
    print()

    # Create and run executor
    executor = ScaleTestExecutor(config)
    report = await executor.run(duration_seconds=config["duration"])

    # Print results
    print("\n" + "="*60)
    print(executor.metrics.format_report(report))
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
