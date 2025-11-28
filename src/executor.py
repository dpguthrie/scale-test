"""Async workload executor for scale testing"""

import asyncio
import random
import time
from typing import Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from src.scenarios import TraceScenario, get_scenario
from src.platforms import get_platform, Platform
from src.metrics import MetricsCollector


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rate"""

    def __init__(self, rate: float, burst: Optional[float] = None):
        """Initialize rate limiter

        Args:
            rate: Requests per second
            burst: Burst capacity (defaults to rate)
        """
        self.rate = rate
        self.burst = burst or rate
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # Wait for next token
            wait_time = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0.0


class ScaleTestExecutor:
    """Executes scale test workload with async workers"""

    def __init__(self, config: Dict):
        """Initialize executor with configuration

        Args:
            config: Configuration dictionary with:
                - concurrency: Number of concurrent workers
                - rate_limit: Max requests per second (optional)
                - query_mix: Dict mapping scenario names to weights
                - platform_config: Platform configuration
                - ramp_up_seconds: Gradual ramp-up period (optional)
        """
        self.concurrency = config["concurrency"]
        self.rate_limit = config.get("rate_limit")
        self.query_mix = config["query_mix"]
        self.ramp_up_seconds = config.get("ramp_up_seconds", 0)

        # Build scenario pool with weights
        self.scenarios: List[TraceScenario] = []
        self.weights: List[float] = []
        for scenario_name, weight in self.query_mix.items():
            self.scenarios.append(get_scenario(scenario_name))
            self.weights.append(weight)

        # Setup rate limiter
        self.rate_limiter = None
        if self.rate_limit:
            self.rate_limiter = TokenBucketRateLimiter(rate=self.rate_limit)

        # Setup platform and OTEL
        self.platform = get_platform(config["platform_config"])
        self.tracer_provider = self._setup_tracer()
        self.tracer = self.tracer_provider.get_tracer(__name__)

        # Metrics
        self.metrics = MetricsCollector()

        # Control flags
        self.should_stop = False

    def _setup_tracer(self) -> TracerProvider:
        """Setup OpenTelemetry tracer with platform exporter"""
        provider = TracerProvider()

        # Setup exporter based on platform
        if self.platform.endpoint:
            exporter = OTLPSpanExporter(
                endpoint=self.platform.endpoint,
                headers=self.platform.get_headers(),
                timeout=120  # 120 second timeout for large payloads
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        return provider

    async def run(self, duration_seconds: float) -> Dict:
        """Run scale test for specified duration

        Args:
            duration_seconds: How long to run test

        Returns:
            Metrics report dictionary
        """
        print(f"Starting scale test: {self.concurrency} workers, {duration_seconds}s duration")

        # Schedule stop
        asyncio.create_task(self._stop_after(duration_seconds))

        # Start workers
        workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.concurrency)
        ]

        # Wait for all workers to complete
        await asyncio.gather(*workers)

        # Shutdown tracer provider to flush remaining spans
        self.tracer_provider.shutdown()

        return self.metrics.report()

    async def _stop_after(self, seconds: float):
        """Stop execution after specified time"""
        await asyncio.sleep(seconds)
        self.should_stop = True

    async def _worker(self, worker_id: int):
        """Worker coroutine that executes scenarios

        Args:
            worker_id: Unique worker identifier
        """
        # Gradual ramp-up: stagger worker starts
        if self.ramp_up_seconds > 0:
            delay = (worker_id / self.concurrency) * self.ramp_up_seconds
            await asyncio.sleep(delay)

        while not self.should_stop:
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Select scenario based on weights
            scenario = random.choices(self.scenarios, weights=self.weights)[0]

            # Execute scenario
            await self._execute_scenario(scenario)

    async def _execute_scenario(self, scenario: TraceScenario):
        """Execute a single scenario and collect metrics

        Args:
            scenario: Scenario to execute
        """
        start_time = time.time()
        data_size = 0

        try:
            # Create root span for trace (required for Braintrust)
            with self.tracer.start_as_current_span(f"trace_{scenario.name}") as root_span:
                root_span.set_attribute("scenario.name", scenario.name)
                root_span.set_attribute("scenario.expected_spans", scenario.expected_span_count)

                # Execute workflow steps
                for step in scenario.workflow_steps:
                    await step.execute(self.tracer, None, self.platform)

                # Estimate data size (rough approximation)
                data_size = scenario.expected_size_kb * 1024

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_success(
                scenario_name=scenario.name,
                latency_ms=latency_ms,
                data_size_bytes=data_size
            )

        except Exception as e:
            # Record failure
            self.metrics.record_failure(
                scenario_name=scenario.name,
                error=str(e)
            )
