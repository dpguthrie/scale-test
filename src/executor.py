"""Async workload executor for scale testing"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from src.scenarios import TraceScenario, get_scenario
from src.platforms import get_platform
from src.metrics import MetricsCollector
from src.metadata import generate_trace_metadata
from src.span_attributes import set_io_attributes, set_span_type_attributes, set_metadata_attributes

# Setup logging
logger = logging.getLogger(__name__)


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
        """Setup OpenTelemetry tracer with platform exporter

        Configuration can be tuned via environment variables:
        - OTEL_BSP_SCHEDULE_DELAY: Milliseconds between exports (default: 5000)
        - OTEL_BSP_MAX_EXPORT_BATCH_SIZE: Max spans per batch (default: 10 for 10MB limit)
        - OTEL_BSP_MAX_QUEUE_SIZE: Max queued spans (default: 4096, increased for high concurrency)
        - OTEL_BSP_EXPORT_TIMEOUT: Export timeout in milliseconds (default: 120000)

        Note: Queue fills when spans are created faster than exported. Increase MAX_QUEUE_SIZE
        or reduce SCHEDULE_DELAY to prevent "Queue is full, likely spans will be dropped" warnings.
        """
        import os

        provider = TracerProvider()

        # Setup exporter based on platform
        if self.platform.endpoint:
            exporter = OTLPSpanExporter(
                endpoint=self.platform.endpoint,
                headers=self.platform.get_headers(),
                timeout=int(os.getenv("OTEL_BSP_EXPORT_TIMEOUT", "120000")) / 1000
            )
            # BatchSpanProcessor with configurable settings
            # Defaults tuned for high-concurrency scale testing with large traces (500KB)
            processor = BatchSpanProcessor(
                exporter,
                max_export_batch_size=int(os.getenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "10")),
                schedule_delay_millis=int(os.getenv("OTEL_BSP_SCHEDULE_DELAY", "5000")),
                max_queue_size=int(os.getenv("OTEL_BSP_MAX_QUEUE_SIZE", "4096")),  # Increased from 2048
            )
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
        logger.info(f"Starting scale test: {self.concurrency} workers, {duration_seconds}s duration")
        print(f"Starting scale test: {self.concurrency} workers, {duration_seconds}s duration")

        # Schedule stop
        asyncio.create_task(self._stop_after(duration_seconds))

        # Schedule periodic progress reporting
        asyncio.create_task(self._progress_reporter(duration_seconds))

        # Start workers
        workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.concurrency)
        ]

        logger.info(f"Started {len(workers)} worker tasks")

        # Wait for all workers to complete
        logger.info("Waiting for workers to complete...")
        await asyncio.gather(*workers)
        logger.info("All workers completed")

        # Get metrics before shutdown
        total_requests = self.metrics.success_count + self.metrics.failure_count
        total_spans_estimate = sum(
            scenario.expected_span_count
            for scenario in self.scenarios
        ) * total_requests / len(self.scenarios) if len(self.scenarios) > 0 else 0

        # Shutdown tracer provider to flush remaining spans
        import os
        skip_shutdown = os.getenv("OTEL_SKIP_SHUTDOWN", "false").lower() == "true"

        if skip_shutdown:
            # Detect if using collector (localhost endpoint)
            endpoint = self.platform.endpoint or ""
            using_collector = "localhost" in endpoint or "127.0.0.1" in endpoint

            if using_collector:
                logger.info("OTEL_SKIP_SHUTDOWN=true - exiting immediately (collector will continue exporting)")
                logger.debug(f"~{int(total_spans_estimate)} spans sent to collector, export continues in background")
            else:
                logger.warning("OTEL_SKIP_SHUTDOWN=true - exiting without flushing remaining spans")
                logger.warning(f"~{int(total_spans_estimate)} spans may not be exported (direct export mode)")
            return self.metrics.report()

        logger.info(f"Shutting down tracer provider to flush ~{int(total_spans_estimate)} spans...")
        logger.info(f"This may take a while with {total_requests} traces...")
        logger.info("Set OTEL_SKIP_SHUTDOWN=true to exit immediately without waiting")

        # Run shutdown in thread pool to avoid blocking event loop indefinitely
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # Give it a generous timeout (30s per 1000 spans, min 30s, max 300s)
                timeout_seconds = max(30, min(300, int(total_spans_estimate / 1000 * 30)))
                logger.info(f"Shutdown timeout set to {timeout_seconds}s")

                future = executor.submit(self.tracer_provider.shutdown)
                future.result(timeout=timeout_seconds)
                logger.info("Tracer provider shutdown complete")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Tracer shutdown timed out after {timeout_seconds}s - some spans may not be exported")
                logger.warning("Consider: reducing SCALE_TEST_DURATION, increasing OTEL_BSP_SCHEDULE_DELAY, or reducing OTEL_BSP_MAX_QUEUE_SIZE")
                logger.warning("Or use OTEL_SKIP_SHUTDOWN=true to exit immediately")

        return self.metrics.report()

    async def _stop_after(self, seconds: float):
        """Stop execution after specified time"""
        await asyncio.sleep(seconds)
        logger.info(f"Duration {seconds}s elapsed, stopping workers...")
        self.should_stop = True

    async def _progress_reporter(self, total_duration: float):
        """Report progress every 10 seconds"""
        start_time = time.time()
        report_interval = 10  # seconds

        while not self.should_stop:
            await asyncio.sleep(report_interval)

            if not self.should_stop:
                elapsed = time.time() - start_time
                requests_so_far = self.metrics.success_count + self.metrics.failure_count
                rate = requests_so_far / elapsed if elapsed > 0 else 0

                logger.info(f"Progress: {int(elapsed)}s/{int(total_duration)}s | {requests_so_far} traces | {rate:.1f} req/s")

    async def _worker(self, worker_id: int):
        """Worker coroutine that executes scenarios

        Args:
            worker_id: Unique worker identifier
        """
        # Gradual ramp-up: stagger worker starts
        if self.ramp_up_seconds > 0:
            delay = (worker_id / self.concurrency) * self.ramp_up_seconds
            logger.debug(f"Worker {worker_id}: Delaying start by {delay:.1f}s for ramp-up")
            await asyncio.sleep(delay)

        logger.info(f"Worker {worker_id}: Starting execution loop")
        iteration = 0

        while not self.should_stop:
            iteration += 1

            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Select scenario based on weights
            scenario = random.choices(self.scenarios, weights=self.weights)[0]

            logger.debug(f"Worker {worker_id}: Iteration {iteration}, executing {scenario.name}")

            # Execute scenario
            await self._execute_scenario(scenario)

        logger.info(f"Worker {worker_id}: Completed {iteration} iterations, stopping")

    async def _execute_scenario(self, scenario: TraceScenario):
        """Execute a single scenario and collect metrics

        Args:
            scenario: Scenario to execute
        """
        start_time = time.time()
        data_size = 0

        try:
            logger.debug(f"Starting scenario: {scenario.name} ({scenario.expected_span_count} spans expected)")

            # Create root span for trace (required for Braintrust)
            with self.tracer.start_as_current_span(f"trace_{scenario.name}") as root_span:
                root_span.set_attribute("scenario.name", scenario.name)
                root_span.set_attribute("scenario.expected_spans", scenario.expected_span_count)
                root_span.set_attribute("span.kind", "server")  # Mark as entry point

                # Add realistic trace-level metadata for filtering
                trace_metadata = generate_trace_metadata()

                # Set input for root span
                user_query = f"Execute {scenario.name} workflow"
                set_io_attributes(root_span, input_value=user_query)

                # Set span type and kind
                set_span_type_attributes(root_span, span_type="task", span_kind="chain")

                # Set metadata (both raw attributes and platform-prefixed)
                set_metadata_attributes(root_span, trace_metadata)

                # Execute workflow steps
                for step in scenario.workflow_steps:
                    await step.execute(self.tracer, None, self.platform)

                # Set output after workflow completes
                workflow_output = f"Completed {scenario.name} with {len(scenario.workflow_steps)} steps"
                set_io_attributes(root_span, output_value=workflow_output)

                # Estimate data size (rough approximation)
                data_size = scenario.expected_size_kb * 1024

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Scenario {scenario.name} completed in {latency_ms:.1f}ms")
            self.metrics.record_success(
                scenario_name=scenario.name,
                latency_ms=latency_ms,
                data_size_bytes=data_size
            )

        except Exception as e:
            # Record failure
            logger.error(f"Scenario {scenario.name} failed: {e}", exc_info=True)
            self.metrics.record_failure(
                scenario_name=scenario.name,
                error=str(e)
            )
