import pytest
import time
from src.metrics import MetricsCollector


def test_metrics_collector_initialization():
    """MetricsCollector initializes with empty state"""
    collector = MetricsCollector()
    assert collector.success_count == 0
    assert collector.failure_count == 0
    assert len(collector.latencies) == 0


def test_record_success():
    """Recording success increments count and tracks latency"""
    collector = MetricsCollector()
    collector.record_success(
        scenario_name="test",
        latency_ms=100.0,
        data_size_bytes=1024
    )
    assert collector.success_count == 1
    assert len(collector.latencies) == 1
    assert collector.latencies[0] == 100.0


def test_record_failure():
    """Recording failure increments failure count"""
    collector = MetricsCollector()
    collector.record_failure(scenario_name="test", error="Test error")
    assert collector.failure_count == 1


def test_metrics_report():
    """Report generates comprehensive statistics"""
    collector = MetricsCollector()
    collector.start_time = time.time() - 10  # 10 seconds ago

    # Record some metrics
    for i in range(100):
        collector.record_success("test", latency_ms=float(i), data_size_bytes=1024 * 150)

    report = collector.report()

    assert "throughput" in report
    assert "latency_p50" in report
    assert "latency_p95" in report
    assert "latency_p99" in report
    assert "total_data_gb" in report
    assert "success_rate" in report
    assert report["success_rate"] == 1.0


def test_per_scenario_breakdown():
    """Report includes per-scenario breakdown"""
    collector = MetricsCollector()
    collector.start_time = time.time() - 10

    collector.record_success("scenario_a", latency_ms=50, data_size_bytes=1024 * 20)
    collector.record_success("scenario_a", latency_ms=60, data_size_bytes=1024 * 25)
    collector.record_success("scenario_b", latency_ms=100, data_size_bytes=1024 * 100)

    report = collector.report()

    assert "per_scenario" in report
    assert "scenario_a" in report["per_scenario"]
    assert "scenario_b" in report["per_scenario"]
    assert report["per_scenario"]["scenario_a"]["count"] == 2
    assert report["per_scenario"]["scenario_b"]["count"] == 1
