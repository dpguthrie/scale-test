"""Metrics collection and reporting for scale tests"""

import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScenarioMetrics:
    """Metrics for a specific scenario"""
    count: int = 0
    latencies: List[float] = field(default_factory=list)
    data_sizes: List[int] = field(default_factory=list)
    failures: int = 0

    def add_success(self, latency_ms: float, data_size_bytes: int):
        """Record successful execution"""
        self.count += 1
        self.latencies.append(latency_ms)
        self.data_sizes.append(data_size_bytes)

    def add_failure(self):
        """Record failed execution"""
        self.failures += 1

    def summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.latencies:
            return {
                "count": self.count,
                "failures": self.failures,
                "avg_latency_ms": 0,
                "avg_data_kb": 0
            }

        return {
            "count": self.count,
            "failures": self.failures,
            "avg_latency_ms": np.mean(self.latencies),
            "p50_latency_ms": np.percentile(self.latencies, 50),
            "p95_latency_ms": np.percentile(self.latencies, 95),
            "avg_data_kb": np.mean(self.data_sizes) / 1024
        }


class MetricsCollector:
    """Collects and reports metrics during scale testing"""

    def __init__(self):
        self.start_time = time.time()
        self.success_count = 0
        self.failure_count = 0
        self.latencies: List[float] = []
        self.data_sizes: List[int] = []
        self.per_scenario: Dict[str, ScenarioMetrics] = defaultdict(ScenarioMetrics)
        self.errors: List[str] = []

    def record_success(
        self,
        scenario_name: str,
        latency_ms: float,
        data_size_bytes: int
    ):
        """Record a successful trace execution

        Args:
            scenario_name: Name of the scenario
            latency_ms: Execution latency in milliseconds
            data_size_bytes: Total data size in bytes
        """
        self.success_count += 1
        self.latencies.append(latency_ms)
        self.data_sizes.append(data_size_bytes)
        self.per_scenario[scenario_name].add_success(latency_ms, data_size_bytes)

    def record_failure(self, scenario_name: str, error: str):
        """Record a failed trace execution

        Args:
            scenario_name: Name of the scenario
            error: Error message
        """
        self.failure_count += 1
        self.per_scenario[scenario_name].add_failure()
        self.errors.append(f"{scenario_name}: {error}")

    def report(self) -> Dict:
        """Generate comprehensive metrics report

        Returns:
            Dictionary with all collected metrics
        """
        elapsed_time = time.time() - self.start_time
        total_requests = self.success_count + self.failure_count

        report = {
            "duration_seconds": elapsed_time,
            "total_requests": total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0,
            "throughput": self.success_count / elapsed_time if elapsed_time > 0 else 0,
        }

        # Latency statistics
        if self.latencies:
            report.update({
                "latency_p50": np.percentile(self.latencies, 50),
                "latency_p95": np.percentile(self.latencies, 95),
                "latency_p99": np.percentile(self.latencies, 99),
                "latency_min": min(self.latencies),
                "latency_max": max(self.latencies),
            })

        # Data volume statistics
        if self.data_sizes:
            total_bytes = sum(self.data_sizes)
            report.update({
                "total_data_gb": total_bytes / (1024 ** 3),
                "avg_data_per_request_kb": (total_bytes / len(self.data_sizes)) / 1024,
            })

        # Per-scenario breakdown
        report["per_scenario"] = {
            name: metrics.summary()
            for name, metrics in self.per_scenario.items()
        }

        return report

    def format_report(self, report: Optional[Dict] = None) -> str:
        """Format report as human-readable string

        Args:
            report: Report dict (or generate new one if None)

        Returns:
            Formatted report string
        """
        if report is None:
            report = self.report()

        lines = [
            "Scale Test Results:",
            f"   Duration: {report['duration_seconds']:.1f}s",
            f"   Total requests: {report['total_requests']}",
            f"   Success rate: {report['success_rate']*100:.1f}%",
            f"   Throughput: {report['throughput']:.1f} req/s",
            "",
        ]

        if "latency_p50" in report:
            lines.extend([
                "Latency:",
                f"   P50: {report['latency_p50']:.1f}ms",
                f"   P95: {report['latency_p95']:.1f}ms",
                f"   P99: {report['latency_p99']:.1f}ms",
                "",
            ])

        if "total_data_gb" in report:
            lines.extend([
                "Data Volume:",
                f"   Total: {report['total_data_gb']:.2f} GB",
                f"   Avg per request: {report['avg_data_per_request_kb']:.1f} KB",
                "",
            ])

        if report.get("per_scenario"):
            lines.append("Query Mix Breakdown:")
            for name, metrics in report["per_scenario"].items():
                if metrics["count"] > 0:
                    lines.append(
                        f"   {name}: {metrics['count']} traces, "
                        f"P50={metrics.get('p50_latency_ms', 0):.1f}ms, "
                        f"{metrics['avg_data_kb']:.1f}KB avg"
                    )

        return "\n".join(lines)
