# AI Observability Scale Test Framework

Scale testing framework for AI observability platforms using realistic OpenTelemetry traces.

## Features

- Framework-free trace generation inspired by real agent workloads
- Multi-platform support: Braintrust, LangSmith, OTLP-compatible backends
- Configurable scale: 10-100 req/sec with async execution
- Realistic characteristics: 100-250KB traces, 50-150 spans, 5-8 nesting levels

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

```bash
python scripts/run_scale_test.py
```

See `docs/plans/2025-11-27-ai-observability-scale-test-design.md` for full design.
