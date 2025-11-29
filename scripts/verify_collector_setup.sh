#!/bin/bash
# Verify OpenTelemetry Collector setup

set -e

echo "🔍 Verifying OpenTelemetry Collector Setup"
echo "==========================================="
echo

# Check Docker is running
echo "1. Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi
echo "✅ Docker is running"
echo

# Check collector container
echo "2. Checking collector container..."
if ! docker ps | grep -q scale-test-otel-collector; then
    echo "❌ Collector container is not running."
    echo "   Run: docker-compose up -d"
    exit 1
fi
echo "✅ Collector container is running"
echo

# Check environment variables
echo "3. Checking environment variables..."
if [ -z "$BRAINTRUST_API_KEY" ]; then
    echo "⚠️  BRAINTRUST_API_KEY is not set"
    echo "   Set it with: export BRAINTRUST_API_KEY='sk-...'"
    echo "   Then restart collector: docker-compose restart"
else
    echo "✅ BRAINTRUST_API_KEY is set (${BRAINTRUST_API_KEY:0:6}...)"
fi

if [ -z "$BRAINTRUST_PROJECT" ]; then
    echo "⚠️  BRAINTRUST_PROJECT is not set (will use default: scale-test)"
else
    echo "✅ BRAINTRUST_PROJECT is set: $BRAINTRUST_PROJECT"
fi
echo

# Check collector is receiving spans
echo "4. Checking collector endpoints..."
if curl -s http://localhost:4318 > /dev/null 2>&1; then
    echo "✅ OTLP HTTP endpoint (4318) is accessible"
else
    echo "❌ Cannot reach OTLP HTTP endpoint (4318)"
    exit 1
fi
echo

# Check .env file
echo "5. Checking .env configuration..."
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "   Create it with: cp .env.example.collector .env"
else
    if grep -q "OTEL_PLATFORM=otlp" .env; then
        echo "✅ .env is configured for collector (OTEL_PLATFORM=otlp)"
    else
        echo "⚠️  .env exists but not configured for collector"
        echo "   Make sure OTEL_PLATFORM=otlp"
    fi
fi
echo

# Check collector logs for errors
echo "6. Checking collector logs for errors..."
if docker-compose logs --tail=20 otel-collector 2>&1 | grep -i "error" > /dev/null; then
    echo "⚠️  Found errors in collector logs:"
    docker-compose logs --tail=20 otel-collector | grep -i "error"
    echo
    echo "   Check full logs with: docker-compose logs otel-collector"
else
    echo "✅ No recent errors in collector logs"
fi
echo

# Summary
echo "==========================================="
echo "📊 Setup Summary"
echo "==========================================="
echo

if [ -z "$BRAINTRUST_API_KEY" ]; then
    echo "🔧 Next steps:"
    echo "   1. export BRAINTRUST_API_KEY='sk-...'"
    echo "   2. docker-compose restart"
    echo "   3. Run this script again to verify"
else
    echo "✅ Ready to run tests!"
    echo
    echo "Quick test:"
    echo "   SCALE_TEST_DURATION=10 SCALE_TEST_CONCURRENCY=5 uv run python scripts/run_scale_test.py"
    echo
    echo "Monitor collector:"
    echo "   docker-compose logs -f otel-collector"
fi
