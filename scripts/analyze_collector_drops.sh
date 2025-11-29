#!/bin/bash
# Analyze OTel Collector logs for dropped spans and platform rejections

echo "🔍 OpenTelemetry Collector Drop Analysis"
echo "=========================================="
echo

# Get logs for analysis
LOGS=$(docker-compose logs --tail=10000 otel-collector 2>&1)

# Count drops by reason
echo "📉 Dropped Spans Summary:"
echo "------------------------"
total_dropped=$(echo "$LOGS" | grep "Dropping data" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
echo "Total spans dropped: ${total_dropped:-0}"
echo


# Break down by HTTP status code
echo "🚫 Platform Rejection Breakdown:"
echo "--------------------------------"
echo "HTTP 413 (Payload Too Large):"
http_413=$(echo "$LOGS" | grep "Status Code 413" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
count_413=$(echo "$LOGS" | grep -c "Status Code 413")
echo "  Occurrences: ${count_413:-0}"
echo "  Spans dropped: ${http_413:-0}"

echo
echo "HTTP 429 (Rate Limited):"
http_429=$(echo "$LOGS" | grep "Status Code 429" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
count_429=$(echo "$LOGS" | grep -c "Status Code 429")
echo "  Occurrences: ${count_429:-0}"
echo "  Spans dropped: ${http_429:-0}"

echo
echo "HTTP 503 (Service Unavailable):"
http_503=$(echo "$LOGS" | grep "Status Code 503" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
count_503=$(echo "$LOGS" | grep -c "Status Code 503")
echo "  Occurrences: ${count_503:-0}"
echo "  Spans dropped: ${http_503:-0}"

echo
echo "HTTP 409 (Conflict):"
http_409=$(echo "$LOGS" | grep "Status Code 409" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')
count_409=$(echo "$LOGS" | grep -c "Status Code 409")
echo "  Occurrences: ${count_409:-0}"
echo "  Spans dropped: ${http_409:-0}"

echo
echo "Other Errors:"
other_errors=$(echo "$LOGS" | grep "Exporting failed. Dropping data" | grep -v "Status Code" | wc -l)
echo "  Occurrences: ${other_errors:-0}"

# Retry attempts
echo
echo "🔄 Retry Statistics:"
echo "-------------------"
retry_count=$(echo "$LOGS" | grep -c "Will retry the request")
echo "Total retry attempts: ${retry_count:-0}"

# Memory issues
echo
echo "💾 Memory Pressure:"
echo "------------------"
mem_warnings=$(echo "$LOGS" | grep -c "Memory usage is above")
echo "Memory warnings: ${mem_warnings:-0}"

# Recent errors (last 10)
echo
echo "🕐 Most Recent Errors (last 5):"
echo "-------------------------------"
echo "$LOGS" | grep "Exporting failed. Dropping data" | tail -5 | while read -r line; do
    timestamp=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
    status=$(echo "$line" | grep -oE 'Status Code [0-9]+' | grep -oE '[0-9]+' || echo "Unknown")
    dropped=$(echo "$line" | grep -oE 'dropped_items":\s*[0-9]+' | grep -oE '[0-9]+')
    echo "  [$timestamp] HTTP $status - Dropped: $dropped spans"
done

echo
echo "=========================================="
echo "💡 Recommendations:"
echo


if [ "${http_413:-0}" -gt 0 ]; then
    echo "⚠️  HTTP 413: Reduce OTEL_BSP_MAX_EXPORT_BATCH_SIZE in collector config"
fi

if [ "${http_429:-0}" -gt 0 ]; then
    echo "⚠️  HTTP 429: Platform rate limiting - increase retry intervals or reduce concurrency"
fi

if [ "${mem_warnings:-0}" -gt 10 ]; then
    echo "⚠️  Memory pressure: Increase memory_limiter.limit_mib in collector config"
fi

if [ "${total_dropped:-0}" -eq 0 ]; then
    echo "✅ No spans dropped - collector is healthy!"
fi
