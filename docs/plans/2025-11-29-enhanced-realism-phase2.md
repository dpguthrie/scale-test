# Enhanced Realism Phase 2: Tool Results, Agents, Faker, Debug Info

**Date:** 2025-11-29
**Status:** Approved
**Goal:** Add MCP-style tool results, agent planning/reflection, Faker integration, and debug trails

## Overview

Building on the Anthropic message format implementation, this phase adds four enhancements:
1. **MCP-style tool result format** - Structured function calling with metadata
2. **Agent planning & reflection** - Explicit reasoning and outcome assessment
3. **Faker integration** - Natural language variety without template repetition
4. **Debug/audit trails** - Full diagnostic context for observability

## Design Decisions

**Tool Results:** Full MCP format
- Complete tool_call_id linking to tool_use blocks
- Structured function with arguments
- Rich result metadata (timing, cache, data sources)
- Execution context with request IDs and retry info

**Agent Planning:** All agent spans
- Every DelegationStep gets plan structure at start
- Reflection added at completion
- Consistent across all agent types (hotel, flight, primary)

**Text Generation:** Faker dependency
- Use Faker for all thinking blocks and text responses
- Natural sentence variety without templates
- No fallback - Faker is required dependency

**Debug Info:** Full diagnostic context
- SQL queries and execution plans
- Cache keys and hit/miss stats
- Upstream API calls with timing
- Performance breakdowns
- Data lineage tracking

## Implementation

### Step 1: Install Faker

```bash
uv add faker
```

### Step 2: Update Tool Result Format (`src/payloads.py`)

**Add helper functions:**

```python
def generate_search_metadata(tool_name: str, result_count: int) -> Dict[str, Any]:
    """Generate search metadata for tool results

    Args:
        tool_name: Name of the tool
        result_count: Number of results returned

    Returns:
        Search metadata dictionary
    """
    total_results = result_count + random.randint(10, 50)

    return {
        "query_time_ms": random.randint(50, 300),
        "cache_hit": random.choice([True, False]),
        "total_results": total_results,
        "filtered_to": result_count,
        "data_sources": random.sample(
            ["gds_amadeus", "gds_sabre", "expedia_api", "booking_com", "internal_db"],
            k=random.randint(1, 3)
        ),
        "pricing_currency": "USD"
    }


def generate_execution_context(tool_name: str) -> Dict[str, Any]:
    """Generate execution context for tool calls

    Args:
        tool_name: Name of the tool

    Returns:
        Execution context dictionary
    """
    return {
        "request_id": f"req_{generate_random_id(16)}",
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": random.randint(50, 500),
        "retry_count": random.choice([0, 0, 0, 1])  # Mostly no retries
    }


def generate_debug_info(tool_name: str, result_data: Any) -> Dict[str, Any]:
    """Generate full diagnostic context for tool execution

    Args:
        tool_name: Name of the tool
        result_data: The result data to generate debug info for

    Returns:
        Debug info dictionary with SQL, cache, upstream calls, timing
    """
    # SQL query based on tool type
    sql_queries = {
        "search_flights": "SELECT * FROM flights WHERE origin = $1 AND destination = $2 AND departure_date >= $3",
        "search_hotels": "SELECT * FROM hotels WHERE city = $1 AND check_in >= $2 AND check_out <= $3",
        "fetch_user": "SELECT * FROM users WHERE user_id = $1",
        "get_price": "SELECT * FROM price_history WHERE service_type = $1 AND service_id = $2"
    }

    base_query = sql_queries.get(tool_name, "SELECT * FROM data WHERE id = $1")

    # Generate realistic execution plan
    execution_plan = {
        "scan_type": random.choice(["index_scan", "bitmap_scan", "seq_scan"]),
        "index_used": f"idx_{tool_name}_{random.choice(['primary', 'composite', 'covering'])}",
        "rows_examined": random.randint(100, 5000),
        "rows_returned": random.randint(10, 100)
    }

    # Cache information
    cache_keys = [
        f"{tool_name}:{generate_random_id(8)}",
        f"result:{generate_random_id(8)}"
    ]
    cache_hit = random.choice([True, False])

    # Upstream API calls
    services = {
        "search_flights": [("gds_amadeus", "/v1/shopping/flight-offers"), ("gds_sabre", "/v2/search/flights")],
        "search_hotels": [("booking_api", "/v1/hotels/search"), ("expedia_api", "/v2/properties")],
        "fetch_user": [("user_service", "/v1/users/:id"), ("preference_service", "/v1/preferences")],
    }

    upstream_calls = []
    for service, endpoint in services.get(tool_name, [("data_service", "/v1/query")]):
        upstream_calls.append({
            "service": service,
            "endpoint": endpoint,
            "duration_ms": random.randint(30, 200),
            "status_code": 200,
            "retry_count": 0
        })

    # Performance breakdown
    total_time = random.randint(100, 500)
    timing_breakdown = {
        "parse_request_ms": random.randint(1, 5),
        "validate_params_ms": random.randint(1, 3),
        "query_database_ms": int(total_time * 0.6),
        "fetch_pricing_ms": int(total_time * 0.25),
        "format_response_ms": int(total_time * 0.1)
    }

    # Data lineage
    data_lineage = {
        "source_systems": random.sample(
            ["amadeus_gds", "sabre_gds", "pricing_engine", "user_db", "cache_layer"],
            k=random.randint(2, 4)
        ),
        "last_updated": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
        "data_freshness_seconds": random.randint(60, 1800)
    }

    return {
        "sql_query": base_query,
        "execution_plan": execution_plan,
        "cache_keys": cache_keys,
        "cache_hit": cache_hit,
        "cache_write_time_ms": random.randint(5, 20) if not cache_hit else 0,
        "upstream_calls": upstream_calls,
        "timing_breakdown": timing_breakdown,
        "data_lineage": data_lineage
    }
```

**Update `generate_tool_result()`:**

```python
def generate_tool_result(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    target_size_kb: int = 5,
    tool_call_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a tool execution result with MCP format

    Args:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        data: Tool result data
        target_size_kb: Target size in KB
        tool_call_id: Optional tool call ID to link to tool_use block

    Returns:
        Tool result dictionary in MCP format
    """
    result_data = data or {}
    result_count = len(result_data.get("flights", result_data.get("hotels", [])))

    # Build MCP-style result
    result = {
        "tool_call_id": tool_call_id or f"call_{generate_random_id(16)}",
        "function": {
            "name": tool_name,
            "arguments": generate_tool_parameters(tool_name, "default")
        },
        "result": {
            **result_data,
            "search_metadata": generate_search_metadata(tool_name, result_count) if result_count > 0 else None
        },
        "execution_context": generate_execution_context(tool_name),
        "debug_info": generate_debug_info(tool_name, result_data)
    }

    # Remove None values
    if result["result"]["search_metadata"] is None:
        del result["result"]["search_metadata"]

    # Pad to target size if needed
    current_size = estimate_json_size_kb(result)
    if current_size < target_size_kb:
        padding_needed = int((target_size_kb - current_size) * 1024)
        # Add realistic verbose field instead of "xxx"
        result["debug_info"]["trace_context"] = {
            "span_id": generate_random_id(16),
            "trace_id": generate_random_id(32),
            "additional_context": "x" * padding_needed
        }

    return result
```

### Step 3: Add Agent Planning (`src/payloads.py`)

```python
def generate_agent_plan(agent_type: str, available_tools: List[str]) -> Dict[str, Any]:
    """Generate agent planning structure

    Args:
        agent_type: Type of agent (hotel, flight, primary, etc.)
        available_tools: List of tools available to agent

    Returns:
        Agent plan dictionary
    """
    plans = {
        "hotel": {
            "goal": "Find hotels matching user preferences and budget constraints",
            "steps": [
                "Parse user preferences from context",
                "Apply budget and amenity filters",
                "Search hotels with geographic constraints",
                "Rank results by preference match score",
                "Format top recommendations with details"
            ]
        },
        "flight": {
            "goal": "Search and recommend flights based on travel requirements",
            "steps": [
                "Parse travel dates and destinations",
                "Apply cabin class and airline preferences",
                "Search available flights",
                "Compare fares and connection times",
                "Recommend best value options"
            ]
        },
        "primary": {
            "goal": "Coordinate multi-service booking workflow",
            "steps": [
                "Understand user request and requirements",
                "Identify required services (flights, hotels, etc.)",
                "Delegate to specialist agents",
                "Coordinate timing and availability",
                "Synthesize recommendations"
            ]
        }
    }

    plan_template = plans.get(agent_type, {
        "goal": f"Execute {agent_type} workflow",
        "steps": ["Parse request", "Execute actions", "Return results"]
    })

    return {
        "goal": plan_template["goal"],
        "steps": plan_template["steps"],
        "estimated_steps": len(plan_template["steps"]),
        "parallel_capable": agent_type != "primary",
        "available_tools": available_tools
    }


def generate_agent_reflection(
    agent_type: str,
    success: bool,
    results_count: int,
    filters_applied: List[str]
) -> Dict[str, Any]:
    """Generate agent reflection after execution

    Args:
        agent_type: Type of agent
        success: Whether execution succeeded
        results_count: Number of results returned
        filters_applied: List of filters that were applied

    Returns:
        Agent reflection dictionary
    """
    quality_score = random.uniform(0.75, 0.95) if success else random.uniform(0.3, 0.6)

    suggestions = [
        "Consider expanding search radius for more options",
        "User prefers boutique properties based on history",
        "Price sensitivity detected - prioritize value options",
        "Loyalty program membership could provide benefits"
    ]

    return {
        "success": success,
        "goal_achieved": success and results_count > 0,
        "quality_score": round(quality_score, 2),
        "results_count": results_count,
        "filters_applied": filters_applied,
        "execution_notes": f"Successfully processed {agent_type} request with {len(filters_applied)} filters",
        "suggestions": random.sample(suggestions, k=random.randint(1, 2))
    }
```

### Step 4: Integrate Faker (`src/payloads.py`)

**Add import:**
```python
from faker import Faker
fake = Faker()
```

**Update text generation functions:**

```python
def generate_thinking_block(scenario_context: str, target_tokens: int) -> Dict[str, str]:
    """Generate a thinking content block with natural language reasoning

    Uses Faker to generate varied, realistic reasoning text.
    """
    thinking_templates = {
        "parse": "Let me analyze the user's request to understand what they need. ",
        "refine": "I need to refine the search parameters based on the user's preferences. ",
        # ... other templates
    }

    base_thinking = thinking_templates.get(scenario_context, "Let me think about how to approach this. ")

    # Use Faker to generate natural reasoning
    target_chars = target_tokens * 4
    thinking_text = base_thinking

    while len(thinking_text) < target_chars:
        thinking_text += fake.paragraph(nb_sentences=3) + " "

    return {
        "type": "thinking",
        "thinking": thinking_text[:target_chars].strip()
    }


def generate_text_block(response_context: str, target_tokens: int) -> Dict[str, str]:
    """Generate a text content block with natural language response

    Uses Faker to generate varied, realistic response text.
    """
    response_templates = {
        "results": "Based on the search results, I found several great options that match your criteria. ",
        "analysis": "After analyzing the available options, here's what I recommend. ",
        # ... other templates
    }

    base_response = response_templates.get(response_context, "Here's what I found. ")

    # Use Faker to generate natural explanatory text
    target_chars = target_tokens * 4
    response_text = base_response

    while len(response_text) < target_chars:
        response_text += fake.paragraph(nb_sentences=4) + " "

    return {
        "type": "text",
        "text": response_text[:target_chars].strip()
    }
```

### Step 5: Update Agent Spans (`src/workflow.py`)

**Modify `DelegationStep.execute()`:**

```python
async def execute(self, tracer, parent_span_context, platform: Optional[Platform] = None):
    """Execute delegation step with nested sub-workflow"""
    # Import at function level to avoid circular dependency
    from src.payloads import generate_agent_plan, generate_agent_reflection
    import json

    # Create agent span for this assistant
    with tracer.start_as_current_span(self.name) as span:
        # Set agent attributes
        tools = [step.name for step in self.sub_workflow if isinstance(step, ToolStep)]
        span.set_attribute("gen_ai.agent.tools", json.dumps(tools))
        span.set_attribute("agent.type", self.assistant_type)

        # Add planning structure at start
        plan = generate_agent_plan(self.assistant_type, tools)
        span.set_attribute("agent.plan", json.dumps(plan))

        if platform:
            set_platform_attributes(
                span,
                platform,
                span_type="agent",
                data={"agent_type": self.assistant_type, "tools": tools}
            )

        # Execute sub-workflow steps
        for step in self.sub_workflow:
            await step.execute(tracer, context.get_current(), platform)

        # Add reflection structure at end
        results_count = len([s for s in self.sub_workflow if isinstance(s, ToolStep)])
        filters_applied = [f"filter_{i}" for i in range(random.randint(2, 5))]
        reflection = generate_agent_reflection(
            self.assistant_type,
            success=True,
            results_count=results_count,
            filters_applied=filters_applied
        )
        span.set_attribute("agent.reflection", json.dumps(reflection))
```

## Files Modified

- `src/payloads.py` - Add MCP format, Faker integration, debug info, agent planning
- `src/workflow.py` - Add planning/reflection to DelegationStep
- `pyproject.toml` - Add Faker dependency

## Testing

1. Install Faker: `uv add faker`
2. Run test with all scenarios
3. Verify in Braintrust/LangSmith:
   - Tool results show MCP format with debug_info
   - Agent spans have plan and reflection attributes
   - Text content is varied (not repetitive)
   - Debug info shows SQL, cache, upstream calls

## Success Criteria

✅ Tool results use MCP format with tool_call_id
✅ All tool results include search_metadata and execution_context
✅ Debug info shows SQL queries, cache hits, upstream calls
✅ All agent spans include plan at start
✅ All agent spans include reflection at end
✅ Text blocks use Faker for natural variety
✅ No repetitive "xxx" padding anywhere
✅ Traces look production-realistic in both platforms
