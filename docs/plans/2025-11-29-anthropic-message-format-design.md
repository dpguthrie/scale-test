# Anthropic Message Format Implementation

**Date:** 2025-11-29
**Status:** Approved
**Goal:** Replace simple text payloads with realistic Anthropic API message format

## Problem

Current LLM payloads use simple text strings with "xxx" padding:
```json
{
  "role": "assistant",
  "content": "Simple response" + "xxxxxx padding"
}
```

This doesn't test:
- Complex JSON parsing
- Multi-part responses (thinking + tool use + text)
- Realistic LLM output structure
- Platform handling of structured content

## Solution

Implement Anthropic's multi-part content format with structured content blocks.

### Design Decisions

**Realism Level:** Core multi-part content
- Focus on content array with text/tool_use/thinking types
- Skip full API metadata (stop_reason, detailed usage breakdown)
- Provides structural complexity without over-engineering

**Text Generation:** Template-based (no Faker initially)
- Use scenario-specific template strings
- Can add Faker later as drop-in replacement
- Keeps dependencies minimal

**Tool Format:** Match Anthropic exactly
- Use full tool_use block format: `{type, id, name, input}`
- Generate realistic tool parameters per scenario
- Most authentic to production usage

**Architecture:** Template-based generation
- Pre-define templates for each scenario type
- Predictable, maintainable, easy to debug
- Fast generation without randomness complexity

## Data Structure

### Message Format

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze the user's flight search request..."
    },
    {
      "type": "tool_use",
      "id": "toolu_01A2B3C4D5E6F7G8H9",
      "name": "search_flights",
      "input": {
        "origin": "JFK",
        "destination": "LHR",
        "departure_date": "2025-06-15",
        "cabin_class": "business"
      }
    },
    {
      "type": "text",
      "text": "I found 20 business class flights from New York to London..."
    }
  ]
}
```

### Content Block Types

1. **Thinking Block** - Chain-of-thought reasoning
```json
{"type": "thinking", "thinking": "reasoning text..."}
```

2. **Tool Use Block** - Structured tool calls
```json
{
  "type": "tool_use",
  "id": "toolu_<22_random_chars>",
  "name": "tool_name",
  "input": {...}
}
```

3. **Text Block** - Natural language response
```json
{"type": "text", "text": "response text..."}
```

## Template Mapping by Scenario

### Simple Query (5 spans, ~4800 tokens)
**Assistant message:** 1 text block (2000 tokens)

### Single Service Search (12 spans, ~18K tokens)
**Parse LLM:** 1 thinking (800t) + 1 text (1200t)
**Refine LLM:** 1 thinking (1000t) + 1 tool_use (200t) + 1 text (800t)
**Analyze LLM:** 1 thinking (1500t) + 1 text (1500t)
**Compare/Present LLMs:** Similar multi-block structure

### Delegated Booking (25 spans, ~40K tokens)
**Primary assistant:** thinking + tool_use + text
**Specialist agents:** Multiple thinking + 2-3 tool_use + text

### Complex Multi-Service (80 spans, ~185K tokens)
**Multiple agents:** Extensive thinking + tool_use sequences
**Coordinator:** Multiple thinking blocks with context

## Implementation

### Step 1: Add Content Block Generators (`src/payloads.py`)

```python
def generate_thinking_block(scenario_context: str, target_tokens: int) -> Dict:
    """Generate a thinking content block

    Args:
        scenario_context: What the agent is thinking about
        target_tokens: Target token count for thinking text

    Returns:
        Thinking block dictionary
    """
    # Template strings based on scenario_context
    # Pad to target_tokens using realistic explanations
    return {
        "type": "thinking",
        "thinking": generated_text
    }

def generate_tool_use_block(tool_name: str, scenario_type: str) -> Dict:
    """Generate a tool_use content block with Anthropic format

    Args:
        tool_name: Name of tool (e.g., "search_flights")
        scenario_type: Scenario context for realistic parameters

    Returns:
        Tool use block dictionary
    """
    return {
        "type": "tool_use",
        "id": f"toolu_{generate_random_id(22)}",
        "name": tool_name,
        "input": generate_tool_parameters(tool_name, scenario_type)
    }

def generate_text_block(response_context: str, target_tokens: int) -> Dict:
    """Generate a text content block

    Args:
        response_context: What the response is about
        target_tokens: Target token count for response text

    Returns:
        Text block dictionary
    """
    return {
        "type": "text",
        "text": generated_text
    }
```

### Step 2: Replace `generate_llm_messages()`

```python
def generate_anthropic_messages(
    scenario_name: str,
    span_type: str,
    target_tokens: int
) -> List[Dict]:
    """Generate messages with Anthropic multi-part content format

    Args:
        scenario_name: Scenario (simple_query, single_service_search, etc.)
        span_type: Span role (parse, refine, analyze, etc.)
        target_tokens: Total target tokens for assistant message

    Returns:
        List of message dictionaries with content arrays
    """
    # Look up template for scenario + span_type
    # Generate content blocks based on template
    # Distribute tokens across blocks
    # Return [user_message, assistant_message]
```

### Step 3: Add Tool Parameter Generators

```python
def generate_tool_parameters(tool_name: str, scenario_type: str) -> Dict:
    """Generate realistic tool parameters based on tool and scenario

    Examples:
    - search_flights: origin, destination, dates, cabin_class, filters
    - search_hotels: location, check_in, check_out, amenities, price_range
    - fetch_user_info: user_id, include_preferences, include_history
    """
    # Map tool names to parameter generators
    # Use realistic values (real airports, dates, etc.)
    # Return structured parameter dictionary
```

### Step 4: Update Call Sites in `workflow.py`

```python
# Old:
messages = generate_llm_messages(
    user_prompt="User request",
    assistant_response="Assistant response",
    target_size_kb=self.tokens_out // 100
)

# New:
messages = generate_anthropic_messages(
    scenario_name=scenario.name,
    span_type=self.name,
    target_tokens=self.tokens_out
)
```

## Token Distribution Logic

**Conversion:** 1 token ≈ 4 characters

**Distribution by block type:**
- Thinking blocks: 30% of total tokens
- Tool use blocks: 10% of total tokens (parameters + structure)
- Text blocks: 60% of total tokens

**Content ordering (matches Claude's actual pattern):**
1. Thinking (if present)
2. Tool use(s) (if present)
3. Text (always present)

Never intermix: thinking → text → thinking would be unrealistic.

## Edge Cases

1. **Empty content:** Always include at least one text block
2. **Multiple tool calls:** thinking → tool_use → tool_use → text
3. **Tool ID uniqueness:** Generate unique IDs per tool_use block
4. **Token accuracy:** Target ±10% of specified token counts
5. **Parameter realism:** Use actual airport codes, realistic dates/prices

## Backward Compatibility

**No changes needed:**
- `instrumentation.py` already iterates over message arrays
- Platform attribute mapping handles complex content
- Existing span attribute logic works with nested content
- Test suite should pass without modifications

## Future Extensions

**Easy to add later:**

1. **Faker integration:**
   - Replace template strings with `faker.paragraph()`
   - Add `USE_FAKER` environment variable
   - Drop-in replacement in block generators

2. **Additional block types:**
   - Image blocks for vision scenarios
   - Document blocks for RAG patterns
   - Add new generator functions as needed

3. **Template library:**
   - Move templates to `templates/` directory
   - Load based on scenario + span type
   - Update templates without code changes

4. **Stop reasons and usage:**
   - Add `stop_reason` field to messages
   - Add `usage` breakdown (text_tokens, thinking_tokens)
   - Makes format match Anthropic API exactly

## Validation

**Testing approach:**
1. Generate sample messages for each scenario type
2. Verify JSON structure matches Anthropic format
3. Check token counts within ±10% of targets
4. Run existing test suite (should pass)
5. Visual inspection of generated traces in Braintrust/LangSmith

## Files Modified

- `src/payloads.py` - Add block generators, replace `generate_llm_messages()`
- `src/workflow.py` - Update `LLMStep.execute()` to use new function

## Files Not Modified

- `src/instrumentation.py` - Already handles message arrays correctly
- `src/scenarios.py` - No changes to scenario definitions
- `src/executor.py` - No changes needed
- Test files - Should work without modifications

## Success Criteria

✅ LLM spans show multi-part content in both Braintrust and LangSmith
✅ Tool use blocks display properly in platform UIs
✅ Thinking blocks visible in trace details
✅ Token counts match expected values (±10%)
✅ No artificial "xxx" padding in payloads
✅ All existing tests pass
✅ Traces look realistic compared to production Claude API output
