"""Utilities for setting platform-specific span attributes"""

from typing import Any, Dict, List, Optional
from opentelemetry.trace import Span

from src.platforms import Platform, BraintrustPlatform, LangSmithPlatform


def set_io_attributes(
    span: Span,
    input_value: Optional[str] = None,
    output_value: Optional[str] = None,
    platforms: Optional[List[Platform]] = None
):
    """Set input/output attributes for one or more platforms

    When using OTLP collector, pass platforms=None to set attributes for ALL platforms.
    When using direct export, pass the specific platform(s).

    Args:
        span: OpenTelemetry span
        input_value: Input value to set
        output_value: Output value to set
        platforms: List of platforms (None = set for all known platforms)
    """
    if platforms is None:
        # OTLP mode: set attributes for all platforms
        platforms = [BraintrustPlatform("", ""), LangSmithPlatform("", "")]

    for platform in platforms:
        mapping = platform.get_attribute_mapping()

        if input_value:
            for key, value in mapping.map_input(input_value).items():
                span.set_attribute(key, value)

        if output_value:
            for key, value in mapping.map_output(output_value).items():
                span.set_attribute(key, value)


def set_span_type_attributes(
    span: Span,
    span_type: str,
    span_kind: Optional[str] = None,
    platforms: Optional[List[Platform]] = None
):
    """Set span type and kind attributes for one or more platforms

    Args:
        span: OpenTelemetry span
        span_type: Span type (e.g., "task", "llm", "tool", "agent")
        span_kind: Span kind (e.g., "chain") for platforms that use it
        platforms: List of platforms (None = set for all known platforms)
    """
    if platforms is None:
        # OTLP mode: set attributes for all platforms
        platforms = [BraintrustPlatform("", ""), LangSmithPlatform("", "")]

    for platform in platforms:
        mapping = platform.get_attribute_mapping()

        # Set span type
        for key, value in mapping.map_span_type(span_type).items():
            span.set_attribute(key, value)

        # Set span kind if provided
        if span_kind:
            for key, value in mapping.map_span_kind(span_kind).items():
                span.set_attribute(key, value)


def set_metadata_attributes(
    span: Span,
    metadata: Dict[str, Any],
    platforms: Optional[List[Platform]] = None
):
    """Set metadata attributes for one or more platforms

    This handles platform-specific metadata prefixes (like langsmith.metadata.*)
    and also sets the raw attributes for platforms that don't use prefixes.

    Args:
        span: OpenTelemetry span
        metadata: Dictionary of metadata key-value pairs
        platforms: List of platforms (None = set for all known platforms)
    """
    # Always set raw attributes (for platforms like Braintrust that don't use prefixes)
    for key, value in metadata.items():
        if isinstance(value, (bool, int, float, str)):
            span.set_attribute(key, value)

    # Set prefixed attributes for platforms that use them
    if platforms is None:
        # OTLP mode: set attributes for all platforms
        platforms = [BraintrustPlatform("", ""), LangSmithPlatform("", "")]

    for platform in platforms:
        mapping = platform.get_attribute_mapping()
        for key, value in mapping.map_metadata(metadata).items():
            span.set_attribute(key, value)
