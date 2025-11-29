"""Platform-specific OTEL configuration for observability backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


class AttributeMapping:
    """Defines how to map semantic attributes to platform-specific attributes"""

    def __init__(self):
        # Attribute mappings for different span types
        self.input_key: Optional[str] = None
        self.output_key: Optional[str] = None
        self.span_type_key: Optional[str] = None
        self.metadata_prefix: Optional[str] = None
        self.span_kind_key: Optional[str] = None

    def map_input(self, value: str) -> Dict[str, Any]:
        """Map input value to platform-specific attributes"""
        if self.input_key:
            return {self.input_key: value}
        return {}

    def map_output(self, value: str) -> Dict[str, Any]:
        """Map output value to platform-specific attributes"""
        if self.output_key:
            return {self.output_key: value}
        return {}

    def map_span_type(self, span_type: str) -> Dict[str, Any]:
        """Map span type to platform-specific attributes"""
        if self.span_type_key:
            return {self.span_type_key: span_type}
        return {}

    def map_span_kind(self, kind: str) -> Dict[str, Any]:
        """Map span kind to platform-specific attributes"""
        if self.span_kind_key:
            return {self.span_kind_key: kind}
        return {}

    def map_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Map metadata dictionary to platform-specific attributes"""
        if not self.metadata_prefix:
            return {}

        result = {}
        for key, value in metadata.items():
            if isinstance(value, (bool, int, float, str)):
                result[f"{self.metadata_prefix}{key}"] = value
        return result


class Platform(ABC):
    """Base class for observability platform configuration"""

    @property
    @abstractmethod
    def endpoint(self) -> Optional[str]:
        """OTLP endpoint URL"""
        pass

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """HTTP headers for OTLP exporter"""
        pass

    @abstractmethod
    def get_span_attributes(self) -> Dict[str, str]:
        """Platform-specific span attributes to add"""
        pass

    @abstractmethod
    def get_attribute_mapping(self) -> AttributeMapping:
        """Get platform-specific attribute mapping configuration"""
        pass


@dataclass
class BraintrustPlatform(Platform):
    """Braintrust OTEL configuration

    Docs: https://www.braintrust.dev/docs/integrations/sdk-integrations/opentelemetry

    Critical requirements:
    - Every trace MUST have a root span
    - Use GenAI semantic conventions OR braintrust.* namespace
    - BatchSpanProcessor recommended for scale
    """
    api_key: str
    project_name: str
    endpoint_url: str = "https://api.braintrust.dev/otel/v1/traces"

    @property
    def endpoint(self) -> str:
        # OTLPSpanExporter requires the full path including /v1/traces
        if not self.endpoint_url.endswith("/v1/traces"):
            return f"{self.endpoint_url}/v1/traces"
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-bt-parent": f"project_name:{self.project_name}"
        }

    def get_span_attributes(self) -> Dict[str, str]:
        """Braintrust-specific attributes for direct field mapping"""
        return {
            "braintrust.project_name": self.project_name,
        }

    def get_attribute_mapping(self) -> AttributeMapping:
        """Braintrust attribute mapping configuration"""
        mapping = AttributeMapping()
        mapping.input_key = "braintrust.input"
        mapping.output_key = "braintrust.output"
        mapping.span_type_key = "braintrust.span_attributes.type"
        mapping.metadata_prefix = None  # Braintrust uses direct attributes, no prefix needed
        mapping.span_kind_key = None  # Not used by Braintrust
        return mapping


@dataclass
class LangSmithPlatform(Platform):
    """LangSmith OTEL configuration

    Docs: https://docs.langchain.com/langsmith/trace-with-opentelemetry

    Critical requirements:
    - Must set langsmith.span.kind attribute (llm/chain/tool/retriever)
    - Supports indexed messages: gen_ai.prompt.{n}.content
    - Or array format: gen_ai.input.messages
    """
    api_key: str
    project_name: str
    endpoint_url: str = "https://api.smith.langchain.com/otel"
    region: str = "us"  # or "eu"

    @property
    def endpoint(self) -> str:
        if self.region == "eu":
            base = "https://eu.api.smith.langchain.com/otel"
        else:
            base = self.endpoint_url

        # OTLPSpanExporter requires the full path including /v1/traces
        if not base.endswith("/v1/traces"):
            return f"{base}/v1/traces"
        return base

    def get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Langsmith-Project": self.project_name
        }

    def get_span_attributes(self) -> Dict[str, str]:
        """LangSmith-specific attributes"""
        return {
            "langsmith.project": self.project_name,
        }

    def get_attribute_mapping(self) -> AttributeMapping:
        """LangSmith attribute mapping configuration"""
        mapping = AttributeMapping()
        mapping.input_key = "input.value"
        mapping.output_key = "output.value"
        mapping.span_type_key = None  # LangSmith doesn't use explicit type attribute
        mapping.metadata_prefix = "langsmith.metadata."
        mapping.span_kind_key = "langsmith.span.kind"
        return mapping


@dataclass
class OTLPPlatform(Platform):
    """Generic OTLP exporter for any compatible backend

    When using OTLP with a collector, attributes for ALL platforms
    are set so the collector can forward to any backend.
    """
    endpoint_url: str
    headers: Optional[Dict[str, str]] = None

    @property
    def endpoint(self) -> str:
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return self.headers or {}

    def get_span_attributes(self) -> Dict[str, str]:
        return {}

    def get_attribute_mapping(self) -> AttributeMapping:
        """OTLP uses a combined mapping for compatibility with multiple backends"""
        # Return empty mapping - we'll set attributes for all platforms directly
        # This allows the collector to forward to any backend
        return AttributeMapping()


@dataclass
class ConsolePlatform(Platform):
    """Console exporter for debugging (no network export)"""

    @property
    def endpoint(self) -> None:
        return None

    def get_headers(self) -> Dict[str, str]:
        return {}

    def get_span_attributes(self) -> Dict[str, str]:
        return {}

    def get_attribute_mapping(self) -> AttributeMapping:
        """Console doesn't need platform-specific attributes"""
        return AttributeMapping()


def get_platform(config: Dict) -> Platform:
    """Factory function to create platform instance from config

    Args:
        config: Configuration dictionary with platform details

    Returns:
        Platform instance

    Raises:
        ValueError: If platform type is unsupported
    """
    platform_type = config.get("platform", "").lower()

    if platform_type == "braintrust":
        return BraintrustPlatform(
            api_key=config["api_key"],
            project_name=config["project_name"]
        )
    elif platform_type == "langsmith":
        return LangSmithPlatform(
            api_key=config["api_key"],
            project_name=config["project_name"]
        )
    elif platform_type == "otlp":
        return OTLPPlatform(
            endpoint_url=config["endpoint"],
            headers=config.get("headers", {})
        )
    elif platform_type == "console":
        return ConsolePlatform()
    else:
        raise ValueError(f"Unsupported platform: {platform_type}")
