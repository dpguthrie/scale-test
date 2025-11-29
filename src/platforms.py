"""Platform-specific OTEL configuration for observability backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


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


@dataclass
class OTLPPlatform(Platform):
    """Generic OTLP exporter for any compatible backend"""
    endpoint_url: str
    headers: Dict[str, str] = None

    @property
    def endpoint(self) -> str:
        return self.endpoint_url

    def get_headers(self) -> Dict[str, str]:
        return self.headers or {}

    def get_span_attributes(self) -> Dict[str, str]:
        return {}


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
